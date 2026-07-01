"""Image Captioner transform for enriching chunks with image descriptions.

Performance Optimizations:
1. Only processes images that are actually referenced in chunk text (via [IMAGE: id] placeholder)
2. Uses caption cache to avoid redundant Vision API calls for the same image
3. Skips chunks without image references entirely
4. Parallel processing of unique images with thread-safe caching
"""

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Optional, Dict, Tuple

from src.core.settings import Settings
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput
from src.libs.llm.llm_factory import LLMFactory
from src.observability.logger import get_logger

logger = get_logger(__name__)

# Regex to find image placeholders: [IMAGE: some_id]
IMAGE_PLACEHOLDER_PATTERN = re.compile(r'\[IMAGE:\s*([^\]]+)\]')

# Default max parallel workers for Vision API calls
DEFAULT_MAX_WORKERS = 3  # Lower than text LLM due to higher cost/latency

# Vision providers commonly reject images whose width or height is <= 10px.
# PDF extraction can produce these from table borders, separators, or hairlines.
MIN_CAPTION_IMAGE_DIMENSION = 10


class ImageCaptioner(BaseTransform):
    """Generates captions for images referenced in chunks using Vision LLM.
    
    This transform identifies chunks containing image references, uses a Vision LLM
    to generate descriptive captions, and enriches the chunk text/metadata with
    these captions to improve retrieval for visual content.
    
    Key Features:
    - Only processes images actually referenced in chunk text (not all images in metadata)
    - Caches captions to avoid redundant Vision API calls
    - Thread-safe caption cache for potential future parallelization
    """
    
    def __init__(
        self, 
        settings: Settings, 
        llm: Optional[BaseVisionLLM] = None
    ):
        self.settings = settings
        self.llm = None
        # Caption cache: image_id -> caption string (thread-safe with lock)
        self._caption_cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()
        
        # Check if vision LLM is enabled in settings
        if self.settings.vision_llm and self.settings.vision_llm.enabled:
             try:
                 self.llm = llm or LLMFactory.create_vision_llm(settings)
             except Exception as e:
                 logger.error(f"Failed to initialize Vision LLM: {e}")
                 # We don't raise here to allow pipeline to continue without captioning
                 # effectively falling back to no-op for this transform
        else:
             logger.warning("Vision LLM is disabled or not configured. ImageCaptioner will skip processing.")
        
        self.prompt = self._load_prompt()
        
    def _load_prompt(self) -> str:
        """Load the image captioning prompt from configuration."""
        # Assuming standard relative path. In production, logic might be robust.
        from src.core.settings import resolve_path
        prompt_path = resolve_path("config/prompts/image_captioning.txt")
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        return "Describe this image in detail for indexing purposes."

    def _find_referenced_image_ids(self, text: str) -> List[str]:
        """Extract image IDs actually referenced in the chunk text.
        
        Args:
            text: Chunk text content
            
        Returns:
            List of image IDs found in [IMAGE: id] placeholders
        """
        matches = IMAGE_PLACEHOLDER_PATTERN.findall(text)
        return [m.strip() for m in matches]

    def _get_caption(
        self, 
        img_id: str, 
        img_path: str, 
        trace: Optional[TraceContext] = None
    ) -> Optional[str]:
        """Get caption for an image, using cache if available. Thread-safe.
        
        Args:
            img_id: Image identifier
            img_path: Path to image file
            trace: Optional trace context
            
        Returns:
            Caption string or None if failed
        """
        # Check cache first (thread-safe read)
        with self._cache_lock:
            if img_id in self._caption_cache:
                logger.debug(f"Caption cache hit for image {img_id}")
                return self._caption_cache[img_id]
        
        # Validate path
        if not img_path or not Path(img_path).exists():
            logger.warning(f"Image path not found: {img_path}")
            return None
        
        try:
            image_input = ImageInput(path=img_path)
            response = self.llm.chat_with_image(
                text=self.prompt,
                image=image_input,
                trace=trace
            )
            caption = response.content
            
            # Cache the result (thread-safe write)
            with self._cache_lock:
                self._caption_cache[img_id] = caption
            logger.debug(f"Generated and cached caption for image {img_id}")
            
            return caption
            
        except Exception as e:
            logger.error(f"Failed to caption image {img_path}: {e}")
            return None

    def _metadata_dimensions(self, img_meta: dict) -> Optional[Tuple[int, int]]:
        """Return image dimensions from loader metadata when available."""
        position = img_meta.get("position")
        if not isinstance(position, dict):
            return None

        width = position.get("width")
        height = position.get("height")
        if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
            return None
        if width <= 0 or height <= 0:
            return None
        return int(width), int(height)

    def _file_dimensions(self, img_path: str) -> Optional[Tuple[int, int]]:
        """Read image dimensions from disk when metadata is missing."""
        try:
            from PIL import Image
        except ImportError:
            return None

        try:
            with Image.open(img_path) as img:
                return img.size
        except Exception as e:
            logger.warning(f"Could not inspect image dimensions for {img_path}: {e}")
            return None

    def _is_captionable_image(
        self,
        img_id: str,
        img_meta: dict,
        img_path: str,
        skipped_images: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Skip tiny extracted images that vision APIs will reject."""
        dimensions = self._metadata_dimensions(img_meta) or self._file_dimensions(img_path)
        if not dimensions:
            return True

        width, height = dimensions
        if width <= MIN_CAPTION_IMAGE_DIMENSION or height <= MIN_CAPTION_IMAGE_DIMENSION:
            logger.info(
                "Skipping image %s for captioning due to small dimensions: %sx%s",
                img_id,
                width,
                height,
            )
            if skipped_images is not None:
                skipped_images.append({
                    "id": img_id,
                    "path": img_path,
                    "width": width,
                    "height": height,
                    "reason": "image_too_small",
                })
            return False

        return True

    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Process chunks and add captions for referenced images.
        
        Only processes images that are actually referenced in chunk text
        via [IMAGE: id] placeholders. Uses caching to avoid redundant API calls.
        Parallel processing for unique images.
        """
        if not self.llm:
            return chunks
        
        # Build image lookup from all chunks' metadata
        image_lookup: Dict[str, dict] = {}
        for chunk in chunks:
            if chunk.metadata and "images" in chunk.metadata:
                for img_meta in chunk.metadata.get("images", []):
                    img_id = img_meta.get("id")
                    if img_id and img_id not in image_lookup:
                        image_lookup[img_id] = img_meta
        
        logger.info(f"Found {len(image_lookup)} unique images in document")
        
        # Clear cache for new document processing
        with self._cache_lock:
            self._caption_cache.clear()
        
        # First pass: collect all unique image IDs that need captioning
        images_to_caption: Dict[str, str] = {}  # img_id -> img_path
        skipped_images: List[Dict[str, Any]] = []
        for chunk in chunks:
            referenced_ids = self._find_referenced_image_ids(chunk.text)
            for img_id in referenced_ids:
                if img_id not in images_to_caption:
                    img_meta = image_lookup.get(img_id)
                    if img_meta and img_meta.get("path"):
                        img_path = img_meta.get("path")
                        if self._is_captionable_image(img_id, img_meta, img_path, skipped_images):
                            images_to_caption[img_id] = img_path

        if trace is not None and skipped_images:
            trace.record_stage("image_captioner_skipped_images", {
                "count": len(skipped_images),
                "images": skipped_images,
            })
        
        # Parallel caption generation for all unique images
        if images_to_caption:
            self._generate_captions_parallel(images_to_caption, trace)
        
        # Second pass: apply captions to chunks
        processed_chunks = []
        total_captions_added = 0
        
        for chunk in chunks:
            referenced_ids = self._find_referenced_image_ids(chunk.text)
            
            if not referenced_ids:
                processed_chunks.append(chunk)
                continue
            
            new_text = chunk.text
            captions = []
            
            for img_id in referenced_ids:
                img_id_stripped = img_id.strip()
                
                # Get caption from cache (already populated by parallel processing)
                with self._cache_lock:
                    caption = self._caption_cache.get(img_id_stripped)
                
                if caption:
                    captions.append({"id": img_id_stripped, "caption": caption})
                    
                    placeholder = f"[IMAGE: {img_id}]"
                    replacement = f"[IMAGE: {img_id}]\n(Description: {caption})"
                    new_text = new_text.replace(placeholder, replacement)
                    total_captions_added += 1
                    
            chunk.text = new_text
            
            if captions:
                if "image_captions" not in chunk.metadata:
                    chunk.metadata["image_captions"] = []
                chunk.metadata["image_captions"].extend(captions)
            
            processed_chunks.append(chunk)
        
        with self._cache_lock:
            api_calls = len(self._caption_cache)
        logger.info(f"Added {total_captions_added} captions, API calls: {api_calls}")
            
        return processed_chunks
    
    def _generate_captions_parallel(
        self, 
        images_to_caption: Dict[str, str],
        trace: Optional[TraceContext] = None
    ) -> None:
        """Generate captions for multiple images in parallel.
        
        Args:
            images_to_caption: Dict of img_id -> img_path
            trace: Optional trace context
        """
        if not images_to_caption:
            return
        
        max_workers = min(DEFAULT_MAX_WORKERS, len(images_to_caption))
        logger.debug(f"Generating captions for {len(images_to_caption)} images (max_workers={max_workers})")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._get_caption, img_id, img_path, trace): img_id
                for img_id, img_path in images_to_caption.items()
            }
            
            for future in as_completed(futures):
                img_id = futures[future]
                try:
                    caption = future.result()
                    if caption:
                        logger.debug(f"Caption generated for {img_id}")
                except Exception as e:
                    logger.error(f"Failed to generate caption for {img_id}: {e}")
