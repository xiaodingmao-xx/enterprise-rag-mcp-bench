"""Metadata enrichment transform: rule-based + optional LLM enhancement."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from src.core.settings import Settings, resolve_path
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.base_llm import BaseLLM, Message
from src.observability.logger import get_logger

logger = get_logger(__name__)

# Default max parallel workers for LLM calls
DEFAULT_MAX_WORKERS = 5


class _LegacyMetadataEnricher(BaseTransform):
    """Enriches chunk metadata with title, summary, and tags.
    
    Processing Pipeline:
        1. Rule-based enrichment: Extract basic metadata from content
        2. (Optional) LLM enrichment: Generate semantic-rich metadata
        3. On LLM failure: Gracefully fallback to rule-based metadata
    
    Output Metadata:
        - title: Brief title/heading for the chunk
        - summary: Concise summary of the content
        - tags: List of relevant keywords/topics
        - enriched_by: "rule" or "llm"
    
    Configuration (via settings.yaml):
        - ingestion.metadata_enricher.use_llm: bool - Enable LLM enhancement
        - ingestion.metadata_enricher.prompt_path: str - Custom prompt file path
    
    Design Principles:
        - Graceful Degradation: LLM errors don't block ingestion
        - Atomic Processing: Each chunk processed independently
        - Observable: Records enriched_by in metadata
    """
    
    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseLLM] = None,
        prompt_path: Optional[str] = None
    ):
        """Initialize MetadataEnricher.
        
        Args:
            settings: Application settings
            llm: Optional LLM instance (for testing; auto-created if None)
            prompt_path: Optional custom prompt file path
        """
        self.settings = settings
        self._llm = llm
        self._prompt_template: Optional[str] = None
        self._prompt_path = prompt_path or str(resolve_path("config/prompts/metadata_enrichment.txt"))
        
        # Determine if LLM should be used
        enricher_config = {}
        if hasattr(settings, 'ingestion') and settings.ingestion is not None:
            ingestion_config = settings.ingestion
            # Check if ingestion has metadata_enricher attribute (dataclass) or dict
            if hasattr(ingestion_config, 'metadata_enricher') and ingestion_config.metadata_enricher:
                enricher_config = ingestion_config.metadata_enricher
            elif isinstance(ingestion_config, dict):
                enricher_config = ingestion_config.get('metadata_enricher', {})
        
        self.use_llm = enricher_config.get('use_llm', False) if enricher_config else False
        
    @property
    def llm(self) -> Optional[BaseLLM]:
        """Lazy-load LLM instance."""
        if self.use_llm and self._llm is None:
            try:
                self._llm = LLMFactory.create(self.settings)
                logger.info("LLM initialized for metadata enrichment")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}. Falling back to rule-based only.")
                self.use_llm = False
        return self._llm
    
    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Transform chunks by enriching their metadata.
        
        Args:
            chunks: List of chunks to enrich
            trace: Optional trace context
            
        Returns:
            List of enriched chunks (same length as input)
        """
        if not chunks:
            return []
        
        # Process chunks in parallel if LLM is enabled
        if self.use_llm and self.llm:
            return self._transform_parallel(chunks, trace)
        else:
            return self._transform_sequential(chunks, trace)
    
    def _enrich_single_chunk(
        self, 
        chunk: Chunk, 
        trace: Optional[TraceContext] = None
    ) -> Tuple[Chunk, str, Optional[str]]:
        """Enrich a single chunk. Thread-safe.
        
        Args:
            chunk: Chunk to enrich
            trace: Optional trace context
            
        Returns:
            Tuple of (enriched_chunk, enriched_by, error_message)
        """
        try:
            # Step 1: Rule-based enrichment
            rule_metadata = self._rule_based_enrich(chunk.text)
            
            # Step 2: LLM enhancement
            if self.use_llm and self.llm:
                llm_metadata = self._llm_enrich(chunk.text, trace)
                
                if llm_metadata:
                    enriched_metadata = llm_metadata
                    enriched_by = "llm"
                else:
                    enriched_metadata = rule_metadata
                    enriched_by = "rule"
                    enriched_metadata['enrich_fallback_reason'] = "llm_failed"
            else:
                enriched_metadata = rule_metadata
                enriched_by = "rule"
            
            final_metadata = {
                **(chunk.metadata or {}),
                **enriched_metadata,
                'enriched_by': enriched_by
            }
            
            enriched_chunk = Chunk(
                id=chunk.id,
                text=chunk.text,
                metadata=final_metadata,
                source_ref=chunk.source_ref
            )
            return (enriched_chunk, enriched_by, None)
            
        except Exception as e:
            logger.error(f"Failed to enrich chunk {chunk.id}: {e}")
            text_preview = ""
            if chunk.text:
                text_preview = chunk.text[:100] + '...' if len(chunk.text) > 100 else chunk.text
            minimal_metadata = {
                **(chunk.metadata or {}),
                'title': 'Untitled',
                'summary': text_preview,
                'tags': [],
                'enriched_by': 'error',
                'enrich_error': str(e)
            }
            enriched_chunk = Chunk(
                id=chunk.id,
                text=chunk.text or "",
                metadata=minimal_metadata,
                source_ref=chunk.source_ref
            )
            return (enriched_chunk, "error", str(e))
    
    def _transform_parallel(
        self, 
        chunks: List[Chunk], 
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Process chunks in parallel using ThreadPoolExecutor."""
        max_workers = min(DEFAULT_MAX_WORKERS, len(chunks))
        enriched_chunks = [None] * len(chunks)
        llm_enhanced_count = 0
        fallback_count = 0
        
        logger.debug(f"Processing {len(chunks)} chunks in parallel (max_workers={max_workers})")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._enrich_single_chunk, chunk, trace): idx
                for idx, chunk in enumerate(chunks)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    enriched_chunk, enriched_by, error = future.result()
                    enriched_chunks[idx] = enriched_chunk
                    
                    if enriched_by == "llm":
                        llm_enhanced_count += 1
                    elif enriched_by == "rule" and error is None:
                        fallback_count += 1
                except Exception as e:
                    logger.error(f"Unexpected error in parallel enrichment: {e}")
                    enriched_chunks[idx] = chunks[idx]
        
        success_count = sum(1 for c in enriched_chunks if c is not None)
        
        if trace:
            trace.record_stage("metadata_enricher", {
                "total_chunks": len(chunks),
                "success_count": success_count,
                "llm_enhanced_count": llm_enhanced_count,
                "fallback_count": fallback_count,
                "use_llm": self.use_llm,
                "parallel": True,
                "max_workers": max_workers
            })
        
        logger.info(
            f"Enriched {success_count}/{len(chunks)} chunks "
            f"(LLM: {llm_enhanced_count}, Fallback: {fallback_count})"
        )
        
        return enriched_chunks
    
    def _transform_sequential(
        self, 
        chunks: List[Chunk], 
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Process chunks sequentially (fallback when LLM disabled)."""
        enriched_chunks = []
        success_count = 0
        llm_enhanced_count = 0
        fallback_count = 0
        
        for chunk in chunks:
            try:
                # Step 1: Rule-based enrichment (always performed)
                rule_metadata = self._rule_based_enrich(chunk.text)
                
                # Step 2: Optional LLM enhancement
                if self.use_llm and self.llm:
                    llm_metadata = self._llm_enrich(chunk.text, trace)
                    
                    if llm_metadata:
                        # LLM success
                        enriched_metadata = llm_metadata
                        enriched_by = "llm"
                        llm_enhanced_count += 1
                    else:
                        # LLM failed, fallback to rule-based
                        enriched_metadata = rule_metadata
                        enriched_by = "rule"
                        fallback_count += 1
                        enriched_metadata['enrich_fallback_reason'] = "llm_failed"
                else:
                    # LLM disabled, use rule-based
                    enriched_metadata = rule_metadata
                    enriched_by = "rule"
                
                # Merge enriched metadata with existing metadata
                final_metadata = {
                    **(chunk.metadata or {}),
                    **enriched_metadata,
                    'enriched_by': enriched_by
                }
                
                # Create enriched chunk
                enriched_chunk = Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=final_metadata,
                    source_ref=chunk.source_ref
                )
                enriched_chunks.append(enriched_chunk)
                success_count += 1
                
            except Exception as e:
                # Atomic failure: log and preserve original with minimal metadata
                logger.error(f"Failed to enrich chunk {chunk.id}: {e}")
                # Handle None text case
                text_preview = ""
                if chunk.text:
                    text_preview = chunk.text[:100] + '...' if len(chunk.text) > 100 else chunk.text
                minimal_metadata = {
                    **(chunk.metadata or {}),
                    'title': 'Untitled',
                    'summary': text_preview,
                    'tags': [],
                    'enriched_by': 'error',
                    'enrich_error': str(e)
                }
                enriched_chunk = Chunk(
                    id=chunk.id,
                    text=chunk.text or "",  # Ensure text is not None
                    metadata=minimal_metadata,
                    source_ref=chunk.source_ref
                )
                enriched_chunks.append(enriched_chunk)
        
        # Record trace
        if trace:
            trace.record_stage("metadata_enricher", {
                "total_chunks": len(chunks),
                "success_count": success_count,
                "llm_enhanced_count": llm_enhanced_count,
                "fallback_count": fallback_count,
                "use_llm": self.use_llm,
                "parallel": False
            })
        
        logger.info(
            f"Enriched {success_count}/{len(chunks)} chunks "
            f"(LLM: {llm_enhanced_count}, Fallback: {fallback_count})"
        )
        
        return enriched_chunks
    
    def _rule_based_enrich(self, text: str) -> Dict[str, Any]:
        """Extract metadata using rule-based heuristics.
        
        Args:
            text: Chunk text content
            
        Returns:
            Dictionary with title, summary, tags
            
        Raises:
            TypeError: If text is None
        """
        if text is None:
            raise TypeError("Chunk text cannot be None")
        
        # Extract title from first heading or first line
        title = self._extract_title(text)
        
        # Generate summary from first sentences
        summary = self._extract_summary(text)
        
        # Extract tags from common patterns
        tags = self._extract_tags(text)
        
        return {
            'title': title,
            'summary': summary,
            'tags': tags
        }
    
    def _extract_title(self, text: str) -> str:
        """Extract title from text using heuristics.
        
        Priority:
            1. Markdown heading (# Title)
            2. First line if short enough
            3. First sentence
            4. First N characters
        """
        if not text:
            return "Untitled"
        
        # Check for markdown heading
        heading_match = re.match(r'^#{1,6}\s+(.+)$', text, re.MULTILINE)
        if heading_match:
            return heading_match.group(1).strip()
        
        # Use first line if it's short and looks like a title
        first_line = text.split('\n')[0].strip()
        if first_line and len(first_line) <= 100 and not first_line.endswith(('.', ',', ';')):
            return first_line
        
        # Use first sentence (without trailing punctuation)
        sentences = re.split(r'[.!?]\s+', text)
        if sentences and sentences[0]:
            title = sentences[0].strip()
            # Remove trailing punctuation if present
            title = re.sub(r'[.!?]+$', '', title)
            if len(title) <= 150:
                return title
            return title[:147] + "..."
        
        # Fallback: first 100 chars
        return text[:100].strip() + ("..." if len(text) > 100 else "")
    
    def _extract_summary(self, text: str, max_sentences: int = 3) -> str:
        """Extract summary from text using first N sentences.
        
        Args:
            text: Source text
            max_sentences: Maximum number of sentences to include
            
        Returns:
            Summary text
        """
        if not text:
            return ""
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Take first N sentences
        summary_sentences = sentences[:max_sentences]
        summary = ' '.join(summary_sentences).strip()
        
        # Limit length
        if len(summary) > 500:
            summary = summary[:497] + "..."
        
        return summary
    
    def _extract_tags(self, text: str, max_tags: int = 10) -> List[str]:
        """Extract tags using keyword extraction heuristics.
        
        Args:
            text: Source text
            max_tags: Maximum number of tags to extract
            
        Returns:
            List of tag strings
        """
        if not text:
            return []
        
        tags = set()
        
        # Extract capitalized words (potential proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        tags.update(capitalized[:5])
        
        # Extract code identifiers (camelCase, snake_case)
        identifiers = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b|\b[a-z]+_[a-z_]+\b', text)
        tags.update(identifiers[:5])
        
        # Extract markdown bold/italic terms (potential keywords)
        markdown_keywords = re.findall(r'\*\*(.+?)\*\*|\*(.+?)\*|__(.+?)__|_(.+?)_', text)
        for match in markdown_keywords[:5]:
            for group in match:
                if group:
                    tags.add(group.strip())
        
        # Convert to list and limit
        tag_list = sorted(list(tags))[:max_tags]
        
        return tag_list
    
    def _llm_enrich(
        self,
        text: str,
        trace: Optional[TraceContext] = None
    ) -> Optional[Dict[str, Any]]:
        """Enrich metadata using LLM.
        
        Args:
            text: Chunk text content
            trace: Optional trace context
            
        Returns:
            Dictionary with title, summary, tags, or None on failure
        """
        if not self.llm:
            return None
        
        try:
            # Load prompt template
            prompt = self._load_prompt()
            
            # Build prompt with text
            formatted_prompt = prompt.replace("{chunk_text}", text[:2000])  # Limit text length
            
            # Call LLM
            messages = [Message(role="user", content=formatted_prompt)]
            response = self.llm.chat(messages)
            
            if not response:
                logger.warning("LLM returned empty response for metadata enrichment")
                return None
            
            # Extract text from response (handle both string and ChatResponse object)
            response_text = response
            if hasattr(response, 'content'):
                response_text = response.content
            elif hasattr(response, 'text'):
                response_text = response.text
            elif not isinstance(response, str):
                response_text = str(response)
            
            # Parse LLM response
            metadata = self._parse_llm_response(response_text)
            
            if trace:
                trace.record_stage("llm_enrich", {
                    "success": True,
                    "response_length": len(response_text)
                })
            
            return metadata
            
        except Exception as e:
            logger.warning(f"LLM enrichment failed: {e}")
            if trace:
                trace.record_stage("llm_enrich", {
                    "success": False,
                    "error": str(e)
                })
            return None
    
    def _load_prompt(self) -> str:
        """Load prompt template from file.
        
        Returns:
            Prompt template string
            
        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        if self._prompt_template is not None:
            return self._prompt_template
        
        prompt_path = Path(self._prompt_path)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self._prompt_path}")
        
        self._prompt_template = prompt_path.read_text(encoding='utf-8')
        logger.info(f"Loaded metadata enrichment prompt from {self._prompt_path}")
        
        return self._prompt_template
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured metadata.
        
        Expected format:
            Title: <title>
            Summary: <summary>
            Tags: <tag1>, <tag2>, <tag3>
        
        Args:
            response: LLM response text
            
        Returns:
            Dictionary with title, summary, tags
        """
        metadata = {
            'title': '',
            'summary': '',
            'tags': []
        }
        
        # Extract title
        title_match = re.search(r'Title:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if title_match:
            metadata['title'] = title_match.group(1).strip()
        
        # Extract summary
        summary_match = re.search(r'Summary:\s*(.+?)(?:\n(?:Tags:|$))', response, re.IGNORECASE | re.DOTALL)
        if summary_match:
            metadata['summary'] = summary_match.group(1).strip()
        
        # Extract tags
        tags_match = re.search(r'Tags:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if tags_match:
            tags_text = tags_match.group(1).strip()
            # Split by comma and clean
            tags = [tag.strip() for tag in tags_text.split(',') if tag.strip()]
            metadata['tags'] = tags
        
        # Validation: ensure non-empty values
        if not metadata['title']:
            metadata['title'] = 'Untitled'
        if not metadata['summary']:
            metadata['summary'] = response[:500]  # Fallback to raw response
        
        return metadata


# ============================================================================
# Structured metadata enrichment implementation
# ============================================================================

import threading

from src.ingestion.transform.entity_extractor import extract_entities
from src.ingestion.transform.metadata_cache import (
    MetadataEnrichmentCache,
    build_cache_key,
    sha256_text,
    stable_config_hash,
)
from src.ingestion.transform.metadata_schema import (
    MetadataEnrichmentConfig,
    add_text_companions,
    build_metadata_enrichment_config,
    extract_json_object,
    normalise_page_range,
    normalise_string_list,
    truncate_string,
    validate_enrichment_metadata,
)
from src.ingestion.transform.question_generator import generate_rule_based_questions


class MetadataEnricher(BaseTransform):
    """Structured metadata enrichment with rule, optional LLM, cache, and budget.

    The public class keeps the same import path as the legacy implementation and
    preserves ``enriched_by`` for backward compatibility. New structured fields
    use ``enrichment_*`` names and include Chroma-friendly ``*_text`` companions.
    """

    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseLLM] = None,
        prompt_path: Optional[str] = None,
        cache: Optional[MetadataEnrichmentCache] = None,
    ) -> None:
        self.settings = settings
        self.config = self._load_config(settings)
        if prompt_path is not None:
            self.config = MetadataEnrichmentConfig(
                **{**self.config.__dict__, "prompt_path": prompt_path}
            )
        self._llm = llm
        self._prompt_template: Optional[str] = None
        self._prompt_path = str(resolve_path(self.config.prompt_path))
        self.use_llm = bool(self.config.enabled and self.config.use_llm)
        self.max_concurrency = max(1, int(self.config.max_concurrency))

        self._cache = cache
        if self._cache is None and self.config.cache_enabled:
            self._cache = MetadataEnrichmentCache(self.config.cache_path)

        self._config_hash = stable_config_hash(self.config.cache_relevant_dict())
        self._budget_lock = threading.Lock()
        self._estimated_spend_usd = 0.0
        self._llm_call_count = 0
        self._budget_fallback_count = 0
        self._llm_fallback_count = 0

    @staticmethod
    def _load_config(settings: Settings) -> MetadataEnrichmentConfig:
        raw: dict[str, Any] = {}
        ingestion = getattr(settings, "ingestion", None)
        if ingestion is not None:
            if isinstance(ingestion, dict):
                new_config = ingestion.get("metadata_enrichment")
                legacy_config = ingestion.get("metadata_enricher")
            else:
                new_config = getattr(ingestion, "metadata_enrichment", None)
                legacy_config = getattr(ingestion, "metadata_enricher", None)
            if isinstance(new_config, dict):
                raw = new_config
            elif isinstance(legacy_config, dict):
                raw = {"cache_enabled": False, **legacy_config}
        return build_metadata_enrichment_config(raw if isinstance(raw, dict) else {})

    @property
    def llm(self) -> Optional[BaseLLM]:
        """Lazy-load LLM instance when LLM enrichment is enabled."""

        if self.use_llm and self._llm is None:
            try:
                self._llm = LLMFactory.create(self.settings)
                logger.info("LLM initialized for metadata enrichment")
            except Exception as exc:
                logger.warning(
                    f"Failed to initialize LLM for metadata enrichment: {exc}. "
                    "Falling back to rule-based enrichment."
                )
                self.use_llm = False
        return self._llm

    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None,
    ) -> List[Chunk]:
        """Enrich chunks while isolating per-chunk failures."""

        if not chunks:
            return []
        if not self.config.enabled:
            return chunks

        if self.use_llm and self.llm:
            enriched = self._transform_parallel(chunks, trace)
        else:
            enriched = self._transform_sequential(chunks, trace)

        self._record_trace(chunks, enriched, trace, parallel=bool(self.use_llm and self.llm))
        logger.info(
            "Metadata enrichment complete: chunks=%d, llm_calls=%d, "
            "estimated_cost=%.6f, budget_fallbacks=%d, llm_fallbacks=%d",
            len(chunks),
            self._llm_call_count,
            self._estimated_spend_usd,
            self._budget_fallback_count,
            self._llm_fallback_count,
        )
        return enriched

    def _transform_parallel(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext],
    ) -> List[Chunk]:
        max_workers = min(self.max_concurrency, len(chunks))
        results: list[Optional[Chunk]] = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._enrich_single_chunk, chunk, trace): index
                for index, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    logger.warning(f"Metadata enrichment worker failed: {exc}")
                    results[index] = self._fallback_chunk(chunks[index], str(exc))
        return [result if result is not None else chunks[index] for index, result in enumerate(results)]

    def _transform_sequential(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext],
    ) -> List[Chunk]:
        return [self._enrich_single_chunk(chunk, trace) for chunk in chunks]

    def _enrich_single_chunk(
        self,
        chunk: Chunk,
        trace: Optional[TraceContext] = None,
    ) -> Chunk:
        try:
            rule_metadata = self._rule_based_enrich(chunk)
            cached = self._read_cache(chunk)
            if cached is not None:
                cached["enrichment_cached"] = True
                return self._build_chunk(chunk, cached)

            enrichment = rule_metadata
            fallback_reason: Optional[str] = None

            if self.use_llm and self.llm:
                can_call, budget_cost = self._reserve_budget(chunk.text or "")
                if can_call:
                    llm_metadata, input_truncated = self._llm_enrich(
                        chunk.text or "",
                        rule_metadata,
                        trace,
                    )
                    if llm_metadata is not None:
                        enrichment = self._merge_rule_and_llm(
                            rule_metadata,
                            llm_metadata,
                            input_truncated=input_truncated,
                        )
                    else:
                        self._llm_fallback_count += 1
                        fallback_reason = "llm_failed"
                        enrichment = dict(rule_metadata)
                        enrichment["enrichment_method"] = "fallback"
                else:
                    self._budget_fallback_count += 1
                    fallback_reason = "budget_exceeded"
                    enrichment = dict(rule_metadata)
                    enrichment["enrichment_budget_exceeded"] = True
                    enrichment["estimated_llm_cost_usd"] = round(budget_cost, 8)

            if fallback_reason:
                enrichment["fallback_reason"] = fallback_reason
                enrichment["enrich_fallback_reason"] = fallback_reason

            enrichment["enrichment_cached"] = False
            self._write_cache(chunk, enrichment)
            return self._build_chunk(chunk, enrichment)
        except Exception as exc:
            logger.warning(f"Failed to enrich chunk {getattr(chunk, 'id', '')}: {exc}")
            return self._fallback_chunk(chunk, str(exc))

    def _build_chunk(self, chunk: Chunk, enrichment: Dict[str, Any]) -> Chunk:
        method = enrichment.get("enrichment_method", "rule_based")
        enriched_by = "llm" if method in {"llm", "hybrid"} else "rule"
        if method == "fallback":
            enriched_by = "rule"
        if enrichment.get("_enriched_by_override"):
            enriched_by = str(enrichment["_enriched_by_override"])

        public_enrichment = {
            key: value for key, value in add_text_companions(enrichment).items()
            if not key.startswith("_")
        }
        final_metadata = {
            **(chunk.metadata or {}),
            **public_enrichment,
            "enriched_by": enriched_by,
        }
        return Chunk(
            id=chunk.id,
            text=chunk.text or "",
            metadata=final_metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        )

    def _fallback_chunk(self, chunk: Chunk, reason: str) -> Chunk:
        text = chunk.text or ""
        try:
            metadata = self._rule_based_enrich(chunk)
        except Exception:
            metadata = validate_enrichment_metadata(
                {
                    "title": "Untitled",
                    "summary": truncate_string(text, 240),
                    "tags": [],
                    "enrichment_method": "fallback",
                    "fallback_reason": reason,
                }
            )
        metadata["enrichment_method"] = "fallback"
        metadata["fallback_reason"] = reason
        metadata["enrich_error"] = reason
        if chunk.text is None:
            metadata["_enriched_by_override"] = "error"
        metadata["enrichment_cached"] = False
        return self._build_chunk(chunk, metadata)

    def _cache_identity(self, chunk: Chunk) -> tuple[str, str, str]:
        text_hash = sha256_text(chunk.text or "")
        cache_key = build_cache_key(chunk.id, text_hash, self._config_hash)
        return cache_key, text_hash, self._config_hash

    def _read_cache(self, chunk: Chunk) -> Optional[dict[str, Any]]:
        if not self.config.cache_enabled or self._cache is None:
            return None
        cache_key, _, _ = self._cache_identity(chunk)
        cached = self._cache.get(cache_key)
        if cached is None:
            return None
        return validate_enrichment_metadata(cached)

    def _write_cache(self, chunk: Chunk, enrichment: dict[str, Any]) -> None:
        if not self.config.cache_enabled or self._cache is None:
            return
        cache_key, text_hash, config_hash = self._cache_identity(chunk)
        to_store = dict(enrichment)
        to_store["enrichment_cached"] = False
        self._cache.set(
            cache_key=cache_key,
            chunk_id=chunk.id,
            text_hash=text_hash,
            config_hash=config_hash,
            enrichment_result=to_store,
        )

    def _reserve_budget(self, text: str) -> tuple[bool, float]:
        estimated_tokens = self._estimate_tokens(text) + 512
        estimated_cost = (
            estimated_tokens / 1000.0 * self.config.estimated_cost_per_1k_tokens
        )
        with self._budget_lock:
            if self._estimated_spend_usd + estimated_cost > self.config.budget_usd_per_run:
                return False, estimated_cost
            self._estimated_spend_usd += estimated_cost
            self._llm_call_count += 1
            return True, estimated_cost

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, int(len(text or "") / 4))

    def _rule_based_enrich(self, chunk_or_text: Any) -> Dict[str, Any]:
        """Generate full structured metadata without calling LLM."""

        if isinstance(chunk_or_text, Chunk):
            chunk = chunk_or_text
            text = chunk.text
            source_metadata = chunk.metadata or {}
        else:
            chunk = None
            text = chunk_or_text
            source_metadata = {}

        if text is None:
            raise TypeError("Chunk text cannot be None")

        heading_path = self._extract_heading_path(text, source_metadata)
        section_path = normalise_string_list(
            source_metadata.get("section_path") or heading_path,
            limit=5,
        )
        title = self._extract_title(text, source_metadata, heading_path)
        summary = self._extract_summary(text)
        page_range = self._extract_page_range(source_metadata)
        table_ids = self._extract_table_ids(text, source_metadata)
        image_ids = self._extract_image_ids(text, source_metadata)
        entities = (
            extract_entities(text, source_metadata, 15)
            if self.config.extract_entities
            else []
        )
        tags = self._extract_tags(
            text,
            source_metadata,
            title=title,
            heading_path=heading_path,
            entities=entities,
        )
        questions = generate_rule_based_questions(
            title=title,
            summary=summary,
            entities=entities,
            tags=tags,
            enabled=self.config.generate_questions,
            max_questions=3,
        )

        return validate_enrichment_metadata(
            {
                "title": title,
                "summary": summary,
                "tags": tags,
                "section_path": section_path,
                "heading_path": heading_path,
                "page_range": page_range,
                "table_ids": table_ids,
                "image_ids": image_ids,
                "entities": entities,
                "questions": questions,
                "enrichment_method": "rule_based",
                "enrichment_cached": False,
            }
        )

    def _extract_title(
        self,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
        heading_path: Optional[list[str]] = None,
    ) -> str:
        metadata = metadata or {}
        heading_path = heading_path or []
        if metadata.get("title"):
            return truncate_string(metadata["title"], 60)
        if heading_path:
            return truncate_string(heading_path[-1], 60)

        heading_match = re.search(r"^#{1,6}\s+(.+)$", text or "", re.MULTILINE)
        if heading_match:
            return truncate_string(heading_match.group(1), 60)

        for line in (text or "").splitlines()[:5]:
            stripped = line.strip()
            if stripped and len(stripped) <= 100 and not stripped.endswith((".", "。", ",", "，")):
                return truncate_string(stripped.lstrip("#").strip(), 60)

        sentence_match = re.search(r"(.+?)[。！？.!?](?:\s|$)", text or "")
        if sentence_match:
            return truncate_string(sentence_match.group(1), 60)

        if not (text or "").strip():
            return "Untitled"

        source_path = metadata.get("source_path") or metadata.get("file_name")
        if source_path:
            return truncate_string(Path(str(source_path)).stem, 60)
        return "Untitled"

    def _extract_summary(self, text: str, max_sentences: int = 3) -> str:
        if not text:
            return ""
        cleaned = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        sentences = re.split(r"(?<=[。！？.!?])\s+", cleaned)
        selected = " ".join(sentence for sentence in sentences[:max_sentences] if sentence)
        return truncate_string(selected or cleaned, 240)

    def _extract_tags(
        self,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
        *,
        title: str = "",
        heading_path: Optional[list[str]] = None,
        entities: Optional[list[str]] = None,
        max_tags: int = 8,
    ) -> List[str]:
        metadata = metadata or {}
        if not (text or "").strip():
            return []

        candidates: list[str] = []
        for key in ("file_type", "doc_type", "language"):
            if metadata.get(key):
                candidates.append(str(metadata[key]))
        candidates.extend(heading_path or [])
        if title:
            candidates.append(title)
        candidates.extend((entities or [])[:5])

        lower = (text or "").lower()
        keyword_map = {
            "error": ("error", "exception", "错误", "报错", "http 4", "http 5"),
            "config": ("config", "configuration", "配置", "settings.yaml"),
            "deployment": ("deploy", "deployment", "docker", "部署"),
            "api": ("api", "/api/"),
            "version": ("version", "v1.", "版本"),
            "table": ("|", "[table:"),
            "image": ("[image:", "image_id"),
        }
        for tag, markers in keyword_map.items():
            if any(marker in lower for marker in markers):
                candidates.append(tag)

        identifiers = re.findall(r"\b[a-z]+(?:[A-Z][A-Za-z0-9]*)+\b|\b[a-z]+_[a-z0-9_]+\b", text or "")
        candidates.extend(identifiers[:5])
        capitalized_terms = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text or "")
        candidates.extend(capitalized_terms[:8])
        return normalise_string_list(candidates, limit=max_tags)

    def _extract_heading_path(
        self,
        text: str,
        metadata: dict[str, Any],
    ) -> list[str]:
        existing = metadata.get("heading_path") or metadata.get("section_path")
        if existing:
            return normalise_string_list(existing, limit=5)

        stack: list[str] = []
        for match in re.finditer(r"^(#{1,6})\s+(.+)$", text or "", re.MULTILINE):
            level = len(match.group(1))
            heading = truncate_string(match.group(2), 80)
            stack = stack[: level - 1]
            stack.append(heading)
        return stack[-5:]

    @staticmethod
    def _extract_page_range(metadata: dict[str, Any]) -> dict[str, int] | None:
        if metadata.get("page_range"):
            return normalise_page_range(metadata.get("page_range"))
        if metadata.get("page_start") is not None or metadata.get("page_end") is not None:
            return normalise_page_range(
                {
                    "start": metadata.get("page_start"),
                    "end": metadata.get("page_end", metadata.get("page_start")),
                }
            )
        for key in ("page_num", "page", "page_number"):
            if metadata.get(key) is not None:
                return normalise_page_range(metadata.get(key))
        return None

    @staticmethod
    def _extract_table_ids(text: str, metadata: dict[str, Any]) -> list[str]:
        values: list[Any] = []
        for key in ("table_ids", "table_refs", "tables"):
            item = metadata.get(key)
            if isinstance(item, list):
                values.extend(item)
            elif item:
                values.append(item)
        values.extend(re.findall(r"\[TABLE:\s*([^\]]+)\]", text or "", flags=re.IGNORECASE))
        return normalise_string_list(values, limit=50)

    @staticmethod
    def _extract_image_ids(text: str, metadata: dict[str, Any]) -> list[str]:
        values: list[Any] = []
        for key in ("image_ids", "image_refs", "images"):
            item = metadata.get(key)
            if isinstance(item, list):
                values.extend(item)
            elif item:
                values.append(item)
        values.extend(re.findall(r"\[IMAGE:\s*([^\]]+)\]", text or "", flags=re.IGNORECASE))
        return normalise_string_list(values, limit=50)

    def _llm_enrich(
        self,
        text: str,
        rule_metadata: dict[str, Any],
        trace: Optional[TraceContext] = None,
    ) -> tuple[Optional[dict[str, Any]], bool]:
        if not self.llm:
            return None, False
        try:
            llm_input, truncated = self._truncate_for_llm(text)
            prompt = self._load_prompt().replace("{chunk_text}", llm_input)
            messages = [Message(role="user", content=prompt)]
            response = self.llm.chat(messages)
            response_text = getattr(response, "content", response)
            response_text = str(response_text)
            if not response_text.strip():
                raise ValueError("empty LLM response")
            if (
                self.config.output_schema.lower() == "json"
                and "{" not in response_text
                and not re.search(r"^\s*Title:", response_text, flags=re.IGNORECASE | re.MULTILINE)
            ):
                raise ValueError("LLM response does not contain JSON metadata")
            parsed = self._parse_llm_response(response_text, fallback=rule_metadata)
            if trace is not None:
                trace.record_stage(
                    "llm_enrich",
                    {"success": True, "input_truncated": truncated},
                )
            return parsed, truncated
        except Exception as exc:
            logger.warning(f"LLM metadata enrichment failed: {exc}")
            if trace is not None:
                trace.record_stage("llm_enrich", {"success": False, "error": str(exc)})
            return None, False

    def _merge_rule_and_llm(
        self,
        rule_metadata: dict[str, Any],
        llm_metadata: dict[str, Any],
        *,
        input_truncated: bool,
    ) -> dict[str, Any]:
        merged = validate_enrichment_metadata(llm_metadata, fallback=rule_metadata)
        for key in ("heading_path", "page_range"):
            if not merged.get(key):
                merged[key] = rule_metadata.get(key)
        for key in ("tags", "entities", "questions", "section_path", "table_ids", "image_ids"):
            merged[key] = normalise_string_list(
                list(merged.get(key, [])) + list(rule_metadata.get(key, [])),
                limit={
                    "tags": 8,
                    "entities": 15,
                    "questions": 5,
                    "section_path": 5,
                    "table_ids": 50,
                    "image_ids": 50,
                }[key],
            )
        merged["enrichment_method"] = "hybrid"
        merged["enrichment_cached"] = False
        merged["enrichment_input_truncated"] = input_truncated
        return add_text_companions(merged)

    def _truncate_for_llm(self, text: str) -> tuple[str, bool]:
        max_chars = max(1, self.config.max_tokens_per_chunk * 4)
        if len(text or "") <= max_chars:
            return text or "", False

        head_chars = int(max_chars * 0.75)
        head = (text or "")[:head_chars].rstrip()
        preserved: list[str] = []
        for line in (text or "").splitlines():
            if re.search(r"\[(?:IMAGE|TABLE):", line, flags=re.IGNORECASE):
                preserved.append(line.strip())
            elif extract_entities(line, {}, 3):
                preserved.append(line.strip())
            if len("\n".join(preserved)) > max_chars - head_chars - 80:
                break

        suffix = "\n\n[TRUNCATED: preserved references]\n" + "\n".join(preserved)
        truncated = (head + suffix)[:max_chars]
        return truncated, True

    def _load_prompt(self) -> str:
        if self._prompt_template is not None:
            return self._prompt_template
        prompt_path = Path(self._prompt_path)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self._prompt_path}")
        self._prompt_template = prompt_path.read_text(encoding="utf-8")
        return self._prompt_template

    def _parse_llm_response(
        self,
        response: str,
        fallback: Optional[dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            parsed = extract_json_object(response)
            return validate_enrichment_metadata(parsed, fallback=fallback)
        except Exception:
            legacy = self._parse_legacy_response(response)
            return validate_enrichment_metadata(legacy, fallback=fallback)

    @staticmethod
    def _parse_legacy_response(response: str) -> dict[str, Any]:
        metadata: dict[str, Any] = {"title": "", "summary": "", "tags": []}
        title_match = re.search(r"Title:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        summary_match = re.search(
            r"Summary:\s*(.+?)(?:\n(?:Tags:|$))",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if summary_match:
            metadata["summary"] = summary_match.group(1).strip()
        tags_match = re.search(r"Tags:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if tags_match:
            metadata["tags"] = [
                tag.strip() for tag in tags_match.group(1).split(",") if tag.strip()
            ]
        if not metadata["summary"]:
            metadata["summary"] = response[:240]
        return metadata

    def _record_trace(
        self,
        original_chunks: List[Chunk],
        enriched_chunks: List[Chunk],
        trace: Optional[TraceContext],
        *,
        parallel: bool,
    ) -> None:
        if trace is None:
            return
        llm_count = sum(
            1 for chunk in enriched_chunks
            if chunk.metadata.get("enrichment_method") in {"llm", "hybrid"}
        )
        fallback_count = sum(
            1 for chunk in enriched_chunks
            if chunk.metadata.get("enrichment_method") == "fallback"
        )
        cache_hits = sum(1 for chunk in enriched_chunks if chunk.metadata.get("enrichment_cached"))
        trace.record_stage(
            "metadata_enricher",
            {
                "total_chunks": len(original_chunks),
                "success_count": len(enriched_chunks),
                "llm_enhanced_count": llm_count,
                "fallback_count": fallback_count,
                "cache_hits": cache_hits,
                "use_llm": self.use_llm,
                "parallel": parallel,
                "max_concurrency": self.max_concurrency,
                "estimated_spend_usd": round(self._estimated_spend_usd, 8),
            },
        )
