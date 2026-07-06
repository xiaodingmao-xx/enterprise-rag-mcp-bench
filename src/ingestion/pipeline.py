"""Ingestion Pipeline orchestrator for the Modular RAG MCP Server.

This module implements the main pipeline that orchestrates the complete
document ingestion flow:
    1. File Integrity Check (SHA256 skip check)
    2. Document Quality Check (PDF text-layer preflight)
    3. Document Loading (multi-format file → Document)
    4. Chunking (Document → Chunks)
    5. Transform (Refine + Enrich + Caption)
    6. Encoding (Dense + Sparse vectors)
    7. Storage (VectorStore + BM25 Index + ImageStorage)

Design Principles:
- Config-Driven: All components configured via settings.yaml
- Observable: Logs progress and stage completion
- Graceful Degradation: LLM failures don't block pipeline
- Idempotent: SHA256-based skip for unchanged files
"""

from pathlib import Path
from typing import Callable, List, Optional, Dict, Any
import hashlib
import time

from src.core.settings import Settings, load_settings, resolve_path
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.observability.logger import get_logger

# Libs layer imports
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.libs.loader.document_quality import (
    DOCUMENT_QUALITY_REJECTION_MESSAGE,
    DocumentQualityReport,
    PdfQualityChecker,
)
from src.libs.loader.loader_factory import LoaderFactory
from src.libs.embedding.embedding_factory import EmbeddingFactory

# Ingestion layer imports
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.embedding.batch_processor import BatchProcessor, BatchResult
from src.ingestion.embedding.embedding_cache import SQLiteEmbeddingCache
from src.ingestion.storage.sparse_indexer_factory import (
    create_sparse_indexer,
    get_sparse_backend,
)
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.ingestion.storage.image_storage import ImageStorage

logger = get_logger(__name__)


class PipelineResult:
    """Result of pipeline execution with detailed statistics.
    
    Attributes:
        success: Whether pipeline completed successfully
        file_path: Path to the processed file
        doc_id: Document ID (SHA256 hash)
        chunk_count: Number of chunks generated
        image_count: Number of images processed
        vector_ids: List of vector IDs stored
        error: Error message if pipeline failed
        stages: Dict of stage names to their individual results
    """
    
    def __init__(
        self,
        success: bool,
        file_path: str,
        doc_id: Optional[str] = None,
        chunk_count: int = 0,
        image_count: int = 0,
        vector_ids: Optional[List[str]] = None,
        error: Optional[str] = None,
        stages: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.file_path = file_path
        self.doc_id = doc_id
        self.chunk_count = chunk_count
        self.image_count = image_count
        self.vector_ids = vector_ids or []
        self.error = error
        self.stages = stages or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "file_path": self.file_path,
            "doc_id": self.doc_id,
            "chunk_count": self.chunk_count,
            "image_count": self.image_count,
            "vector_ids_count": len(self.vector_ids),
            "error": self.error,
            "stages": self.stages
        }


class IngestionPipeline:
    """Main pipeline orchestrator for document ingestion.
    
    This class coordinates all stages of the ingestion process:
    - File integrity checking for incremental processing
    - Document quality preflight for noisy or unreadable PDFs
    - Document loading (PDF with image extraction)
    - Text chunking with configurable splitter
    - Chunk refinement (rule-based + LLM)
    - Metadata enrichment (rule-based + LLM)
    - Image captioning (Vision LLM)
    - Dense embedding (Azure text-embedding-ada-002)
    - Sparse encoding (BM25 term statistics)
    - Vector storage (ChromaDB)
    - BM25 index building
    
    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings("config/settings.yaml")
        >>> pipeline = IngestionPipeline(settings)
        >>> result = pipeline.run("documents/report.pdf", collection="contracts")
        >>> print(f"Processed {result.chunk_count} chunks")
    """
    
    def __init__(
        self,
        settings: Settings,
        collection: str = "default",
        force: bool = False
    ):
        """Initialize pipeline with all components.
        
        Args:
            settings: Application settings from settings.yaml
            collection: Collection name for organizing documents
            force: If True, re-process even if file was previously processed
        """
        self.settings = settings
        self.collection = collection
        self.force = force
        
        # Initialize all components
        logger.info("Initializing Ingestion Pipeline components...")
        
        # Stage 1: File Integrity
        self.integrity_checker = SQLiteIntegrityChecker(db_path=str(resolve_path("data/db/ingestion_history.db")))
        logger.info("  ✓ FileIntegrityChecker initialized")

        # Stage 2: Document Quality
        quality_config = {}
        if settings.ingestion is not None:
            if isinstance(settings.ingestion, dict):
                maybe_quality_config = settings.ingestion.get("document_quality")
            else:
                maybe_quality_config = getattr(settings.ingestion, "document_quality", None)
            if isinstance(maybe_quality_config, dict):
                quality_config = maybe_quality_config

        self.document_quality_checker = PdfQualityChecker(
            enabled=quality_config.get("enabled", True),
            sample_pages=quality_config.get("sample_pages", 3),
            min_effective_char_ratio=quality_config.get("min_effective_char_ratio", 0.8),
            min_recognizable_text_density=quality_config.get(
                "min_recognizable_text_density",
                20.0,
            ),
        )
        logger.info(
            "  ✓ PdfQualityChecker initialized "
            f"(enabled={self.document_quality_checker.enabled}, "
            f"min_ratio={self.document_quality_checker.min_effective_char_ratio})"
        )
        
        # Stage 3: LoaderFactory
        self.loader = None
        self.loader_factory = LoaderFactory
        self.loader_options = {
            "extract_images": True,
            "image_storage_dir": str(resolve_path(f"data/images/{collection}")),
        }
        supported_extensions = self.loader_factory.get_supported_extensions(settings)
        logger.info(
            "  ✓ LoaderFactory initialized "
            f"(supported={', '.join(supported_extensions)})"
        )
        
        # Stage 4: Chunker
        self.chunker = DocumentChunker(settings)
        logger.info("  ✓ DocumentChunker initialized")
        
        # Stage 5: Transforms
        self.chunk_refiner = ChunkRefiner(settings)
        logger.info(f"  ✓ ChunkRefiner initialized (use_llm={self.chunk_refiner.use_llm})")
        
        self.metadata_enricher = MetadataEnricher(settings)
        logger.info(f"  ✓ MetadataEnricher initialized (use_llm={self.metadata_enricher.use_llm})")
        
        self.image_captioner = ImageCaptioner(settings)
        has_vision = self.image_captioner.llm is not None
        logger.info(f"  ✓ ImageCaptioner initialized (vision_enabled={has_vision})")
        
        # Stage 6: Encoders
        embedding = EmbeddingFactory.create(settings)
        batch_size = settings.ingestion.batch_size if settings.ingestion else 100
        self.dense_encoder = DenseEncoder(embedding, batch_size=batch_size)
        logger.info(f"  ✓ DenseEncoder initialized (provider={settings.embedding.provider})")
        
        self.sparse_encoder = SparseEncoder()
        logger.info("  ✓ SparseEncoder initialized")
        
        self.batch_processor = BatchProcessor(
            dense_encoder=self.dense_encoder,
            sparse_encoder=self.sparse_encoder,
            batch_size=batch_size
        )
        logger.info(f"  ✓ BatchProcessor initialized (batch_size={batch_size})")
        
        embedding_cache_config = getattr(
            getattr(settings, "performance", None),
            "embedding_cache",
            None,
        )
        embedding_cache_enabled = (
            True if embedding_cache_config is None
            else getattr(embedding_cache_config, "enabled", True)
        )
        self.embedding_cache: Optional[SQLiteEmbeddingCache] = None
        if embedding_cache_enabled:
            cache_db_path = (
                "./data/db/embedding_cache.db"
                if embedding_cache_config is None
                else getattr(
                    embedding_cache_config,
                    "db_path",
                    "./data/db/embedding_cache.db",
                )
            )
            self.embedding_cache = SQLiteEmbeddingCache(cache_db_path)
            logger.info(f"  EmbeddingCache initialized ({cache_db_path})")
        else:
            logger.info("  EmbeddingCache disabled")

        # Stage 7: Storage
        self.vector_upserter = VectorUpserter(settings, collection_name=collection)
        logger.info(f"  ✓ VectorUpserter initialized (provider={settings.vector_store.provider}, collection={collection})")
        
        self.sparse_indexer = create_sparse_indexer(settings, collection=collection)
        self.bm25_indexer = self.sparse_indexer
        logger.info(f"  SparseIndexer initialized (backend={get_sparse_backend(settings)})")
        
        self.image_storage = ImageStorage(
            db_path=str(resolve_path("data/db/image_index.db")),
            images_root=str(resolve_path("data/images"))
        )
        logger.info("  ✓ ImageStorage initialized")
        
        logger.info("Pipeline initialization complete!")

    def _process_embeddings(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None,
    ) -> BatchResult:
        """Encode chunks while reusing cached dense vectors when available."""
        if self.embedding_cache is None:
            return self.batch_processor.process(chunks, trace)

        start_time = time.perf_counter()
        self.embedding_cache.annotate_chunks(chunks)

        provider = self.settings.embedding.provider
        model = self.settings.embedding.model
        dimensions = self.settings.embedding.dimensions

        cached_vectors = self.embedding_cache.get_many(
            chunks,
            collection=self.collection,
            provider=provider,
            model=model,
            dimensions=dimensions,
        )

        dense_vectors: List[Optional[List[float]]] = [None] * len(chunks)
        for idx, vector in cached_vectors.items():
            dense_vectors[idx] = vector

        missing_indices = [idx for idx, vector in enumerate(dense_vectors) if vector is None]
        missing_chunks = [chunks[idx] for idx in missing_indices]
        computed_vectors: List[List[float]] = []

        if missing_chunks:
            computed_vectors = self.dense_encoder.encode(missing_chunks, trace=trace)
            self.embedding_cache.set_many(
                missing_chunks,
                computed_vectors,
                collection=self.collection,
                provider=provider,
                model=model,
                dimensions=dimensions,
            )

            for idx, vector in zip(missing_indices, computed_vectors):
                dense_vectors[idx] = vector

        sparse_stats = self.sparse_encoder.encode(chunks, trace=trace)
        total_time = time.perf_counter() - start_time

        if trace is not None:
            trace.record_stage(
                "embedding_cache",
                {
                    "enabled": True,
                    "cache_hits": len(cached_vectors),
                    "cache_misses": len(missing_indices),
                    "provider": provider,
                    "model": model,
                    "dimensions": dimensions,
                },
            )
            trace.record_stage(
                "batch_processing",
                {
                    "total_chunks": len(chunks),
                    "dense_batches": self.dense_encoder.get_batch_count(len(missing_chunks)),
                    "batch_size": self.dense_encoder.batch_size,
                    "successful_chunks": len(chunks),
                    "failed_chunks": 0,
                    "total_time_seconds": total_time,
                    "cache_hits": len(cached_vectors),
                    "cache_misses": len(missing_indices),
                },
            )

        resolved_dense_vectors = [
            vector for vector in dense_vectors
            if vector is not None
        ]
        if len(resolved_dense_vectors) != len(chunks):
            raise RuntimeError("Embedding cache did not resolve all dense vectors")

        return BatchResult(
            dense_vectors=resolved_dense_vectors,
            sparse_stats=sparse_stats,
            batch_count=self.dense_encoder.get_batch_count(len(missing_chunks)),
            total_time=total_time,
            successful_chunks=len(chunks),
            failed_chunks=0,
            cache_hits=len(cached_vectors),
            cache_misses=len(missing_indices),
        )

    def _delete_existing_vectors_for_source(
        self,
        source_path: str,
        trace: Optional[TraceContext] = None,
    ) -> int:
        """Remove stale vector records for a source before writing current chunks."""
        delete_by_metadata = getattr(
            self.vector_upserter.vector_store,
            "delete_by_metadata",
            None,
        )
        if not callable(delete_by_metadata):
            return 0

        try:
            return int(delete_by_metadata({"source_path": source_path}, trace=trace))
        except Exception as exc:
            logger.warning(f"Failed to delete stale vectors for {source_path}: {exc}")
            return 0

    def _get_loader_for_file(self, file_path: Path):
        """Create the proper loader for a file, unless a test injected one."""

        injected_loader = getattr(self, "loader", None)
        if injected_loader is not None:
            return injected_loader

        return self.loader_factory.get_loader(
            file_path,
            settings=self.settings,
            **getattr(self, "loader_options", {}),
        )
    
    def run(
        self,
        file_path: str,
        trace: Optional[TraceContext] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> PipelineResult:
        """Execute the full ingestion pipeline on a file.
        
        Args:
            file_path: Path to the file to process (e.g., PDF)
            trace: Optional trace context for observability
            on_progress: Optional callback ``(stage_name, current, total)``
                invoked when each pipeline stage completes.  *current* is
                the 1-based index of the completed stage; *total* is the
                number of stages (currently 7).
        
        Returns:
            PipelineResult with success status and statistics
        """
        file_path = Path(file_path)
        stages: Dict[str, Any] = {}
        _total_stages = 7

        def _notify(stage_name: str, step: int) -> None:
            if on_progress is not None:
                on_progress(stage_name, step, _total_stages)
        
        logger.info(f"=" * 60)
        logger.info(f"Starting Ingestion Pipeline for: {file_path}")
        logger.info(f"Collection: {self.collection}")
        logger.info(f"=" * 60)
        
        try:
            # ─────────────────────────────────────────────────────────────
            # Stage 1: File Integrity Check
            # ─────────────────────────────────────────────────────────────
            logger.info("\n📋 Stage 1: File Integrity Check")
            _notify("integrity", 1)
            
            _t0 = time.monotonic()
            file_hash = self.integrity_checker.compute_sha256(str(file_path))
            logger.info(f"  File hash: {file_hash[:16]}...")
            
            if not self.force and self.integrity_checker.should_skip(file_hash):
                logger.info(f"  ⏭️  File already processed, skipping (use force=True to reprocess)")
                integrity_data = {
                    "method": "sha256",
                    "file_hash": file_hash,
                    "skipped": True,
                    "reason": "already_processed",
                    "message": "File already processed; ingestion skipped.",
                    "collection": self.collection,
                }
                stages["integrity"] = integrity_data
                if trace is not None:
                    trace.metadata["ingestion_status"] = "skipped"
                    trace.metadata["skip_reason"] = "already_processed"
                    trace.record_stage(
                        "integrity",
                        integrity_data,
                        elapsed_ms=(time.monotonic() - _t0) * 1000.0,
                    )
                return PipelineResult(
                    success=True,
                    file_path=str(file_path),
                    doc_id=file_hash,
                    stages=stages,
                )
            
            stages["integrity"] = {
                "method": "sha256",
                "file_hash": file_hash,
                "skipped": False,
                "collection": self.collection,
            }
            if trace is not None:
                trace.record_stage(
                    "integrity",
                    stages["integrity"],
                    elapsed_ms=(time.monotonic() - _t0) * 1000.0,
                )
            logger.info("  ✓ File needs processing")
            
            # ─────────────────────────────────────────────────────────────
            # Stage 2: Document Quality Check
            # ─────────────────────────────────────────────────────────────
            logger.info("\n🔎 Stage 2: Document Quality Check")
            _notify("quality", 2)

            _t0 = time.monotonic()
            quality_checker = getattr(self, "document_quality_checker", None)
            if file_path.suffix.lower() != ".pdf":
                quality_report = DocumentQualityReport(
                    checked=False,
                    passed=True,
                    reason="non_pdf_quality_check_skipped",
                    file_path=str(file_path),
                )
            elif quality_checker is not None:
                quality_report = quality_checker.check(file_path)
            else:
                quality_report = DocumentQualityReport(
                    checked=False,
                    passed=True,
                    reason="checker_unavailable",
                    file_path=str(file_path),
                )
            _elapsed = (time.monotonic() - _t0) * 1000.0

            quality_data = quality_report.to_dict()
            quality_data["method"] = "pdf_quality_checker"
            stages["quality"] = quality_data
            if trace is not None:
                trace.record_stage("quality", quality_data, elapsed_ms=_elapsed)

            if quality_report.passed:
                logger.info(
                    "  ✓ Quality check passed "
                    f"(ratio={quality_report.effective_char_ratio}, "
                    f"density={quality_report.recognizable_text_density})"
                )
            else:
                logger.warning(
                    "  ✗ Quality check failed "
                    f"(reason={quality_report.reason}, "
                    f"ratio={quality_report.effective_char_ratio}, "
                    f"density={quality_report.recognizable_text_density})"
                )
                self.integrity_checker.mark_failed(
                    file_hash,
                    str(file_path),
                    DOCUMENT_QUALITY_REJECTION_MESSAGE,
                )
                return PipelineResult(
                    success=False,
                    file_path=str(file_path),
                    doc_id=file_hash,
                    error=DOCUMENT_QUALITY_REJECTION_MESSAGE,
                    stages=stages,
                )

            # ─────────────────────────────────────────────────────────────
            # Stage 3: Document Loading
            # ─────────────────────────────────────────────────────────────
            logger.info("\n📄 Stage 3: Document Loading")
            _notify("load", 3)
            
            _t0 = time.monotonic()
            get_loader = getattr(self, "_get_loader_for_file", None)
            loader = get_loader(file_path) if callable(get_loader) else self.loader
            document = loader.load(str(file_path))
            _elapsed = (time.monotonic() - _t0) * 1000.0
            
            text_preview = document.text[:200].replace('\n', ' ') + "..." if len(document.text) > 200 else document.text
            image_count = len(document.metadata.get("images", []))
            
            logger.info(f"  Document ID: {document.id}")
            logger.info(f"  Text length: {len(document.text)} chars")
            logger.info(f"  Images extracted: {image_count}")
            logger.info(f"  Preview: {text_preview[:100]}...")
            
            stages["loading"] = {
                "doc_id": document.id,
                "loader": loader.__class__.__name__,
                "file_type": document.metadata.get("file_type", document.metadata.get("doc_type", "")),
                "text_length": len(document.text),
                "image_count": image_count
            }
            if trace is not None:
                trace.record_stage("load", {
                    "method": loader.__class__.__name__,
                    "doc_id": document.id,
                    "file_type": document.metadata.get("file_type", document.metadata.get("doc_type", "")),
                    "text_length": len(document.text),
                    "image_count": image_count,
                    "text_preview": document.text,
                }, elapsed_ms=_elapsed)
            
            # ─────────────────────────────────────────────────────────────
            # Stage 4: Chunking
            # ─────────────────────────────────────────────────────────────
            logger.info("\n✂️  Stage 4: Document Chunking")
            _notify("split", 4)
            
            _t0 = time.monotonic()
            chunks = self.chunker.split_document(document)
            _elapsed = (time.monotonic() - _t0) * 1000.0
            
            logger.info(f"  Chunks generated: {len(chunks)}")
            if chunks:
                logger.info(f"  First chunk ID: {chunks[0].id}")
                logger.info(f"  First chunk preview: {chunks[0].text[:100]}...")
            
            stages["chunking"] = {
                "chunk_count": len(chunks),
                "avg_chunk_size": sum(len(c.text) for c in chunks) // len(chunks) if chunks else 0
            }
            if trace is not None:
                trace.record_stage("split", {
                    "method": "recursive",
                    "chunk_count": len(chunks),
                    "avg_chunk_size": sum(len(c.text) for c in chunks) // len(chunks) if chunks else 0,
                    "chunks": [
                        {
                            "chunk_id": c.id,
                            "text": c.text,
                            "char_len": len(c.text),
                            "chunk_index": c.metadata.get("chunk_index", i),
                        }
                        for i, c in enumerate(chunks)
                    ],
                }, elapsed_ms=_elapsed)
            
            # ─────────────────────────────────────────────────────────────
            # Stage 5: Transform Pipeline
            # ─────────────────────────────────────────────────────────────
            logger.info("\n🔄 Stage 5: Transform Pipeline")
            _notify("transform", 5)
            
            # 4a: Chunk Refinement
            logger.info("  4a. Chunk Refinement...")
            _t0_transform = time.monotonic()
            # snapshot before refinement
            _pre_refine_texts = {c.id: c.text for c in chunks}
            chunks = self.chunk_refiner.transform(chunks, trace)
            refined_by_llm = sum(1 for c in chunks if c.metadata.get("refined_by") == "llm")
            refined_by_rule = sum(1 for c in chunks if c.metadata.get("refined_by") == "rule")
            logger.info(f"      LLM refined: {refined_by_llm}, Rule refined: {refined_by_rule}")
            
            # 4b: Metadata Enrichment
            logger.info("  4b. Metadata Enrichment...")
            chunks = self.metadata_enricher.transform(chunks, trace)
            enriched_by_llm = sum(1 for c in chunks if c.metadata.get("enriched_by") == "llm")
            enriched_by_rule = sum(1 for c in chunks if c.metadata.get("enriched_by") == "rule")
            logger.info(f"      LLM enriched: {enriched_by_llm}, Rule enriched: {enriched_by_rule}")
            
            # 4c: Image Captioning
            logger.info("  4c. Image Captioning...")
            chunks = self.image_captioner.transform(chunks, trace)
            captioned = sum(1 for c in chunks if c.metadata.get("image_captions"))
            logger.info(f"      Chunks with captions: {captioned}")
            
            stages["transform"] = {
                "chunk_refiner": {"llm": refined_by_llm, "rule": refined_by_rule},
                "metadata_enricher": {"llm": enriched_by_llm, "rule": enriched_by_rule},
                "image_captioner": {"captioned_chunks": captioned}
            }
            _elapsed_transform = (time.monotonic() - _t0_transform) * 1000.0
            if trace is not None:
                trace.record_stage("transform", {
                    "method": "refine+enrich+caption",
                    "refined_by_llm": refined_by_llm,
                    "refined_by_rule": refined_by_rule,
                    "enriched_by_llm": enriched_by_llm,
                    "enriched_by_rule": enriched_by_rule,
                    "captioned_chunks": captioned,
                    "chunks": [
                        {
                            "chunk_id": c.id,
                            "text_before": _pre_refine_texts.get(c.id, ""),
                            "text_after": c.text,
                            "char_len": len(c.text),
                            "refined_by": c.metadata.get("refined_by", ""),
                            "enriched_by": c.metadata.get("enriched_by", ""),
                            "title": c.metadata.get("title", ""),
                            "tags": c.metadata.get("tags", []),
                            "summary": c.metadata.get("summary", ""),
                        }
                        for c in chunks
                    ],
                }, elapsed_ms=_elapsed_transform)
            
            # ─────────────────────────────────────────────────────────────
            # Stage 6: Encoding
            # ─────────────────────────────────────────────────────────────
            logger.info("\n🔢 Stage 6: Encoding")
            _notify("embed", 6)
            
            # Process through BatchProcessor
            _t0 = time.monotonic()
            process_embeddings = getattr(self, "_process_embeddings", None)
            if callable(process_embeddings):
                batch_result = process_embeddings(chunks, trace)
            else:
                batch_result = self.batch_processor.process(chunks, trace)
            _elapsed = (time.monotonic() - _t0) * 1000.0
            
            dense_vectors = batch_result.dense_vectors
            sparse_stats = batch_result.sparse_stats
            cache_hits = getattr(batch_result, "cache_hits", 0)
            cache_misses = getattr(batch_result, "cache_misses", 0)
            
            logger.info(f"  Dense vectors: {len(dense_vectors)} (dim={len(dense_vectors[0]) if dense_vectors else 0})")
            logger.info(f"  Sparse stats: {len(sparse_stats)} documents")
            
            stages["encoding"] = {
                "dense_vector_count": len(dense_vectors),
                "dense_dimension": len(dense_vectors[0]) if dense_vectors else 0,
                "sparse_doc_count": len(sparse_stats),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
            }
            if trace is not None:
                # Build per-chunk encoding details (both dense & sparse)
                chunk_details = []
                for idx, c in enumerate(chunks):
                    detail: dict = {
                        "chunk_id": c.id,
                        "char_len": len(c.text),
                    }
                    # Dense: vector dimension (same for all, but confirm per-chunk)
                    if idx < len(dense_vectors):
                        detail["dense_dim"] = len(dense_vectors[idx])
                    # Sparse: BM25 term stats
                    if idx < len(sparse_stats):
                        ss = sparse_stats[idx]
                        detail["doc_length"] = ss.get("doc_length", 0)
                        detail["unique_terms"] = ss.get("unique_terms", 0)
                        # Top-10 terms by frequency for inspection
                        tf = ss.get("term_frequencies", {})
                        top_terms = sorted(tf.items(), key=lambda x: x[1], reverse=True)[:10]
                        detail["top_terms"] = [{"term": t, "freq": f} for t, f in top_terms]
                    chunk_details.append(detail)

                trace.record_stage("embed", {
                    "method": "batch_processor",
                    "dense_vector_count": len(dense_vectors),
                    "dense_dimension": len(dense_vectors[0]) if dense_vectors else 0,
                    "sparse_doc_count": len(sparse_stats),
                    "cache_hits": cache_hits,
                    "cache_misses": cache_misses,
                    "chunks": chunk_details,
                }, elapsed_ms=_elapsed)
            
            # ─────────────────────────────────────────────────────────────
            # Stage 7: Storage
            # ─────────────────────────────────────────────────────────────
            logger.info("\n💾 Stage 7: Storage")
            _notify("upsert", 7)
            
            # 7a: Vector Upsert
            logger.info("  7a. Vector Storage (ChromaDB)...")
            _t0_storage = time.monotonic()
            _t0_chroma = time.monotonic()
            delete_existing = getattr(self, "_delete_existing_vectors_for_source", None)
            if callable(delete_existing):
                deleted_vector_count = delete_existing(str(file_path), trace=trace)
            else:
                deleted_vector_count = 0
            vector_ids = self.vector_upserter.upsert(chunks, dense_vectors, trace)
            logger.info(f"      Stored {len(vector_ids)} vectors")
            _elapsed_chroma = (time.monotonic() - _t0_chroma) * 1000.0
            if trace is not None:
                trace.record_stage("chroma_upsert", {
                    "method": "chroma_upsert",
                    "backend": "ChromaDB",
                    "collection": self.collection,
                    "count": len(vector_ids),
                    "deleted_stale_vectors": deleted_vector_count,
                    "path": "data/db/chroma/",
                }, elapsed_ms=_elapsed_chroma)

            # Align BM25 chunk_ids with Chroma vector IDs so the SparseRetriever
            # can look up BM25 hits in the vector store after retrieval.
            for stat, vid in zip(sparse_stats, vector_ids):
                stat["chunk_id"] = vid

            # 7b: Sparse Index
            sparse_backend = get_sparse_backend(getattr(self, "settings", None))
            logger.info(f"  7b. Sparse Index ({sparse_backend})...")
            _t0_sparse = time.monotonic()
            sparse_indexer = getattr(
                self,
                "sparse_indexer",
                getattr(self, "bm25_indexer", None),
            )
            if getattr(sparse_indexer, "requires_chunk_text", False):
                sparse_indexer.add_documents(
                    sparse_stats,
                    chunks=chunks,
                    chunk_ids=vector_ids,
                    collection=self.collection,
                    doc_id=file_hash,
                    source_path=str(file_path),
                    trace=trace,
                )
            else:
                source_hash = hashlib.sha256(str(file_path).encode("utf-8")).hexdigest()[:8]
                sparse_indexer.add_documents(
                    sparse_stats,
                    collection=self.collection,
                    doc_id=source_hash,
                    trace=trace,
                )
            _elapsed_sparse = (time.monotonic() - _t0_sparse) * 1000.0
            if trace is not None:
                trace.record_stage("sparse_index", {
                    "method": "sparse_index",
                    "backend": sparse_backend,
                    "collection": self.collection,
                    "count": len(sparse_stats),
                    "path": (
                        "data/db/sparse_fts5.db"
                        if sparse_backend in {"sqlite_fts5", "fts5"}
                        else f"data/db/bm25/{self.collection}/"
                    ),
                }, elapsed_ms=_elapsed_sparse)
            logger.info(f"      Index built for {len(sparse_stats)} documents")
            
            # 7c: Register images in image storage index
            # Note: Images are already saved by PdfLoader, we just need to index them
            logger.info("  7c. Image Storage Index...")
            _t0_image = time.monotonic()
            images = document.metadata.get("images", [])
            for img in images:
                img_path = Path(img["path"])
                if img_path.exists():
                    self.image_storage.register_image(
                        image_id=img["id"],
                        file_path=img_path,
                        collection=self.collection,
                        doc_hash=file_hash,
                        page_num=img.get("page", 0)
                    )
            logger.info(f"      Indexed {len(images)} images")
            _elapsed_image = (time.monotonic() - _t0_image) * 1000.0
            if trace is not None:
                trace.record_stage("image_index", {
                    "method": "image_index",
                    "backend": "ImageStorage (JSON index)",
                    "count": len(images),
                }, elapsed_ms=_elapsed_image)
            
            stages["storage"] = {
                "vector_count": len(vector_ids),
                "bm25_docs": len(sparse_stats),
                "images_indexed": len(images),
                "deleted_stale_vectors": deleted_vector_count,
            }
            _elapsed_storage = (time.monotonic() - _t0_storage) * 1000.0
            if trace is not None:
                # Per-chunk storage mapping: chunk_id → vector_id
                chunk_storage = [
                    {
                        "chunk_id": c.id,
                        "vector_id": vector_ids[i] if i < len(vector_ids) else "—",
                        "collection": self.collection,
                        "store": "ChromaDB",
                    }
                    for i, c in enumerate(chunks)
                ]
                # Image storage details
                image_storage_details = [
                    {
                        "image_id": img["id"],
                        "file_path": str(img["path"]),
                        "page": img.get("page", 0),
                        "doc_hash": file_hash,
                    }
                    for img in images
                ]
                trace.record_stage("upsert", {
                    "method": "vector+bm25+image_storage",
                    "dense_store": {
                        "backend": "ChromaDB",
                        "collection": self.collection,
                        "count": len(vector_ids),
                        "deleted_stale_vectors": deleted_vector_count,
                        "path": "data/db/chroma/",
                    },
                    "sparse_store": {
                        "backend": sparse_backend,
                        "collection": self.collection,
                        "count": len(sparse_stats),
                        "path": (
                            "data/db/sparse_fts5.db"
                            if sparse_backend in {"sqlite_fts5", "fts5"}
                            else f"data/db/bm25/{self.collection}/"
                        ),
                    },
                    "image_store": {
                        "backend": "ImageStorage (JSON index)",
                        "count": len(images),
                        "images": image_storage_details,
                    },
                    "chunk_mapping": chunk_storage,
                }, elapsed_ms=_elapsed_storage)
            
            # ─────────────────────────────────────────────────────────────
            # Mark Success
            # ─────────────────────────────────────────────────────────────
            self.integrity_checker.mark_success(file_hash, str(file_path), self.collection)
            
            logger.info("\n" + "=" * 60)
            logger.info("✅ Pipeline completed successfully!")
            logger.info(f"   Chunks: {len(chunks)}")
            logger.info(f"   Vectors: {len(vector_ids)}")
            logger.info(f"   Images: {len(images)}")
            logger.info("=" * 60)
            
            return PipelineResult(
                success=True,
                file_path=str(file_path),
                doc_id=file_hash,
                chunk_count=len(chunks),
                image_count=len(images),
                vector_ids=vector_ids,
                stages=stages
            )
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
            failed_hash = locals().get("file_hash")
            if failed_hash:
                self.integrity_checker.mark_failed(failed_hash, str(file_path), str(e))
            
            return PipelineResult(
                success=False,
                file_path=str(file_path),
                doc_id=failed_hash,
                error=str(e),
                stages=stages
            )
    
    def close(self) -> None:
        """Clean up resources."""
        self.image_storage.close()


def run_pipeline(
    file_path: str,
    settings_path: Optional[str] = None,
    collection: str = "default",
    force: bool = False
) -> PipelineResult:
    """Convenience function to run the pipeline.
    
    Args:
        file_path: Path to file to process
        settings_path: Path to settings.yaml (default: <repo>/config/settings.yaml)
        collection: Collection name
        force: Force reprocessing
    
    Returns:
        PipelineResult with execution details
    """
    settings = load_settings(settings_path)
    pipeline = IngestionPipeline(settings, collection=collection, force=force)
    
    try:
        return pipeline.run(file_path)
    finally:
        pipeline.close()
