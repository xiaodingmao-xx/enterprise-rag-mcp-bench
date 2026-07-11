"""
Response Module.

This package contains response building components:
- Response builder
- Citation generator
- Multimodal assembler
"""

from src.core.response.citation_generator import Citation, CitationGenerator
from src.core.response.citation import CitationRecord
from src.core.response.claim_extractor import Claim, RuleBasedClaimExtractor
from src.core.response.citation_verifier import CitationVerifier, CitationVerificationResult
from src.core.response.confidence import AnswerConfidence, AnswerConfidenceScorer
from src.core.response.refusal_policy import RefusalDecision, RefusalPolicy

try:
    from src.core.response.response_builder import MCPToolResponse, ResponseBuilder
except ImportError:
    MCPToolResponse = None
    ResponseBuilder = None

try:
    from src.core.response.multimodal_assembler import (
        ImageContent,
        ImageReference,
        MultimodalAssembler,
    )
except ImportError:
    ImageContent = None
    ImageReference = None
    MultimodalAssembler = None

__all__ = [
    "Citation",
    "CitationGenerator",
    "CitationRecord",
    "Claim",
    "RuleBasedClaimExtractor",
    "CitationVerifier",
    "CitationVerificationResult",
    "AnswerConfidence",
    "AnswerConfidenceScorer",
    "RefusalDecision",
    "RefusalPolicy",
    "ImageContent",
    "ImageReference",
    "MCPToolResponse",
    "MultimodalAssembler",
    "ResponseBuilder",
]
