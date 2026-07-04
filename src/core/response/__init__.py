"""
Response Module.

This package contains response building components:
- Response builder
- Citation generator
- Multimodal assembler
"""

from src.core.response.citation_generator import Citation, CitationGenerator

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
    "ImageContent",
    "ImageReference",
    "MCPToolResponse",
    "MultimodalAssembler",
    "ResponseBuilder",
]
