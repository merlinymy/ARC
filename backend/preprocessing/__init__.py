"""Preprocessing module for PDF extraction and chunking."""

from .models import (
    ChunkType,
    PaperMetadata,
    Chunk,
    embed_text_from_payload,
    display_text_from_payload,
)
from .section_detector import SectionDetector, Section
from .chunker import PaperChunker
from . import numeric_facts
from .pdf_processor import PDFProcessor, EnhancedPDFProcessor, PDFChunk
from . import paper_record

__all__ = [
    # Models
    "ChunkType",
    "PaperMetadata",
    "Chunk",
    "embed_text_from_payload",
    "display_text_from_payload",
    # Section detection
    "SectionDetector",
    "Section",
    # Chunking
    "PaperChunker",
    "numeric_facts",
    # The paper record (W1) -- structured extraction persisted per paper
    "paper_record",
    # PDF Processing
    "PDFProcessor",
    "EnhancedPDFProcessor",
    "PDFChunk",
]
