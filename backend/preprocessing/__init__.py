"""Preprocessing module for PDF extraction and chunking."""

from .models import ChunkType, PaperMetadata, Chunk
from .section_detector import SectionDetector, Section
from .chunker import PaperChunker
from .table_extractor import TableExtractor
from .caption_extractor import CaptionExtractor
from .pdf_processor import PDFProcessor, EnhancedPDFProcessor, PDFChunk
from . import paper_record

__all__ = [
    # Models
    "ChunkType",
    "PaperMetadata",
    "Chunk",
    # Section detection
    "SectionDetector",
    "Section",
    # Chunking
    "PaperChunker",
    # Extraction
    "TableExtractor",
    "CaptionExtractor",
    # The paper record (W1) -- structured extraction persisted per paper
    "paper_record",
    # PDF Processing
    "PDFProcessor",
    "EnhancedPDFProcessor",
    "PDFChunk",
]
