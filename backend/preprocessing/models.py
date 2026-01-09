"""Data models for PDF processing."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class ChunkType(str, Enum):
    """Types of chunks extracted from papers."""
    ABSTRACT = "abstract"
    SECTION = "section"
    FINE = "fine"
    FULL = "full"
    CAPTION = "caption"
    TABLE = "table"


@dataclass
class PaperMetadata:
    """Metadata for a research paper."""
    paper_id: str
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    file_name: str = ""
    num_pages: int = 0
    # New fields for filtering
    project_tag: Optional[str] = None  # e.g., "ERα_inhibitors", "LL37_imaging"
    research_area: Optional[str] = None  # e.g., "peptide_imaging", "protein_purification"


@dataclass
class Chunk:
    """A single chunk of content from a paper."""
    chunk_id: str
    paper_id: str
    chunk_type: ChunkType
    text: str

    # Metadata
    section_name: Optional[str] = None  # Parent section: "methods", "results", etc.
    subsection_name: Optional[str] = None  # Subsection header if applicable
    parent_chunk_id: Optional[str] = None
    figure_id: Optional[str] = None
    page_numbers: List[int] = field(default_factory=list)

    # Paper-level metadata (denormalized for retrieval)
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None

    # New fields
    project_tag: Optional[str] = None
    research_area: Optional[str] = None
    file_name: str = ""  # Original PDF filename for linking

    # Processing metadata
    token_count: int = 0

    def to_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant payload format."""
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "chunk_type": self.chunk_type.value,
            "text": self.text,
            "section_name": self.section_name,
            "subsection_name": self.subsection_name,
            "parent_chunk_id": self.parent_chunk_id,
            "figure_id": self.figure_id,
            "page_numbers": self.page_numbers,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "doi": self.doi,
            "project_tag": self.project_tag,
            "research_area": self.research_area,
            "file_name": self.file_name,
            "token_count": self.token_count,
        }
