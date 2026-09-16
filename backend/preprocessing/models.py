"""Data models for PDF processing.

The chunk contract (W2)
-----------------------
``Chunk.text`` used to serve four consumers with incompatible needs, and was
built as ``context_header + source_text`` — so the Citations API would have
returned quotes containing ``[Methods]`` and ``char_location`` offsets would
have pointed into a synthetic string.  The field is now split three ways:

======================  =================================================  ==========================
Field                   Contents                                           Consumer
======================  =================================================  ==========================
``text``                the **verbatim** source span, byte-identical to     Citations API, quote display
                        ``paper_record.full_text(record)[char_start:char_end]``
``embed_text``          context line(s) + the span                         Voyage, BM25, answer prompt
``display_text``        clean readable form                                 ``SourceCard`` in the UI
======================  =================================================  ==========================

``embed_text`` and ``display_text`` are **properties**, composed from the
stored parts, so there is one definition of how they are built and no way for a
stored copy to drift from ``text``.  Only ``text`` may be sent as a citation
document: nothing synthetic is ever prepended to it.

Storage: the payload keeps ``text`` plus the *differences* (``context_line``,
``llm_context``, ``embed_body``, ``display_text``), each null when it is just
``text`` again.  Use :func:`embed_text_from_payload` /
:func:`display_text_from_payload` to recompose them on the read side — that is
how ``qdrant_store`` gets the BM25 document text, and it is exact.

Positions (R6).  ``char_start`` / ``char_end`` are offsets into the paper
record's document text.  ``page_start`` / ``page_end`` / ``page_numbers`` are
**1-indexed** human page numbers (``record`` block ``page_idx`` + 1), because
that is what a PDF viewer and the Citations API ``page_location`` both use.
A chunk may legitimately span pages — W1 splits paragraphs that straddle a page
break (R18) — so ``page_end`` is not always ``page_start``.
"""

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


#: Joined between the context lines and the body inside ``embed_text``.
CONTEXT_SEP = "\n"


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
    """A single chunk of content from a paper.

    ``text`` is the verbatim source span and nothing else.  See the module
    docstring for the three-field text contract and the position fields.
    """
    chunk_id: str
    paper_id: str
    chunk_type: ChunkType
    text: str

    # --- the text contract (W2) -------------------------------------------
    #: Deterministic header: ``[Title — Section > Subsection, p.N]``.  Goes into
    #: ``embed_text`` only, never into ``text``.
    context_line: Optional[str] = None
    #: One LLM-written context sentence per chunk (contextual retrieval, §3b.2).
    llm_context: Optional[str] = None
    #: Body to embed when it should differ from the verbatim span — a truncated
    #: section, a linearized table, a cleaned abstract.  ``None`` means ``text``.
    embed_body: Optional[str] = None
    #: Readable body when it should differ from the verbatim span — a rendered
    #: table, a cleaned abstract.  ``None`` means ``text``.
    display_body: Optional[str] = None
    #: False only for chunks with no single source span (``full``), which must
    #: therefore never be sent as a citation document.
    text_is_verbatim: bool = True

    # --- positions and provenance (R6) ------------------------------------
    page_start: Optional[int] = None      # 1-indexed
    page_end: Optional[int] = None        # 1-indexed, inclusive; may differ (R18)
    char_start: Optional[int] = None      # offset into paper_record.full_text
    char_end: Optional[int] = None
    #: MinerU bbox of the first block, per-mille of page size, top-left origin.
    bbox: Optional[List[int]] = None
    #: Record block index range this chunk covers, ``[block_start, block_end)``.
    block_start: Optional[int] = None
    block_end: Optional[int] = None

    # Metadata
    section_name: Optional[str] = None  # Parent section: "methods", "results", etc.
    subsection_name: Optional[str] = None  # Subsection header if applicable
    parent_chunk_id: Optional[str] = None
    figure_id: Optional[str] = None
    page_numbers: List[int] = field(default_factory=list)  # 1-indexed, inclusive

    # Paper-level metadata (denormalized for retrieval)
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None

    # New fields
    project_tag: Optional[str] = None
    research_area: Optional[str] = None
    file_name: str = ""  # Original PDF filename for linking

    #: value + unit + entity triples lifted out of tables and prose.
    numeric_facts: List[Dict[str, Any]] = field(default_factory=list)

    # Processing metadata
    token_count: int = 0

    # ------------------------------------------------------------------
    # Derived text
    # ------------------------------------------------------------------

    @property
    def embed_text(self) -> str:
        """Context line(s) + the body.  What Voyage, BM25 and the prompt see."""
        parts = [p for p in (self.context_line, self.llm_context) if p]
        parts.append(self.embed_body if self.embed_body is not None else self.text)
        return CONTEXT_SEP.join(p for p in parts if p)

    @property
    def display_text(self) -> str:
        """Clean readable form.  What ``SourceCard`` shows the user."""
        return self.display_body if self.display_body is not None else self.text

    def spans_pages(self) -> bool:
        return (self.page_start is not None and self.page_end is not None
                and self.page_end > self.page_start)

    def to_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant payload format.

        ``text`` is verbatim.  ``embed_text`` is *not* stored — it is recomposed
        by :func:`embed_text_from_payload` from the parts below, so the long body
        string is never duplicated in the payload.
        """
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "chunk_type": self.chunk_type.value,
            # --- the text contract ---
            "text": self.text,
            "display_text": self.display_body,   # null == same as `text`
            "embed_body": self.embed_body,       # null == same as `text`
            "context_line": self.context_line,
            "llm_context": self.llm_context,
            "text_is_verbatim": self.text_is_verbatim,
            # --- positions (1-indexed pages; char offsets into the record) ---
            "page_start": self.page_start,
            "page_end": self.page_end,
            "page_numbers": self.page_numbers,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "bbox": self.bbox,
            "block_start": self.block_start,
            "block_end": self.block_end,
            # --- metadata ---
            "section_name": self.section_name,
            "subsection_name": self.subsection_name,
            "parent_chunk_id": self.parent_chunk_id,
            "figure_id": self.figure_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "doi": self.doi,
            "project_tag": self.project_tag,
            "research_area": self.research_area,
            "file_name": self.file_name,
            "token_count": self.token_count,
            # --- numeric facts ---
            "numeric_facts": self.numeric_facts,
            "numeric_properties": sorted({f["property"] for f in self.numeric_facts
                                          if f.get("property")}),
            "numeric_entities": sorted({f["entity"].lower() for f in self.numeric_facts
                                        if f.get("entity")}),
            "numeric_units": sorted({f["unit"] for f in self.numeric_facts
                                     if f.get("unit")}),
            "has_numeric_facts": bool(self.numeric_facts),
        }


# ---------------------------------------------------------------------------
# Read side: recompose the derived text fields from a stored payload
# ---------------------------------------------------------------------------

#: The payload fields :func:`embed_text_from_payload` reads.  Kept here so a
#: caller that scrolls Qdrant with a field projection (``scripts/rebuild_bm25_index``)
#: cannot silently fetch too few fields and rebuild BM25 from the wrong string.
BM25_PAYLOAD_FIELDS = ["text", "embed_body", "context_line", "llm_context"]


def embed_text_from_payload(payload: Dict[str, Any]) -> str:
    """Rebuild ``embed_text`` from a Qdrant payload.

    Used by ``qdrant_store.upsert_chunks`` (BM25 document vectors) and by
    ``scripts/rebuild_bm25_index.py``, so that what BM25 indexes is exactly what
    Voyage embedded.  Falls back to ``text`` for pre-W2 payloads.
    """
    parts = [payload.get("context_line"), payload.get("llm_context")]
    body = payload.get("embed_body")
    if body is None:
        body = payload.get("text", "")
    parts.append(body)
    return CONTEXT_SEP.join(p for p in parts if p)


def display_text_from_payload(payload: Dict[str, Any]) -> str:
    """Rebuild ``display_text`` from a Qdrant payload."""
    display = payload.get("display_text")
    if display:
        return display
    return payload.get("text", "")
