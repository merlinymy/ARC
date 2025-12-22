# Phase 1 Implementation Guide: Validation Pipeline

## Prerequisites

**IMPORTANT: Complete Phase 0 first!**

Before starting Phase 1, you must complete [phase_0_impl.md](phase_0_impl.md) to filter the ~22,000 PDFs down to only research papers. Phase 1 expects cleaned data in `backend/data/classified/papers/`.

Verify Phase 0 is complete:
- [ ] `backend/data/classified/papers/` exists and contains filtered PDFs
- [ ] Cleaning logs exist in `backend/data/cleaning_logs/`
- [ ] Paper count is documented (expected: 15,000-20,000 papers)

---

## Agent Instructions

**READ THIS FIRST - Instructions for LLM Code Agents**

This document guides the implementation of Phase 1 (Validation) of the Research Paper RAG System. Follow these rules:

1. **Work sequentially through sections** - Each section builds on previous ones. Do not skip ahead.
2. **Mark tasks complete** - After completing each task, update the checkbox from `[ ]` to `[x]` in this file.
3. **Verify before proceeding** - Each section has verification steps. Run them before moving to the next section.
4. **Commit atomically** - Commit after completing each major section (not individual tasks).
5. **Handle errors explicitly** - If a task fails, document the error in the "Issues Log" section at the bottom before attempting fixes.
6. **Preserve existing code** - Extend, don't replace, unless explicitly instructed. The project has existing code in `backend/` that should be enhanced.

**Current State Assessment:**
- `backend/preprocessing/pdf_processor.py` - Basic PDF extraction exists but lacks multi-chunk-type support
- `backend/preprocessing/process_pdfs.py` - Basic pipeline exists but uses simple chunking
- `backend/api/main.py` - Query API exists but lacks reranker
- `backend/config.py` - Configuration exists, needs extension

**Phase 1 Goal:** Prove the pipeline works on 50 papers before scaling to 22,000.

---

## Section 0: Environment Setup

### 0.1 Dependencies

Update `backend/requirements.txt` to include all required packages:

```
# Core API Framework
fastapi
uvicorn[standard]
python-multipart

# AI/ML Libraries
anthropic
voyageai
cohere

# PDF Processing
marker-pdf
pymupdf
pdfplumber
camelot-py[base]

# Vector Database
qdrant-client

# Utilities
python-dotenv
pydantic
pydantic-settings

# HTTP Client
httpx
aiofiles

# Data Processing
numpy
pillow
tqdm
tiktoken

# Testing & Evaluation
pytest
pytest-asyncio
pandas
```

**Tasks:**
- [ ] Update `backend/requirements.txt` with the packages above
- [ ] Run `pip install -r requirements.txt` in the virtual environment
- [ ] Verify installations: `python -c "import cohere; import camelot; print('OK')"`

### 0.2 Configuration Updates

Extend `backend/config.py` to include Phase 1 settings:

**Add these fields to the Settings class:**

```python
# Reranker Settings
cohere_api_key: str
reranker_model: str = "rerank-v3.5"
rerank_top_n: int = 15

# Chunk Type Settings
abstract_max_tokens: int = 300
section_max_tokens: int = 2000
fine_chunk_tokens: int = 500
fine_chunk_overlap: int = 128

# Retrieval Settings
retrieval_top_k: int = 50  # Per chunk type before reranking
final_top_k: int = 15      # After reranking

# Phase 1 Settings
validation_sample_size: int = 50
test_queries_path: Path = Path("./data/test_queries.json")
evaluation_results_path: Path = Path("./data/evaluation_results.json")
```

**Tasks:**
- [ ] Add the fields above to `backend/config.py`
- [ ] Add `COHERE_API_KEY` to `.env.example`
- [ ] Create `.env` file with actual API keys (do not commit)

### 0.3 Directory Structure

Create the following directory structure:

```
backend/
├── config.py                 # Extended configuration
├── requirements.txt          # Updated dependencies
├── preprocessing/
│   ├── __init__.py
│   ├── pdf_processor.py      # Enhanced PDF extraction
│   ├── chunker.py            # NEW: Multi-type chunking logic
│   ├── table_extractor.py    # NEW: Table extraction with Camelot
│   └── process_pdfs.py       # Enhanced pipeline
├── retrieval/
│   ├── __init__.py
│   ├── embedder.py           # NEW: Voyage embedding wrapper
│   ├── reranker.py           # NEW: Cohere reranker wrapper
│   ├── qdrant_store.py       # NEW: Qdrant operations
│   └── query_engine.py       # NEW: Full query pipeline
├── evaluation/
│   ├── __init__.py
│   ├── test_queries.py       # NEW: Test query definitions
│   └── evaluator.py          # NEW: Retrieval quality metrics
├── api/
│   └── main.py               # Enhanced API
└── data/
    ├── sample_papers/        # NEW: 50 validation PDFs
    ├── test_queries.json     # NEW: 50 test queries
    └── evaluation_results.json
```

**Tasks:**
- [ ] Create `backend/preprocessing/__init__.py`
- [ ] Create `backend/retrieval/` directory with `__init__.py`
- [ ] Create `backend/evaluation/` directory with `__init__.py`
- [ ] Create `backend/data/` directory
- [ ] Create `backend/data/sample_papers/` directory

---

## Section 1: Enhanced PDF Extraction

### 1.1 Data Models

Create `backend/preprocessing/models.py` with chunk data structures:

```python
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


@dataclass
class Chunk:
    """A single chunk of content from a paper."""
    chunk_id: str
    paper_id: str
    chunk_type: ChunkType
    text: str

    # Metadata
    section_name: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    figure_id: Optional[str] = None
    page_numbers: List[int] = field(default_factory=list)

    # Paper-level metadata (denormalized for retrieval)
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None

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
            "parent_chunk_id": self.parent_chunk_id,
            "figure_id": self.figure_id,
            "page_numbers": self.page_numbers,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "token_count": self.token_count,
        }
```

**Tasks:**
- [ ] Create `backend/preprocessing/models.py` with the code above

### 1.2 Section Detection

Create `backend/preprocessing/section_detector.py` to identify paper sections:

```python
"""Detect and classify sections in research papers."""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Section:
    """A detected section in a paper."""
    name: str
    normalized_name: str  # e.g., "methods", "results", "discussion"
    start_idx: int
    end_idx: int
    text: str
    level: int  # 1 for main sections, 2 for subsections


# Common section headers in scientific papers (case-insensitive patterns)
SECTION_PATTERNS = [
    # Main sections
    (r"^(?:1\.?\s*)?(?:introduction|background)", "introduction", 1),
    (r"^(?:2\.?\s*)?(?:materials?\s*(?:and|&)\s*methods?|methods?|experimental)", "methods", 1),
    (r"^(?:3\.?\s*)?(?:results?)", "results", 1),
    (r"^(?:4\.?\s*)?(?:discussion)", "discussion", 1),
    (r"^(?:5\.?\s*)?(?:conclusion|conclusions|summary)", "conclusion", 1),
    (r"^abstract", "abstract", 1),
    (r"^(?:references?|bibliography|literature\s*cited)", "references", 1),
    (r"^(?:acknowledg|funding|support)", "acknowledgments", 1),
    (r"^(?:supplementa|supporting\s*information|appendix)", "supplementary", 1),
    # Subsections (level 2)
    (r"^\d+\.\d+\.?\s+", "subsection", 2),
]


class SectionDetector:
    """Detect sections in extracted PDF text."""

    def __init__(self):
        self.patterns = [(re.compile(p, re.IGNORECASE | re.MULTILINE), name, level)
                         for p, name, level in SECTION_PATTERNS]

    def detect_sections(self, text: str) -> List[Section]:
        """Detect all sections in the text.

        Args:
            text: Full text of the paper

        Returns:
            List of Section objects in order of appearance
        """
        # Find all potential section headers
        candidates = []
        lines = text.split('\n')
        current_pos = 0

        for line_idx, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                current_pos += len(line) + 1
                continue

            # Check if line matches any section pattern
            for pattern, normalized_name, level in self.patterns:
                if pattern.match(line_stripped):
                    candidates.append({
                        'name': line_stripped,
                        'normalized_name': normalized_name,
                        'start_idx': current_pos,
                        'level': level,
                        'line_idx': line_idx
                    })
                    break

            current_pos += len(line) + 1

        # Build sections with end indices
        sections = []
        for i, candidate in enumerate(candidates):
            end_idx = candidates[i + 1]['start_idx'] if i + 1 < len(candidates) else len(text)
            section_text = text[candidate['start_idx']:end_idx].strip()

            sections.append(Section(
                name=candidate['name'],
                normalized_name=candidate['normalized_name'],
                start_idx=candidate['start_idx'],
                end_idx=end_idx,
                text=section_text,
                level=candidate['level']
            ))

        return sections

    def extract_abstract(self, text: str) -> Optional[str]:
        """Extract abstract from paper text.

        Handles common formats:
        - "Abstract" followed by text
        - "Abstract:" followed by text
        - Text before "Introduction" if paper starts with abstract
        """
        # Try to find explicit abstract section
        abstract_pattern = re.compile(
            r'abstract[:\s]*\n*(.*?)(?=\n\s*(?:introduction|keywords?|1\.|background)|\Z)',
            re.IGNORECASE | re.DOTALL
        )
        match = abstract_pattern.search(text[:5000])  # Abstract should be near start

        if match:
            abstract = match.group(1).strip()
            # Clean up: remove excessive whitespace
            abstract = re.sub(r'\s+', ' ', abstract)
            return abstract if len(abstract) > 50 else None

        return None
```

**Tasks:**
- [ ] Create `backend/preprocessing/section_detector.py` with the code above

### 1.3 Multi-Type Chunker

Create `backend/preprocessing/chunker.py` implementing the 6 chunk types:

```python
"""Multi-type chunking for research papers."""

import tiktoken
import hashlib
from typing import List, Optional, Dict
from pathlib import Path

from .models import Chunk, ChunkType, PaperMetadata
from .section_detector import SectionDetector, Section


class PaperChunker:
    """Create multiple chunk types from a paper.

    Chunk Types:
    - ABSTRACT: ~200-300 tokens, paper discovery
    - SECTION: ~2000 tokens, literature review context
    - FINE: ~500 tokens with 100-150 overlap, specific fact retrieval
    - FULL: Mean-pooled embedding from all sections
    - CAPTION: Variable, figure/table discovery
    - TABLE: Variable, structured table content
    """

    def __init__(
        self,
        fine_chunk_tokens: int = 500,
        fine_chunk_overlap: int = 128,
        section_max_tokens: int = 2000,
        abstract_max_tokens: int = 300,
    ):
        self.fine_chunk_tokens = fine_chunk_tokens
        self.fine_chunk_overlap = fine_chunk_overlap
        self.section_max_tokens = section_max_tokens
        self.abstract_max_tokens = abstract_max_tokens

        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.section_detector = SectionDetector()

    def _generate_chunk_id(self, paper_id: str, chunk_type: ChunkType, index: int) -> str:
        """Generate unique chunk ID."""
        raw = f"{paper_id}_{chunk_type.value}_{index}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to max tokens."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.tokenizer.decode(tokens[:max_tokens])

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Split on sentence-ending punctuation followed by space and capital
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _create_fine_chunks(
        self,
        text: str,
        paper_id: str,
        metadata: PaperMetadata,
        section_name: Optional[str],
        parent_chunk_id: Optional[str],
        page_numbers: List[int]
    ) -> List[Chunk]:
        """Create fine-grained chunks with overlap."""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_sentences = []
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            if current_tokens + sentence_tokens > self.fine_chunk_tokens and current_sentences:
                # Create chunk
                chunk_text = ' '.join(current_sentences)
                chunk_id = self._generate_chunk_id(paper_id, ChunkType.FINE, chunk_index)

                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    chunk_type=ChunkType.FINE,
                    text=chunk_text,
                    section_name=section_name,
                    parent_chunk_id=parent_chunk_id,
                    page_numbers=page_numbers,
                    title=metadata.title,
                    authors=metadata.authors,
                    year=metadata.year,
                    token_count=current_tokens,
                ))

                # Calculate overlap
                overlap_sentences = []
                overlap_tokens = 0
                for s in reversed(current_sentences):
                    s_tokens = self._count_tokens(s)
                    if overlap_tokens + s_tokens > self.fine_chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tokens

                current_sentences = overlap_sentences
                current_tokens = overlap_tokens
                chunk_index += 1

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        # Final chunk
        if current_sentences:
            chunk_text = ' '.join(current_sentences)
            chunk_id = self._generate_chunk_id(paper_id, ChunkType.FINE, chunk_index)

            chunks.append(Chunk(
                chunk_id=chunk_id,
                paper_id=paper_id,
                chunk_type=ChunkType.FINE,
                text=chunk_text,
                section_name=section_name,
                parent_chunk_id=parent_chunk_id,
                page_numbers=page_numbers,
                title=metadata.title,
                authors=metadata.authors,
                year=metadata.year,
                token_count=current_tokens,
            ))

        return chunks

    def chunk_paper(
        self,
        full_text: str,
        metadata: PaperMetadata,
        captions: List[Dict] = None,
        tables: List[Dict] = None,
    ) -> List[Chunk]:
        """Create all chunk types for a paper.

        Args:
            full_text: Full extracted text of the paper
            metadata: Paper metadata
            captions: List of figure/table captions with 'text' and 'figure_id' keys
            tables: List of extracted tables with 'markdown' and 'table_id' keys

        Returns:
            List of all chunks for this paper
        """
        chunks = []
        captions = captions or []
        tables = tables or []

        # 1. ABSTRACT chunk
        abstract_text = self.section_detector.extract_abstract(full_text)
        if abstract_text:
            abstract_text = self._truncate_to_tokens(abstract_text, self.abstract_max_tokens)
            chunks.append(Chunk(
                chunk_id=self._generate_chunk_id(metadata.paper_id, ChunkType.ABSTRACT, 0),
                paper_id=metadata.paper_id,
                chunk_type=ChunkType.ABSTRACT,
                text=abstract_text,
                section_name="abstract",
                title=metadata.title,
                authors=metadata.authors,
                year=metadata.year,
                token_count=self._count_tokens(abstract_text),
            ))

        # 2. SECTION chunks
        sections = self.section_detector.detect_sections(full_text)
        section_chunk_index = 0

        for section in sections:
            # Skip references and supplementary
            if section.normalized_name in ('references', 'acknowledgments', 'supplementary'):
                continue

            section_text = self._truncate_to_tokens(section.text, self.section_max_tokens)
            section_chunk_id = self._generate_chunk_id(
                metadata.paper_id, ChunkType.SECTION, section_chunk_index
            )

            chunks.append(Chunk(
                chunk_id=section_chunk_id,
                paper_id=metadata.paper_id,
                chunk_type=ChunkType.SECTION,
                text=section_text,
                section_name=section.normalized_name,
                title=metadata.title,
                authors=metadata.authors,
                year=metadata.year,
                token_count=self._count_tokens(section_text),
            ))

            # 3. FINE chunks within this section
            fine_chunks = self._create_fine_chunks(
                text=section.text,
                paper_id=metadata.paper_id,
                metadata=metadata,
                section_name=section.normalized_name,
                parent_chunk_id=section_chunk_id,
                page_numbers=[],  # TODO: Track page numbers per section
            )
            chunks.extend(fine_chunks)

            section_chunk_index += 1

        # 4. FULL paper chunk (for mean-pooling later)
        # We store the text but embedding will be computed as mean of section embeddings
        full_text_truncated = self._truncate_to_tokens(full_text, 8000)  # Reasonable limit
        chunks.append(Chunk(
            chunk_id=self._generate_chunk_id(metadata.paper_id, ChunkType.FULL, 0),
            paper_id=metadata.paper_id,
            chunk_type=ChunkType.FULL,
            text=full_text_truncated,
            title=metadata.title,
            authors=metadata.authors,
            year=metadata.year,
            token_count=self._count_tokens(full_text_truncated),
        ))

        # 5. CAPTION chunks
        for idx, caption in enumerate(captions):
            caption_text = caption.get('text', '')
            if not caption_text or len(caption_text) < 20:
                continue

            chunks.append(Chunk(
                chunk_id=self._generate_chunk_id(metadata.paper_id, ChunkType.CAPTION, idx),
                paper_id=metadata.paper_id,
                chunk_type=ChunkType.CAPTION,
                text=caption_text,
                figure_id=caption.get('figure_id'),
                title=metadata.title,
                authors=metadata.authors,
                year=metadata.year,
                token_count=self._count_tokens(caption_text),
            ))

        # 6. TABLE chunks
        for idx, table in enumerate(tables):
            table_text = table.get('markdown', '')
            if not table_text or len(table_text) < 20:
                continue

            chunks.append(Chunk(
                chunk_id=self._generate_chunk_id(metadata.paper_id, ChunkType.TABLE, idx),
                paper_id=metadata.paper_id,
                chunk_type=ChunkType.TABLE,
                text=table_text,
                figure_id=table.get('table_id'),
                title=metadata.title,
                authors=metadata.authors,
                year=metadata.year,
                token_count=self._count_tokens(table_text),
            ))

        return chunks
```

**Tasks:**
- [ ] Create `backend/preprocessing/chunker.py` with the code above

### 1.4 Table Extractor

Create `backend/preprocessing/table_extractor.py`:

```python
"""Extract tables from PDFs using multiple strategies."""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import re

logger = logging.getLogger(__name__)


class TableExtractor:
    """Extract tables from PDF files.

    Uses pdfplumber as primary method (already in dependencies).
    Camelot can be added for better accuracy on complex tables.
    """

    def __init__(self, use_camelot: bool = False):
        """Initialize table extractor.

        Args:
            use_camelot: Whether to use Camelot (requires ghostscript)
        """
        self.use_camelot = use_camelot

        if use_camelot:
            try:
                import camelot
                self.camelot = camelot
            except ImportError:
                logger.warning("Camelot not available, falling back to pdfplumber")
                self.use_camelot = False

    def extract_tables_pdfplumber(self, pdf_path: Path) -> List[Dict]:
        """Extract tables using pdfplumber."""
        import pdfplumber

        tables = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()

                    for table_idx, table_data in enumerate(page_tables):
                        if not table_data or len(table_data) < 2:
                            continue

                        markdown = self._table_to_markdown(table_data)
                        if markdown:
                            tables.append({
                                'table_id': f"Table_p{page_num + 1}_{table_idx + 1}",
                                'page_number': page_num + 1,
                                'markdown': markdown,
                                'row_count': len(table_data),
                                'col_count': len(table_data[0]) if table_data else 0,
                            })
        except Exception as e:
            logger.error(f"Error extracting tables from {pdf_path}: {e}")

        return tables

    def extract_tables_camelot(self, pdf_path: Path) -> List[Dict]:
        """Extract tables using Camelot (higher accuracy)."""
        if not self.use_camelot:
            return self.extract_tables_pdfplumber(pdf_path)

        tables = []

        try:
            # Camelot returns TableList object
            table_list = self.camelot.read_pdf(
                str(pdf_path),
                pages='all',
                flavor='lattice'  # Use 'stream' for borderless tables
            )

            for idx, table in enumerate(table_list):
                df = table.df
                if df.empty or len(df) < 2:
                    continue

                markdown = df.to_markdown(index=False)
                tables.append({
                    'table_id': f"Table_{idx + 1}",
                    'page_number': table.page,
                    'markdown': markdown,
                    'row_count': len(df),
                    'col_count': len(df.columns),
                    'accuracy': table.accuracy,
                })
        except Exception as e:
            logger.warning(f"Camelot failed for {pdf_path}, falling back to pdfplumber: {e}")
            return self.extract_tables_pdfplumber(pdf_path)

        return tables

    def extract(self, pdf_path: Path) -> List[Dict]:
        """Extract tables using best available method."""
        if self.use_camelot:
            return self.extract_tables_camelot(pdf_path)
        return self.extract_tables_pdfplumber(pdf_path)

    @staticmethod
    def _table_to_markdown(table_data: List[List]) -> Optional[str]:
        """Convert table data to markdown format."""
        if not table_data or len(table_data) < 2:
            return None

        # Clean cells
        def clean_cell(cell):
            if cell is None:
                return ""
            return str(cell).replace('\n', ' ').strip()

        lines = []

        # Header
        headers = [clean_cell(h) for h in table_data[0]]
        lines.append("| " + " | ".join(headers) + " |")

        # Separator
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        # Data rows
        for row in table_data[1:]:
            cells = [clean_cell(c) for c in row]
            # Ensure row has same number of columns as header
            while len(cells) < len(headers):
                cells.append("")
            cells = cells[:len(headers)]
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)
```

**Tasks:**
- [ ] Create `backend/preprocessing/table_extractor.py` with the code above

### 1.5 Caption Extractor

Create `backend/preprocessing/caption_extractor.py`:

```python
"""Extract figure and table captions from PDFs."""

import re
import logging
from pathlib import Path
from typing import List, Dict
import pymupdf

logger = logging.getLogger(__name__)


class CaptionExtractor:
    """Extract figure and table captions from PDF text."""

    # Patterns to match captions
    CAPTION_PATTERNS = [
        # "Figure 1: Description" or "Figure 1. Description"
        (r'(Fig(?:ure)?\.?\s*\d+[a-zA-Z]?)[:\.\s]+(.+?)(?=\n\n|\nFig|\nTable|$)', 'figure'),
        # "Table 1: Description" or "Table 1. Description"
        (r'(Table\.?\s*\d+[a-zA-Z]?)[:\.\s]+(.+?)(?=\n\n|\nFig|\nTable|$)', 'table'),
        # "Scheme 1: Description" (common in chemistry papers)
        (r'(Scheme\.?\s*\d+[a-zA-Z]?)[:\.\s]+(.+?)(?=\n\n|\nFig|\nScheme|$)', 'scheme'),
    ]

    def __init__(self):
        self.patterns = [(re.compile(p, re.IGNORECASE | re.DOTALL), t)
                         for p, t in self.CAPTION_PATTERNS]

    def extract_from_text(self, text: str) -> List[Dict]:
        """Extract captions from full paper text.

        Args:
            text: Full text of the paper

        Returns:
            List of caption dictionaries with 'figure_id', 'text', 'type'
        """
        captions = []

        for pattern, caption_type in self.patterns:
            matches = pattern.findall(text)

            for match in matches:
                if len(match) >= 2:
                    figure_id = match[0].strip()
                    caption_text = match[1].strip()

                    # Clean up caption text
                    caption_text = re.sub(r'\s+', ' ', caption_text)
                    caption_text = caption_text[:1000]  # Limit length

                    if len(caption_text) > 20:  # Skip very short captions
                        captions.append({
                            'figure_id': figure_id,
                            'text': f"{figure_id}: {caption_text}",
                            'type': caption_type,
                        })

        # Deduplicate by figure_id
        seen = set()
        unique_captions = []
        for cap in captions:
            if cap['figure_id'] not in seen:
                seen.add(cap['figure_id'])
                unique_captions.append(cap)

        return unique_captions

    def extract_from_pdf(self, pdf_path: Path) -> List[Dict]:
        """Extract captions directly from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of caption dictionaries
        """
        try:
            doc = pymupdf.open(pdf_path)
            full_text = ""

            for page in doc:
                full_text += page.get_text() + "\n"

            doc.close()

            return self.extract_from_text(full_text)

        except Exception as e:
            logger.error(f"Error extracting captions from {pdf_path}: {e}")
            return []
```

**Tasks:**
- [ ] Create `backend/preprocessing/caption_extractor.py` with the code above

### 1.6 Enhanced PDF Processor

Update `backend/preprocessing/pdf_processor.py` to integrate all components:

```python
"""Enhanced PDF processing with multi-type chunking."""

import pymupdf
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .models import Chunk, ChunkType, PaperMetadata
from .chunker import PaperChunker
from .table_extractor import TableExtractor
from .caption_extractor import CaptionExtractor

logger = logging.getLogger(__name__)


class EnhancedPDFProcessor:
    """Process PDFs into multiple chunk types for RAG.

    This processor:
    1. Extracts text using PyMuPDF
    2. Extracts tables using pdfplumber
    3. Extracts figure/table captions
    4. Creates 6 chunk types: abstract, section, fine, full, caption, table
    """

    def __init__(
        self,
        fine_chunk_tokens: int = 500,
        fine_chunk_overlap: int = 128,
        section_max_tokens: int = 2000,
        use_camelot: bool = False,
    ):
        self.chunker = PaperChunker(
            fine_chunk_tokens=fine_chunk_tokens,
            fine_chunk_overlap=fine_chunk_overlap,
            section_max_tokens=section_max_tokens,
        )
        self.table_extractor = TableExtractor(use_camelot=use_camelot)
        self.caption_extractor = CaptionExtractor()

        logger.info(f"Initialized EnhancedPDFProcessor")

    def extract_text_and_metadata(self, pdf_path: Path) -> Tuple[str, PaperMetadata]:
        """Extract text and metadata from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (full_text, PaperMetadata)
        """
        doc = pymupdf.open(pdf_path)

        # Extract metadata
        meta = doc.metadata

        # Try to parse year from various fields
        year = None
        for field in ['creationDate', 'modDate']:
            if field in meta and meta[field]:
                try:
                    # Format: D:YYYYMMDDHHmmSS
                    date_str = meta[field]
                    if date_str.startswith('D:'):
                        year = int(date_str[2:6])
                        break
                except (ValueError, IndexError):
                    pass

        # Extract authors (often in 'author' field)
        authors = []
        if meta.get('author'):
            # Split by common delimiters
            import re
            authors = re.split(r'[,;]|\band\b', meta['author'])
            authors = [a.strip() for a in authors if a.strip()]

        metadata = PaperMetadata(
            paper_id="",  # Set by caller
            title=meta.get('title', pdf_path.stem),
            authors=authors,
            year=year,
            file_name=pdf_path.name,
            num_pages=len(doc),
        )

        # Extract full text
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n\n"

        doc.close()

        return full_text, metadata

    def process_pdf(self, pdf_path: Path, paper_id: str) -> List[Chunk]:
        """Process a single PDF into all chunk types.

        Args:
            pdf_path: Path to PDF file
            paper_id: Unique identifier for this paper

        Returns:
            List of Chunk objects (all types)
        """
        logger.info(f"Processing: {pdf_path.name}")

        # Extract text and metadata
        full_text, metadata = self.extract_text_and_metadata(pdf_path)
        metadata.paper_id = paper_id

        # Extract tables
        tables = self.table_extractor.extract(pdf_path)
        logger.debug(f"Extracted {len(tables)} tables")

        # Extract captions
        captions = self.caption_extractor.extract_from_text(full_text)
        logger.debug(f"Extracted {len(captions)} captions")

        # Create all chunk types
        chunks = self.chunker.chunk_paper(
            full_text=full_text,
            metadata=metadata,
            captions=captions,
            tables=tables,
        )

        # Log chunk type distribution
        type_counts = {}
        for chunk in chunks:
            type_counts[chunk.chunk_type.value] = type_counts.get(chunk.chunk_type.value, 0) + 1

        logger.info(f"Created {len(chunks)} chunks: {type_counts}")

        return chunks

    def validate_extraction(self, pdf_path: Path) -> Dict:
        """Validate extraction quality for a single PDF.

        Returns a report of extraction quality metrics.
        """
        full_text, metadata = self.extract_text_and_metadata(pdf_path)
        tables = self.table_extractor.extract(pdf_path)
        captions = self.caption_extractor.extract_from_text(full_text)

        # Quality checks
        issues = []

        # Check for common extraction problems
        if 'ÿ' in full_text or '�' in full_text:
            issues.append("Contains encoding errors (replacement characters)")

        if len(full_text) < 1000:
            issues.append("Suspiciously short text extraction")

        # Check for Greek letter corruption
        greek_patterns = ['α', 'β', 'γ', 'δ', 'μ', 'Δ']
        has_greek = any(g in full_text for g in greek_patterns)

        # Check for subscript/superscript patterns
        has_subscripts = 'CO2' in full_text or 'H2O' in full_text

        return {
            'file_name': pdf_path.name,
            'text_length': len(full_text),
            'num_pages': metadata.num_pages,
            'num_tables': len(tables),
            'num_captions': len(captions),
            'has_title': bool(metadata.title and metadata.title != pdf_path.stem),
            'has_authors': len(metadata.authors) > 0,
            'has_year': metadata.year is not None,
            'has_greek_chars': has_greek,
            'has_subscripts': has_subscripts,
            'issues': issues,
        }
```

**Tasks:**
- [ ] Replace `backend/preprocessing/pdf_processor.py` with the enhanced version above
- [ ] Update `backend/preprocessing/__init__.py` to export new classes

**Verification:**
```bash
cd backend && python -c "
from preprocessing.pdf_processor import EnhancedPDFProcessor
from preprocessing.models import ChunkType
print('All preprocessing modules imported successfully')
"
```

---

## Section 2: Retrieval Components

### 2.1 Embedding Service

Create `backend/retrieval/embedder.py`:

```python
"""Voyage AI embedding service."""

import logging
from typing import List, Optional
import voyageai

logger = logging.getLogger(__name__)


class VoyageEmbedder:
    """Wrapper for Voyage AI embeddings.

    Uses voyage-3-large for scientific text (1024 dimensions).
    """

    def __init__(self, api_key: str, model: str = "voyage-3-large"):
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        self.dimension = 1024  # voyage-3-large dimension

        logger.info(f"Initialized VoyageEmbedder with model {model}")

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 128
    ) -> List[List[float]]:
        """Embed documents (for indexing).

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            result = self.client.embed(
                texts=batch,
                model=self.model,
                input_type="document"
            )
            all_embeddings.extend(result.embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query (for retrieval).

        Args:
            text: Query text

        Returns:
            Embedding vector
        """
        result = self.client.embed(
            texts=[text],
            model=self.model,
            input_type="query"
        )
        return result.embeddings[0]

    def compute_mean_embedding(self, embeddings: List[List[float]]) -> List[float]:
        """Compute mean-pooled embedding.

        Used for FULL paper embeddings (average of section embeddings).

        Args:
            embeddings: List of embedding vectors

        Returns:
            Mean embedding vector
        """
        import numpy as np

        if not embeddings:
            return [0.0] * self.dimension

        embeddings_array = np.array(embeddings)
        mean_embedding = np.mean(embeddings_array, axis=0)

        # Normalize
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm

        return mean_embedding.tolist()
```

**Tasks:**
- [ ] Create `backend/retrieval/embedder.py` with the code above

### 2.2 Reranker Service

Create `backend/retrieval/reranker.py`:

```python
"""Cohere reranker service."""

import logging
from typing import List, Dict, Any
import cohere

logger = logging.getLogger(__name__)


class CohereReranker:
    """Rerank search results using Cohere.

    Critical for accuracy - the design doc emphasizes this is the
    biggest accuracy gain for minimal effort.
    """

    def __init__(self, api_key: str, model: str = "rerank-v3.5"):
        self.client = cohere.Client(api_key=api_key)
        self.model = model

        logger.info(f"Initialized CohereReranker with model {model}")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = 15,
        text_field: str = "text"
    ) -> List[Dict[str, Any]]:
        """Rerank documents by relevance to query.

        Args:
            query: The search query
            documents: List of documents with 'text' field and other metadata
            top_n: Number of top results to return
            text_field: Field name containing the text to rerank on

        Returns:
            Top N documents sorted by rerank score, with 'rerank_score' added
        """
        if not documents:
            return []

        # Extract texts for reranking
        texts = [doc[text_field] for doc in documents]

        # Call Cohere rerank
        response = self.client.rerank(
            query=query,
            documents=texts,
            model=self.model,
            top_n=min(top_n, len(documents))
        )

        # Build result with rerank scores
        reranked = []
        for result in response.results:
            doc = documents[result.index].copy()
            doc['rerank_score'] = result.relevance_score
            reranked.append(doc)

        return reranked
```

**Tasks:**
- [ ] Create `backend/retrieval/reranker.py` with the code above

### 2.3 Qdrant Store

Create `backend/retrieval/qdrant_store.py`:

```python
"""Qdrant vector store operations."""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

logger = logging.getLogger(__name__)


class QdrantStore:
    """Qdrant vector store for research paper chunks.

    Single collection with all 6 chunk types, differentiated by metadata.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "research_papers",
        embedding_dimension: int = 1024,
    ):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension

        logger.info(f"Connected to Qdrant at {host}:{port}")

    def ensure_collection(self) -> bool:
        """Create collection if it doesn't exist.

        Returns:
            True if collection was created, False if it already existed
        """
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.collection_name}")
            return True

        logger.info(f"Collection {self.collection_name} already exists")
        return False

    def upsert_chunks(
        self,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        payloads: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """Upsert chunks to collection.

        Args:
            chunk_ids: Unique IDs for each chunk
            embeddings: Embedding vectors
            payloads: Metadata payloads (from Chunk.to_payload())
            batch_size: Number of points per upsert call

        Returns:
            Total number of points upserted
        """
        total = 0

        for i in range(0, len(chunk_ids), batch_size):
            batch_ids = chunk_ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_payloads = payloads[i:i + batch_size]

            points = [
                PointStruct(
                    id=idx,  # Use numeric ID for Qdrant
                    vector=embedding,
                    payload={**payload, '_chunk_id': chunk_id}
                )
                for idx, (chunk_id, embedding, payload)
                in enumerate(zip(batch_ids, batch_embeddings, batch_payloads), start=i)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            total += len(points)

        logger.info(f"Upserted {total} chunks to {self.collection_name}")
        return total

    def search(
        self,
        query_embedding: List[float],
        limit: int = 50,
        chunk_types: Optional[List[str]] = None,
        paper_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum results to return
            chunk_types: Filter by chunk types (e.g., ["abstract", "fine"])
            paper_ids: Filter by paper IDs

        Returns:
            List of results with payload and score
        """
        # Build filter
        filter_conditions = []

        if chunk_types:
            filter_conditions.append(
                FieldCondition(
                    key="chunk_type",
                    match=MatchValue(value=chunk_types[0]) if len(chunk_types) == 1
                          else {"any": chunk_types}
                )
            )

        if paper_ids:
            filter_conditions.append(
                FieldCondition(
                    key="paper_id",
                    match=MatchValue(value=paper_ids[0]) if len(paper_ids) == 1
                          else {"any": paper_ids}
                )
            )

        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=query_filter,
        )

        return [
            {
                'score': r.score,
                **r.payload
            }
            for r in results
        ]

    def search_all_types(
        self,
        query_embedding: List[float],
        per_type_limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search all chunk types in parallel, then merge.

        Per the design doc: Search ALL chunk types in parallel (top 50 each),
        then merge results for reranking.

        Args:
            query_embedding: Query embedding vector
            per_type_limit: Results per chunk type

        Returns:
            Merged results from all chunk types
        """
        chunk_types = ["abstract", "section", "fine", "full", "caption", "table"]
        all_results = []

        for chunk_type in chunk_types:
            results = self.search(
                query_embedding=query_embedding,
                limit=per_type_limit,
                chunk_types=[chunk_type]
            )
            all_results.extend(results)

        # Sort by score descending
        all_results.sort(key=lambda x: x['score'], reverse=True)

        return all_results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection_name)

        return {
            'total_points': info.points_count,
            'status': info.status,
            'vector_dimension': self.embedding_dimension,
        }

    def delete_collection(self) -> bool:
        """Delete the collection (use with caution)."""
        self.client.delete_collection(self.collection_name)
        logger.warning(f"Deleted collection: {self.collection_name}")
        return True
```

**Tasks:**
- [ ] Create `backend/retrieval/qdrant_store.py` with the code above

### 2.4 Query Engine

Create `backend/retrieval/query_engine.py`:

```python
"""Full query pipeline with retrieval, reranking, and answer generation."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from anthropic import Anthropic

from .embedder import VoyageEmbedder
from .reranker import CohereReranker
from .qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result from a query."""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    retrieval_count: int
    reranked_count: int


class QueryEngine:
    """Full RAG query pipeline.

    Implements the workflow from design doc:
    1. Embed query (Voyage)
    2. Search ALL chunk types in parallel (top 50 each)
    3. Merge results + Rerank → top 15
    4. Expand fine chunks → pull parent section for context
    5. Deduplicate by paper_id
    6. Generate answer (Claude)
    7. Self-verify (optional)
    """

    def __init__(
        self,
        embedder: VoyageEmbedder,
        reranker: CohereReranker,
        store: QdrantStore,
        anthropic_client: Anthropic,
        claude_model: str = "claude-sonnet-4-20250514",
        retrieval_top_k: int = 50,
        rerank_top_n: int = 15,
    ):
        self.embedder = embedder
        self.reranker = reranker
        self.store = store
        self.anthropic = anthropic_client
        self.claude_model = claude_model
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_n = rerank_top_n

    def _expand_fine_chunks(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Expand fine chunks by pulling parent section for context.

        For fine chunks, we add the parent section text to provide
        more context for answer generation.
        """
        expanded = []
        parent_ids_seen = set()

        for result in results:
            if result.get('chunk_type') == 'fine':
                parent_id = result.get('parent_chunk_id')
                if parent_id and parent_id not in parent_ids_seen:
                    # Could query for parent section here
                    # For now, just mark that context expansion is needed
                    result['needs_context_expansion'] = True
                    parent_ids_seen.add(parent_id)

            expanded.append(result)

        return expanded

    def _deduplicate_by_paper(
        self,
        results: List[Dict[str, Any]],
        max_per_paper: int = 3
    ) -> List[Dict[str, Any]]:
        """Deduplicate results, keeping max N chunks per paper."""
        paper_counts = {}
        deduplicated = []

        for result in results:
            paper_id = result.get('paper_id', '')
            current_count = paper_counts.get(paper_id, 0)

            if current_count < max_per_paper:
                deduplicated.append(result)
                paper_counts[paper_id] = current_count + 1

        return deduplicated

    def _build_context(self, sources: List[Dict[str, Any]]) -> str:
        """Build context string from sources."""
        context_parts = []

        for idx, source in enumerate(sources, 1):
            title = source.get('title', 'Unknown')
            chunk_type = source.get('chunk_type', 'unknown')
            text = source.get('text', '')

            context_parts.append(
                f"[Source {idx}] {title} ({chunk_type})\n{text}\n"
            )

        return "\n---\n".join(context_parts)

    def _generate_answer(
        self,
        query: str,
        context: str,
        temperature: float = 0.3
    ) -> str:
        """Generate answer using Claude."""
        prompt = f"""You are a PhD-level research assistant specializing in biochemistry, chemistry, and biology.

Based on the following excerpts from research papers, answer the user's question. Be precise and cite specific sources when making claims. If information is insufficient, acknowledge limitations.

Context from papers:
{context}

User's question: {query}

Provide a comprehensive, accurate answer. Reference sources as [Source N] when citing specific information."""

        message = self.anthropic.messages.create(
            model=self.claude_model,
            max_tokens=2048,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        return message.content[0].text

    def _verify_answer(
        self,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Self-verify: Does cited chunk support this claim?

        Returns verification results for each citation.
        """
        # Extract citations from answer
        import re
        citations = re.findall(r'\[Source (\d+)\]', answer)

        verification = {
            'citations_found': len(citations),
            'sources_available': len(sources),
            'verified': True,  # Would implement actual verification
        }

        return verification

    def query(
        self,
        question: str,
        verify: bool = False
    ) -> QueryResult:
        """Execute full query pipeline.

        Args:
            question: User's question
            verify: Whether to run self-verification step

        Returns:
            QueryResult with answer, sources, and metadata
        """
        # 1. Embed query
        query_embedding = self.embedder.embed_query(question)

        # 2. Search all chunk types
        raw_results = self.store.search_all_types(
            query_embedding=query_embedding,
            per_type_limit=self.retrieval_top_k
        )

        retrieval_count = len(raw_results)
        logger.info(f"Retrieved {retrieval_count} chunks")

        if not raw_results:
            return QueryResult(
                answer="I couldn't find relevant information to answer your question.",
                sources=[],
                query=question,
                retrieval_count=0,
                reranked_count=0,
            )

        # 3. Rerank
        reranked = self.reranker.rerank(
            query=question,
            documents=raw_results,
            top_n=self.rerank_top_n
        )

        # 4. Expand fine chunks
        expanded = self._expand_fine_chunks(reranked)

        # 5. Deduplicate by paper
        deduplicated = self._deduplicate_by_paper(expanded)

        reranked_count = len(deduplicated)
        logger.info(f"After reranking and dedup: {reranked_count} chunks")

        # 6. Generate answer
        context = self._build_context(deduplicated)
        answer = self._generate_answer(question, context)

        # 7. Optional verification
        if verify:
            verification = self._verify_answer(answer, deduplicated)
            logger.info(f"Verification: {verification}")

        return QueryResult(
            answer=answer,
            sources=deduplicated,
            query=question,
            retrieval_count=retrieval_count,
            reranked_count=reranked_count,
        )
```

**Tasks:**
- [ ] Create `backend/retrieval/query_engine.py` with the code above
- [ ] Update `backend/retrieval/__init__.py` to export all classes

**Verification:**
```bash
cd backend && python -c "
from retrieval.embedder import VoyageEmbedder
from retrieval.reranker import CohereReranker
from retrieval.qdrant_store import QdrantStore
from retrieval.query_engine import QueryEngine
print('All retrieval modules imported successfully')
"
```

---

## Section 3: Processing Pipeline

### 3.1 Enhanced Pipeline

Update `backend/preprocessing/process_pdfs.py`:

```python
"""Main pipeline to process PDFs and store in Qdrant."""

import sys
import logging
import argparse
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from config import settings
from preprocessing.pdf_processor import EnhancedPDFProcessor
from preprocessing.models import Chunk, ChunkType
from retrieval.embedder import VoyageEmbedder
from retrieval.qdrant_store import QdrantStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFPipeline:
    """Pipeline to process PDFs and store in vector database."""

    def __init__(self):
        self.processor = EnhancedPDFProcessor(
            fine_chunk_tokens=settings.fine_chunk_tokens,
            fine_chunk_overlap=settings.fine_chunk_overlap,
            section_max_tokens=settings.section_max_tokens,
        )

        self.embedder = VoyageEmbedder(
            api_key=settings.voyage_api_key,
            model=settings.embedding_model
        )

        self.store = QdrantStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection_name,
            embedding_dimension=settings.embedding_dimension,
        )

        self.store.ensure_collection()

    def get_pdf_files(self, source_dir: Path, limit: Optional[int] = None) -> List[Path]:
        """Get list of PDF files to process."""
        if not source_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {source_dir}")

        pdf_files = list(source_dir.glob("**/*.pdf"))

        if limit:
            pdf_files = pdf_files[:limit]

        logger.info(f"Found {len(pdf_files)} PDF files")
        return pdf_files

    def process_pdf(self, pdf_path: Path, paper_id: str) -> int:
        """Process single PDF and store in Qdrant.

        Returns number of chunks created.
        """
        # Extract and chunk
        chunks = self.processor.process_pdf(pdf_path, paper_id)

        if not chunks:
            logger.warning(f"No chunks from {pdf_path.name}")
            return 0

        # Separate FULL chunks for mean pooling
        regular_chunks = [c for c in chunks if c.chunk_type != ChunkType.FULL]
        full_chunks = [c for c in chunks if c.chunk_type == ChunkType.FULL]

        # Embed regular chunks
        texts = [c.text for c in regular_chunks]
        embeddings = self.embedder.embed_documents(texts, batch_size=100)

        # Create mean-pooled embedding for FULL chunk
        if full_chunks:
            # Get section embeddings for mean pooling
            section_chunks = [c for c in regular_chunks if c.chunk_type == ChunkType.SECTION]
            if section_chunks:
                section_texts = [c.text for c in section_chunks]
                section_embeddings = self.embedder.embed_documents(section_texts)
                mean_embedding = self.embedder.compute_mean_embedding(section_embeddings)

                # Add full chunk with mean embedding
                regular_chunks.extend(full_chunks)
                embeddings.append(mean_embedding)

        # Store in Qdrant
        chunk_ids = [c.chunk_id for c in regular_chunks]
        payloads = [c.to_payload() for c in regular_chunks]

        self.store.upsert_chunks(chunk_ids, embeddings, payloads)

        return len(regular_chunks)

    def process_all(
        self,
        source_dir: Path,
        limit: Optional[int] = None,
        skip_existing: bool = True
    ) -> dict:
        """Process all PDFs in directory.

        Returns summary statistics.
        """
        pdf_files = self.get_pdf_files(source_dir, limit=limit)

        stats = {
            'total_pdfs': len(pdf_files),
            'processed': 0,
            'failed': 0,
            'total_chunks': 0,
            'errors': []
        }

        for idx, pdf_path in enumerate(tqdm(pdf_files, desc="Processing PDFs")):
            paper_id = f"paper_{idx:06d}"

            try:
                chunk_count = self.process_pdf(pdf_path, paper_id)
                stats['processed'] += 1
                stats['total_chunks'] += chunk_count
            except Exception as e:
                logger.error(f"Failed: {pdf_path.name}: {e}")
                stats['failed'] += 1
                stats['errors'].append({'file': pdf_path.name, 'error': str(e)})

        # Get final stats from Qdrant
        collection_stats = self.store.get_collection_stats()
        stats['qdrant_total_points'] = collection_stats['total_points']

        logger.info(f"Pipeline complete: {stats}")
        return stats

    def validate_sample(self, source_dir: Path, sample_size: int = 5) -> List[dict]:
        """Validate extraction on a sample of PDFs.

        Returns validation reports for manual review.
        """
        pdf_files = self.get_pdf_files(source_dir, limit=sample_size)
        reports = []

        for pdf_path in pdf_files:
            report = self.processor.validate_extraction(pdf_path)
            reports.append(report)

            # Print summary
            issues = report.get('issues', [])
            status = 'OK' if not issues else f"ISSUES: {issues}"
            logger.info(f"{report['file_name']}: {status}")

        return reports


def main():
    parser = argparse.ArgumentParser(description="Process PDFs for RAG")
    parser.add_argument("--source-dir", type=Path, default=settings.pdf_source_dir)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validate", action="store_true", help="Run validation only")
    parser.add_argument("--sample", type=int, default=5, help="Validation sample size")

    args = parser.parse_args()

    pipeline = PDFPipeline()

    if args.validate:
        reports = pipeline.validate_sample(args.source_dir, args.sample)
        print(f"\nValidation complete. {len(reports)} PDFs checked.")
    else:
        stats = pipeline.process_all(args.source_dir, limit=args.limit)
        print(f"\nProcessing complete: {stats}")


if __name__ == "__main__":
    main()
```

**Tasks:**
- [ ] Replace `backend/preprocessing/process_pdfs.py` with the code above

---

## Section 4: Evaluation Framework

### 4.1 Test Queries

Create `backend/evaluation/test_queries.py`:

```python
"""Test query definitions for Phase 1 validation."""

import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class QueryType(str, Enum):
    """Types of test queries."""
    FACTUAL = "factual"           # Specific fact retrieval
    CONCEPTUAL = "conceptual"     # Explanation/understanding
    COMPARATIVE = "comparative"   # Compare entities/methods
    METHODOLOGICAL = "methodological"  # How-to, protocols
    QUANTITATIVE = "quantitative" # Numbers, measurements
    ENTITY = "entity"             # Chemical/biological entity lookup


@dataclass
class TestQuery:
    """A test query for evaluation."""
    query_id: str
    query: str
    query_type: QueryType
    expected_topics: List[str]  # Topics that should be covered
    expected_entities: List[str] = None  # Entities that should be found
    notes: str = ""


# Phase 1 test queries - focused on biochemistry/chemistry domain
PHASE_1_QUERIES: List[TestQuery] = [
    # Factual queries
    TestQuery(
        query_id="f01",
        query="What is the IC50 value of compound X against target Y?",
        query_type=QueryType.QUANTITATIVE,
        expected_topics=["IC50", "inhibition", "binding"],
        notes="Template - replace X and Y with actual compounds from your papers"
    ),
    TestQuery(
        query_id="f02",
        query="What enzymes are involved in dopamine synthesis?",
        query_type=QueryType.FACTUAL,
        expected_topics=["tyrosine hydroxylase", "DOPA decarboxylase", "dopamine"],
        expected_entities=["L-DOPA", "tyrosine"]
    ),
    TestQuery(
        query_id="f03",
        query="What is the molecular weight of the protein complex?",
        query_type=QueryType.QUANTITATIVE,
        expected_topics=["molecular weight", "kDa", "protein"]
    ),

    # Conceptual queries
    TestQuery(
        query_id="c01",
        query="Explain the mechanism of CRISPR-Cas9 gene editing",
        query_type=QueryType.CONCEPTUAL,
        expected_topics=["Cas9", "guide RNA", "PAM", "double-strand break"],
        expected_entities=["sgRNA", "Cas9"]
    ),
    TestQuery(
        query_id="c02",
        query="How do kinase inhibitors work?",
        query_type=QueryType.CONCEPTUAL,
        expected_topics=["ATP binding", "competitive inhibition", "phosphorylation"]
    ),

    # Methodological queries
    TestQuery(
        query_id="m01",
        query="What buffer conditions were used for protein crystallization?",
        query_type=QueryType.METHODOLOGICAL,
        expected_topics=["buffer", "pH", "crystallization", "precipitant"]
    ),
    TestQuery(
        query_id="m02",
        query="How was cell viability measured in the cytotoxicity assays?",
        query_type=QueryType.METHODOLOGICAL,
        expected_topics=["MTT", "cell viability", "cytotoxicity", "IC50"]
    ),
    TestQuery(
        query_id="m03",
        query="What incubation temperature and time were used?",
        query_type=QueryType.METHODOLOGICAL,
        expected_topics=["temperature", "incubation", "hours", "degrees"]
    ),

    # Comparative queries
    TestQuery(
        query_id="cmp01",
        query="Compare the selectivity of different kinase inhibitors",
        query_type=QueryType.COMPARATIVE,
        expected_topics=["selectivity", "kinase", "inhibitor", "comparison"]
    ),
    TestQuery(
        query_id="cmp02",
        query="What are the differences between Type I and Type II inhibitors?",
        query_type=QueryType.COMPARATIVE,
        expected_topics=["Type I", "Type II", "binding mode", "DFG"]
    ),

    # Entity-focused queries (tests synonym handling)
    TestQuery(
        query_id="e01",
        query="What papers discuss BRCA1 mutations?",
        query_type=QueryType.ENTITY,
        expected_topics=["BRCA1", "mutation", "breast cancer"],
        expected_entities=["BRCA1", "BRCA2"]
    ),
    TestQuery(
        query_id="e02",
        query="Find information about reactive oxygen species",
        query_type=QueryType.ENTITY,
        expected_topics=["ROS", "oxidative stress", "antioxidant"],
        expected_entities=["ROS", "H2O2", "superoxide"]
    ),
]


def save_test_queries(queries: List[TestQuery], path: Path):
    """Save test queries to JSON file."""
    data = [asdict(q) for q in queries]

    # Convert enum to string
    for item in data:
        item['query_type'] = item['query_type'].value

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_test_queries(path: Path) -> List[TestQuery]:
    """Load test queries from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)

    queries = []
    for item in data:
        item['query_type'] = QueryType(item['query_type'])
        if item.get('expected_entities') is None:
            item['expected_entities'] = []
        queries.append(TestQuery(**item))

    return queries


if __name__ == "__main__":
    # Generate initial test queries file
    output_path = Path("../data/test_queries.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_test_queries(PHASE_1_QUERIES, output_path)
    print(f"Saved {len(PHASE_1_QUERIES)} test queries to {output_path}")
```

**Tasks:**
- [ ] Create `backend/evaluation/test_queries.py` with the code above
- [ ] Run the script to generate `backend/data/test_queries.json`
- [ ] Review and customize queries based on your actual paper corpus

### 4.2 Evaluator

Create `backend/evaluation/evaluator.py`:

```python
"""Evaluate retrieval and answer quality."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from .test_queries import TestQuery, load_test_queries

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating a single query."""
    query_id: str
    query: str
    query_type: str

    # Retrieval metrics
    retrieval_count: int
    reranked_count: int
    unique_papers: int
    chunk_types_retrieved: List[str]

    # Quality signals
    expected_topics_found: List[str]
    expected_topics_missing: List[str]
    expected_entities_found: List[str]
    expected_entities_missing: List[str]

    # Answer
    answer: str
    answer_length: int
    has_citations: bool

    # Timing
    latency_ms: float

    # Manual review fields (to be filled later)
    manual_relevance_score: Optional[int] = None  # 1-5
    manual_answer_quality: Optional[int] = None   # 1-5
    manual_notes: str = ""


class Evaluator:
    """Evaluate retrieval quality on test queries."""

    def __init__(self, query_engine):
        """Initialize evaluator.

        Args:
            query_engine: QueryEngine instance for running queries
        """
        self.query_engine = query_engine

    def _check_topics(
        self,
        text: str,
        expected: List[str]
    ) -> tuple[List[str], List[str]]:
        """Check which expected topics are present in text."""
        text_lower = text.lower()
        found = []
        missing = []

        for topic in expected:
            if topic.lower() in text_lower:
                found.append(topic)
            else:
                missing.append(topic)

        return found, missing

    def evaluate_query(self, test_query: TestQuery) -> EvaluationResult:
        """Evaluate a single test query.

        Args:
            test_query: TestQuery to evaluate

        Returns:
            EvaluationResult with metrics
        """
        import time

        start_time = time.time()
        result = self.query_engine.query(test_query.query)
        latency_ms = (time.time() - start_time) * 1000

        # Analyze retrieved chunks
        chunk_types = list(set(s.get('chunk_type', 'unknown') for s in result.sources))
        unique_papers = len(set(s.get('paper_id', '') for s in result.sources))

        # Combine all source text for topic/entity checking
        all_source_text = ' '.join(s.get('text', '') for s in result.sources)
        combined_text = all_source_text + ' ' + result.answer

        # Check expected topics
        topics_found, topics_missing = self._check_topics(
            combined_text,
            test_query.expected_topics
        )

        # Check expected entities
        entities_found, entities_missing = self._check_topics(
            combined_text,
            test_query.expected_entities or []
        )

        # Check for citations in answer
        import re
        has_citations = bool(re.search(r'\[Source \d+\]', result.answer))

        return EvaluationResult(
            query_id=test_query.query_id,
            query=test_query.query,
            query_type=test_query.query_type.value,
            retrieval_count=result.retrieval_count,
            reranked_count=result.reranked_count,
            unique_papers=unique_papers,
            chunk_types_retrieved=chunk_types,
            expected_topics_found=topics_found,
            expected_topics_missing=topics_missing,
            expected_entities_found=entities_found,
            expected_entities_missing=entities_missing,
            answer=result.answer,
            answer_length=len(result.answer),
            has_citations=has_citations,
            latency_ms=latency_ms,
        )

    def evaluate_all(
        self,
        test_queries: List[TestQuery],
        output_path: Optional[Path] = None
    ) -> List[EvaluationResult]:
        """Evaluate all test queries.

        Args:
            test_queries: List of TestQuery objects
            output_path: Optional path to save results

        Returns:
            List of EvaluationResult objects
        """
        results = []

        for tq in test_queries:
            logger.info(f"Evaluating: {tq.query_id} - {tq.query[:50]}...")

            try:
                result = self.evaluate_query(tq)
                results.append(result)

                # Log summary
                topic_coverage = len(result.expected_topics_found) / max(len(tq.expected_topics), 1)
                logger.info(f"  Topic coverage: {topic_coverage:.0%}, "
                           f"Latency: {result.latency_ms:.0f}ms")

            except Exception as e:
                logger.error(f"  Failed: {e}")

        # Save results
        if output_path:
            self._save_results(results, output_path)

        return results

    def _save_results(self, results: List[EvaluationResult], path: Path):
        """Save evaluation results to JSON."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(results),
            'results': [asdict(r) for r in results],
            'summary': self._compute_summary(results)
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved evaluation results to {path}")

    def _compute_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Compute summary statistics."""
        if not results:
            return {}

        # Topic coverage
        total_expected = sum(
            len(r.expected_topics_found) + len(r.expected_topics_missing)
            for r in results
        )
        total_found = sum(len(r.expected_topics_found) for r in results)

        # Entity coverage
        total_entities_expected = sum(
            len(r.expected_entities_found) + len(r.expected_entities_missing)
            for r in results
        )
        total_entities_found = sum(len(r.expected_entities_found) for r in results)

        return {
            'avg_latency_ms': sum(r.latency_ms for r in results) / len(results),
            'avg_retrieval_count': sum(r.retrieval_count for r in results) / len(results),
            'avg_reranked_count': sum(r.reranked_count for r in results) / len(results),
            'topic_coverage': total_found / max(total_expected, 1),
            'entity_coverage': total_entities_found / max(total_entities_expected, 1),
            'citation_rate': sum(1 for r in results if r.has_citations) / len(results),
            'by_query_type': self._summarize_by_type(results),
        }

    def _summarize_by_type(self, results: List[EvaluationResult]) -> Dict[str, Dict]:
        """Summarize results by query type."""
        by_type = {}

        for r in results:
            if r.query_type not in by_type:
                by_type[r.query_type] = {'count': 0, 'latencies': [], 'topic_hits': 0, 'topic_total': 0}

            by_type[r.query_type]['count'] += 1
            by_type[r.query_type]['latencies'].append(r.latency_ms)
            by_type[r.query_type]['topic_hits'] += len(r.expected_topics_found)
            by_type[r.query_type]['topic_total'] += len(r.expected_topics_found) + len(r.expected_topics_missing)

        # Compute averages
        for qt, data in by_type.items():
            data['avg_latency_ms'] = sum(data['latencies']) / len(data['latencies'])
            data['topic_coverage'] = data['topic_hits'] / max(data['topic_total'], 1)
            del data['latencies']

        return by_type
```

**Tasks:**
- [ ] Create `backend/evaluation/evaluator.py` with the code above
- [ ] Update `backend/evaluation/__init__.py` to export classes

---

## Section 5: Run Phase 1 Validation

### 5.1 Select Sample Papers

**Tasks:**
- [ ] Copy 50 representative PDFs to `backend/data/sample_papers/`
- [ ] Ensure diversity: different topics, years, journals
- [ ] Document selection criteria in `backend/data/sample_papers/README.md`

### 5.2 Validate Extraction Quality

Run extraction validation:

```bash
cd backend
python -m preprocessing.process_pdfs --source-dir data/sample_papers --validate --sample 20
```

**Tasks:**
- [ ] Run validation on 20 papers
- [ ] Review output for extraction issues
- [ ] Document issues in the Issues Log section below
- [ ] Decide if extraction quality is acceptable

### 5.3 Process Sample Papers

Start Qdrant first:

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Then process papers:

```bash
cd backend
python -m preprocessing.process_pdfs --source-dir data/sample_papers --limit 50
```

**Tasks:**
- [ ] Start Qdrant container
- [ ] Process 50 sample papers
- [ ] Verify chunks in Qdrant (check stats endpoint)

### 5.4 Customize Test Queries

**Tasks:**
- [ ] Review `backend/data/test_queries.json`
- [ ] Add 40 more queries specific to your papers (total 50)
- [ ] Include queries for:
  - [ ] At least 5 factual queries
  - [ ] At least 5 quantitative queries (IC50, concentrations)
  - [ ] At least 5 methodological queries (protocols)
  - [ ] At least 5 entity queries (chemical names, genes)
  - [ ] At least 5 comparative queries

### 5.5 Run Evaluation

Create `backend/run_evaluation.py`:

```python
"""Run Phase 1 evaluation."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config import settings
from anthropic import Anthropic
from retrieval.embedder import VoyageEmbedder
from retrieval.reranker import CohereReranker
from retrieval.qdrant_store import QdrantStore
from retrieval.query_engine import QueryEngine
from evaluation.test_queries import load_test_queries
from evaluation.evaluator import Evaluator


def main():
    # Initialize components
    embedder = VoyageEmbedder(api_key=settings.voyage_api_key)
    reranker = CohereReranker(api_key=settings.cohere_api_key)
    store = QdrantStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection_name
    )
    anthropic = Anthropic(api_key=settings.anthropic_api_key)

    query_engine = QueryEngine(
        embedder=embedder,
        reranker=reranker,
        store=store,
        anthropic_client=anthropic,
        claude_model=settings.claude_model,
        retrieval_top_k=settings.retrieval_top_k,
        rerank_top_n=settings.rerank_top_n,
    )

    # Load test queries
    queries = load_test_queries(settings.test_queries_path)
    print(f"Loaded {len(queries)} test queries")

    # Run evaluation
    evaluator = Evaluator(query_engine)
    results = evaluator.evaluate_all(
        queries,
        output_path=settings.evaluation_results_path
    )

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    with open(settings.evaluation_results_path) as f:
        import json
        data = json.load(f)
        summary = data['summary']

        print(f"Total queries: {data['total_queries']}")
        print(f"Avg latency: {summary['avg_latency_ms']:.0f}ms")
        print(f"Topic coverage: {summary['topic_coverage']:.1%}")
        print(f"Entity coverage: {summary['entity_coverage']:.1%}")
        print(f"Citation rate: {summary['citation_rate']:.1%}")

        print("\nBy query type:")
        for qt, stats in summary['by_query_type'].items():
            print(f"  {qt}: {stats['topic_coverage']:.1%} coverage, {stats['avg_latency_ms']:.0f}ms")


if __name__ == "__main__":
    main()
```

**Tasks:**
- [ ] Create `backend/run_evaluation.py` with the code above
- [ ] Run: `python run_evaluation.py`
- [ ] Review results in `backend/data/evaluation_results.json`

### 5.6 Analyze Failure Modes

After running evaluation, analyze failures:

**Tasks:**
- [ ] Review queries with low topic coverage
- [ ] Identify patterns in failures:
  - [ ] Synonym/entity issues?
  - [ ] Chunk type issues?
  - [ ] Extraction quality issues?
- [ ] Document findings below

---

## Section 6: Phase 1 Completion Checklist

### Environment
- [ ] All dependencies installed
- [ ] API keys configured (.env file)
- [ ] Qdrant running

### Preprocessing
- [ ] `models.py` created with data structures
- [ ] `section_detector.py` created
- [ ] `chunker.py` created with 6 chunk types
- [ ] `table_extractor.py` created
- [ ] `caption_extractor.py` created
- [ ] `pdf_processor.py` enhanced
- [ ] Extraction validated on sample papers

### Retrieval
- [ ] `embedder.py` created (Voyage)
- [ ] `reranker.py` created (Cohere)
- [ ] `qdrant_store.py` created
- [ ] `query_engine.py` created with full pipeline

### Evaluation
- [ ] `test_queries.py` created
- [ ] 50 test queries defined
- [ ] `evaluator.py` created
- [ ] Evaluation run completed

### Analysis
- [ ] Extraction error rate documented
- [ ] Failing query types identified
- [ ] Entity/synonym issues assessed
- [ ] Recommendations for Phase 2 documented

---

## Issues Log

Document issues encountered during implementation:

| Date | Section | Issue | Resolution | Status |
|------|---------|-------|------------|--------|
| | | | | |

---

## Phase 1 Results Summary

*Fill in after completing evaluation*

**Extraction Quality:**
- Error rate: ____%
- Common issues:

**Retrieval Performance:**
- Avg topic coverage: ____%
- Avg latency: ____ms
- Citation rate: ____%

**Failure Modes:**
1.
2.
3.

**Recommendations for Phase 2:**
1.
2.
3.

---

## Notes for Improvement

This implementation document could be enhanced by:

1. **Adding code verification snippets** - After each code block, include a Python snippet that verifies the module works correctly in isolation.

2. **Including rollback instructions** - For each section, describe how to undo changes if something goes wrong.

3. **Adding performance benchmarks** - Specify expected performance targets (e.g., "embedding 100 chunks should take < 30s").

4. **Including error handling patterns** - Provide standard try/except patterns for common failure modes.

5. **Adding integration tests** - Include pytest test cases that verify end-to-end functionality.

6. **Providing sample data** - Include 2-3 sample PDFs in the repo for testing without needing the full corpus.

7. **Adding observability** - Include structured logging and metrics collection for debugging.
