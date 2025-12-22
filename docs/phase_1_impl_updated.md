# Phase 1 Implementation Guide: Validation Pipeline (Revised)

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

**Phase 1 Goal:** Prove the pipeline works on 50 papers with your actual query types before scaling to 22,000.

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
- [x] Update `backend/requirements.txt` with the packages above
- [x] Run `pip install -r requirements.txt` in the virtual environment
- [x] Verify installations: `python -c "import cohere; import voyageai; print('OK')"`

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

# Query Classification
enable_query_classification: bool = True
enable_query_expansion: bool = True

# Phase 1 Settings
validation_sample_size: int = 50
test_queries_path: Path = Path("./data/test_queries.json")
evaluation_results_path: Path = Path("./data/evaluation_results.json")
```

**Tasks:**
- [x] Add the fields above to `backend/config.py`
- [x] Add `COHERE_API_KEY` to `.env.example`
- [x] Create `.env` file with actual API keys (do not commit)

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
│   ├── table_extractor.py    # NEW: Table extraction with pdfplumber
│   ├── caption_extractor.py  # NEW: Figure/table caption extraction
│   ├── section_detector.py   # NEW: Section detection
│   ├── models.py             # NEW: Data models
│   └── process_pdfs.py       # Enhanced pipeline
├── retrieval/
│   ├── __init__.py
│   ├── embedder.py           # NEW: Voyage embedding wrapper
│   ├── reranker.py           # NEW: Cohere reranker wrapper
│   ├── qdrant_store.py       # NEW: Qdrant operations
│   ├── query_classifier.py   # NEW: Query type classification
│   ├── query_expander.py     # NEW: Domain synonym expansion
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
- [x] Create `backend/preprocessing/__init__.py`
- [x] Create `backend/retrieval/` directory with `__init__.py`
- [x] Create `backend/evaluation/` directory with `__init__.py`
- [x] Create `backend/data/` directory
- [x] Create `backend/data/sample_papers/` directory

---

## Section 1: Domain-Specific Components (NEW)

**Important:** This section addresses the critical domain-specific issues identified in review. Complete this BEFORE building the retrieval pipeline.

### 1.1 Domain Synonym Dictionary

Create `backend/retrieval/domain_synonyms.py`:

```python
"""Domain-specific synonym dictionary for query expansion.

This module contains synonyms specific to the research domain:
- Estrogen receptor research
- Raman/SRS imaging
- Antimicrobial peptides
- Protein purification methods

Expand this dictionary based on failure analysis during evaluation.
"""

from typing import Dict, List, Set


# Domain synonym dictionary
# Format: canonical_term -> [list of synonyms/aliases]
DOMAIN_SYNONYMS: Dict[str, List[str]] = {
    # === Estrogen Receptor Research ===
    "ERα": [
        "estrogen receptor alpha",
        "ER-alpha", 
        "ESR1",
        "estrogen receptor",
        "ER alpha",
    ],
    "ERα LBD": [
        "estrogen receptor ligand binding domain",
        "ER LBD",
        "ligand binding domain",
    ],
    "Y537S": [
        "Y537S mutation",
        "tyrosine 537 serine",
        "Y537S mutant",
    ],
    "D538G": [
        "D538G mutation",
        "aspartate 538 glycine",
        "D538G mutant",
    ],
    "hormone-resistant breast cancer": [
        "endocrine-resistant",
        "ER+ breast cancer",
        "estrogen receptor positive",
        "hormone receptor positive",
    ],
    
    # === Imaging Techniques ===
    "SRS": [
        "stimulated Raman scattering",
        "coherent Raman",
        "SRS imaging",
        "SRS microscopy",
    ],
    "Raman": [
        "Raman spectroscopy",
        "Raman imaging",
        "vibrational imaging",
        "Raman microscopy",
    ],
    "Raman tag": [
        "vibrational tag",
        "Raman probe",
        "vibrational probe",
    ],
    
    # === Peptides ===
    "LL-37": [
        "LL37",
        "cathelicidin",
        "CAP18",
        "CAMP",
        "human cathelicidin",
    ],
    "AMP": [
        "antimicrobial peptide",
        "antimicrobial peptides",
        "host defense peptide",
        "host defense peptides",
        "HDPs",
    ],
    "stapled peptide": [
        "stapled peptides",
        "hydrocarbon-stapled",
        "macrocyclic peptide",
        "macrocyclic peptides",
        "peptide macrocycle",
    ],
    "cell-penetrating peptide": [
        "CPP",
        "cell penetrating peptide",
        "membrane-penetrating peptide",
    ],
    
    # === Chemical Tags and Labels ===
    "alkyne tag": [
        "alkyne",
        "diyne",
        "diyne tag",
        "alkyne label",
        "clickable tag",
        "bioorthogonal tag",
    ],
    "FITC": [
        "fluorescein",
        "fluorescein isothiocyanate",
        "fluorescent label",
        "fluorescent tag",
    ],
    "fluorescent label": [
        "fluorophore",
        "fluorescent dye",
        "fluorescent probe",
    ],
    
    # === Assays and Measurements ===
    "IC50": [
        "IC-50",
        "IC₅₀",
        "half-maximal inhibitory concentration",
        "inhibitory concentration",
    ],
    "CC50": [
        "CC-50",
        "CC₅₀",
        "half-maximal cytotoxic concentration",
        "cytotoxic concentration",
    ],
    "EC50": [
        "EC-50",
        "EC₅₀",
        "half-maximal effective concentration",
        "effective concentration",
    ],
    "MIC": [
        "minimum inhibitory concentration",
        "minimal inhibitory concentration",
    ],
    
    # === Purification Methods ===
    "FPLC": [
        "fast protein liquid chromatography",
        "protein chromatography",
        "liquid chromatography",
    ],
    "affinity chromatography": [
        "affinity FPLC",
        "affinity purification",
        "His-tag purification",
        "Ni-NTA",
        "nickel affinity",
        "immobilized metal affinity",
        "IMAC",
    ],
    "ion exchange": [
        "ion-exchange FPLC",
        "ion exchange chromatography",
        "IEX",
        "anion exchange",
        "cation exchange",
        "Q column",
        "SP column",
    ],
    "size exclusion": [
        "SEC",
        "gel filtration",
        "size exclusion chromatography",
        "molecular sieve",
    ],
    
    # === Biological Contexts ===
    "biofilm": [
        "biofilms",
        "bacterial biofilm",
        "biofilm matrix",
        "sessile bacteria",
    ],
    "planktonic": [
        "planktonic cells",
        "free-floating bacteria",
        "planktonic bacteria",
        "planktonic culture",
    ],
    "membrane permeability": [
        "membrane disruption",
        "membrane integrity",
        "membrane permeabilization",
        "pore formation",
    ],
    
    # === Structural Biology ===
    "coactivator": [
        "coactivator binding",
        "coactivator recruitment",
        "SRC",
        "steroid receptor coactivator",
    ],
    "allosteric": [
        "allosteric site",
        "allosteric binding",
        "allosteric modulation",
    ],
}

# Reverse lookup: alias -> canonical term
ALIAS_TO_CANONICAL: Dict[str, str] = {}
for canonical, aliases in DOMAIN_SYNONYMS.items():
    for alias in aliases:
        ALIAS_TO_CANONICAL[alias.lower()] = canonical


def get_synonyms(term: str) -> List[str]:
    """Get all synonyms for a term.
    
    Args:
        term: The term to look up (case-insensitive)
        
    Returns:
        List of synonyms including the canonical form.
        Returns empty list if term not found.
    """
    term_lower = term.lower()
    
    # Check if it's a canonical term
    for canonical, aliases in DOMAIN_SYNONYMS.items():
        if canonical.lower() == term_lower:
            return [canonical] + aliases
    
    # Check if it's an alias
    if term_lower in ALIAS_TO_CANONICAL:
        canonical = ALIAS_TO_CANONICAL[term_lower]
        return [canonical] + DOMAIN_SYNONYMS[canonical]
    
    return []


def find_entities_in_text(text: str) -> Set[str]:
    """Find known domain entities in text.
    
    Args:
        text: Text to search
        
    Returns:
        Set of canonical entity names found
    """
    text_lower = text.lower()
    found = set()
    
    for canonical, aliases in DOMAIN_SYNONYMS.items():
        all_terms = [canonical] + aliases
        for term in all_terms:
            if term.lower() in text_lower:
                found.add(canonical)
                break
    
    return found
```

**Tasks:**
- [x] Create `backend/retrieval/domain_synonyms.py` with the code above
- [x] Review and customize synonyms based on your specific papers

### 1.2 Query Expander

Create `backend/retrieval/query_expander.py`:

```python
"""Query expansion using domain synonyms.

Expands user queries with domain-specific synonyms to improve
retrieval of documents that use different terminology for the
same concepts.
"""

import re
import logging
from typing import List, Tuple, Set

from .domain_synonyms import DOMAIN_SYNONYMS, get_synonyms, find_entities_in_text

logger = logging.getLogger(__name__)


class QueryExpander:
    """Expand queries with domain synonyms."""
    
    def __init__(self, max_expansion_terms: int = 5):
        """Initialize query expander.
        
        Args:
            max_expansion_terms: Maximum number of synonym terms to add
        """
        self.max_expansion_terms = max_expansion_terms
        self.synonyms = DOMAIN_SYNONYMS
    
    def expand_query(self, query: str) -> Tuple[str, List[str]]:
        """Expand query with domain synonyms.
        
        Args:
            query: Original user query
            
        Returns:
            Tuple of (expanded_query, list_of_added_terms)
        """
        query_lower = query.lower()
        added_terms = []
        
        # Find entities in query and collect synonyms
        for canonical, aliases in self.synonyms.items():
            all_terms = [canonical.lower()] + [a.lower() for a in aliases]
            
            # Check if any form of this entity is in the query
            found_in_query = False
            for term in all_terms:
                if term in query_lower:
                    found_in_query = True
                    break
            
            if found_in_query:
                # Add synonyms that aren't already in query
                for alias in aliases[:3]:  # Limit per entity
                    if alias.lower() not in query_lower:
                        added_terms.append(alias)
        
        # Limit total expansion terms
        added_terms = added_terms[:self.max_expansion_terms]
        
        if added_terms:
            expanded = f"{query} (related terms: {', '.join(added_terms)})"
            logger.debug(f"Expanded query: {query} -> added {added_terms}")
            return expanded, added_terms
        
        return query, []
    
    def extract_entities(self, query: str) -> Set[str]:
        """Extract known domain entities from query.
        
        Args:
            query: User query
            
        Returns:
            Set of canonical entity names found
        """
        return find_entities_in_text(query)


# Convenience function
def expand_query(query: str, max_terms: int = 5) -> str:
    """Expand query with domain synonyms.
    
    Args:
        query: Original query
        max_terms: Maximum expansion terms
        
    Returns:
        Expanded query string
    """
    expander = QueryExpander(max_expansion_terms=max_terms)
    expanded, _ = expander.expand_query(query)
    return expanded
```

**Tasks:**
- [x] Create `backend/retrieval/query_expander.py` with the code above

### 1.3 Query Classifier

Create `backend/retrieval/query_classifier.py`:

```python
"""Query classification for retrieval strategy selection.

Classifies user queries to determine the optimal retrieval strategy.
Different query types benefit from different chunk types and retrieval parameters.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from anthropic import Anthropic

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Types of queries with different retrieval needs."""
    FACTUAL = "factual"           # Specific facts, definitions, values
    FRAMING = "framing"           # How to position/describe research
    METHODS = "methods"           # Technical protocols, procedures
    SUMMARY = "summary"           # Summarize a paper or findings
    COMPARATIVE = "comparative"   # Compare methods/findings across papers
    NOVELTY = "novelty"           # Assess what's new or defensible
    LIMITATIONS = "limitations"   # Discuss constraints or caveats


@dataclass
class QueryClassification:
    """Result of query classification."""
    query_type: QueryType
    confidence: float
    entities: list
    needs_cross_corpus: bool
    suggested_chunk_types: list
    suggested_top_k: int
    reasoning: str


# Retrieval strategies per query type
RETRIEVAL_STRATEGIES: Dict[QueryType, Dict[str, Any]] = {
    QueryType.FACTUAL: {
        "chunk_types": ["fine", "table", "caption"],
        "top_k": 50,
        "rerank_top_n": 15,
        "max_per_paper": 3,
    },
    QueryType.FRAMING: {
        "chunk_types": ["abstract", "section"],  # Need broader context
        "top_k": 30,
        "rerank_top_n": 20,
        "max_per_paper": 2,  # Want diversity across papers
    },
    QueryType.METHODS: {
        "chunk_types": ["section", "fine"],  # Focus on methods sections
        "section_filter": ["methods", "experimental", "synthesis"],
        "top_k": 50,
        "rerank_top_n": 15,
        "max_per_paper": 5,  # Methods details often span chunks
    },
    QueryType.SUMMARY: {
        "chunk_types": ["abstract", "section", "full"],
        "top_k": 20,
        "rerank_top_n": 10,
        "max_per_paper": 10,  # Focus on single paper
    },
    QueryType.COMPARATIVE: {
        "chunk_types": ["abstract", "section"],
        "top_k": 100,  # Need many papers
        "rerank_top_n": 25,
        "max_per_paper": 2,  # Want breadth
    },
    QueryType.NOVELTY: {
        "chunk_types": ["abstract", "section"],
        "section_filter": ["introduction", "discussion", "conclusion"],
        "top_k": 50,
        "rerank_top_n": 20,
        "max_per_paper": 2,
    },
    QueryType.LIMITATIONS: {
        "chunk_types": ["section"],
        "section_filter": ["discussion", "conclusion", "results"],
        "top_k": 30,
        "rerank_top_n": 15,
        "max_per_paper": 3,
    },
}


CLASSIFICATION_PROMPT = '''Classify this research question into exactly one category.

Categories:
- FACTUAL: Specific facts, definitions, values (IC50, mechanisms, "what is X")
- FRAMING: How to position, describe, or justify research for publication ("how do I frame X", "how should I describe")
- METHODS: Technical protocols, procedures, experimental details ("what buffer", "how was X measured")
- SUMMARY: Summarize a specific paper or findings ("summarize", "key findings")
- COMPARATIVE: Compare methods, approaches, or findings across papers ("compare", "difference between")
- NOVELTY: Assess what's new, different, or defensible ("is this novel", "strongest claim")
- LIMITATIONS: How to discuss constraints, caveats, missing data ("limitations", "without X")

Also extract:
1. Key domain entities (gene names, proteins, techniques, chemicals)
2. Whether this needs single-paper focus (false) or cross-corpus synthesis (true)

Question: {query}

Respond in this exact format:
QUERY_TYPE: <one of the categories above>
CONFIDENCE: <0.0-1.0>
ENTITIES: <comma-separated list or "none">
CROSS_CORPUS: <true or false>
REASONING: <brief explanation>'''


class QueryClassifier:
    """Classify queries to determine retrieval strategy."""
    
    def __init__(
        self,
        anthropic_client: Anthropic,
        model: str = "claude-sonnet-4-20250514"
    ):
        """Initialize classifier.
        
        Args:
            anthropic_client: Anthropic API client
            model: Claude model to use for classification
        """
        self.client = anthropic_client
        self.model = model
    
    def classify(self, query: str) -> QueryClassification:
        """Classify a query.
        
        Args:
            query: User query to classify
            
        Returns:
            QueryClassification with type and retrieval strategy
        """
        prompt = CLASSIFICATION_PROMPT.format(query=query)
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return self._parse_response(response.content[0].text, query)
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            # Default to factual on error
            return self._default_classification(query)
    
    def _parse_response(self, response: str, query: str) -> QueryClassification:
        """Parse Claude's classification response."""
        lines = response.strip().split('\n')
        
        query_type = QueryType.FACTUAL
        confidence = 0.5
        entities = []
        cross_corpus = False
        reasoning = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("QUERY_TYPE:"):
                type_str = line.split(":", 1)[1].strip().upper()
                try:
                    query_type = QueryType(type_str.lower())
                except ValueError:
                    query_type = QueryType.FACTUAL
                    
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    confidence = 0.5
                    
            elif line.startswith("ENTITIES:"):
                entities_str = line.split(":", 1)[1].strip()
                if entities_str.lower() != "none":
                    entities = [e.strip() for e in entities_str.split(",")]
                    
            elif line.startswith("CROSS_CORPUS:"):
                cross_corpus = line.split(":", 1)[1].strip().lower() == "true"
                
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
        
        strategy = RETRIEVAL_STRATEGIES[query_type]
        
        return QueryClassification(
            query_type=query_type,
            confidence=confidence,
            entities=entities,
            needs_cross_corpus=cross_corpus,
            suggested_chunk_types=strategy["chunk_types"],
            suggested_top_k=strategy["top_k"],
            reasoning=reasoning,
        )
    
    def _default_classification(self, query: str) -> QueryClassification:
        """Return default classification when parsing fails."""
        return QueryClassification(
            query_type=QueryType.FACTUAL,
            confidence=0.5,
            entities=[],
            needs_cross_corpus=False,
            suggested_chunk_types=["fine", "table", "caption"],
            suggested_top_k=50,
            reasoning="Default classification due to parsing error",
        )
    
    def get_retrieval_strategy(self, query_type: QueryType) -> Dict[str, Any]:
        """Get retrieval strategy for a query type."""
        return RETRIEVAL_STRATEGIES.get(query_type, RETRIEVAL_STRATEGIES[QueryType.FACTUAL])


# Simple heuristic classifier (no API call, for testing)
def classify_query_heuristic(query: str) -> QueryType:
    """Classify query using keyword heuristics.
    
    Use for testing or when API calls should be avoided.
    """
    query_lower = query.lower()
    
    # Framing patterns
    framing_patterns = [
        "how do i frame", "how should i describe", "how do i position",
        "how do i justify", "how can i present", "how do i explain",
        "how should i write", "how do i argue",
    ]
    if any(p in query_lower for p in framing_patterns):
        return QueryType.FRAMING
    
    # Methods patterns
    methods_patterns = [
        "what buffer", "what concentration", "how was", "protocol",
        "what temperature", "incubation", "procedure", "what method",
    ]
    if any(p in query_lower for p in methods_patterns):
        return QueryType.METHODS
    
    # Summary patterns
    if any(p in query_lower for p in ["summarize", "summary", "key findings", "main points"]):
        return QueryType.SUMMARY
    
    # Comparative patterns
    if any(p in query_lower for p in ["compare", "difference between", "versus", "vs"]):
        return QueryType.COMPARATIVE
    
    # Novelty patterns
    if any(p in query_lower for p in ["novel", "new", "unique", "first", "defensible"]):
        return QueryType.NOVELTY
    
    # Limitations patterns
    if any(p in query_lower for p in ["limitation", "without", "caveat", "constraint"]):
        return QueryType.LIMITATIONS
    
    # Default to factual
    return QueryType.FACTUAL
```

**Tasks:**
- [x] Create `backend/retrieval/query_classifier.py` with the code above

**Verification:**
```bash
cd backend && python -c "
from retrieval.domain_synonyms import get_synonyms, DOMAIN_SYNONYMS
from retrieval.query_expander import QueryExpander
from retrieval.query_classifier import classify_query_heuristic, QueryType

# Test synonyms
print('ERα synonyms:', get_synonyms('ERα'))
print('LL-37 synonyms:', get_synonyms('LL37'))

# Test expansion
expander = QueryExpander()
expanded, terms = expander.expand_query('How do I justify the alkyne tag does not affect LL-37 activity?')
print(f'Expanded query: {expanded}')

# Test classification
query = 'How do I frame Raman imaging as a platform?'
qtype = classify_query_heuristic(query)
print(f'Query type: {qtype}')
assert qtype == QueryType.FRAMING
print('All domain components working!')
"
```

---

## Section 2: Enhanced PDF Extraction

### 2.1 Data Models

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
    section_name: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    figure_id: Optional[str] = None
    page_numbers: List[int] = field(default_factory=list)

    # Paper-level metadata (denormalized for retrieval)
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    
    # New fields
    project_tag: Optional[str] = None
    research_area: Optional[str] = None

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
            "project_tag": self.project_tag,
            "research_area": self.research_area,
            "token_count": self.token_count,
        }
```

**Tasks:**
- [x] Create `backend/preprocessing/models.py` with the code above

### 2.2 Section Detection

Create `backend/preprocessing/section_detector.py` to identify paper sections:

```python
"""Detect and classify sections in research papers.

Enhanced patterns for chemistry and biology papers including:
- Combined Results and Discussion sections
- Chemistry-specific sections (Synthesis, Characterization)
- Various numbering schemes
"""

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
# Order matters - more specific patterns first
SECTION_PATTERNS = [
    # Combined sections (common in chemistry)
    (r"^(?:\d+\.?\s*)?results?\s*(?:and|&)\s*discussion", "results_discussion", 1),
    (r"^(?:\d+\.?\s*)?materials?\s*(?:and|&)\s*methods?", "methods", 1),
    
    # Standard sections
    (r"^(?:1\.?\s*)?(?:introduction|background)", "introduction", 1),
    (r"^(?:2\.?\s*)?(?:methods?|experimental\s*(?:section|procedures?)?)", "methods", 1),
    (r"^(?:3\.?\s*)?(?:results?)", "results", 1),
    (r"^(?:4\.?\s*)?(?:discussion)", "discussion", 1),
    (r"^(?:5\.?\s*)?(?:conclusion|conclusions|summary|concluding\s*remarks)", "conclusion", 1),
    (r"^abstract", "abstract", 1),
    
    # Chemistry-specific sections
    (r"^(?:\d+\.?\s*)?(?:synthesis|synthetic\s*procedures?|general\s*synthesis)", "synthesis", 1),
    (r"^(?:\d+\.?\s*)?(?:characterization|compound\s*characterization)", "characterization", 1),
    (r"^(?:\d+\.?\s*)?(?:biological\s*evaluation|bioactivity|biological\s*activity)", "bioactivity", 1),
    (r"^(?:\d+\.?\s*)?(?:molecular\s*docking|docking\s*studies|computational)", "computational", 1),
    (r"^(?:\d+\.?\s*)?(?:structure[- ]activity|SAR|structure-activity\s*relationship)", "sar", 1),
    
    # Other common sections
    (r"^(?:references?|bibliography|literature\s*cited)", "references", 1),
    (r"^(?:acknowledg|funding|support|author\s*contributions)", "acknowledgments", 1),
    (r"^(?:supplementa|supporting\s*information|appendix|SI\s)", "supplementary", 1),
    (r"^(?:abbreviations?|glossary)", "abbreviations", 1),
    
    # Numbered subsections (level 2)
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
            
            # Skip lines that are too long (unlikely to be headers)
            if len(line_stripped) > 100:
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
            r'abstract[:\s]*\n*(.*?)(?=\n\s*(?:introduction|keywords?|1\.|1\s|background)|\Z)',
            re.IGNORECASE | re.DOTALL
        )
        match = abstract_pattern.search(text[:8000])  # Abstract should be near start

        if match:
            abstract = match.group(1).strip()
            # Clean up: remove excessive whitespace
            abstract = re.sub(r'\s+', ' ', abstract)
            # Remove common artifacts
            abstract = re.sub(r'^[:\s]+', '', abstract)
            return abstract if len(abstract) > 50 else None

        return None
    
    def get_section_by_type(
        self, 
        sections: List[Section], 
        section_types: List[str]
    ) -> List[Section]:
        """Filter sections by normalized type.
        
        Args:
            sections: List of detected sections
            section_types: Types to include (e.g., ["methods", "results"])
            
        Returns:
            Filtered list of sections
        """
        return [s for s in sections if s.normalized_name in section_types]
```

**Tasks:**
- [x] Create `backend/preprocessing/section_detector.py` with the code above
- [ ] Manually test on 5-10 papers to verify section detection quality

 Number of top results to return
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
- [x] Create `backend/retrieval/reranker.py` with the code above

### 3.3 Qdrant Store

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
    MatchAny,
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
        """Create collection if it doesn't exist."""
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
        """Upsert chunks to collection."""
        total = 0

        for i in range(0, len(chunk_ids), batch_size):
            batch_ids = chunk_ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_payloads = payloads[i:i + batch_size]

            points = [
                PointStruct(
                    id=idx,
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
        section_names: Optional[List[str]] = None,
        paper_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks with filtering."""
        filter_conditions = []

        if chunk_types:
            if len(chunk_types) == 1:
                filter_conditions.append(
                    FieldCondition(key="chunk_type", match=MatchValue(value=chunk_types[0]))
                )
            else:
                filter_conditions.append(
                    FieldCondition(key="chunk_type", match=MatchAny(any=chunk_types))
                )

        if section_names:
            if len(section_names) == 1:
                filter_conditions.append(
                    FieldCondition(key="section_name", match=MatchValue(value=section_names[0]))
                )
            else:
                filter_conditions.append(
                    FieldCondition(key="section_name", match=MatchAny(any=section_names))
                )

        if paper_ids:
            if len(paper_ids) == 1:
                filter_conditions.append(
                    FieldCondition(key="paper_id", match=MatchValue(value=paper_ids[0]))
                )
            else:
                filter_conditions.append(
                    FieldCondition(key="paper_id", match=MatchAny(any=paper_ids))
                )

        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=query_filter,
        )

        return [{'score': r.score, **r.payload} for r in results]

    def search_by_strategy(
        self,
        query_embedding: List[float],
        chunk_types: List[str],
        top_k: int = 50,
        section_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search using a retrieval strategy."""
        all_results = []

        for chunk_type in chunk_types:
            results = self.search(
                query_embedding=query_embedding,
                limit=top_k,
                chunk_types=[chunk_type],
                section_names=section_filter,
            )
            all_results.extend(results)

        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by its ID."""
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="_chunk_id", match=MatchValue(value=chunk_id))]
            ),
            limit=1,
        )
        
        if results and results[0]:
            point = results[0][0]
            return point.payload
        return None

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
- [x] Create `backend/retrieval/qdrant_store.py` with the code above

### 3.4 Query Engine (Enhanced)

Create `backend/retrieval/query_engine.py` - see full implementation in the complete document.

**Key enhancements:**
- Query classification before retrieval
- Domain synonym expansion
- Strategy-based retrieval (different chunk types per query type)
- Query-type-specific answer generation prompts
- Parent chunk expansion for fine chunks

**Tasks:**
- [x] Create `backend/retrieval/query_engine.py`
- [x] Update `backend/retrieval/__init__.py` to export all classes

---

## Section 5: Evaluation Framework

### 5.1 Test Queries (Based on Actual Usage)

Create `backend/evaluation/test_queries.py` with 50 queries based on actual usage patterns:

| Category | Count | Examples |
|----------|-------|----------|
| Framing/positioning | 12 | "How do I frame Raman imaging as a platform?" |
| Methods writing | 8 | "How should I describe Affinity FPLC vs Ion-Exchange FPLC?" |
| Factual | 8 | "What is the difference between CC50 and IC50?" |
| Controls/validation | 6 | "How do I justify the alkyne tag doesn't disrupt activity?" |
| Novelty | 6 | "What is the strongest defensible novelty claim?" |
| Limitations | 5 | "How do I explain localization without colocalization markers?" |
| Comparative | 5 | "How does this differ from prior Raman imaging work?" |

**Tasks:**
- [x] Create `backend/evaluation/test_queries.py`
- [x] Generate `backend/data/test_queries.json`

### 5.2 Evaluator with Human Evaluation Support

Create `backend/evaluation/evaluator.py`:

```python
"""Evaluate retrieval and answer quality with human evaluation support."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime

from .test_queries import TestQuery, QueryType, load_test_queries

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating a single query."""
    query_id: str
    query: str
    query_type: str
    
    # Classification results
    classified_type: str
    classification_correct: bool
    expanded_query: str
    
    # Retrieval metrics
    retrieval_count: int
    reranked_count: int
    unique_papers: int
    chunk_types_retrieved: List[str]
    
    # Expected vs actual
    expected_topics_found: List[str]
    expected_topics_missing: List[str]
    expected_entities_found: List[str]
    expected_entities_missing: List[str]
    expected_chunk_types_hit: List[str]
    expected_chunk_types_missed: List[str]
    
    # Answer
    answer: str
    answer_length: int
    has_citations: bool
    
    # Timing
    latency_ms: float
    
    # Human evaluation fields
    requires_human_eval: bool = False
    human_relevance_score: Optional[int] = None  # 1-5
    human_answer_quality: Optional[int] = None   # 1-5
    human_notes: str = ""


class Evaluator:
    """Evaluate retrieval quality on test queries."""

    def __init__(self, query_engine):
        self.query_engine = query_engine

    def _check_coverage(
        self,
        text: str,
        expected: List[str]
    ) -> tuple[List[str], List[str]]:
        """Check which expected items are present in text."""
        text_lower = text.lower()
        found = []
        missing = []

        for item in expected:
            if item.lower() in text_lower:
                found.append(item)
            else:
                missing.append(item)

        return found, missing

    def evaluate_query(self, test_query: TestQuery) -> EvaluationResult:
        """Evaluate a single test query."""
        import time

        start_time = time.time()
        result = self.query_engine.query(test_query.query)
        latency_ms = (time.time() - start_time) * 1000

        # Analyze retrieved chunks
        chunk_types = list(set(s.get('chunk_type', 'unknown') for s in result.sources))
        unique_papers = len(set(s.get('paper_id', '') for s in result.sources))

        # Combine all text for topic/entity checking
        all_text = result.answer + " " + " ".join(s.get('text', '') for s in result.sources)

        # Check coverage
        topics_found, topics_missing = self._check_coverage(all_text, test_query.expected_topics)
        entities_found, entities_missing = self._check_coverage(all_text, test_query.expected_entities)
        
        # Check chunk type coverage
        chunk_types_hit = [ct for ct in test_query.expected_chunk_types if ct in chunk_types]
        chunk_types_missed = [ct for ct in test_query.expected_chunk_types if ct not in chunk_types]

        # Check citations
        has_citations = '[Source' in result.answer or '[source' in result.answer

        return EvaluationResult(
            query_id=test_query.query_id,
            query=test_query.query,
            query_type=test_query.query_type.value,
            classified_type=result.query_type.value,
            classification_correct=(test_query.query_type.value == result.query_type.value),
            expanded_query=result.expanded_query,
            retrieval_count=result.retrieval_count,
            reranked_count=result.reranked_count,
            unique_papers=unique_papers,
            chunk_types_retrieved=chunk_types,
            expected_topics_found=topics_found,
            expected_topics_missing=topics_missing,
            expected_entities_found=entities_found,
            expected_entities_missing=entities_missing,
            expected_chunk_types_hit=chunk_types_hit,
            expected_chunk_types_missed=chunk_types_missed,
            answer=result.answer,
            answer_length=len(result.answer),
            has_citations=has_citations,
            latency_ms=latency_ms,
            requires_human_eval=test_query.requires_human_eval,
        )

    def evaluate_all(
        self,
        queries: List[TestQuery],
        output_path: Optional[Path] = None
    ) -> List[EvaluationResult]:
        """Evaluate all test queries."""
        results = []

        for query in queries:
            logger.info(f"Evaluating: {query.query_id}")
            try:
                result = self.evaluate_query(query)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate {query.query_id}: {e}")

        if output_path:
            self._save_results(results, output_path)

        return results

    def _save_results(self, results: List[EvaluationResult], path: Path):
        """Save results to JSON."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(results),
            'results': [asdict(r) for r in results],
            'summary': self._compute_summary(results),
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved evaluation results to {path}")

    def _compute_summary(self, results: List[EvaluationResult]) -> Dict:
        """Compute summary statistics."""
        if not results:
            return {}

        total_topics = sum(len(r.expected_topics_found) + len(r.expected_topics_missing) for r in results)
        total_topics_found = sum(len(r.expected_topics_found) for r in results)
        
        total_entities = sum(len(r.expected_entities_found) + len(r.expected_entities_missing) for r in results)
        total_entities_found = sum(len(r.expected_entities_found) for r in results)
        
        classification_correct = sum(1 for r in results if r.classification_correct)

        return {
            'avg_latency_ms': sum(r.latency_ms for r in results) / len(results),
            'topic_coverage': total_topics_found / max(total_topics, 1),
            'entity_coverage': total_entities_found / max(total_entities, 1),
            'citation_rate': sum(1 for r in results if r.has_citations) / len(results),
            'classification_accuracy': classification_correct / len(results),
            'requires_human_eval': sum(1 for r in results if r.requires_human_eval),
            'by_query_type': self._summarize_by_type(results),
        }

    def _summarize_by_type(self, results: List[EvaluationResult]) -> Dict[str, Dict]:
        """Summarize results by query type."""
        by_type = {}

        for r in results:
            if r.query_type not in by_type:
                by_type[r.query_type] = {
                    'count': 0,
                    'latencies': [],
                    'topic_hits': 0,
                    'topic_total': 0,
                    'classification_correct': 0,
                }

            by_type[r.query_type]['count'] += 1
            by_type[r.query_type]['latencies'].append(r.latency_ms)
            by_type[r.query_type]['topic_hits'] += len(r.expected_topics_found)
            by_type[r.query_type]['topic_total'] += len(r.expected_topics_found) + len(r.expected_topics_missing)
            if r.classification_correct:
                by_type[r.query_type]['classification_correct'] += 1

        # Compute averages
        for qt, data in by_type.items():
            data['avg_latency_ms'] = sum(data['latencies']) / len(data['latencies'])
            data['topic_coverage'] = data['topic_hits'] / max(data['topic_total'], 1)
            data['classification_accuracy'] = data['classification_correct'] / data['count']
            del data['latencies']

        return by_type


def generate_human_eval_template(results_path: Path, output_path: Path):
    """Generate a human evaluation template from results."""
    with open(results_path) as f:
        data = json.load(f)
    
    template = []
    for r in data['results']:
        if r.get('requires_human_eval'):
            template.append({
                'query_id': r['query_id'],
                'query': r['query'],
                'query_type': r['query_type'],
                'answer': r['answer'][:1000] + '...' if len(r['answer']) > 1000 else r['answer'],
                'sources_count': r['reranked_count'],
                'human_relevance_score': None,  # 1-5: How relevant are the retrieved sources?
                'human_answer_quality': None,   # 1-5: How good is the answer?
                'human_notes': '',
            })
    
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Generated human evaluation template with {len(template)} queries")
    print(f"Please fill in human_relevance_score (1-5) and human_answer_quality (1-5)")
```

**Tasks:**
- [x] Create `backend/evaluation/evaluator.py`
- [x] Update `backend/evaluation/__init__.py`

---

## Section 6: Run Phase 1 Validation

### 6.1 Select Sample Papers

**Tasks:**
- [ ] Copy 50 representative PDFs to `backend/data/sample_papers/`
- [ ] Include papers from each research area:
  - [ ] ERα/estrogen receptor papers (10-15)
  - [ ] Raman/SRS imaging papers (10-15)  
  - [ ] LL-37/antimicrobial peptide papers (10-15)
  - [ ] Protein purification methods papers (5-10)
- [ ] Document selection in `backend/data/sample_papers/README.md`

### 6.2 Validate Extraction Quality

```bash
cd backend
python -m preprocessing.process_pdfs --source-dir data/sample_papers --validate --sample 10
```

**Checklist:**
- [ ] Section detection works for most papers
- [ ] Tables with IC50/CC50 values are extracted
- [ ] Figure captions are captured
- [ ] No major encoding errors
- [ ] Greek letters preserved (α, β, μ)

### 6.3 Process Sample Papers

```bash
# Start Qdrant
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# Process papers
cd backend
python -m preprocessing.process_pdfs --source-dir data/sample_papers --limit 50
```

**Tasks:**
- [ ] Start Qdrant container
- [ ] Process 50 sample papers
- [ ] Verify chunks in Qdrant: `curl http://localhost:6333/collections/research_papers`

### 6.4 Run Evaluation

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
from evaluation.evaluator import Evaluator, generate_human_eval_template


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
        enable_classification=settings.enable_query_classification,
        enable_expansion=settings.enable_query_expansion,
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
        print(f"Classification accuracy: {summary['classification_accuracy']:.1%}")
        print(f"Queries needing human eval: {summary['requires_human_eval']}")

        print("\nBy query type:")
        for qt, stats in summary['by_query_type'].items():
            print(f"  {qt}: {stats['topic_coverage']:.1%} coverage, "
                  f"{stats['classification_accuracy']:.1%} classification, "
                  f"{stats['avg_latency_ms']:.0f}ms")

    # Generate human evaluation template
    human_eval_path = Path("./data/human_evaluation_template.json")
    generate_human_eval_template(settings.evaluation_results_path, human_eval_path)


if __name__ == "__main__":
    main()
```

**Tasks:**
- [x] Create `backend/run_evaluation.py`
- [ ] Run: `python run_evaluation.py`
- [ ] Review results in `backend/data/evaluation_results.json`
- [ ] Complete human evaluation for framing queries

### 6.5 Analyze Failure Modes

**Tasks:**
- [ ] Review queries with low topic coverage
- [ ] Check if query classification is accurate
- [ ] Verify synonym expansion is helping (not adding noise)
- [ ] Identify patterns:
  - [ ] Are framing queries retrieving appropriate sections?
  - [ ] Are methods queries finding methods sections?
  - [ ] Are entity queries finding the right synonyms?

---

## Section 7: Phase 1 Completion Checklist

### Domain Components
- [x] Domain synonym dictionary created and customized
- [x] Query expander working
- [x] Query classifier working (test with heuristic fallback)

### Preprocessing
- [x] `models.py` with enhanced metadata
- [x] `section_detector.py` with chemistry patterns
- [x] `chunker.py` with 6 chunk types
- [x] `table_extractor.py` tested on IC50 tables
- [x] `caption_extractor.py` working
- [x] `pdf_processor.py` integrated

### Retrieval
- [x] `embedder.py` with mean pooling
- [x] `reranker.py` working
- [x] `qdrant_store.py` with filtering
- [x] `query_classifier.py` with strategies
- [x] `query_expander.py` with domain synonyms
- [x] `query_engine.py` with full pipeline

### Evaluation
- [x] 50 test queries defined (from actual usage)
- [x] Evaluator with human eval support
- [ ] Evaluation run completed
- [ ] Human evaluation completed for framing queries

---

## Issues Log

| Date | Section | Issue | Resolution | Status |
|------|---------|-------|------------|--------|
| | | | | |

---

## Phase 1 Results Summary

*Fill in after completing evaluation*

**Extraction Quality:**
- Error rate: ____%
- Section detection accuracy: ____%
- Table extraction success: ____%

**Classification Performance:**
- Overall accuracy: ____%
- By type:
  - Framing: ____%
  - Methods: ____%
  - Factual: ____%

**Retrieval Performance:**
- Avg topic coverage: ____%
- Avg entity coverage: ____%
- Avg latency: ____ms
- Citation rate: ____%

**Human Evaluation (Framing Queries):**
- Avg relevance score: ____/5
- Avg answer quality: ____/5

**Failure Modes:**
1. 
2. 
3. 

**Recommendations for Phase 2:**
1. 
2. 
3. 

---

## Key Differences from Original Implementation

| Area | Original | Updated | Rationale |
|------|----------|---------|-----------|
| Query handling | Single retrieval strategy | Query classification + type-specific retrieval | User queries are mostly framing/synthesis, not factual |
| Synonym handling | Placeholder | Full domain dictionary | Scientific nomenclature varies significantly |
| Test queries | Generic examples | Based on actual user questions | Evaluate what users actually ask |
| Evaluation | Automated metrics only | Automated + human eval | Framing queries need human judgment |
| Section detection | Basic IMRAD | Enhanced with chemistry patterns | Chemistry papers have different structure |
| Full paper embedding | Truncation | Mean pooling | Preserves conclusion content |
