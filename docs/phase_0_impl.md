# Phase 0 Implementation Guide: Data Cleaning & Paper Classification

## Agent Instructions

**READ THIS FIRST - Instructions for LLM Code Agents**

This document guides the data cleaning phase before Phase 1. The goal is to filter ~22,000 PDFs down to only research papers and scientific articles.

1. **Work sequentially through sections** - Each section builds on previous ones.
2. **Mark tasks complete** - After completing each task, update `[ ]` to `[x]`.
3. **Run in stages** - Use fast filters first, then progressively more expensive ones.
4. **Preserve originals** - Never delete source files. Create filtered output directories.
5. **Log everything** - Every classification decision should be logged for review.
6. **Sample before scaling** - Test each filter on 100 files before running on all 22k.

**Source Directory:** `/Users/merlin/projects/researchPaperAgent/resource/Total Backup 2025-09-20-recover-Converted.Data/PDF/`

**Observed File Types (from sampling):**

- Research papers (target): `acs.biomac.7b01245.pdf`, `1-s2.0-S0032386114002857-main.pdf`
- Receipts: `Walmart 2010-0703.pdf`, `Costco 2011-09-23.pdf`
- Tax/financial: `Etrade 2017 consolidated 1099.pdf`
- Homework/lectures: `hw0.pdf`, `lecture5.pdf`, `physics 43.pdf`
- Personal scans: `2014-07-14 Live Scan.pdf`
- Legal docs: `Traffic court trial checklist.pdf`
- Order confirmations: `Sigma-AldrichOrderConfirmation.pdf`
- Music sheets: `bwv855a-let.pdf`

---

## Section 0: Setup

### 0.1 Directory Structure

```
backend/
├── data_cleaning/
│   ├── __init__.py
│   ├── classifier.py       # Main classification logic
│   ├── filters/
│   │   ├── __init__.py
│   │   ├── filename_filter.py
│   │   ├── metadata_filter.py
│   │   ├── content_filter.py
│   │   └── llm_filter.py
│   ├── run_cleaning.py     # Main script
│   └── review_tool.py      # Manual review helper
├── data/
│   ├── cleaning_logs/      # Classification logs
│   ├── classified/
│   │   ├── papers/         # Confirmed research papers
│   │   ├── rejected/       # Non-papers (symlinks for review)
│   │   └── uncertain/      # Needs manual review
│   └── cleaning_config.json
```

**Tasks:**

- [ X ] Create `backend/data_cleaning/` directory
- [ X ] Create `backend/data_cleaning/filters/` directory
- [ X ] Create `backend/data/cleaning_logs/` directory
- [ X ] Create `backend/data/classified/papers/` directory
- [ X ] Create `backend/data/classified/rejected/` directory
- [ X ] Create `backend/data/classified/uncertain/` directory
- [ X ] Create all `__init__.py` files

### 0.2 Configuration

Create `backend/data/cleaning_config.json`:

```json
{
  "source_dir": "/Users/merlin/projects/researchPaperAgent/resource/Total Backup 2025-09-20-recover-Converted.Data/PDF",
  "output_dir": "/Users/merlin/projects/researchPaperAgent/backend/data/classified",
  "log_dir": "/Users/merlin/projects/researchPaperAgent/backend/data/cleaning_logs",
  "use_symlinks": true,
  "filters": {
    "filename": { "enabled": true, "order": 1 },
    "metadata": { "enabled": true, "order": 2 },
    "content": { "enabled": true, "order": 3 },
    "llm": { "enabled": true, "order": 4, "only_uncertain": true }
  },
  "thresholds": {
    "min_pages": 2,
    "max_pages": 100,
    "min_text_length": 1000,
    "confidence_threshold": 0.7
  }
}
```

**Tasks:**

- [ X ] Create `backend/data/cleaning_config.json` with content above

---

## Section 1: Data Models

Create `backend/data_cleaning/models.py`:

```python
"""Data models for PDF classification."""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
from pathlib import Path
import json
from datetime import datetime


class Classification(str, Enum):
    """Classification result for a PDF."""
    PAPER = "paper"           # Research paper / scientific article
    REJECTED = "rejected"     # Clearly not a paper
    UNCERTAIN = "uncertain"   # Needs manual review


class RejectionReason(str, Enum):
    """Reasons for rejecting a PDF."""
    RECEIPT = "receipt"
    TAX_FORM = "tax_form"
    HOMEWORK = "homework"
    LECTURE_NOTES = "lecture_notes"
    PERSONAL_SCAN = "personal_scan"
    LEGAL_DOCUMENT = "legal_document"
    ORDER_CONFIRMATION = "order_confirmation"
    MUSIC_SHEET = "music_sheet"
    PRESENTATION = "presentation"
    MANUAL = "manual"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    NO_TEXT = "no_text"
    OTHER = "other"


@dataclass
class FilterResult:
    """Result from a single filter."""
    filter_name: str
    classification: Classification
    confidence: float  # 0.0 to 1.0
    reason: Optional[RejectionReason] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassificationResult:
    """Complete classification result for a PDF."""
    file_path: str
    file_name: str
    file_size_kb: float

    # Final classification
    classification: Classification
    confidence: float
    rejection_reason: Optional[RejectionReason] = None

    # Filter results
    filter_results: List[FilterResult] = field(default_factory=list)

    # Metadata extracted
    num_pages: Optional[int] = None
    text_length: Optional[int] = None
    has_abstract: bool = False
    has_references: bool = False
    has_doi: bool = False
    detected_language: str = "unknown"

    # Processing info
    processed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_time_ms: float = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['classification'] = self.classification.value
        if self.rejection_reason:
            d['rejection_reason'] = self.rejection_reason.value
        d['filter_results'] = [
            {**asdict(fr), 'classification': fr.classification.value,
             'reason': fr.reason.value if fr.reason else None}
            for fr in self.filter_results
        ]
        return d


class ClassificationLog:
    """Log for all classification results."""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log_file = self.log_dir / f"classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.stats = {"paper": 0, "rejected": 0, "uncertain": 0, "total": 0}

    def log(self, result: ClassificationResult):
        """Log a classification result."""
        with open(self.current_log_file, 'a') as f:
            f.write(json.dumps(result.to_dict()) + '\n')

        self.stats[result.classification.value] += 1
        self.stats['total'] += 1

    def get_stats(self) -> Dict[str, int]:
        """Get current statistics."""
        return self.stats.copy()

    def save_summary(self):
        """Save summary statistics."""
        summary_file = self.log_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'stats': self.stats,
                'log_file': str(self.current_log_file),
                'completed_at': datetime.now().isoformat()
            }, f, indent=2)
```

**Tasks:**

- [ X ] Create `backend/data_cleaning/models.py` with the code above

---

## Section 2: Filename Filter (Fast, First Pass)

The filename filter is the fastest and should catch obvious non-papers.

Create `backend/data_cleaning/filters/filename_filter.py`:

```python
"""Filter PDFs based on filename patterns."""

import re
from pathlib import Path
from typing import Tuple, Optional

from ..models import Classification, RejectionReason, FilterResult


# Patterns that strongly indicate a research paper
PAPER_PATTERNS = [
    # DOI-based filenames
    r'10\.\d{4,}',                          # DOI prefix
    r'1-s2\.0-S\d+',                        # Elsevier/ScienceDirect
    r'acs\.[a-z]+\.\d+',                    # ACS journals
    r'jp[a-z]*\d{6}',                       # J. Phys. Chem
    r'ja\d{7}',                             # JACS
    r'nl\d{6}',                             # Nano Letters
    r'nn\d{6}',                             # ACS Nano
    r'ao\d{6}',                             # ACS Omega

    # Author-year patterns
    r'[A-Z][a-z]+-\d{4}-',                  # Author-2020-Title
    r'[A-Z][a-z]+\s+et\s+al',               # Author et al
    r'[A-Z][a-z]+\d{4}[a-z]?\.pdf$',        # Author2020.pdf

    # Journal identifiers
    r'pnas\.',                              # PNAS
    r'nature\d+',                           # Nature
    r'science\.',                           # Science
    r'cell\.',                              # Cell
    r'bip\.\d+',                            # Biopolymers
    r'jbc\.',                               # J. Biol. Chem.
]

# Patterns that strongly indicate NOT a paper
REJECT_PATTERNS = [
    # Receipts and financial
    (r'walmart|costco|target|safeway|grocery', RejectionReason.RECEIPT),
    (r'\d{4}-\d{2}-\d{2}.*receipt', RejectionReason.RECEIPT),
    (r'1099|w-?2|tax.*form|consolidated.*1099', RejectionReason.TAX_FORM),
    (r'etrade|schwab|fidelity|vanguard.*\d{4}', RejectionReason.TAX_FORM),

    # Academic non-papers
    (r'^hw\d|homework|problem.*set|pset', RejectionReason.HOMEWORK),
    (r'lecture[\s_]?\d|lecture[\s_]notes|notes[\s_]?week', RejectionReason.LECTURE_NOTES),
    (r'exam\s*(1|2|3|final|mid)|midterm|final.*exam', RejectionReason.HOMEWORK),
    (r'syllabus|course.*outline', RejectionReason.LECTURE_NOTES),

    # Scans and personal docs
    (r'live\s*scan|scan.*\d{4}', RejectionReason.PERSONAL_SCAN),
    (r'\d{4}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2}', RejectionReason.PERSONAL_SCAN),  # Timestamp scans

    # Legal
    (r'court|trial|legal|checklist|subpoena', RejectionReason.LEGAL_DOCUMENT),
    (r'contract|agreement|lease', RejectionReason.LEGAL_DOCUMENT),

    # Orders and confirmations
    (r'order.*confirmation|confirmation.*order', RejectionReason.ORDER_CONFIRMATION),
    (r'invoice|shipping|tracking', RejectionReason.ORDER_CONFIRMATION),
    (r'sigma.*aldrich|fisher.*scientific', RejectionReason.ORDER_CONFIRMATION),

    # Music
    (r'bwv\d+|sheet.*music|piano|guitar|violin', RejectionReason.MUSIC_SHEET),

    # Presentations
    (r'presentation|slides|powerpoint|pptx?', RejectionReason.PRESENTATION),
]


class FilenameFilter:
    """Fast filter based on filename patterns."""

    def __init__(self):
        self.paper_patterns = [re.compile(p, re.IGNORECASE) for p in PAPER_PATTERNS]
        self.reject_patterns = [(re.compile(p, re.IGNORECASE), r) for p, r in REJECT_PATTERNS]

    def filter(self, file_path: Path) -> FilterResult:
        """Classify PDF based on filename.

        Args:
            file_path: Path to PDF file

        Returns:
            FilterResult with classification
        """
        filename = file_path.name.lower()
        stem = file_path.stem.lower()

        # Check reject patterns first (high confidence rejects)
        for pattern, reason in self.reject_patterns:
            if pattern.search(filename):
                return FilterResult(
                    filter_name="filename",
                    classification=Classification.REJECTED,
                    confidence=0.9,
                    reason=reason,
                    details={"matched_pattern": pattern.pattern}
                )

        # Check paper patterns
        paper_matches = []
        for pattern in self.paper_patterns:
            if pattern.search(filename):
                paper_matches.append(pattern.pattern)

        if paper_matches:
            return FilterResult(
                filter_name="filename",
                classification=Classification.PAPER,
                confidence=0.7 + (0.1 * min(len(paper_matches), 3)),  # Max 1.0
                details={"matched_patterns": paper_matches}
            )

        # Uncertain - needs further analysis
        return FilterResult(
            filter_name="filename",
            classification=Classification.UNCERTAIN,
            confidence=0.5,
            details={"reason": "no_pattern_match"}
        )


def test_filter():
    """Test the filename filter with sample filenames."""
    filter = FilenameFilter()

    test_cases = [
        # Papers
        "acs.biomac.7b01245.pdf",
        "1-s2.0-S0032386114002857-main.pdf",
        "Park-2020-Carboxylic Acid.pdf",
        "Smith et al. - 2019 - Title.pdf",
        "ja0437050.pdf",

        # Rejects
        "Walmart 2010-0703.pdf",
        "hw0.pdf",
        "lecture5.pdf",
        "Etrade 2017 consolidated 1099.pdf",
        "Traffic court trial checklist.pdf",
        "bwv855a-let.pdf",
        "2014-07-14 Live Scan.pdf",
        "Sigma-AldrichOrderConfirmation.pdf",
    ]

    print("Filename Filter Test Results:")
    print("=" * 60)

    for filename in test_cases:
        result = filter.filter(Path(filename))
        print(f"{filename[:40]:<40} -> {result.classification.value:<10} ({result.confidence:.2f})")
        if result.reason:
            print(f"{'':40}    Reason: {result.reason.value}")


if __name__ == "__main__":
    test_filter()
```

**Tasks:**

- [ X ] Create `backend/data_cleaning/filters/filename_filter.py` with the code above
- [ X ] Run the test: `python -m data_cleaning.filters.filename_filter`
- [ X ] Review results and adjust patterns as needed

---

## Section 3: Metadata Filter (Medium Speed)

Create `backend/data_cleaning/filters/metadata_filter.py`:

```python
"""Filter PDFs based on PDF metadata and basic properties."""

import pymupdf
from pathlib import Path
from typing import Optional, Dict, Any
import re

from ..models import Classification, RejectionReason, FilterResult


class MetadataFilter:
    """Filter based on PDF metadata and document properties."""

    def __init__(
        self,
        min_pages: int = 2,
        max_pages: int = 100,
        min_text_length: int = 1000,
    ):
        self.min_pages = min_pages
        self.max_pages = max_pages
        self.min_text_length = min_text_length

        # Keywords in metadata that suggest a paper
        self.paper_keywords = [
            'journal', 'doi', 'abstract', 'article', 'publication',
            'research', 'study', 'university', 'institute', 'department',
            'proceedings', 'conference', 'volume', 'issue'
        ]

        # Keywords that suggest NOT a paper
        self.reject_keywords = [
            'receipt', 'invoice', 'order', 'confirmation', 'statement',
            'tax', 'w-2', '1099', 'scan', 'copy'
        ]

    def _extract_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract metadata from PDF."""
        try:
            doc = pymupdf.open(pdf_path)

            metadata = doc.metadata or {}
            num_pages = len(doc)

            # Get text from first few pages
            text_sample = ""
            for i in range(min(3, num_pages)):
                text_sample += doc[i].get_text()

            # Get total text length estimate
            total_text = ""
            for page in doc:
                total_text += page.get_text()

            doc.close()

            return {
                'metadata': metadata,
                'num_pages': num_pages,
                'text_sample': text_sample,
                'text_length': len(total_text),
                'file_size': pdf_path.stat().st_size,
            }

        except Exception as e:
            return {
                'error': str(e),
                'num_pages': 0,
                'text_length': 0,
            }

    def _check_paper_indicators(self, text: str, metadata: Dict) -> Dict[str, bool]:
        """Check for indicators that this is a research paper."""
        text_lower = text.lower()

        indicators = {
            'has_abstract': bool(re.search(r'\babstract\b', text_lower)),
            'has_introduction': bool(re.search(r'\bintroduction\b', text_lower)),
            'has_references': bool(re.search(r'\breferences\b|\bbibliography\b', text_lower)),
            'has_methods': bool(re.search(r'\bmethods?\b|\bmaterials?\b', text_lower)),
            'has_results': bool(re.search(r'\bresults?\b', text_lower)),
            'has_conclusion': bool(re.search(r'\bconclusions?\b', text_lower)),
            'has_doi': bool(re.search(r'10\.\d{4,}/[^\s]+', text)),
            'has_issn': bool(re.search(r'ISSN[:\s]*\d{4}-\d{4}', text, re.IGNORECASE)),
            'has_copyright': bool(re.search(r'©|\(c\)|copyright', text_lower)),
            'has_journal_name': any(j in text_lower for j in [
                'journal', 'proceedings', 'transactions', 'letters', 'review'
            ]),
        }

        # Check metadata for paper keywords
        meta_text = ' '.join(str(v) for v in metadata.values() if v).lower()
        indicators['metadata_suggests_paper'] = any(
            kw in meta_text for kw in self.paper_keywords
        )

        return indicators

    def filter(self, file_path: Path) -> FilterResult:
        """Classify PDF based on metadata and properties.

        Args:
            file_path: Path to PDF file

        Returns:
            FilterResult with classification
        """
        extracted = self._extract_metadata(file_path)

        # Handle extraction errors
        if 'error' in extracted:
            return FilterResult(
                filter_name="metadata",
                classification=Classification.UNCERTAIN,
                confidence=0.3,
                details={"error": extracted['error']}
            )

        num_pages = extracted['num_pages']
        text_length = extracted['text_length']

        # Quick rejects based on size
        if num_pages < self.min_pages:
            return FilterResult(
                filter_name="metadata",
                classification=Classification.REJECTED,
                confidence=0.8,
                reason=RejectionReason.TOO_SHORT,
                details={"num_pages": num_pages}
            )

        if num_pages > self.max_pages:
            return FilterResult(
                filter_name="metadata",
                classification=Classification.UNCERTAIN,
                confidence=0.4,
                reason=RejectionReason.TOO_LONG,
                details={"num_pages": num_pages, "note": "Could be a book or thesis"}
            )

        if text_length < self.min_text_length:
            return FilterResult(
                filter_name="metadata",
                classification=Classification.REJECTED,
                confidence=0.7,
                reason=RejectionReason.NO_TEXT,
                details={"text_length": text_length, "note": "Possibly scanned without OCR"}
            )

        # Check paper indicators
        indicators = self._check_paper_indicators(
            extracted['text_sample'],
            extracted['metadata']
        )

        positive_count = sum(1 for v in indicators.values() if v)
        total_checks = len(indicators)

        # Strong paper indicators
        if indicators['has_abstract'] and indicators['has_references']:
            return FilterResult(
                filter_name="metadata",
                classification=Classification.PAPER,
                confidence=0.85,
                details={
                    "indicators": indicators,
                    "num_pages": num_pages,
                    "positive_ratio": positive_count / total_checks
                }
            )

        # Moderate paper indicators
        if positive_count >= 4:
            return FilterResult(
                filter_name="metadata",
                classification=Classification.PAPER,
                confidence=0.7,
                details={
                    "indicators": indicators,
                    "num_pages": num_pages,
                    "positive_ratio": positive_count / total_checks
                }
            )

        # Weak indicators - uncertain
        if positive_count >= 2:
            return FilterResult(
                filter_name="metadata",
                classification=Classification.UNCERTAIN,
                confidence=0.5,
                details={
                    "indicators": indicators,
                    "num_pages": num_pages,
                    "positive_ratio": positive_count / total_checks
                }
            )

        # Few indicators - likely not a paper
        return FilterResult(
            filter_name="metadata",
            classification=Classification.REJECTED,
            confidence=0.6,
            reason=RejectionReason.OTHER,
            details={
                "indicators": indicators,
                "num_pages": num_pages,
                "positive_ratio": positive_count / total_checks,
                "note": "Few paper indicators found"
            }
        )
```

**Tasks:**

- [ X ] Create `backend/data_cleaning/filters/metadata_filter.py` with the code above

---

## Section 4: Content Filter (Slower, More Accurate)

Create `backend/data_cleaning/filters/content_filter.py`:

```python
"""Deep content analysis filter for PDF classification."""

import re
from pathlib import Path
from typing import Dict, List, Tuple
import pymupdf

from ..models import Classification, RejectionReason, FilterResult


class ContentFilter:
    """Analyze PDF content structure to classify as paper or not."""

    def __init__(self):
        # Section headers typical in research papers
        self.paper_sections = [
            r'\babstract\b',
            r'\bintroduction\b',
            r'\bbackground\b',
            r'\bliterature\s+review\b',
            r'\bmethods?\b',
            r'\bmaterials?\s*(and|&)\s*methods?\b',
            r'\bexperimental\b',
            r'\bresults?\b',
            r'\bdiscussion\b',
            r'\bconclusions?\b',
            r'\backnowledg',
            r'\breferences\b',
            r'\bbibliography\b',
            r'\bsupplementary\b',
            r'\bappendix\b',
        ]

        # Content patterns for rejects
        self.reject_content = {
            RejectionReason.RECEIPT: [
                r'total\s*:?\s*\$[\d,]+\.\d{2}',
                r'subtotal',
                r'tax\s*:?\s*\$',
                r'payment\s+method',
                r'visa.*\*{4}',
                r'mastercard.*\*{4}',
                r'change\s+due',
            ],
            RejectionReason.TAX_FORM: [
                r'form\s+(?:1099|w-?2)',
                r'payer.*identification',
                r'recipient.*identification',
                r'social\s+security',
                r'employer\s+identification',
                r'wages.*tips',
                r'federal\s+income\s+tax\s+withheld',
            ],
            RejectionReason.HOMEWORK: [
                r'problem\s+\d',
                r'question\s+\d',
                r'(?:show\s+)?your\s+work',
                r'due\s+date',
                r'points?\s*:?\s*\d+',
                r'name\s*:?\s*_{2,}',
                r'student\s+id',
            ],
            RejectionReason.LECTURE_NOTES: [
                r'lecture\s+\d+',
                r'week\s+\d+',
                r'chapter\s+\d+\s+notes',
                r'reading\s+assignment',
                r'office\s+hours',
                r'instructor',
            ],
            RejectionReason.ORDER_CONFIRMATION: [
                r'order\s+(?:number|#|confirmation)',
                r'shipping\s+address',
                r'billing\s+address',
                r'tracking\s+number',
                r'estimated\s+delivery',
                r'qty\s+(?:ordered|shipped)',
            ],
        }

    def _extract_text(self, pdf_path: Path, max_pages: int = 10) -> str:
        """Extract text from PDF."""
        try:
            doc = pymupdf.open(pdf_path)
            text = ""

            for i in range(min(max_pages, len(doc))):
                text += doc[i].get_text() + "\n\n"

            doc.close()
            return text

        except Exception as e:
            return f"ERROR: {e}"

    def _count_section_matches(self, text: str) -> Tuple[int, List[str]]:
        """Count how many paper section patterns match."""
        text_lower = text.lower()
        matches = []

        for pattern in self.paper_sections:
            if re.search(pattern, text_lower):
                matches.append(pattern)

        return len(matches), matches

    def _check_reject_patterns(self, text: str) -> Tuple[bool, RejectionReason, List[str]]:
        """Check for patterns indicating this is NOT a paper."""
        text_lower = text.lower()

        for reason, patterns in self.reject_content.items():
            matched = []
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    matched.append(pattern)

            # If multiple reject patterns match, it's likely that type
            if len(matched) >= 2:
                return True, reason, matched

        return False, None, []

    def _analyze_structure(self, text: str) -> Dict[str, any]:
        """Analyze document structure."""
        lines = text.split('\n')

        # Count approximate paragraph count
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]

        # Check for citations pattern [1], [2], etc.
        citation_pattern = re.findall(r'\[\d+\]', text)

        # Check for figure/table references
        figure_refs = len(re.findall(r'(?:Figure|Fig\.?|Table)\s*\d+', text, re.IGNORECASE))

        # Check for equations (simple heuristic)
        has_equations = bool(re.search(r'[=∑∫∂]|\\frac|\\sum', text))

        return {
            'paragraph_count': len(paragraphs),
            'citation_count': len(citation_pattern),
            'figure_table_refs': figure_refs,
            'has_equations': has_equations,
            'avg_paragraph_length': sum(len(p) for p in paragraphs) / max(len(paragraphs), 1),
        }

    def filter(self, file_path: Path) -> FilterResult:
        """Classify PDF based on deep content analysis.

        Args:
            file_path: Path to PDF file

        Returns:
            FilterResult with classification
        """
        text = self._extract_text(file_path)

        if text.startswith("ERROR:"):
            return FilterResult(
                filter_name="content",
                classification=Classification.UNCERTAIN,
                confidence=0.3,
                details={"error": text}
            )

        # Check reject patterns first
        is_reject, reason, reject_matches = self._check_reject_patterns(text)
        if is_reject:
            return FilterResult(
                filter_name="content",
                classification=Classification.REJECTED,
                confidence=0.85,
                reason=reason,
                details={"matched_patterns": reject_matches}
            )

        # Count paper section matches
        section_count, section_matches = self._count_section_matches(text)

        # Analyze structure
        structure = self._analyze_structure(text)

        # Scoring
        score = 0.0

        # Section matches (0-0.4)
        score += min(section_count / 8, 0.4)

        # Citations (0-0.2)
        if structure['citation_count'] > 5:
            score += 0.2
        elif structure['citation_count'] > 0:
            score += 0.1

        # Figure/table references (0-0.15)
        if structure['figure_table_refs'] > 2:
            score += 0.15
        elif structure['figure_table_refs'] > 0:
            score += 0.08

        # Paragraph structure (0-0.15)
        if structure['paragraph_count'] > 10 and structure['avg_paragraph_length'] > 200:
            score += 0.15
        elif structure['paragraph_count'] > 5:
            score += 0.08

        # Equations (0-0.1)
        if structure['has_equations']:
            score += 0.1

        # Determine classification
        if score >= 0.6:
            classification = Classification.PAPER
            confidence = 0.5 + score * 0.5  # 0.8-1.0
        elif score >= 0.3:
            classification = Classification.UNCERTAIN
            confidence = 0.4 + score * 0.3  # 0.49-0.58
        else:
            classification = Classification.REJECTED
            confidence = 0.5 + (0.3 - score) * 0.5  # 0.5-0.65

        return FilterResult(
            filter_name="content",
            classification=classification,
            confidence=confidence,
            details={
                "section_matches": section_matches,
                "section_count": section_count,
                "structure": structure,
                "score": score,
            }
        )
```

**Tasks:**

- [x] Create `backend/data_cleaning/filters/content_filter.py` with the code above

---

## Section 5: LLM Filter (Slowest, Most Accurate)

Create `backend/data_cleaning/filters/llm_filter.py`:

```python
"""LLM-based filter for uncertain cases."""

import pymupdf
from pathlib import Path
from typing import Optional
import json
import re

from anthropic import Anthropic

from ..models import Classification, RejectionReason, FilterResult


CLASSIFICATION_PROMPT = """Analyze this PDF excerpt and determine if it is a RESEARCH PAPER or SCIENTIFIC ARTICLE.

Research papers typically have:
- Abstract, Introduction, Methods, Results, Discussion, Conclusion sections
- References/Bibliography
- Author affiliations (universities, institutions)
- Scientific content (data, experiments, analysis)
- DOI or journal information

NOT research papers (examples):
- Receipts, invoices, order confirmations
- Tax forms, financial statements
- Homework assignments, lecture notes, exams
- Personal scans, legal documents
- Presentations, manuals, user guides
- Books, book chapters (these are borderline - mark as uncertain)

---

FILENAME: {filename}

CONTENT (first ~2000 chars):
{content}

---

Respond with a JSON object:
{{
    "classification": "paper" | "rejected" | "uncertain",
    "confidence": 0.0 to 1.0,
    "reason": "brief explanation",
    "rejection_type": null | "receipt" | "tax_form" | "homework" | "lecture_notes" | "personal_scan" | "legal_document" | "order_confirmation" | "other"
}}

Only JSON, no other text."""


class LLMFilter:
    """Use Claude to classify uncertain PDFs."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_text_chars: int = 2000,
    ):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_text_chars = max_text_chars

    def _extract_text_sample(self, pdf_path: Path) -> str:
        """Extract text sample from PDF."""
        try:
            doc = pymupdf.open(pdf_path)
            text = ""

            # Get text from first few pages
            for i in range(min(3, len(doc))):
                text += doc[i].get_text()
                if len(text) > self.max_text_chars:
                    break

            doc.close()
            return text[:self.max_text_chars]

        except Exception as e:
            return f"[Error extracting text: {e}]"

    def _parse_response(self, response_text: str) -> dict:
        """Parse LLM response JSON."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"classification": "uncertain", "confidence": 0.5, "reason": "Failed to parse response"}
        except json.JSONDecodeError:
            return {"classification": "uncertain", "confidence": 0.5, "reason": "Invalid JSON response"}

    def filter(self, file_path: Path) -> FilterResult:
        """Classify PDF using LLM.

        Args:
            file_path: Path to PDF file

        Returns:
            FilterResult with classification
        """
        text_sample = self._extract_text_sample(file_path)

        prompt = CLASSIFICATION_PROMPT.format(
            filename=file_path.name,
            content=text_sample
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            result = self._parse_response(response.content[0].text)

            # Map to our enums
            classification_map = {
                "paper": Classification.PAPER,
                "rejected": Classification.REJECTED,
                "uncertain": Classification.UNCERTAIN,
            }

            reason_map = {
                "receipt": RejectionReason.RECEIPT,
                "tax_form": RejectionReason.TAX_FORM,
                "homework": RejectionReason.HOMEWORK,
                "lecture_notes": RejectionReason.LECTURE_NOTES,
                "personal_scan": RejectionReason.PERSONAL_SCAN,
                "legal_document": RejectionReason.LEGAL_DOCUMENT,
                "order_confirmation": RejectionReason.ORDER_CONFIRMATION,
                "other": RejectionReason.OTHER,
            }

            classification = classification_map.get(
                result.get("classification", "uncertain"),
                Classification.UNCERTAIN
            )

            rejection_reason = None
            if classification == Classification.REJECTED:
                rejection_reason = reason_map.get(
                    result.get("rejection_type"),
                    RejectionReason.OTHER
                )

            return FilterResult(
                filter_name="llm",
                classification=classification,
                confidence=result.get("confidence", 0.7),
                reason=rejection_reason,
                details={
                    "llm_reason": result.get("reason", ""),
                    "model": self.model,
                }
            )

        except Exception as e:
            return FilterResult(
                filter_name="llm",
                classification=Classification.UNCERTAIN,
                confidence=0.3,
                details={"error": str(e)}
            )
```

**Tasks:**

- [ X ] Create `backend/data_cleaning/filters/llm_filter.py` with the code above

---

## Section 6: Main Classifier

Create `backend/data_cleaning/classifier.py`:

```python
"""Main classifier that orchestrates all filters."""

import logging
from pathlib import Path
from typing import List, Optional
import time

from .models import Classification, ClassificationResult, ClassificationLog, FilterResult
from .filters.filename_filter import FilenameFilter
from .filters.metadata_filter import MetadataFilter
from .filters.content_filter import ContentFilter
from .filters.llm_filter import LLMFilter

logger = logging.getLogger(__name__)


class PDFClassifier:
    """Classify PDFs as research papers or not.

    Uses a cascade of filters:
    1. Filename filter (fast, catches obvious cases)
    2. Metadata filter (medium, checks PDF properties)
    3. Content filter (slow, analyzes text structure)
    4. LLM filter (slowest, for uncertain cases only)

    Stops early if a filter has high confidence.
    """

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        use_llm: bool = True,
        confidence_threshold: float = 0.8,
        log_dir: Optional[Path] = None,
    ):
        self.filename_filter = FilenameFilter()
        self.metadata_filter = MetadataFilter()
        self.content_filter = ContentFilter()

        self.use_llm = use_llm and anthropic_api_key
        if self.use_llm:
            self.llm_filter = LLMFilter(api_key=anthropic_api_key)
        else:
            self.llm_filter = None

        self.confidence_threshold = confidence_threshold

        if log_dir:
            self.log = ClassificationLog(log_dir)
        else:
            self.log = None

    def _combine_results(self, results: List[FilterResult]) -> tuple[Classification, float]:
        """Combine filter results into final classification.

        Uses weighted voting with confidence scores.
        """
        if not results:
            return Classification.UNCERTAIN, 0.5

        # Weight by confidence
        paper_score = 0.0
        reject_score = 0.0
        total_weight = 0.0

        for result in results:
            weight = result.confidence
            total_weight += weight

            if result.classification == Classification.PAPER:
                paper_score += weight
            elif result.classification == Classification.REJECTED:
                reject_score += weight

        if total_weight == 0:
            return Classification.UNCERTAIN, 0.5

        paper_ratio = paper_score / total_weight
        reject_ratio = reject_score / total_weight

        if paper_ratio > 0.6:
            return Classification.PAPER, paper_ratio
        elif reject_ratio > 0.6:
            return Classification.REJECTED, reject_ratio
        else:
            return Classification.UNCERTAIN, 0.5

    def classify(self, file_path: Path) -> ClassificationResult:
        """Classify a single PDF.

        Args:
            file_path: Path to PDF file

        Returns:
            ClassificationResult with full details
        """
        start_time = time.time()
        filter_results = []

        # 1. Filename filter
        filename_result = self.filename_filter.filter(file_path)
        filter_results.append(filename_result)

        # Early exit if high confidence reject
        if (filename_result.classification == Classification.REJECTED and
            filename_result.confidence >= self.confidence_threshold):
            return self._build_result(file_path, filter_results, start_time)

        # 2. Metadata filter
        metadata_result = self.metadata_filter.filter(file_path)
        filter_results.append(metadata_result)

        # Early exit if high confidence
        combined_class, combined_conf = self._combine_results(filter_results)
        if combined_conf >= self.confidence_threshold and combined_class != Classification.UNCERTAIN:
            return self._build_result(file_path, filter_results, start_time)

        # 3. Content filter
        content_result = self.content_filter.filter(file_path)
        filter_results.append(content_result)

        # Check if we need LLM
        combined_class, combined_conf = self._combine_results(filter_results)

        # 4. LLM filter (only for uncertain cases)
        if (self.use_llm and
            self.llm_filter and
            combined_class == Classification.UNCERTAIN):
            llm_result = self.llm_filter.filter(file_path)
            filter_results.append(llm_result)

        return self._build_result(file_path, filter_results, start_time)

    def _build_result(
        self,
        file_path: Path,
        filter_results: List[FilterResult],
        start_time: float
    ) -> ClassificationResult:
        """Build final classification result."""

        # Combine filter results
        final_class, final_conf = self._combine_results(filter_results)

        # Get rejection reason from highest confidence reject filter
        rejection_reason = None
        if final_class == Classification.REJECTED:
            reject_results = [r for r in filter_results
                           if r.classification == Classification.REJECTED and r.reason]
            if reject_results:
                best = max(reject_results, key=lambda r: r.confidence)
                rejection_reason = best.reason

        # Extract some metadata indicators
        has_abstract = False
        has_references = False
        has_doi = False
        num_pages = None
        text_length = None

        for fr in filter_results:
            if fr.filter_name == "metadata":
                indicators = fr.details.get("indicators", {})
                has_abstract = indicators.get("has_abstract", False)
                has_references = indicators.get("has_references", False)
                has_doi = indicators.get("has_doi", False)
                num_pages = fr.details.get("num_pages")
            elif fr.filter_name == "content":
                structure = fr.details.get("structure", {})

        result = ClassificationResult(
            file_path=str(file_path),
            file_name=file_path.name,
            file_size_kb=file_path.stat().st_size / 1024,
            classification=final_class,
            confidence=final_conf,
            rejection_reason=rejection_reason,
            filter_results=filter_results,
            num_pages=num_pages,
            text_length=text_length,
            has_abstract=has_abstract,
            has_references=has_references,
            has_doi=has_doi,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

        # Log result
        if self.log:
            self.log.log(result)

        return result

    def classify_batch(
        self,
        file_paths: List[Path],
        progress_callback=None
    ) -> List[ClassificationResult]:
        """Classify multiple PDFs.

        Args:
            file_paths: List of PDF paths
            progress_callback: Optional callback(current, total, result)

        Returns:
            List of ClassificationResults
        """
        results = []
        total = len(file_paths)

        for i, path in enumerate(file_paths):
            try:
                result = self.classify(path)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, total, result)

            except Exception as e:
                logger.error(f"Error classifying {path}: {e}")
                # Create error result
                error_result = ClassificationResult(
                    file_path=str(path),
                    file_name=path.name,
                    file_size_kb=0,
                    classification=Classification.UNCERTAIN,
                    confidence=0.0,
                    filter_results=[],
                )
                results.append(error_result)

        return results
```

**Tasks:**

- [ X ] Create `backend/data_cleaning/classifier.py` with the code above

---

## Section 7: Main Runner Script

Create `backend/data_cleaning/run_cleaning.py`:

```python
"""Main script to run data cleaning pipeline."""

import sys
import json
import logging
import argparse
import shutil
from pathlib import Path
from typing import Optional
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from config import settings
from data_cleaning.classifier import PDFClassifier
from data_cleaning.models import Classification

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load cleaning configuration."""
    with open(config_path) as f:
        return json.load(f)


def get_pdf_files(source_dir: Path, limit: Optional[int] = None) -> list[Path]:
    """Get all PDF files from source directory."""
    pdf_files = list(source_dir.glob("**/*.pdf")) + list(source_dir.glob("**/*.PDF"))

    if limit:
        pdf_files = pdf_files[:limit]

    logger.info(f"Found {len(pdf_files)} PDF files")
    return pdf_files


def organize_file(
    result,
    output_dir: Path,
    use_symlinks: bool = True
) -> Path:
    """Organize file based on classification."""
    source = Path(result.file_path)

    # Determine target directory
    if result.classification == Classification.PAPER:
        target_dir = output_dir / "papers"
    elif result.classification == Classification.REJECTED:
        target_dir = output_dir / "rejected"
    else:
        target_dir = output_dir / "uncertain"

    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / source.name

    # Handle duplicates
    if target_path.exists():
        stem = source.stem
        suffix = source.suffix
        counter = 1
        while target_path.exists():
            target_path = target_dir / f"{stem}_{counter}{suffix}"
            counter += 1

    # Create symlink or copy
    if use_symlinks:
        target_path.symlink_to(source)
    else:
        shutil.copy2(source, target_path)

    return target_path


def run_cleaning(
    source_dir: Path,
    output_dir: Path,
    log_dir: Path,
    use_llm: bool = True,
    use_symlinks: bool = True,
    limit: Optional[int] = None,
    sample_only: bool = False,
):
    """Run the full cleaning pipeline."""

    # Initialize classifier
    classifier = PDFClassifier(
        anthropic_api_key=settings.anthropic_api_key if use_llm else None,
        use_llm=use_llm,
        log_dir=log_dir,
    )

    # Get files
    pdf_files = get_pdf_files(source_dir, limit=limit)

    if sample_only:
        logger.info("Sample mode: processing first 100 files only")
        pdf_files = pdf_files[:100]

    # Process
    stats = {"paper": 0, "rejected": 0, "uncertain": 0, "errors": 0}

    with tqdm(total=len(pdf_files), desc="Classifying PDFs") as pbar:
        for pdf_path in pdf_files:
            try:
                result = classifier.classify(pdf_path)

                # Organize file
                organize_file(result, output_dir, use_symlinks)

                # Update stats
                stats[result.classification.value] += 1

                # Update progress bar
                pbar.set_postfix({
                    'papers': stats['paper'],
                    'rejected': stats['rejected'],
                    'uncertain': stats['uncertain']
                })
                pbar.update(1)

            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                stats['errors'] += 1
                pbar.update(1)

    # Save final stats
    if classifier.log:
        classifier.log.save_summary()

    # Print summary
    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    print(f"Total processed: {sum(stats.values())}")
    print(f"  Papers: {stats['paper']}")
    print(f"  Rejected: {stats['rejected']}")
    print(f"  Uncertain: {stats['uncertain']}")
    print(f"  Errors: {stats['errors']}")
    print(f"\nResults saved to: {output_dir}")
    print(f"Logs saved to: {log_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Clean and classify PDFs")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("data/cleaning_config.json"),
        help="Path to config file"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Override source directory from config"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample mode: process only 100 files"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM filter (faster but less accurate)"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks"
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent.parent / args.config
    if config_path.exists():
        config = load_config(config_path)
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")
        config = {}

    # Determine paths
    source_dir = args.source_dir or Path(config.get(
        "source_dir",
        "/Users/merlin/projects/researchPaperAgent/resource/Total Backup 2025-09-20-recover-Converted.Data/PDF"
    ))

    output_dir = Path(config.get(
        "output_dir",
        "/Users/merlin/projects/researchPaperAgent/backend/data/classified"
    ))

    log_dir = Path(config.get(
        "log_dir",
        "/Users/merlin/projects/researchPaperAgent/backend/data/cleaning_logs"
    ))

    # Run
    run_cleaning(
        source_dir=source_dir,
        output_dir=output_dir,
        log_dir=log_dir,
        use_llm=not args.no_llm,
        use_symlinks=not args.copy,
        limit=args.limit,
        sample_only=args.sample,
    )


if __name__ == "__main__":
    main()
```

**Tasks:**

- [x] Create `backend/data_cleaning/run_cleaning.py` with the code above
- [x] Update `backend/config.py` to ensure `anthropic_api_key` is accessible

---

## Section 8: Manual Review Tool

Create `backend/data_cleaning/review_tool.py`:

```python
"""Tool for manually reviewing uncertain classifications."""

import json
import sys
from pathlib import Path
from typing import Optional
import subprocess

sys.path.append(str(Path(__file__).parent.parent))


def load_log(log_file: Path) -> list[dict]:
    """Load classification log."""
    results = []
    with open(log_file) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def open_pdf(pdf_path: str):
    """Open PDF in default viewer."""
    subprocess.run(["open", pdf_path], check=True)


def review_uncertain(log_dir: Path, output_file: Path):
    """Interactive review of uncertain classifications."""

    # Find most recent log
    log_files = sorted(log_dir.glob("classification_*.jsonl"), reverse=True)
    if not log_files:
        print("No log files found")
        return

    log_file = log_files[0]
    print(f"Loading: {log_file}")

    results = load_log(log_file)
    uncertain = [r for r in results if r['classification'] == 'uncertain']

    print(f"Found {len(uncertain)} uncertain files to review")

    reviewed = []

    for i, result in enumerate(uncertain):
        print("\n" + "=" * 60)
        print(f"[{i+1}/{len(uncertain)}] {result['file_name']}")
        print(f"Size: {result['file_size_kb']:.1f} KB")
        print(f"Pages: {result.get('num_pages', 'unknown')}")
        print(f"Confidence: {result['confidence']:.2f}")

        # Show filter results
        print("\nFilter results:")
        for fr in result.get('filter_results', []):
            print(f"  {fr['filter_name']}: {fr['classification']} ({fr['confidence']:.2f})")

        print("\nOptions:")
        print("  p = paper")
        print("  r = reject")
        print("  o = open PDF")
        print("  s = skip")
        print("  q = quit and save")

        while True:
            choice = input("\nYour choice: ").strip().lower()

            if choice == 'o':
                open_pdf(result['file_path'])
            elif choice == 'p':
                reviewed.append({
                    'file_path': result['file_path'],
                    'file_name': result['file_name'],
                    'manual_classification': 'paper'
                })
                break
            elif choice == 'r':
                reason = input("Rejection reason (or Enter to skip): ").strip()
                reviewed.append({
                    'file_path': result['file_path'],
                    'file_name': result['file_name'],
                    'manual_classification': 'rejected',
                    'manual_reason': reason or None
                })
                break
            elif choice == 's':
                break
            elif choice == 'q':
                # Save and quit
                with open(output_file, 'w') as f:
                    json.dump(reviewed, f, indent=2)
                print(f"\nSaved {len(reviewed)} reviews to {output_file}")
                return

    # Save all reviews
    with open(output_file, 'w') as f:
        json.dump(reviewed, f, indent=2)
    print(f"\nSaved {len(reviewed)} reviews to {output_file}")


def apply_reviews(review_file: Path, classified_dir: Path):
    """Apply manual reviews to move files."""
    with open(review_file) as f:
        reviews = json.load(f)

    for review in reviews:
        source = Path(review['file_path'])
        file_name = review['file_name']

        # Find current location (might be in uncertain/)
        current = classified_dir / "uncertain" / file_name

        if not current.exists():
            print(f"Skipping {file_name}: not found in uncertain/")
            continue

        # Determine target
        if review['manual_classification'] == 'paper':
            target_dir = classified_dir / "papers"
        else:
            target_dir = classified_dir / "rejected"

        target = target_dir / file_name

        # Move
        current.rename(target)
        print(f"Moved {file_name} -> {review['manual_classification']}")

    print(f"\nApplied {len(reviews)} reviews")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Review uncertain PDFs")
    parser.add_argument("action", choices=["review", "apply"])
    parser.add_argument("--log-dir", type=Path, default=Path("data/cleaning_logs"))
    parser.add_argument("--classified-dir", type=Path, default=Path("data/classified"))
    parser.add_argument("--reviews", type=Path, default=Path("data/manual_reviews.json"))

    args = parser.parse_args()

    if args.action == "review":
        review_uncertain(args.log_dir, args.reviews)
    elif args.action == "apply":
        apply_reviews(args.reviews, args.classified_dir)
```

**Tasks:**

- [ X ] Create `backend/data_cleaning/review_tool.py` with the code above

---

## Section 9: Execution Plan

### 9.1 Test on Sample

```bash
cd backend

# Test filename filter
python -m data_cleaning.filters.filename_filter

# Run on 100 files without LLM (fast test)
python -m data_cleaning.run_cleaning --sample --no-llm
```

**Tasks:**

- [x] Run filename filter test
- [x] Run sample classification without LLM
- [x] Review results in `data/classified/`
- [x] Check logs in `data/cleaning_logs/`

### 9.2 Run Without LLM (Fast Pass)

```bash
# Process all files without LLM
python -m data_cleaning.run_cleaning --no-llm
```

This will:

- Classify ~22,000 files using filename, metadata, and content filters
- Sort into papers/, rejected/, uncertain/
- Create detailed logs

**Tasks:**

- [ X ] Run full classification without LLM
- [ X ] Review statistics
- [ X ] Sample-check some papers/ files manually
- [ X ] Sample-check some rejected/ files manually

### 9.3 Run LLM on Uncertain Files

```bash
# Re-run with LLM for uncertain only
python -m data_cleaning.run_cleaning --source-dir data/classified/uncertain
```

**Tasks:**

- [ X ] Run LLM filter on uncertain files
- [ X ] Review results

### 9.4 Manual Review

```bash
# Review remaining uncertain files
python -m data_cleaning.review_tool review

# Apply manual decisions
python -m data_cleaning.review_tool apply
```

**Tasks:**

- [ X ] Review uncertain files manually (aim for <100 remaining)
- [ X ] Apply manual reviews

### 9.5 Finalize File Organization

After classification is complete, you have two options for organizing files:

**Option A: Keep Symlinks (Recommended for Phase 1)**

Symlinks preserve original file locations while providing organized access. Good for validation phase.

```bash
# Verify symlinks are working
ls -la data/classified/papers/ | head -20

# Count papers ready for Phase 1
find data/classified/papers -name "*.pdf" -type l | wc -l
```

**Option B: Copy Files to Dedicated Directory**

For Phase 1 validation or if you want actual copies:

```bash
# Copy papers to a clean directory for Phase 1
mkdir -p data/phase1_papers

# Copy all classified papers (following symlinks)
cd backend
find data/classified/papers -name "*.pdf" -exec cp -L {} data/phase1_papers/ \;

# Or use rsync for better handling
rsync -avL data/classified/papers/ data/phase1_papers/
```

**Option C: Move Original Files (Permanent Organization)**

If you want to permanently reorganize the source files:

```bash
# WARNING: This modifies original file locations!
# Only do this after you're confident in classification accuracy

python -m data_cleaning.organize_files --mode move
```

Create `backend/data_cleaning/organize_files.py`:

```python
"""Organize classified files - copy or move to final destinations."""

import sys
import shutil
import argparse
import json
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))


def load_classification_log(log_dir: Path) -> list[dict]:
    """Load the most recent classification log."""
    log_files = sorted(log_dir.glob("classification_*.jsonl"), reverse=True)
    if not log_files:
        raise FileNotFoundError("No classification logs found")

    results = []
    with open(log_files[0]) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    print(f"Loaded {len(results)} classifications from {log_files[0].name}")
    return results


def organize_files(
    log_dir: Path,
    output_dir: Path,
    mode: str = "copy",  # "copy", "move", or "symlink"
    papers_only: bool = False,
):
    """Organize files based on classification.

    Args:
        log_dir: Directory containing classification logs
        output_dir: Target directory for organized files
        mode: "copy", "move", or "symlink"
        papers_only: If True, only process papers (skip rejected/uncertain)
    """
    results = load_classification_log(log_dir)

    # Create output directories
    papers_dir = output_dir / "papers"
    rejected_dir = output_dir / "rejected"
    uncertain_dir = output_dir / "uncertain"

    papers_dir.mkdir(parents=True, exist_ok=True)
    if not papers_only:
        rejected_dir.mkdir(parents=True, exist_ok=True)
        uncertain_dir.mkdir(parents=True, exist_ok=True)

    stats = {"papers": 0, "rejected": 0, "uncertain": 0, "errors": 0}

    for result in tqdm(results, desc=f"Organizing files ({mode})"):
        source = Path(result['file_path'])

        if not source.exists():
            print(f"Warning: Source not found: {source}")
            stats['errors'] += 1
            continue

        # Determine target directory
        classification = result['classification']

        if classification == "paper":
            target_dir = papers_dir
        elif classification == "rejected":
            if papers_only:
                continue
            target_dir = rejected_dir
        else:  # uncertain
            if papers_only:
                continue
            target_dir = uncertain_dir

        target = target_dir / source.name

        # Handle duplicate filenames
        if target.exists():
            stem = source.stem
            suffix = source.suffix
            counter = 1
            while target.exists():
                target = target_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        # Perform the operation
        try:
            if mode == "copy":
                shutil.copy2(source, target)
            elif mode == "move":
                shutil.move(source, target)
            elif mode == "symlink":
                target.symlink_to(source.resolve())

            stats[classification] += 1

        except Exception as e:
            print(f"Error processing {source.name}: {e}")
            stats['errors'] += 1

    print("\n" + "=" * 50)
    print("ORGANIZATION COMPLETE")
    print("=" * 50)
    print(f"Mode: {mode}")
    print(f"Papers: {stats['papers']}")
    print(f"Rejected: {stats['rejected']}")
    print(f"Uncertain: {stats['uncertain']}")
    print(f"Errors: {stats['errors']}")
    print(f"\nOutput: {output_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Organize classified PDFs")
    parser.add_argument(
        "--mode",
        choices=["copy", "move", "symlink"],
        default="copy",
        help="How to organize files (default: copy)"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("data/cleaning_logs"),
        help="Directory containing classification logs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/organized"),
        help="Output directory for organized files"
    )
    parser.add_argument(
        "--papers-only",
        action="store_true",
        help="Only organize papers (skip rejected/uncertain)"
    )
    parser.add_argument(
        "--phase1-prep",
        action="store_true",
        help="Prepare papers for Phase 1 (copies to data/phase1_papers/)"
    )

    args = parser.parse_args()

    # Special mode for Phase 1 preparation
    if args.phase1_prep:
        organize_files(
            log_dir=args.log_dir,
            output_dir=Path("data/phase1_papers"),
            mode="copy",
            papers_only=True,
        )
        print("\nPapers ready for Phase 1 in: data/phase1_papers/")
        return

    organize_files(
        log_dir=args.log_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        papers_only=args.papers_only,
    )


if __name__ == "__main__":
    main()
```

**Tasks:**

- [x] Create `backend/data_cleaning/organize_files.py` with the code above
- [ ] Choose organization strategy (symlink/copy/move)
- [ ] Run organization script

### 9.6 Prepare for Phase 1

Copy classified papers to Phase 1 input directory:

```bash
cd backend

# Prepare papers for Phase 1
python -m data_cleaning.organize_files --phase1-prep

# Verify
ls data/phase1_papers/ | wc -l
echo "Papers ready for Phase 1"
```

Alternatively, select a random sample of 50 papers for Phase 1 validation:

```bash
# Create sample directory
mkdir -p data/sample_papers

# Copy 50 random papers for Phase 1 validation
cd backend
python -c "
import random
import shutil
from pathlib import Path

papers = list(Path('data/classified/papers').glob('*.pdf'))
sample = random.sample(papers, min(50, len(papers)))

for p in sample:
    target = Path('data/sample_papers') / p.name
    if p.is_symlink():
        shutil.copy2(p.resolve(), target)
    else:
        shutil.copy2(p, target)

print(f'Copied {len(sample)} papers to data/sample_papers/')
"
```

**Tasks:**

- [ ] Run Phase 1 preparation script
- [ ] Verify paper count in output directory
- [ ] (Optional) Create 50-paper sample for initial validation

---

## Section 10: Completion Checklist

### Setup

- [x] Directory structure created
- [x] Configuration file created
- [x] All `__init__.py` files created

### Filters

- [x] `models.py` created
- [x] `filename_filter.py` created and tested
- [x] `metadata_filter.py` created
- [x] `content_filter.py` created
- [x] `llm_filter.py` created

### Main Pipeline

- [x] `classifier.py` created
- [x] `run_cleaning.py` created
- [x] `review_tool.py` created

### Execution

- [x] Sample test completed
- [ ] Full classification without LLM completed
- [ ] LLM classification on uncertain completed
- [ ] Manual review completed

### File Organization

- [x] `organize_files.py` created
- [ ] Organization strategy chosen (symlink/copy/move)
- [ ] Files organized to final directories
- [ ] Phase 1 preparation completed

### Results

- [ ] Final paper count: **\_\_**
- [ ] Final rejected count: **\_\_**
- [ ] Final uncertain count: **\_\_**
- [ ] Papers location for Phase 1: **\_\_**

---

## Statistics Log

Track progress here:

| Stage               | Papers | Rejected | Uncertain | Notes                 |
| ------------------- | ------ | -------- | --------- | --------------------- |
| Initial             | -      | -        | 22,401    | All files             |
| Sample run (no LLM) | 53     | 34       | 13        | 100 files, 2025-12-17 |
| After filename      |        |          |           |                       |
| After metadata      |        |          |           |                       |
| After content       |        |          |           |                       |
| After LLM           |        |          |           |                       |
| After manual review |        |          |           |                       |
| **Final**           |        |          |           |                       |

---

## Notes

**Estimated Processing Time:**

- Filename filter: ~1 sec / 1000 files
- Metadata filter: ~5 sec / 100 files
- Content filter: ~10 sec / 100 files
- LLM filter: ~2 sec / file (API calls)

**Cost Estimate for LLM:**

- If 5,000 files need LLM: ~$5-10 at Claude Haiku rates
- Consider using Haiku for faster/cheaper classification

**Tips:**

- Run without LLM first to minimize API costs
- Most files should be classifiable without LLM
- Use LLM only for truly ambiguous cases
- Sample-check results at each stage
