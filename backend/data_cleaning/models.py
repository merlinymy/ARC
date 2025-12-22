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