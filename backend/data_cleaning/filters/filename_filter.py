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