"""Deep content analysis filter for PDF classification."""

import re
import signal
from pathlib import Path
from typing import Dict, List, Tuple
import pymupdf

from ..models import Classification, RejectionReason, FilterResult


class TimeoutError(Exception):
    """Raised when PDF extraction times out."""
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("PDF extraction timed out")


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

    def _extract_text(self, pdf_path: Path, max_pages: int = 10, timeout_seconds: int = 30) -> str:
        """Extract text from PDF with timeout."""
        doc = None
        try:
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)

            doc = pymupdf.open(pdf_path)
            text = ""

            for i in range(min(max_pages, len(doc))):
                text += doc[i].get_text() + "\n\n"

            doc.close()
            signal.alarm(0)  # Cancel timeout
            return text

        except TimeoutError:
            if doc:
                doc.close()
            return "ERROR: PDF extraction timed out"
        except Exception as e:
            if doc:
                doc.close()
            return f"ERROR: {e}"
        finally:
            signal.alarm(0)  # Ensure timeout is cancelled

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