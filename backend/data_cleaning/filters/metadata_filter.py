"""Filter PDFs based on PDF metadata and basic properties."""

import pymupdf
from pathlib import Path
from typing import Optional, Dict, Any
import re
import signal

from ..models import Classification, RejectionReason, FilterResult


class TimeoutError(Exception):
    """Raised when PDF extraction times out."""
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("PDF extraction timed out")


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

    def _extract_metadata(self, pdf_path: Path, timeout_seconds: int = 30) -> Dict[str, Any]:
        """Extract metadata from PDF with timeout."""
        doc = None
        try:
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)

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
            signal.alarm(0)  # Cancel timeout

            return {
                'metadata': metadata,
                'num_pages': num_pages,
                'text_sample': text_sample,
                'text_length': len(total_text),
                'file_size': pdf_path.stat().st_size,
            }

        except TimeoutError:
            if doc:
                doc.close()
            return {
                'error': 'PDF extraction timed out',
                'num_pages': 0,
                'text_length': 0,
            }
        except Exception as e:
            if doc:
                doc.close()
            return {
                'error': str(e),
                'num_pages': 0,
                'text_length': 0,
            }
        finally:
            signal.alarm(0)  # Ensure timeout is cancelled

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