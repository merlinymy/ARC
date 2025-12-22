"""Figure and table caption extraction from PDFs using MinerU.

Extracts captions from research papers:
- Figure captions (Figure 1, Fig. 1, etc.)
- Table captions (Table 1, etc.)
- Scheme captions (Scheme 1, etc.)

Captions often contain key experimental details and are valuable for retrieval.
"""

import re
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


# Patterns for detecting caption starts
CAPTION_PATTERNS = [
    # Figure patterns
    r'(?:Figure|Fig\.?)\s*(\d+[a-zA-Z]?)\s*[\.:\-–—]?\s*',
    r'(?:Figure|Fig\.?)\s*S?(\d+[a-zA-Z]?)\s*[\.:\-–—]?\s*',  # Supplementary

    # Table patterns
    r'Table\s*(\d+[a-zA-Z]?)\s*[\.:\-–—]?\s*',
    r'Table\s*S?(\d+[a-zA-Z]?)\s*[\.:\-–—]?\s*',  # Supplementary

    # Scheme patterns (chemistry papers)
    r'Scheme\s*(\d+[a-zA-Z]?)\s*[\.:\-–—]?\s*',

    # Chart patterns
    r'Chart\s*(\d+[a-zA-Z]?)\s*[\.:\-–—]?\s*',
]


class CaptionExtractor:
    """Extract figure and table captions from PDFs using MinerU."""

    def __init__(self, max_caption_length: int = 1000):
        """Initialize caption extractor.

        Args:
            max_caption_length: Maximum characters per caption
        """
        self.max_caption_length = max_caption_length
        self.patterns = [re.compile(p, re.IGNORECASE) for p in CAPTION_PATTERNS]
        self._extractor = None

    def _get_extractor(self):
        """Lazy load MinerU extractor."""
        if self._extractor is None:
            from .pdf_processor import MinerUExtractor
            self._extractor = MinerUExtractor()
        return self._extractor

    def extract_captions(self, pdf_path: Path) -> List[str]:
        """Extract all captions from a PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of caption strings
        """
        try:
            extractor = self._get_extractor()
            content = extractor.extract(pdf_path)

            # Get captions from MinerU extraction
            captions = list(content.captions)

            # Also extract from figures that have captions
            for fig in content.figures:
                caption = fig.get('caption', '')
                if caption and len(caption) >= 20:
                    captions.append(caption[:self.max_caption_length])

            # Also try to extract from full text as backup
            text_captions = self._extract_captions_from_text(content.full_text)
            captions.extend(text_captions)

            # Deduplicate while preserving order
            seen = set()
            unique_captions = []
            for caption in captions:
                caption_key = caption[:100].lower()
                if caption_key not in seen:
                    seen.add(caption_key)
                    unique_captions.append(caption)

            logger.debug(f"Extracted {len(unique_captions)} unique captions from {pdf_path.name}")
            return unique_captions

        except Exception as e:
            logger.error(f"Error extracting captions from {pdf_path}: {e}")
            return []

    def extract_captions_from_text(self, text: str) -> List[str]:
        """Extract captions from raw text.

        Useful for extracting from already-parsed PDF text.

        Args:
            text: Raw text from PDF

        Returns:
            List of caption strings
        """
        return self._extract_captions_from_text(text)

    def _extract_captions_from_text(self, text: str) -> List[str]:
        """Internal method to extract captions from text."""
        captions = []
        lines = text.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check if line starts a caption
            caption_match = None
            for pattern in self.patterns:
                match = pattern.match(line)
                if match:
                    caption_match = match
                    break

            if caption_match:
                # Found a caption start, collect the full caption
                caption_lines = [line]
                j = i + 1

                # Continue collecting lines until we hit another caption or empty lines
                while j < len(lines):
                    next_line = lines[j].strip()

                    # Stop conditions
                    if not next_line:
                        # Allow one empty line, but stop at two
                        if j + 1 < len(lines) and not lines[j + 1].strip():
                            break
                        j += 1
                        continue

                    # Check if next line is a new caption
                    is_new_caption = False
                    for pattern in self.patterns:
                        if pattern.match(next_line):
                            is_new_caption = True
                            break

                    if is_new_caption:
                        break

                    # Check if line looks like main text (long paragraph)
                    if len(next_line) > 200 and not next_line.endswith('.'):
                        break

                    caption_lines.append(next_line)
                    j += 1

                    # Limit caption length
                    current_length = sum(len(l) for l in caption_lines)
                    if current_length > self.max_caption_length:
                        break

                # Join caption lines
                full_caption = ' '.join(caption_lines)
                full_caption = re.sub(r'\s+', ' ', full_caption).strip()

                if len(full_caption) >= 20:  # Minimum caption length
                    captions.append(full_caption[:self.max_caption_length])

                i = j
            else:
                i += 1

        return captions

    def extract_figure_captions(self, pdf_path: Path) -> List[str]:
        """Extract only figure captions.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of figure caption strings
        """
        all_captions = self.extract_captions(pdf_path)

        figure_pattern = re.compile(r'^(?:Figure|Fig\.?)\s*S?\d+', re.IGNORECASE)
        return [c for c in all_captions if figure_pattern.match(c)]

    def extract_table_captions(self, pdf_path: Path) -> List[str]:
        """Extract only table captions.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of table caption strings
        """
        all_captions = self.extract_captions(pdf_path)

        table_pattern = re.compile(r'^Table\s*S?\d+', re.IGNORECASE)
        return [c for c in all_captions if table_pattern.match(c)]
