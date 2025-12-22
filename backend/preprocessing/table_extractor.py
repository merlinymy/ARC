"""Table extraction from PDFs using MinerU.

Extracts tables from research papers, particularly focusing on:
- IC50/CC50/EC50 value tables
- Compound activity tables
- Results summary tables

Uses MinerU's StructEqTable for accurate table detection and extraction.
"""

import logging
from pathlib import Path
from typing import List, Optional
import re

logger = logging.getLogger(__name__)


class TableExtractor:
    """Extract tables from PDF files using MinerU backend."""

    def __init__(self, min_rows: int = 2, min_cols: int = 2):
        """Initialize table extractor.

        Args:
            min_rows: Minimum rows for a valid table
            min_cols: Minimum columns for a valid table
        """
        self.min_rows = min_rows
        self.min_cols = min_cols
        self._extractor = None

        # Patterns for identifying valuable tables
        self.value_patterns = [
            r'\d+\.?\d*\s*[nμµ]?[mM]',  # IC50, EC50 values
            r'IC50|CC50|EC50|MIC',       # Common assay headers
            r'\d+\s*±\s*\d+',            # Values with error bars
            r'compound|inhibitor|peptide',  # Common table subjects
        ]

    def _get_extractor(self):
        """Lazy load MinerU extractor."""
        if self._extractor is None:
            from .pdf_processor import MinerUExtractor
            self._extractor = MinerUExtractor()
        return self._extractor

    def extract_tables(self, pdf_path: Path) -> List[str]:
        """Extract all tables from a PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of table contents as formatted strings
        """
        try:
            extractor = self._get_extractor()
            content = extractor.extract(pdf_path)

            tables = []
            for idx, table in enumerate(content.tables):
                if table and self._is_valid_table(table):
                    formatted = self._format_table(table, idx)
                    if formatted:
                        tables.append(formatted)

            logger.debug(f"Extracted {len(tables)} tables from {pdf_path.name}")
            return tables

        except Exception as e:
            logger.error(f"Error extracting tables from {pdf_path}: {e}")
            return []

    def _is_valid_table(self, table_text: str) -> bool:
        """Check if extracted table meets minimum requirements."""
        if not table_text:
            return False

        # Check if it has enough content
        if len(table_text) < 50:
            return False

        # Check for table-like structure (pipes or tabs)
        lines = table_text.strip().split('\n')
        if len(lines) < self.min_rows:
            return False

        return True

    def _format_table(self, table_text: str, table_idx: int) -> Optional[str]:
        """Format table for embedding.

        Args:
            table_text: Raw table text (markdown/LaTeX from MinerU)
            table_idx: Table index

        Returns:
            Formatted table string or None if invalid
        """
        if not table_text:
            return None

        # Add header
        formatted = f"[Table {table_idx + 1}]\n{table_text}"

        # Check if table has meaningful content
        if len(formatted) < 50:
            return None

        return formatted

    def _has_scientific_content(self, table_text: str) -> bool:
        """Check if table contains valuable scientific data.

        Args:
            table_text: Formatted table text

        Returns:
            True if table appears to contain scientific data
        """
        for pattern in self.value_patterns:
            if re.search(pattern, table_text, re.IGNORECASE):
                return True

        return False

    def extract_ic50_tables(self, pdf_path: Path) -> List[str]:
        """Extract tables that contain IC50/activity data.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of tables containing IC50 or similar data
        """
        all_tables = self.extract_tables(pdf_path)

        ic50_patterns = [
            r'IC50|IC-50|IC₅₀',
            r'CC50|CC-50|CC₅₀',
            r'EC50|EC-50|EC₅₀',
            r'MIC|minimal inhibitory',
            r'inhibition|inhibitory',
            r'\d+\.?\d*\s*[nμµ]M',
        ]

        ic50_tables = []
        for table in all_tables:
            for pattern in ic50_patterns:
                if re.search(pattern, table, re.IGNORECASE):
                    ic50_tables.append(table)
                    break

        logger.debug(f"Found {len(ic50_tables)} IC50-related tables in {pdf_path.name}")
        return ic50_tables
