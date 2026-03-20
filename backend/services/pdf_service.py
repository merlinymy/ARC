"""PDF service for converting and sending PDF files to Claude API.

This module handles:
- Fetching PDF files from disk
- Converting PDFs to base64
- Formatting PDFs for Claude API document blocks
- Validating file sizes and page limits
"""

import base64
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class PDFService:
    """Service for handling PDF documents for Claude API."""

    # Claude API limits (leaving buffer for request overhead)
    MAX_REQUEST_SIZE_MB = 30  # 32MB limit, use 30MB for safety
    MAX_PAGES = 100

    def __init__(self, paper_library_service: Any):
        """Initialize PDF service.

        Args:
            paper_library_service: PaperLibraryService instance for accessing PDFs
        """
        self.paper_library = paper_library_service

    def convert_pdf_to_base64(self, pdf_path: Path) -> Optional[str]:
        """Convert a PDF file to base64 string.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Base64 encoded string or None if conversion fails
        """
        try:
            with open(pdf_path, 'rb') as pdf_file:
                pdf_bytes = pdf_file.read()
                base64_pdf = base64.standard_b64encode(pdf_bytes).decode('utf-8')
                return base64_pdf
        except Exception as e:
            logger.error(f"Failed to convert PDF to base64: {pdf_path}, error: {e}")
            return None

    def get_pdf_info(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get PDF file information.

        Args:
            paper_id: The paper ID

        Returns:
            Dictionary with pdf_path, size_mb, and page_count, or None if paper not found
        """
        try:
            # Get paper metadata
            paper = self.paper_library.get_paper(paper_id)
            if not paper:
                logger.warning(f"Paper not found: {paper_id}")
                return None

            # Get PDF path
            pdf_path = self.paper_library.get_pdf_path(paper_id)
            if not pdf_path or not pdf_path.exists():
                logger.warning(f"PDF file not found for paper: {paper_id}")
                return None

            # Get file size
            size_bytes = pdf_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)

            return {
                'pdf_path': pdf_path,
                'size_mb': size_mb,
                'page_count': paper.page_count,
                'title': paper.title,
            }
        except Exception as e:
            logger.error(f"Failed to get PDF info for paper {paper_id}: {e}")
            return None

    def validate_pdfs(self, paper_ids: List[str]) -> Dict[str, Any]:
        """Validate that PDFs can be sent to Claude within limits.

        Args:
            paper_ids: List of paper IDs

        Returns:
            Dictionary with:
                - valid: bool, whether PDFs are within limits
                - total_size_mb: float, total size in MB
                - total_pages: int, total page count
                - reason: str, reason if invalid
                - valid_papers: List of valid paper info dicts
        """
        total_size_mb = 0.0
        total_pages = 0
        valid_papers = []

        for paper_id in paper_ids:
            pdf_info = self.get_pdf_info(paper_id)
            if pdf_info:
                total_size_mb += pdf_info['size_mb']
                total_pages += pdf_info['page_count']
                valid_papers.append(pdf_info)
            else:
                logger.warning(f"Skipping paper {paper_id} - PDF not found")

        # Validate limits
        if total_size_mb > self.MAX_REQUEST_SIZE_MB:
            return {
                'valid': False,
                'total_size_mb': total_size_mb,
                'total_pages': total_pages,
                'reason': f"Total size ({total_size_mb:.1f}MB) exceeds {self.MAX_REQUEST_SIZE_MB}MB limit",
                'valid_papers': [],
            }

        if total_pages > self.MAX_PAGES:
            return {
                'valid': False,
                'total_size_mb': total_size_mb,
                'total_pages': total_pages,
                'reason': f"Total pages ({total_pages}) exceeds {self.MAX_PAGES} page limit",
                'valid_papers': [],
            }

        return {
            'valid': True,
            'total_size_mb': total_size_mb,
            'total_pages': total_pages,
            'reason': None,
            'valid_papers': valid_papers,
        }

    def create_document_blocks(self, paper_ids: List[str]) -> List[Dict[str, Any]]:
        """Create Claude API document content blocks for PDFs.

        Args:
            paper_ids: List of paper IDs to include

        Returns:
            List of document content blocks for Claude API, empty if validation fails
        """
        # Validate PDFs
        validation = self.validate_pdfs(paper_ids)

        if not validation['valid']:
            logger.warning(f"PDF validation failed: {validation['reason']}")
            return []

        logger.info(
            f"Preparing {len(validation['valid_papers'])} PDFs "
            f"({validation['total_size_mb']:.1f}MB, {validation['total_pages']} pages)"
        )

        # Create document blocks
        document_blocks = []

        for pdf_info in validation['valid_papers']:
            # Convert PDF to base64
            base64_pdf = self.convert_pdf_to_base64(pdf_info['pdf_path'])

            if base64_pdf:
                # Create document block following Claude API format
                document_block = {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": base64_pdf,
                    },
                    # Optional: Add cache control for repeated queries
                    "cache_control": {"type": "ephemeral"}
                }
                document_blocks.append(document_block)
                logger.info(f"Added PDF document: {pdf_info['title']}")
            else:
                logger.warning(f"Failed to convert PDF: {pdf_info['title']}")

        logger.info(f"Created {len(document_blocks)} document blocks for Claude API")
        return document_blocks

    def create_user_message_with_pdfs(
        self,
        query: str,
        paper_ids: List[str],
        sources_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a user message with both PDF documents and text content.

        Args:
            query: The user's question
            paper_ids: List of paper IDs to include as PDFs
            sources_text: Optional RAG sources text to include

        Returns:
            A user message dict with both document blocks and text content
        """
        # Get PDF document blocks
        document_blocks = self.create_document_blocks(paper_ids)

        # Build content array - PDFs first (Claude best practice), then text
        content = []

        # Add PDFs first
        content.extend(document_blocks)

        # Build text content
        if sources_text:
            text_content = f"""Question: {query}

Retrieved Sources from Uploaded Papers (RAG):
{sources_text}

IMPORTANT: You have access to both the full PDF documents AND the retrieved source chunks. Use both to provide a comprehensive answer with proper citations using [Source N] format."""
        else:
            text_content = f"""Question: {query}

IMPORTANT: You have access to the full PDF documents. Please analyze them to answer the question comprehensively."""

        # Add text content
        content.append({
            "type": "text",
            "text": text_content
        })

        return {
            "role": "user",
            "content": content
        }
