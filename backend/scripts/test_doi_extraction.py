#!/usr/bin/env python3
"""
Test DOI Extraction on Individual PDFs

This script tests DOI extraction on one or more PDF files to diagnose issues.

Usage:
    python test_doi_extraction.py path/to/paper.pdf
    python test_doi_extraction.py path/to/papers/*.pdf
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.pdf_processor import EnhancedPDFProcessor
import pypdfium2 as pdfium
import re


def analyze_pdf(pdf_path: Path):
    """Analyze a PDF for DOI extraction."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {pdf_path.name}")
    print(f"{'='*80}")

    doc = None
    try:
        # Check file size
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")

        # Open PDF
        doc = pdfium.PdfDocument(pdf_path)
        print(f"Pages: {len(doc)}")

        # Check metadata
        print(f"\nMetadata:")
        metadata = doc.get_metadata_dict()
        for key, value in metadata.items():
            if value:
                print(f"  {key}: {value[:100]}")

        # Extract text from first 3 pages
        print(f"\nText extraction from first 3 pages:")
        pages_to_check = min(3, len(doc))

        for page_num in range(pages_to_check):
            page = doc[page_num]
            textpage = page.get_textpage()
            page_text = textpage.get_text_bounded()
            textpage.close()  # Close textpage to free resources

            print(f"\n--- Page {page_num + 1} (first 500 chars) ---")
            print(page_text[:500])

            # Look for DOI patterns
            doi_patterns = [
                (r'doi\.org/([0-9]{2}\.[0-9]{4,}/[^\s\"\'\)]+)', 'doi.org/...'),
                (r'DOI:?\s*([0-9]{2}\.[0-9]{4,}/[^\s\"\'\)]+)', 'DOI: ...'),
                (r'doi:?\s*([0-9]{2}\.[0-9]{4,}/[^\s\"\'\)]+)', 'doi: ...'),
                (r'\bhttps?://dx\.doi\.org/([0-9]{2}\.[0-9]{4,}/[^\s\"\'\)]+)', 'dx.doi.org/...'),
                (r'\b([0-9]{2}\.[0-9]{4,}/[A-Za-z0-9\.\-_\(\)/]+)', 'raw 10.xxxx/...'),
            ]

            found_any = False
            for pattern, name in doi_patterns:
                matches = re.finditer(pattern, page_text, re.IGNORECASE)
                for match in matches:
                    if not found_any:
                        print(f"\n  Found DOI patterns on page {page_num + 1}:")
                        found_any = True
                    doi = match.group(1) if len(match.groups()) > 0 else match.group(0)
                    doi = doi.strip().rstrip('.,;:\)\"\' ')
                    doi = re.sub(r'[.,;:\)\"\'\s]+$', '', doi)
                    print(f"    [{name}] {doi}")

            if not found_any and page_num == 0:
                print(f"  ⚠ No DOI patterns found on page {page_num + 1}")

        # Try extraction with processor
        print(f"\n{'='*80}")
        print(f"Using EnhancedPDFProcessor._extract_doi_from_pdf():")
        print(f"{'='*80}")
        processor = EnhancedPDFProcessor()
        doi = processor._extract_doi_from_pdf(pdf_path)

        if doi:
            print(f"✓ EXTRACTED DOI: {doi}")
        else:
            print(f"✗ NO DOI EXTRACTED")

            # Provide diagnosis
            print(f"\nPossible reasons:")
            print(f"  1. PDF is a scanned image without text layer")
            print(f"  2. DOI is not on the first 3 pages")
            print(f"  3. DOI format is not recognized by patterns")
            print(f"  4. Paper doesn't have a DOI (pre-2000 papers)")

    except Exception as e:
        print(f"✗ Error analyzing PDF: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Always close the PDF document to prevent file descriptor leaks
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_doi_extraction.py path/to/paper.pdf")
        print("       python test_doi_extraction.py path/to/papers/*.pdf")
        sys.exit(1)

    # Get PDF paths from arguments
    pdf_paths = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        if path.is_file() and path.suffix.lower() == '.pdf':
            pdf_paths.append(path)
        elif path.is_dir():
            pdf_paths.extend(path.glob('*.pdf'))

    if not pdf_paths:
        print("No PDF files found")
        sys.exit(1)

    print(f"Testing {len(pdf_paths)} PDF(s)...")

    for pdf_path in pdf_paths:
        analyze_pdf(pdf_path)

    print(f"\n{'='*80}")
    print(f"Summary: Tested {len(pdf_paths)} PDF(s)")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
