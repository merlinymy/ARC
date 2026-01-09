#!/usr/bin/env python3
"""
Batch DOI Extraction and Metadata Update Script

This script processes all papers in the library to:
1. Extract DOI from PDFs that don't have one
2. Fetch metadata from CrossRef using the DOI
3. Update paper metadata in the database

Features:
- Rate limiting to respect CrossRef API limits (1 request/second)
- Progress tracking
- Error handling and logging
- Dry-run mode for testing

Usage:
    python batch_extract_doi_metadata.py [--dry-run] [--api-url http://localhost:8000]
"""

import argparse
import requests
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys


class DOIBatchProcessor:
    """Batch processor for extracting DOIs and updating metadata."""

    def __init__(self, api_url: str = "http://localhost:8000", dry_run: bool = False, rate_limit: float = 1.0):
        """
        Initialize the batch processor.

        Args:
            api_url: Base URL of the API
            dry_run: If True, don't actually update papers
            rate_limit: Seconds to wait between CrossRef requests (default: 1.0)
        """
        self.api_url = api_url.rstrip('/')
        self.dry_run = dry_run
        self.rate_limit = rate_limit
        self.stats = {
            'total': 0,
            'processed': 0,
            'doi_extracted': 0,
            'metadata_updated': 0,
            'errors': 0,
            'skipped': 0
        }
        self.last_crossref_request = 0.0

    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    def rate_limit_wait(self):
        """Wait to respect rate limits for CrossRef API."""
        elapsed = time.time() - self.last_crossref_request
        if elapsed < self.rate_limit:
            wait_time = self.rate_limit - elapsed
            time.sleep(wait_time)
        self.last_crossref_request = time.time()

    def fetch_all_papers(self) -> List[Dict[str, Any]]:
        """Fetch all papers from the API."""
        self.log("Fetching all papers from API...")
        try:
            all_papers = []
            offset = 0
            limit = 100

            while True:
                response = requests.get(
                    f"{self.api_url}/papers",
                    params={'offset': offset, 'limit': limit},
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()

                papers = data.get('papers', [])
                all_papers.extend(papers)

                self.log(f"Fetched {len(papers)} papers (total: {len(all_papers)})")

                # Check if there are more papers (API uses snake_case: has_more)
                if not data.get('has_more', False):
                    break

                offset += limit

            self.log(f"✓ Successfully fetched {len(all_papers)} papers")
            return all_papers

        except Exception as e:
            self.log(f"✗ Failed to fetch papers: {e}", "ERROR")
            raise

    def should_process_paper(self, paper: Dict[str, Any]) -> tuple[bool, str]:
        """
        Determine if a paper should be processed.

        Returns:
            (should_process, reason)
        """
        # Skip if paper already has DOI and good metadata
        has_doi = paper.get('doi') is not None and paper.get('doi') != ''
        has_authors = len(paper.get('authors', [])) > 0
        has_proper_title = paper.get('title', '') != '' and paper.get('title', '') != 'Unknown'

        if has_doi and has_authors and has_proper_title:
            return False, "Already has DOI and good metadata"

        if not has_doi:
            return True, "Missing DOI"

        if not has_authors:
            return True, "Missing authors"

        if not has_proper_title:
            return True, "Poor title quality"

        return False, "Unknown"

    def extract_doi(self, paper_id: str) -> Optional[str]:
        """Extract DOI from a paper's PDF."""
        try:
            response = requests.get(
                f"{self.api_url}/papers/{paper_id}/extract-doi",
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get('doi')

        except Exception as e:
            self.log(f"  ✗ Failed to extract DOI: {e}", "ERROR")
            return None

    def fetch_metadata_from_crossref(self, doi: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata from CrossRef using DOI."""
        self.rate_limit_wait()

        try:
            response = requests.get(
                f"{self.api_url}/metadata/doi/{requests.utils.quote(doi, safe='')}",
                timeout=30
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                self.log(f"  ℹ DOI not found in CrossRef: {doi}", "WARN")
            else:
                self.log(f"  ✗ CrossRef API error: {e}", "ERROR")
            return None
        except Exception as e:
            self.log(f"  ✗ Failed to fetch metadata from CrossRef: {e}", "ERROR")
            return None

    def update_paper_metadata(self, paper_id: str, updates: Dict[str, Any]) -> bool:
        """Update paper metadata via API."""
        if self.dry_run:
            self.log(f"  [DRY RUN] Would update paper {paper_id} with: {updates}")
            return True

        try:
            response = requests.patch(
                f"{self.api_url}/papers/{paper_id}",
                json=updates,
                timeout=30
            )
            response.raise_for_status()
            return True

        except Exception as e:
            self.log(f"  ✗ Failed to update paper metadata: {e}", "ERROR")
            return False

    def process_paper(self, paper: Dict[str, Any]) -> bool:
        """
        Process a single paper.

        Returns:
            True if successful, False otherwise
        """
        paper_id = paper['paper_id']
        paper_title = paper.get('title', 'Unknown')[:50]

        self.log(f"\nProcessing: {paper_title}... (ID: {paper_id})")

        # Check if we should process this paper
        should_process, reason = self.should_process_paper(paper)
        if not should_process:
            self.log(f"  ⊘ Skipping: {reason}")
            self.stats['skipped'] += 1
            return True

        self.log(f"  → Reason: {reason}")

        # Step 1: Extract DOI if missing
        doi = paper.get('doi')
        if not doi:
            self.log("  → Extracting DOI from PDF...")
            doi = self.extract_doi(paper_id)

            if doi:
                self.log(f"  ✓ Found DOI: {doi}")
                self.stats['doi_extracted'] += 1
            else:
                self.log("  ℹ No DOI found in PDF")
                return True  # Not an error, just no DOI available

        # Step 2: Fetch metadata from CrossRef
        self.log(f"  → Fetching metadata from CrossRef for DOI: {doi}")
        metadata = self.fetch_metadata_from_crossref(doi)

        if not metadata:
            self.log("  ✗ Could not fetch metadata from CrossRef")
            self.stats['errors'] += 1
            return False

        # Step 3: Prepare updates
        updates = {}

        if metadata.get('title') and len(metadata['title']) > 10:
            current_title = paper.get('title', '')
            if current_title == 'Unknown' or current_title == '' or current_title == 'No Job Name':
                updates['title'] = metadata['title']
                self.log(f"  → Will update title: {metadata['title'][:50]}...")

        if metadata.get('authors') and len(metadata['authors']) > 0:
            if len(paper.get('authors', [])) == 0:
                updates['authors'] = metadata['authors']
                self.log(f"  → Will update authors: {', '.join(metadata['authors'][:2])}...")

        if metadata.get('year'):
            if not paper.get('year'):
                updates['year'] = metadata['year']
                self.log(f"  → Will update year: {metadata['year']}")

        # Generate proper filename if we have author and title
        if 'authors' in updates and 'title' in updates:
            first_author_last_name = updates['authors'][0].split()[-1] if updates['authors'] else 'Author'
            year = updates.get('year', metadata.get('year', ''))
            title_slug = updates['title'][:50].replace(' ', '-').replace('/', '-')
            # Clean up special characters
            title_slug = ''.join(c for c in title_slug if c.isalnum() or c == '-')
            new_filename = f"{first_author_last_name}-{year}-{title_slug}.pdf"
            updates['filename'] = new_filename
            self.log(f"  → Will update filename: {new_filename}")

        # Step 4: Update paper
        if updates:
            self.log(f"  → Updating paper metadata ({len(updates)} fields)...")
            if self.update_paper_metadata(paper_id, updates):
                self.log("  ✓ Successfully updated paper metadata")
                self.stats['metadata_updated'] += 1
                return True
            else:
                self.stats['errors'] += 1
                return False
        else:
            self.log("  ℹ No metadata improvements available")
            return True

    def run(self):
        """Run the batch processing."""
        self.log("=" * 70)
        self.log("DOI Batch Extraction and Metadata Update")
        self.log("=" * 70)
        if self.dry_run:
            self.log("⚠ DRY RUN MODE - No changes will be made", "WARN")
        self.log(f"API URL: {self.api_url}")
        self.log(f"Rate limit: {self.rate_limit}s between CrossRef requests")
        self.log("")

        try:
            # Fetch all papers
            papers = self.fetch_all_papers()
            self.stats['total'] = len(papers)

            if self.stats['total'] == 0:
                self.log("No papers found to process.")
                return

            self.log("")
            self.log("=" * 70)
            self.log(f"Starting processing of {self.stats['total']} papers...")
            self.log("=" * 70)

            # Process each paper
            for idx, paper in enumerate(papers, 1):
                self.log(f"\n[{idx}/{self.stats['total']}]", "INFO")
                self.process_paper(paper)
                self.stats['processed'] += 1

                # Show progress every 10 papers
                if idx % 10 == 0:
                    self.log("")
                    self.log("-" * 70)
                    self.log(f"Progress: {idx}/{self.stats['total']} papers processed")
                    self.log(f"Stats: DOIs extracted: {self.stats['doi_extracted']}, "
                            f"Metadata updated: {self.stats['metadata_updated']}, "
                            f"Errors: {self.stats['errors']}, "
                            f"Skipped: {self.stats['skipped']}")
                    self.log("-" * 70)

            # Final summary
            self.log("")
            self.log("=" * 70)
            self.log("FINAL SUMMARY")
            self.log("=" * 70)
            self.log(f"Total papers: {self.stats['total']}")
            self.log(f"Processed: {self.stats['processed']}")
            self.log(f"DOIs extracted: {self.stats['doi_extracted']}")
            self.log(f"Metadata updated: {self.stats['metadata_updated']}")
            self.log(f"Skipped (already good): {self.stats['skipped']}")
            self.log(f"Errors: {self.stats['errors']}")
            self.log("=" * 70)

            if self.dry_run:
                self.log("\n⚠ This was a DRY RUN - no actual changes were made")

        except KeyboardInterrupt:
            self.log("\n\n⚠ Interrupted by user", "WARN")
            self.log("Partial progress:")
            self.log(f"  Processed: {self.stats['processed']}/{self.stats['total']}")
            self.log(f"  DOIs extracted: {self.stats['doi_extracted']}")
            self.log(f"  Metadata updated: {self.stats['metadata_updated']}")
            sys.exit(1)
        except Exception as e:
            self.log(f"\n\n✗ Fatal error: {e}", "ERROR")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch extract DOIs and update metadata for all papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be updated
  python batch_extract_doi_metadata.py --dry-run

  # Actually process all papers
  python batch_extract_doi_metadata.py

  # Use custom API URL
  python batch_extract_doi_metadata.py --api-url http://localhost:8000

  # Custom rate limit (2 seconds between requests)
  python batch_extract_doi_metadata.py --rate-limit 2.0
        """
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run in dry-run mode (no actual updates)'
    )
    parser.add_argument(
        '--api-url',
        default='http://localhost:8000',
        help='Base URL of the API (default: http://localhost:8000)'
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=1.0,
        help='Seconds to wait between CrossRef requests (default: 1.0)'
    )

    args = parser.parse_args()

    processor = DOIBatchProcessor(
        api_url=args.api_url,
        dry_run=args.dry_run,
        rate_limit=args.rate_limit
    )
    processor.run()


if __name__ == '__main__':
    main()
