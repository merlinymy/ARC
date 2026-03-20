"""Paper Library Service for managing research papers."""

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from config import settings
from preprocessing import EnhancedPDFProcessor, Chunk, ChunkType
from retrieval.embedder import VoyageEmbedder
from retrieval.qdrant_store import QdrantStore
from retrieval.bm25 import BM25Vectorizer

logger = logging.getLogger(__name__)

# Checkpoint file path (same as index_papers.py)
CHECKPOINT_FILE = Path("data/indexing_checkpoint.json")

# Cache for checkpoint data with modification time tracking
_checkpoint_cache: Dict[str, Any] = {}
_checkpoint_mtime: float = 0.0


@dataclass
class PaperInfo:
    """Information about a paper in the library."""
    paper_id: str
    title: str
    authors: List[str]
    year: Optional[int]
    doi: Optional[str]
    filename: str
    page_count: int
    chunk_count: int
    chunk_stats: Dict[str, int]
    indexed_at: Optional[datetime]
    status: str  # 'indexed', 'indexing', 'error', 'pending'
    error_message: Optional[str] = None
    file_size_bytes: int = 0  # Actual PDF file size in bytes


@dataclass
class PaginatedPapers:
    """Paginated list of papers."""
    papers: List[PaperInfo]
    total: int
    offset: int
    limit: int
    has_more: bool


class PaperLibraryService:
    """Service for managing the paper library.

    Provides CRUD operations for papers:
    - List all papers
    - Get paper details
    - Upload and index new papers
    - Delete papers (file + Qdrant chunks)
    """

    def __init__(
        self,
        qdrant_store: QdrantStore,
        upload_dir: Optional[Path] = None,
        embedder: Optional[VoyageEmbedder] = None,
        bm25_vectorizer: Optional[BM25Vectorizer] = None,
    ):
        """Initialize the paper library service.

        Args:
            qdrant_store: QdrantStore instance for vector operations
            upload_dir: Directory for uploaded PDFs (defaults to settings.upload_dir)
            embedder: Optional VoyageEmbedder for indexing
            bm25_vectorizer: Optional BM25Vectorizer for hybrid search
        """
        self.store = qdrant_store
        self.upload_dir = upload_dir or getattr(settings, 'upload_dir', Path('./uploads'))
        self.embedder = embedder
        self.bm25_vectorizer = bm25_vectorizer

        # Reusable PDF processor - avoids re-loading MinerU ML models per upload
        self._processor: Optional["EnhancedPDFProcessor"] = None

        # Ensure upload directory exists
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def _get_processor(self) -> "EnhancedPDFProcessor":
        """Get or create the reusable PDF processor.

        Reuses a single EnhancedPDFProcessor (and its MinerU models) across
        all uploads to avoid re-loading multi-GB ML models per paper.
        """
        if self._processor is None:
            self._processor = EnhancedPDFProcessor(
                abstract_max_tokens=settings.abstract_max_tokens,
                section_max_tokens=settings.section_max_tokens,
                fine_chunk_tokens=settings.fine_chunk_tokens,
                fine_chunk_overlap=settings.fine_chunk_overlap,
                extraction_timeout=settings.pdf_extraction_timeout,
            )
        return self._processor

    def _generate_paper_id(self, filename: str) -> str:
        """Generate a unique paper ID from filename (same as index_papers.py)."""
        return hashlib.md5(filename.encode()).hexdigest()[:12]

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _compute_content_hash(self, content: bytes) -> str:
        """Compute SHA256 hash from file content bytes."""
        return hashlib.sha256(content).hexdigest()

    def check_duplicate_hashes(self, hashes: List[str]) -> Dict[str, Optional[str]]:
        """Check which file hashes already exist in the library.

        Args:
            hashes: List of SHA256 file hashes to check

        Returns:
            Dict mapping hash -> paper_id if exists, None if not
        """
        checkpoint = self._load_checkpoint()
        file_hashes = checkpoint.get("file_hashes", {})

        result = {}
        for h in hashes:
            result[h] = file_hashes.get(h)
        return result

    def get_paper_by_hash(self, file_hash: str) -> Optional[str]:
        """Get paper_id by file hash, or None if not found."""
        checkpoint = self._load_checkpoint()
        file_hashes = checkpoint.get("file_hashes", {})
        return file_hashes.get(file_hash)

    def _store_file_hash(self, file_hash: str, paper_id: str) -> None:
        """Store a file hash -> paper_id mapping in checkpoint."""
        checkpoint = self._load_checkpoint()
        if "file_hashes" not in checkpoint:
            checkpoint["file_hashes"] = {}
        checkpoint["file_hashes"][file_hash] = paper_id
        self._save_checkpoint(checkpoint)

    def backfill_file_hashes(self) -> Dict[str, Any]:
        """Compute and store file hashes for all existing papers.

        This is used to populate hashes for papers indexed before
        duplicate detection was implemented.

        Returns:
            Dict with counts of processed, skipped, and failed papers
        """
        checkpoint = self._load_checkpoint()
        indexed_papers = checkpoint.get("indexed_papers", [])
        existing_hashes = checkpoint.get("file_hashes", {})

        # Create reverse lookup: paper_id -> hash
        papers_with_hash = set(existing_hashes.values())

        result = {"processed": 0, "skipped": 0, "failed": 0, "papers": []}

        for paper_id in indexed_papers:
            if paper_id in papers_with_hash:
                result["skipped"] += 1
                continue

            # Find the PDF file
            pdf_path = self._get_pdf_path(paper_id)
            if not pdf_path or not pdf_path.exists():
                result["failed"] += 1
                continue

            try:
                file_hash = self._compute_file_hash(pdf_path)
                self._store_file_hash(file_hash, paper_id)
                result["processed"] += 1
                result["papers"].append({"paper_id": paper_id, "hash": file_hash[:16] + "..."})
            except Exception as e:
                logger.error(f"Failed to compute hash for {paper_id}: {e}")
                result["failed"] += 1

        return result

    def backfill_paper_metadata(self, force: bool = False) -> Dict[str, Any]:
        """One-time backfill of paper metadata cache from Qdrant.

        This scans Qdrant once to build a cache of paper metadata.
        After this runs, list_papers() becomes instant.

        Args:
            force: If True, rebuild cache even if it exists

        Returns:
            Dict with status and count of papers backfilled
        """
        checkpoint = self._load_checkpoint()

        # Check if paper_metadata key exists (not just if it's non-empty)
        # This prevents backfill from running after every upload when cache is being built incrementally
        if "paper_metadata" in checkpoint and not force:
            existing_metadata = checkpoint.get("paper_metadata", {})
            return {"status": "skipped", "count": len(existing_metadata), "message": "Cache already exists"}

        logger.info("Backfilling paper metadata cache from Qdrant...")

        # Scan Qdrant once to get all papers (this is slow but only runs once)
        papers_data, total_count = self.store.get_papers_paginated(offset=0, limit=None)

        paper_metadata = {}
        for paper_data in papers_data:
            paper_id = paper_data['paper_id']

            # Get chunk stats (still slow, but only during backfill)
            chunk_stats = self.store.get_paper_chunk_stats(paper_id)

            paper_metadata[paper_id] = {
                "title": paper_data.get('title', 'Unknown'),
                "authors": paper_data.get('authors', []),
                "year": paper_data.get('year'),
                "filename": paper_data.get('file_name', ''),
                "chunk_count": sum(chunk_stats.values()),
                "chunk_stats": chunk_stats,
                "page_count": 0,  # Skip expensive page count query
                "indexed_at": datetime.now().isoformat(),
            }

        checkpoint["paper_metadata"] = paper_metadata
        self._save_checkpoint(checkpoint)

        logger.info(f"Backfilled metadata for {len(paper_metadata)} papers")
        return {"status": "success", "count": len(paper_metadata)}

    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load the indexing checkpoint file with caching and auto-reload.

        Automatically detects when the checkpoint file is modified by external scripts
        (like index_papers.py) and reloads the cache to stay in sync.
        """
        global _checkpoint_cache, _checkpoint_mtime

        if CHECKPOINT_FILE.exists():
            try:
                # Check if file was modified since last load
                current_mtime = CHECKPOINT_FILE.stat().st_mtime
                cache_is_stale = current_mtime > _checkpoint_mtime

                if not _checkpoint_cache or cache_is_stale:
                    # Cache is empty or file was modified - reload from disk
                    with open(CHECKPOINT_FILE) as f:
                        _checkpoint_cache = json.load(f)
                        _checkpoint_mtime = current_mtime

                    if cache_is_stale:
                        logger.info("Checkpoint file was modified externally - reloaded cache")

                return _checkpoint_cache.copy()  # Return a copy to prevent mutations

            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

        return {"indexed_papers": [], "failed_papers": {}, "stats": {}}

    def _save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save the indexing checkpoint file and update cache."""
        global _checkpoint_cache, _checkpoint_mtime

        CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        checkpoint["stats"]["last_updated"] = datetime.now().isoformat()

        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(checkpoint, f, indent=2)

        # Update cache and mtime to reflect our own write
        _checkpoint_cache = checkpoint.copy()
        if CHECKPOINT_FILE.exists():
            _checkpoint_mtime = CHECKPOINT_FILE.stat().st_mtime

    def _get_pdf_path(self, paper_id: str) -> Optional[Path]:
        """Find the PDF file for a given paper ID.

        Searches in both upload_dir and pdf_source_dir.
        """
        # Search in upload directory
        for pdf_file in self.upload_dir.glob("*.pdf"):
            if self._generate_paper_id(pdf_file.name) == paper_id:
                return pdf_file

        # Search in source directory if configured
        if settings.pdf_source_dir and settings.pdf_source_dir.exists():
            for pdf_file in settings.pdf_source_dir.glob("*.pdf"):
                if self._generate_paper_id(pdf_file.name) == paper_id:
                    return pdf_file

        return None

    def list_papers(
        self,
        offset: int = 0,
        limit: Optional[int] = None,
        search: Optional[str] = None,
        sort_by: str = "indexed_at",
        sort_order: str = "desc",
    ) -> PaginatedPapers:
        """List papers in the library with pagination and optional filtering.

        Uses cached metadata from checkpoint file for fast retrieval.
        Automatically backfills cache from Qdrant if empty.

        Args:
            offset: Number of papers to skip
            limit: Maximum papers to return (None for all - backwards compatible)
            search: Optional search query to filter by title, filename, or author (case-insensitive)
            sort_by: Field to sort by (title, year, chunk_count, indexed_at)
            sort_order: Sort order (asc or desc)

        Returns:
            PaginatedPapers with papers list and pagination metadata
        """
        checkpoint = self._load_checkpoint()

        # Backfill cache if it doesn't exist (one-time slow operation)
        # Note: We check if the key exists, not if it's empty, because the cache is built incrementally
        if "paper_metadata" not in checkpoint:
            self.backfill_paper_metadata()
            checkpoint = self._load_checkpoint()

        paper_metadata = checkpoint.get("paper_metadata", {})

        indexed_set = set(checkpoint.get("indexed_papers", []))
        failed_papers = checkpoint.get("failed_papers", {})

        # Start with all paper IDs
        all_paper_ids = list(paper_metadata.keys())

        # Apply text-based filtering if search query provided
        if search:
            search_lower = search.lower()
            filtered_paper_ids = []
            for pid in all_paper_ids:
                meta = paper_metadata[pid]
                # Search in title, filename, and authors
                title = meta.get("title", "").lower()
                filename = meta.get("filename", "").lower()
                authors = " ".join(meta.get("authors", [])).lower()

                if (search_lower in title or
                    search_lower in filename or
                    search_lower in authors):
                    filtered_paper_ids.append(pid)
            all_paper_ids = filtered_paper_ids

        # Apply sorting based on parameters
        def get_sort_key(pid: str):
            meta = paper_metadata[pid]
            if sort_by == "title":
                return meta.get("title", "").lower()
            elif sort_by == "year":
                return meta.get("year", 0) or 0
            elif sort_by == "chunk_count":
                chunk_stats = meta.get("chunk_stats", {})
                return meta.get("chunk_count", sum(chunk_stats.values()))
            elif sort_by == "indexed_at":
                return meta.get("indexed_at", "")
            else:
                # Default to indexed_at
                return meta.get("indexed_at", "")

        all_paper_ids = sorted(
            all_paper_ids,
            key=get_sort_key,
            reverse=(sort_order == "desc")
        )

        total_count = len(all_paper_ids)

        # Apply pagination
        actual_limit = limit if limit is not None else total_count
        paginated_ids = all_paper_ids[offset:offset + actual_limit] if limit else all_paper_ids[offset:]

        papers = []
        for paper_id in paginated_ids:
            meta = paper_metadata[paper_id]
            chunk_stats = meta.get("chunk_stats", {})
            chunk_count = meta.get("chunk_count", sum(chunk_stats.values()))

            # Determine status
            if paper_id in failed_papers:
                status = "error"
                error_msg = failed_papers[paper_id]
            elif chunk_count > 0 or paper_id in indexed_set:
                status = "indexed"
                error_msg = None
            else:
                status = "pending"
                error_msg = None

            indexed_at_str = meta.get("indexed_at")
            indexed_at = datetime.fromisoformat(indexed_at_str) if indexed_at_str else None

            # Get actual file size from disk
            file_size_bytes = 0
            pdf_path = self.get_pdf_path(paper_id)
            if pdf_path and pdf_path.exists():
                file_size_bytes = pdf_path.stat().st_size

            papers.append(PaperInfo(
                paper_id=paper_id,
                title=meta.get("title", "Unknown"),
                authors=meta.get("authors", []),
                year=meta.get("year"),
                doi=meta.get("doi"),
                filename=meta.get("filename", ""),
                page_count=meta.get("page_count", 0),
                chunk_count=chunk_count,
                chunk_stats=chunk_stats,
                indexed_at=indexed_at,
                status=status,
                error_message=error_msg,
                file_size_bytes=file_size_bytes,
            ))

        has_more = (offset + len(papers)) < total_count
        return PaginatedPapers(
            papers=papers,
            total=total_count,
            offset=offset,
            limit=actual_limit,
            has_more=has_more,
        )

    def search_papers(
        self, query: str, limit: int = 25, offset: int = 0
    ) -> tuple[List[Dict[str, Any]], int]:
        """Semantic search for papers using the query.

        Searches using FULL chunk embeddings (paper-level) for best results,
        then finds the best matching chunk within each paper for preview.

        Args:
            query: Natural language search query
            limit: Maximum number of results to return per page
            offset: Number of results to skip (for pagination)

        Returns:
            Tuple of (paginated results list, total count of all matching papers)
        """
        if not self.embedder:
            raise ValueError("Embedder not configured for search")

        # Embed the query
        query_embedding = self.embedder.embed_query(query)

        # Fetch all matching papers (up to a reasonable max) for accurate total count
        max_results = 1000

        # Search for FULL chunks (paper-level embeddings) for best ranking
        results = self.store.search(
            query_embedding=query_embedding,
            limit=max_results,
            chunk_types=["full"],
        )

        # If no FULL chunks, fall back to searching abstracts
        if not results:
            results = self.store.search(
                query_embedding=query_embedding,
                limit=max_results,
                chunk_types=["abstract"],
            )

        # Deduplicate by paper_id and keep highest score
        seen_papers = {}
        for result in results:
            paper_id = result.get('paper_id')
            if paper_id not in seen_papers or result['score'] > seen_papers[paper_id]['score']:
                seen_papers[paper_id] = result

        # Sort by relevance score descending to get consistent ordering
        sorted_papers = sorted(
            seen_papers.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )

        total_count = len(sorted_papers)

        # Apply pagination
        paginated_papers = sorted_papers[offset:offset + limit]
        paginated_paper_ids = [p[0] for p in paginated_papers]

        # Search for best matching chunks within paginated papers for preview
        preview_chunks = {}
        if paginated_paper_ids:
            chunk_results = self.store.search(
                query_embedding=query_embedding,
                limit=len(paginated_paper_ids) * 3,
                chunk_types=["fine", "section", "abstract"],
                paper_ids=paginated_paper_ids,
            )
            # Keep best chunk per paper for preview
            for chunk in chunk_results:
                paper_id = chunk.get('paper_id')
                if paper_id and (paper_id not in preview_chunks or chunk['score'] > preview_chunks[paper_id]['score']):
                    preview_chunks[paper_id] = chunk

        # Build response with paper info and preview
        search_results = []
        for paper_id, result in paginated_papers:
            chunk_stats = self.store.get_paper_chunk_stats(paper_id)
            chunk_count = sum(chunk_stats.values())

            # Get preview from best matching chunk
            preview = preview_chunks.get(paper_id, {})
            chunk_text = preview.get('text', '')
            preview_text = chunk_text[:300] + '...' if len(chunk_text) > 300 else chunk_text

            # Get actual file size from disk
            file_size_bytes = 0
            pdf_path = self.get_pdf_path(paper_id)
            if pdf_path and pdf_path.exists():
                file_size_bytes = pdf_path.stat().st_size

            search_results.append({
                'paper_id': paper_id,
                'title': result.get('title', 'Unknown'),
                'authors': result.get('authors', []),
                'year': result.get('year'),
                'filename': result.get('file_name', ''),
                'relevance_score': result['score'],
                'chunk_count': chunk_count,
                'status': 'indexed' if chunk_count > 0 else 'pending',
                'file_size_bytes': file_size_bytes,
                # Preview info
                'preview_text': preview_text,
                'preview_section': preview.get('section_name'),
                'preview_subsection': preview.get('subsection_name'),
                'preview_chunk_type': preview.get('chunk_type'),
            })

        return search_results, total_count

    def get_paper(self, paper_id: str) -> Optional[PaperInfo]:
        """Get detailed information about a specific paper."""
        # Get chunks for this paper
        chunks = self.store.get_chunks_by_paper(paper_id)
        if not chunks:
            return None

        # Get first chunk for metadata
        first_chunk = chunks[0]

        # Get chunk statistics
        chunk_stats = self.store.get_paper_chunk_stats(paper_id)
        chunk_count = sum(chunk_stats.values())

        # Get page count
        page_count = 0
        for chunk in chunks:
            page_numbers = chunk.get('page_numbers', [])
            if page_numbers:
                page_count = max(page_count, max(page_numbers))

        # Check status from checkpoint
        checkpoint = self._load_checkpoint()
        indexed_set = set(checkpoint.get("indexed_papers", []))
        failed_papers = checkpoint.get("failed_papers", {})

        if paper_id in failed_papers:
            status = "error"
            error_msg = failed_papers[paper_id]
        elif chunk_count > 0:
            # Chunks exist in Qdrant = successfully indexed
            status = "indexed"
            error_msg = None
        elif paper_id in indexed_set:
            status = "indexed"
            error_msg = None
        else:
            status = "pending"
            error_msg = None

        # Get actual file size from disk
        file_size_bytes = 0
        pdf_path = self.get_pdf_path(paper_id)
        if pdf_path and pdf_path.exists():
            file_size_bytes = pdf_path.stat().st_size

        return PaperInfo(
            paper_id=paper_id,
            title=first_chunk.get('title', 'Unknown'),
            authors=first_chunk.get('authors', []),
            year=first_chunk.get('year'),
            doi=first_chunk.get('doi'),
            filename=first_chunk.get('file_name', ''),
            page_count=page_count,
            chunk_count=chunk_count,
            chunk_stats=chunk_stats,
            indexed_at=None,
            status=status,
            error_message=error_msg,
            file_size_bytes=file_size_bytes,
        )

    def get_pdf_path(self, paper_id: str) -> Optional[Path]:
        """Get the file path for a paper's PDF."""
        return self._get_pdf_path(paper_id)

    def update_paper_metadata(
        self,
        paper_id: str,
        title: Optional[str] = None,
        authors: Optional[List[str]] = None,
        year: Optional[int] = None,
        filename: Optional[str] = None,
    ) -> Optional[PaperInfo]:
        """Update paper metadata.

        Updates:
        1. Checkpoint cache
        2. All chunks in Qdrant

        Args:
            paper_id: Paper ID
            title: New title (if provided)
            authors: New authors list (if provided)
            year: New year (if provided)
            filename: New filename (if provided)

        Returns:
            Updated PaperInfo object or None if not found
        """
        checkpoint = self._load_checkpoint()
        paper_metadata = checkpoint.get("paper_metadata", {})

        if paper_id not in paper_metadata:
            return None

        # Update checkpoint metadata
        meta = paper_metadata[paper_id]
        if title is not None:
            meta["title"] = title
        if authors is not None:
            meta["authors"] = authors
        if year is not None:
            meta["year"] = year
        if filename is not None:
            meta["filename"] = filename

        # Save checkpoint
        checkpoint["paper_metadata"] = paper_metadata
        self._save_checkpoint(checkpoint)

        # Update all chunks in Qdrant
        try:
            update_fields = {}
            if title is not None:
                update_fields["title"] = title
            if authors is not None:
                update_fields["authors"] = authors
            if year is not None:
                update_fields["year"] = year
            if filename is not None:
                update_fields["file_name"] = filename

            if update_fields:
                self.store.update_paper_chunks_metadata(paper_id, update_fields)
                logger.info(f"Updated metadata for paper {paper_id}: {update_fields}")

        except Exception as e:
            logger.error(f"Failed to update chunks in Qdrant: {e}")
            # Continue anyway - checkpoint is updated

        # Return updated paper
        indexed_set = set(checkpoint.get("indexed_papers", []))
        failed_papers = checkpoint.get("failed_papers", {})
        chunk_stats = meta.get("chunk_stats", {})
        chunk_count = meta.get("chunk_count", sum(chunk_stats.values()))

        if paper_id in failed_papers:
            status = "error"
            error_msg = failed_papers[paper_id]
        elif chunk_count > 0 or paper_id in indexed_set:
            status = "indexed"
            error_msg = None
        else:
            status = "pending"
            error_msg = None

        return PaperInfo(
            paper_id=paper_id,
            title=meta.get("title", "Unknown"),
            authors=meta.get("authors", []),
            year=meta.get("year"),
            doi=meta.get("doi"),
            filename=meta.get("filename", ""),
            page_count=meta.get("page_count", 0),
            chunk_count=chunk_count,
            chunk_stats=chunk_stats,
            indexed_at=meta.get("indexed_at"),
            status=status,
            error_message=error_msg,
        )

    def delete_paper(self, paper_id: str) -> Dict[str, Any]:
        """Delete a paper and all its associated data.

        Deletes:
        1. The PDF file (if found)
        2. All chunks from Qdrant
        3. Entry from checkpoint file

        Returns:
            Dict with deletion results
        """
        result = {
            "paper_id": paper_id,
            "pdf_deleted": False,
            "chunks_deleted": 0,
            "checkpoint_updated": False,
        }

        # Delete PDF file
        pdf_path = self._get_pdf_path(paper_id)
        if pdf_path and pdf_path.exists():
            try:
                pdf_path.unlink()
                result["pdf_deleted"] = True
                logger.info(f"Deleted PDF file: {pdf_path}")
            except Exception as e:
                logger.error(f"Failed to delete PDF {pdf_path}: {e}")

        # Delete chunks from Qdrant
        chunks_deleted = self.store.delete_paper_chunks(paper_id)
        result["chunks_deleted"] = chunks_deleted

        # Update checkpoint
        checkpoint = self._load_checkpoint()
        indexed_papers = set(checkpoint.get("indexed_papers", []))
        failed_papers = checkpoint.get("failed_papers", {})

        if paper_id in indexed_papers:
            indexed_papers.discard(paper_id)
            checkpoint["indexed_papers"] = list(indexed_papers)

        if paper_id in failed_papers:
            del failed_papers[paper_id]
            checkpoint["failed_papers"] = failed_papers

        # Remove from paper_metadata cache
        paper_metadata = checkpoint.get("paper_metadata", {})
        if paper_id in paper_metadata:
            del paper_metadata[paper_id]
            checkpoint["paper_metadata"] = paper_metadata

        # Remove file hash so the same file can be re-uploaded
        file_hashes = checkpoint.get("file_hashes", {})
        hashes_to_remove = [h for h, pid in file_hashes.items() if pid == paper_id]
        for h in hashes_to_remove:
            del file_hashes[h]
        if hashes_to_remove:
            checkpoint["file_hashes"] = file_hashes
            logger.info(f"Removed {len(hashes_to_remove)} file hash(es) for paper {paper_id}")
        else:
            logger.warning(f"No file hashes found for paper {paper_id} during deletion")

        self._save_checkpoint(checkpoint)
        result["checkpoint_updated"] = True

        logger.info(f"Deleted paper {paper_id}: {result}")
        return result

    def save_uploaded_file(self, file_content: bytes, filename: str) -> Path:
        """Save an uploaded PDF file to the upload directory.

        Returns:
            Path to the saved file
        """
        # Sanitize filename
        safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
        if not safe_filename.lower().endswith('.pdf'):
            safe_filename += '.pdf'

        file_path = self.upload_dir / safe_filename

        # Handle duplicates
        counter = 1
        while file_path.exists():
            stem = safe_filename[:-4]  # Remove .pdf
            file_path = self.upload_dir / f"{stem}_{counter}.pdf"
            counter += 1

        with open(file_path, 'wb') as f:
            f.write(file_content)

        logger.info(f"Saved uploaded file: {file_path}")
        return file_path

    def index_paper(
        self,
        pdf_path: Path,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """Index a single PDF file.

        Args:
            pdf_path: Path to the PDF file
            progress_callback: Optional callback for progress updates
                              Called with (step_name, data_dict)

        Returns:
            Dict with indexing results
        """
        if not self.embedder:
            raise ValueError("Embedder not configured for indexing")

        paper_id = self._generate_paper_id(pdf_path.name)
        result = {
            "paper_id": paper_id,
            "filename": pdf_path.name,
            "success": False,
            "chunks": 0,
            "error": None,
        }

        def emit_progress(step: str, data: Dict[str, Any] = None):
            if progress_callback:
                progress_callback(step, data or {})

        try:
            emit_progress("processing", {"message": "Processing PDF...", "sub_progress": 0.0})

            # Reuse processor to avoid re-loading MinerU ML models per upload
            processor = self._get_processor()

            # Process PDF into chunks
            # MinerU uses tqdm progress bars internally (Layout Predict, MFR Predict,
            # OCR-det, etc.). We intercept them to forward real extraction progress
            # to the frontend, keeping the SSE connection alive during long extractions.
            #
            # IMPORTANT: MinerU does `from tqdm import tqdm` at import time, creating
            # local references. Replacing tqdm_module.tqdm with a subclass does NOT
            # affect those references. Instead, we patch __init__, update, and close
            # on the ORIGINAL tqdm class so all existing references pick up the hooks.
            from tqdm import tqdm as _tqdm_cls

            emit_progress("extracting", {"message": "Extracting text from PDF...", "sub_progress": 0.0})

            # Known MinerU tqdm stages and their weight in overall extraction
            _STAGE_WEIGHTS = {
                "Layout Predict": 0.25,
                "MFD Predict": 0.15,
                "MFR Predict": 0.15,
                "OCR-det": 0.10,
                "OCR-det Predict": 0.05,
                "Table-ocr det": 0.05,
                "Table-wired Predict": 0.05,
            }
            _extraction_sub = 0.0  # tracks cumulative sub_progress (0.0 - 0.7)

            # Save original methods
            _orig_init = _tqdm_cls.__init__
            _orig_update = _tqdm_cls.update
            _orig_close = _tqdm_cls.close

            def _patched_init(self, *args, **kwargs):
                _orig_init(self, *args, **kwargs)
                # Attach tracking attributes after normal init
                desc = getattr(self, 'desc', '') or ''
                self._arc_stage_weight = 0.0
                self._arc_stage_desc = desc
                for prefix, weight in _STAGE_WEIGHTS.items():
                    if desc.startswith(prefix):
                        self._arc_stage_weight = weight
                        break
                self._arc_last_reported = -1

            def _patched_update(self, n=1):
                _orig_update(self, n)
                nonlocal _extraction_sub
                weight = getattr(self, '_arc_stage_weight', 0)
                if weight > 0 and self.total and self.total > 0:
                    pct = int(self.n / self.total * 100)
                    last = getattr(self, '_arc_last_reported', -1)
                    # Emit every 10% to avoid flooding
                    if pct // 10 > last // 10:
                        self._arc_last_reported = pct
                        frac = self.n / self.total
                        current_sub = min(0.7, _extraction_sub + weight * frac)
                        desc = getattr(self, '_arc_stage_desc', '')
                        emit_progress("extracting", {
                            "message": f"{desc}... {pct}%",
                            "sub_progress": current_sub,
                        })

            def _patched_close(self):
                nonlocal _extraction_sub
                weight = getattr(self, '_arc_stage_weight', 0)
                if weight > 0:
                    _extraction_sub = min(0.7, _extraction_sub + weight)
                    desc = getattr(self, '_arc_stage_desc', '')
                    emit_progress("extracting", {
                        "message": f"{desc} complete",
                        "sub_progress": _extraction_sub,
                    })
                _orig_close(self)

            # Patch methods on the original class (affects all existing references)
            _tqdm_cls.__init__ = _patched_init
            _tqdm_cls.update = _patched_update
            _tqdm_cls.close = _patched_close
            try:
                chunks: List[Chunk] = processor.process_pdf(pdf_path, paper_id)
            finally:
                # Restore original methods
                _tqdm_cls.__init__ = _orig_init
                _tqdm_cls.update = _orig_update
                _tqdm_cls.close = _orig_close

            if not chunks:
                result["error"] = "No chunks extracted from PDF"
                return result

            emit_progress("extracting", {"message": "Parsing document structure...", "sub_progress": 0.8})

            # Count chunk types for reporting
            chunk_type_counts: Dict[str, int] = {}
            for c in chunks:
                ct = c.chunk_type.value if hasattr(c.chunk_type, 'value') else str(c.chunk_type)
                chunk_type_counts[ct] = chunk_type_counts.get(ct, 0) + 1

            emit_progress("chunking", {
                "message": f"Creating chunks... ({len(chunks)} chunks, {len(chunk_type_counts)} types)",
                "sub_progress": 0.5,
            })

            emit_progress("embedding", {"message": f"Generating embeddings for {len(chunks)} chunks...", "sub_progress": 0.0})

            # Generate embeddings with per-batch progress
            texts = [chunk.text for chunk in chunks]

            def embedding_progress(batch_idx: int, total_batches: int):
                sub = batch_idx / total_batches if total_batches > 0 else 1.0
                emit_progress("embedding", {
                    "message": f"Generating embeddings (batch {batch_idx}/{total_batches})...",
                    "sub_progress": sub,
                })

            embeddings = self.embedder.embed_documents(texts, progress_callback=embedding_progress)

            # Create full-paper embedding
            pooling_texts = [c.text for c in chunks if c.chunk_type in [ChunkType.ABSTRACT, ChunkType.SECTION]]
            if pooling_texts:
                full_embedding = self.embedder.compute_mean_pooled_embedding(pooling_texts)

                full_chunk = Chunk(
                    chunk_id=f"{paper_id}_full",
                    paper_id=paper_id,
                    chunk_type=ChunkType.FULL,
                    text=f"[Full paper: {chunks[0].title}]",
                    title=chunks[0].title,
                    authors=chunks[0].authors,
                    year=chunks[0].year,
                    project_tag=chunks[0].project_tag,
                    research_area=chunks[0].research_area,
                    file_name=chunks[0].file_name,
                )
                chunks.append(full_chunk)
                embeddings.append(full_embedding)

            emit_progress("indexing", {
                "message": f"Storing {len(chunks)} chunks in vector database...",
                "sub_progress": 0.0,
            })

            # Prepare and upsert to Qdrant
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            payloads = [chunk.to_payload() for chunk in chunks]

            self.store.upsert_chunks(chunk_ids, embeddings, payloads)

            emit_progress("indexing", {
                "message": "Updating metadata...",
                "sub_progress": 0.7,
            })

            # Update checkpoint
            checkpoint = self._load_checkpoint()
            indexed_papers = set(checkpoint.get("indexed_papers", []))
            indexed_papers.add(paper_id)
            checkpoint["indexed_papers"] = list(indexed_papers)

            # Remove from failed if present
            failed_papers = checkpoint.get("failed_papers", {})
            if paper_id in failed_papers:
                del failed_papers[paper_id]
                checkpoint["failed_papers"] = failed_papers

            # Update stats
            stats = checkpoint.get("stats", {})
            stats["total_chunks"] = stats.get("total_chunks", 0) + len(chunks)
            stats["total_papers_attempted"] = stats.get("total_papers_attempted", 0) + 1
            checkpoint["stats"] = stats

            # Store paper metadata for fast retrieval (avoids scanning Qdrant)
            if "paper_metadata" not in checkpoint:
                checkpoint["paper_metadata"] = {}

            # Compute chunk stats
            chunk_stats: Dict[str, int] = {}
            max_page = 0
            for chunk in chunks:
                chunk_type = chunk.chunk_type.value if hasattr(chunk.chunk_type, 'value') else str(chunk.chunk_type)
                chunk_stats[chunk_type] = chunk_stats.get(chunk_type, 0) + 1
                if chunk.page_numbers:
                    max_page = max(max_page, max(chunk.page_numbers))

            checkpoint["paper_metadata"][paper_id] = {
                "title": chunks[0].title if chunks else "Unknown",
                "authors": chunks[0].authors if chunks else [],
                "year": chunks[0].year if chunks else None,
                "doi": chunks[0].doi if chunks else None,
                "filename": chunks[0].file_name if chunks else pdf_path.name,
                "chunk_count": len(chunks),
                "chunk_stats": chunk_stats,
                "page_count": max_page,
                "indexed_at": datetime.now().isoformat(),
            }

            self._save_checkpoint(checkpoint)

            # Compute and store file hash for duplicate detection
            try:
                file_hash = self._compute_file_hash(pdf_path)
                self._store_file_hash(file_hash, paper_id)
                result["file_hash"] = file_hash
            except Exception as hash_err:
                logger.warning(f"Failed to compute file hash for {paper_id}: {hash_err}")

            result["success"] = True
            result["chunks"] = len(chunks)

            emit_progress("complete", {
                "message": f"Successfully indexed {len(chunks)} chunks",
                "chunks": len(chunks),
            })

        except Exception as e:
            error_msg = str(e)
            result["error"] = error_msg
            logger.error(f"Failed to index {pdf_path.name}: {error_msg}")

            # Update checkpoint with failure
            checkpoint = self._load_checkpoint()
            failed_papers = checkpoint.get("failed_papers", {})
            failed_papers[paper_id] = error_msg
            checkpoint["failed_papers"] = failed_papers
            self._save_checkpoint(checkpoint)

            emit_progress("error", {"message": error_msg})

        return result
