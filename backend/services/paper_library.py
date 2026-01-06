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


@dataclass
class PaperInfo:
    """Information about a paper in the library."""
    paper_id: str
    title: str
    authors: List[str]
    year: Optional[int]
    filename: str
    page_count: int
    chunk_count: int
    chunk_stats: Dict[str, int]
    indexed_at: Optional[datetime]
    status: str  # 'indexed', 'indexing', 'error', 'pending'
    error_message: Optional[str] = None


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

        # Ensure upload directory exists
        self.upload_dir.mkdir(parents=True, exist_ok=True)

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
        existing_metadata = checkpoint.get("paper_metadata", {})

        # Skip if already populated (unless forced)
        if existing_metadata and not force:
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
        """Load the indexing checkpoint file."""
        if CHECKPOINT_FILE.exists():
            try:
                with open(CHECKPOINT_FILE) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return {"indexed_papers": [], "failed_papers": {}, "stats": {}}

    def _save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save the indexing checkpoint file."""
        CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        checkpoint["stats"]["last_updated"] = datetime.now().isoformat()
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(checkpoint, f, indent=2)

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
        self, offset: int = 0, limit: Optional[int] = None
    ) -> PaginatedPapers:
        """List papers in the library with pagination.

        Uses cached metadata from checkpoint file for fast retrieval.
        Automatically backfills cache from Qdrant if empty.

        Args:
            offset: Number of papers to skip
            limit: Maximum papers to return (None for all - backwards compatible)

        Returns:
            PaginatedPapers with papers list and pagination metadata
        """
        checkpoint = self._load_checkpoint()
        paper_metadata = checkpoint.get("paper_metadata", {})

        # Backfill cache if empty (one-time slow operation)
        if not paper_metadata:
            self.backfill_paper_metadata()
            checkpoint = self._load_checkpoint()
            paper_metadata = checkpoint.get("paper_metadata", {})

        indexed_set = set(checkpoint.get("indexed_papers", []))
        failed_papers = checkpoint.get("failed_papers", {})

        # Get all paper IDs sorted by indexed_at (newest first)
        all_paper_ids = sorted(
            paper_metadata.keys(),
            key=lambda pid: paper_metadata[pid].get("indexed_at", ""),
            reverse=True
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

            papers.append(PaperInfo(
                paper_id=paper_id,
                title=meta.get("title", "Unknown"),
                authors=meta.get("authors", []),
                year=meta.get("year"),
                filename=meta.get("filename", ""),
                page_count=meta.get("page_count", 0),
                chunk_count=chunk_count,
                chunk_stats=chunk_stats,
                indexed_at=indexed_at,
                status=status,
                error_message=error_msg,
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

            search_results.append({
                'paper_id': paper_id,
                'title': result.get('title', 'Unknown'),
                'authors': result.get('authors', []),
                'year': result.get('year'),
                'filename': result.get('file_name', ''),
                'relevance_score': result['score'],
                'chunk_count': chunk_count,
                'status': 'indexed' if chunk_count > 0 else 'pending',
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

        return PaperInfo(
            paper_id=paper_id,
            title=first_chunk.get('title', 'Unknown'),
            authors=first_chunk.get('authors', []),
            year=first_chunk.get('year'),
            filename=first_chunk.get('file_name', ''),
            page_count=page_count,
            chunk_count=chunk_count,
            chunk_stats=chunk_stats,
            indexed_at=None,
            status=status,
            error_message=error_msg,
        )

    def get_pdf_path(self, paper_id: str) -> Optional[Path]:
        """Get the file path for a paper's PDF."""
        return self._get_pdf_path(paper_id)

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
            emit_progress("processing", {"message": "Processing PDF..."})

            # Initialize processor
            processor = EnhancedPDFProcessor(
                abstract_max_tokens=settings.abstract_max_tokens,
                section_max_tokens=settings.section_max_tokens,
                fine_chunk_tokens=settings.fine_chunk_tokens,
                fine_chunk_overlap=settings.fine_chunk_overlap,
                extraction_timeout=settings.pdf_extraction_timeout,
            )

            # Process PDF into chunks
            emit_progress("extracting", {"message": "Extracting content..."})
            chunks: List[Chunk] = processor.process_pdf(pdf_path, paper_id)

            if not chunks:
                result["error"] = "No chunks extracted from PDF"
                return result

            emit_progress("embedding", {"message": f"Generating embeddings for {len(chunks)} chunks..."})

            # Generate embeddings
            texts = [chunk.text for chunk in chunks]
            embeddings = self.embedder.embed_documents(texts)

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

            emit_progress("indexing", {"message": "Storing in vector database..."})

            # Prepare and upsert to Qdrant
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            payloads = [chunk.to_payload() for chunk in chunks]

            self.store.upsert_chunks(chunk_ids, embeddings, payloads)

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
