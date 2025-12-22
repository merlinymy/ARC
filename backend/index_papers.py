"""Index PDF papers into Qdrant for retrieval.

NOTE: Set MINERU_VIRTUAL_VRAM_SIZE before importing MinerU to allocate GPU memory.

This script connects all preprocessing and retrieval components:

Pipeline:
    PDF files
        ↓ EnhancedPDFProcessor (MinerU)
    Extracts text, tables, captions
        ↓ PaperChunker
    Creates 6 chunk types (abstract, section, fine, caption, table, full)
        ↓ VoyageEmbedder
    Generates embeddings (+ mean-pooled full-paper embedding)
        ↓ QdrantStore
    Stores vectors with metadata for retrieval

Features:
    - Checkpointing: Saves progress to resume if interrupted
    - Rate limiting: Handles Voyage API limits with retry
    - Progress tracking: Shows real-time progress

Usage:
    python index_papers.py                     # Index all papers (resumes if checkpoint exists)
    python index_papers.py --limit 5           # Index only 5 papers (for testing)
    python index_papers.py --reset             # Clear collection AND checkpoint, start fresh
    python index_papers.py --papers-dir PATH   # Use custom papers directory
"""

import os

# Configure MinerU GPU memory allocation BEFORE importing preprocessing
# This allows MinerU to process pages in larger batches for better performance
os.environ.setdefault("MINERU_VIRTUAL_VRAM_SIZE", "8")  # 8GB for Apple Silicon

import sys
import argparse
import logging
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Set
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from preprocessing import EnhancedPDFProcessor, Chunk, ChunkType
from retrieval.embedder import VoyageEmbedder
from retrieval.qdrant_store import QdrantStore

# Checkpoint file for resuming interrupted indexing
CHECKPOINT_FILE = Path("data/indexing_checkpoint.json")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("mineru").setLevel(logging.WARNING)


def generate_paper_id(pdf_path: Path) -> str:
    """Generate a unique paper ID from the file path."""
    return hashlib.md5(pdf_path.name.encode()).hexdigest()[:12]


def chunk_to_payload(chunk: Chunk) -> Dict[str, Any]:
    """Convert Chunk to Qdrant payload using the built-in method."""
    return chunk.to_payload()


class IndexingCheckpoint:
    """Manages checkpointing for resumable indexing."""

    def __init__(self, checkpoint_path: Path = CHECKPOINT_FILE):
        self.checkpoint_path = checkpoint_path
        self.indexed_papers: Set[str] = set()
        self.failed_papers: Dict[str, str] = {}  # paper_id -> error message
        self.stats: Dict[str, Any] = {
            "started_at": None,
            "last_updated": None,
            "total_chunks": 0,
            "total_papers_attempted": 0,
        }
        self._load()

    def _load(self):
        """Load checkpoint from disk if it exists."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path) as f:
                    data = json.load(f)
                    self.indexed_papers = set(data.get("indexed_papers", []))
                    self.failed_papers = data.get("failed_papers", {})
                    self.stats = data.get("stats", self.stats)
                logger.info(f"Loaded checkpoint: {len(self.indexed_papers)} papers already indexed")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

    def save(self):
        """Save checkpoint to disk."""
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.stats["last_updated"] = datetime.now().isoformat()

        data = {
            "indexed_papers": list(self.indexed_papers),
            "failed_papers": self.failed_papers,
            "stats": self.stats,
        }

        with open(self.checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)

    def mark_indexed(self, paper_id: str, num_chunks: int):
        """Mark a paper as successfully indexed."""
        self.indexed_papers.add(paper_id)
        self.stats["total_chunks"] += num_chunks
        self.stats["total_papers_attempted"] += 1
        # Remove from failed if it was there
        self.failed_papers.pop(paper_id, None)

    def mark_failed(self, paper_id: str, error: str):
        """Mark a paper as failed."""
        self.failed_papers[paper_id] = error
        self.stats["total_papers_attempted"] += 1

    def is_indexed(self, paper_id: str) -> bool:
        """Check if a paper has been indexed."""
        return paper_id in self.indexed_papers

    def reset(self):
        """Reset the checkpoint."""
        self.indexed_papers = set()
        self.failed_papers = {}
        self.stats = {
            "started_at": datetime.now().isoformat(),
            "last_updated": None,
            "total_chunks": 0,
            "total_papers_attempted": 0,
        }
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        logger.info("Checkpoint reset")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of indexing progress."""
        return {
            "indexed": len(self.indexed_papers),
            "failed": len(self.failed_papers),
            "total_chunks": self.stats["total_chunks"],
            "started_at": self.stats.get("started_at"),
            "last_updated": self.stats.get("last_updated"),
        }


def index_single_paper(
    pdf_path: Path,
    processor: EnhancedPDFProcessor,
    embedder: VoyageEmbedder,
    store: QdrantStore,
    checkpoint: IndexingCheckpoint,
) -> Dict[str, Any]:
    """Process and index a single PDF.

    Args:
        pdf_path: Path to PDF file
        processor: PDF processor instance
        embedder: Embedding client
        store: Qdrant store
        checkpoint: Checkpoint manager

    Returns:
        Dict with indexing stats
    """
    paper_id = generate_paper_id(pdf_path)
    stats = {"paper_id": paper_id, "file": pdf_path.name, "chunks": 0, "success": False}

    # Skip if already indexed
    if checkpoint.is_indexed(paper_id):
        stats["success"] = True
        stats["skipped"] = True
        return stats

    try:
        # Step 1: Process PDF into chunks (uses MinerU + PaperChunker)
        chunks: List[Chunk] = processor.process_pdf(pdf_path, paper_id)

        if not chunks:
            logger.warning(f"No chunks extracted from {pdf_path.name}")
            checkpoint.mark_failed(paper_id, "No chunks extracted")
            return stats

        # Step 2: Generate embeddings for all chunks
        texts = [chunk.text for chunk in chunks]
        embeddings = embedder.embed_documents(texts)

        # Step 3: Create full-paper embedding via mean pooling
        # Use section and abstract chunks for the full embedding (skip fine to avoid duplication)
        pooling_texts = [c.text for c in chunks if c.chunk_type in [ChunkType.ABSTRACT, ChunkType.SECTION]]
        if pooling_texts:
            full_embedding = embedder.compute_mean_pooled_embedding(pooling_texts)

            # Create a FULL chunk for paper-level retrieval
            full_chunk = Chunk(
                chunk_id=f"{paper_id}_full",
                paper_id=paper_id,
                chunk_type=ChunkType.FULL,
                text=f"[Full paper: {chunks[0].title}]",  # Placeholder text
                title=chunks[0].title,
                authors=chunks[0].authors,
                year=chunks[0].year,
                project_tag=chunks[0].project_tag,
                research_area=chunks[0].research_area,
            )
            chunks.append(full_chunk)
            embeddings.append(full_embedding)

        # Step 4: Prepare data for Qdrant
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        payloads = [chunk_to_payload(chunk) for chunk in chunks]

        # Step 5: Upsert to Qdrant
        store.upsert_chunks(chunk_ids, embeddings, payloads)

        stats["chunks"] = len(chunks)
        stats["success"] = True

        # Count by type
        type_counts = {}
        for chunk in chunks:
            t = chunk.chunk_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        stats["by_type"] = type_counts

        # Mark as indexed and save checkpoint
        checkpoint.mark_indexed(paper_id, len(chunks))
        checkpoint.save()

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to index {pdf_path.name}: {error_msg}")
        stats["error"] = error_msg
        checkpoint.mark_failed(paper_id, error_msg)
        checkpoint.save()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Index papers into Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python index_papers.py                     # Index all papers (resumes from checkpoint)
    python index_papers.py --limit 5           # Test with 5 papers
    python index_papers.py --reset             # Reset collection and checkpoint, start fresh
    python index_papers.py --retry-failed      # Retry only previously failed papers
        """
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of papers to index (for testing)"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Reset collection AND checkpoint, start fresh"
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Retry only previously failed papers"
    )
    parser.add_argument(
        "--papers-dir", type=str, default="data/sample_papers",
        help="Directory containing PDF files"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("RESEARCH PAPER INDEXING PIPELINE")
    print("="*60)

    # Initialize checkpoint
    checkpoint = IndexingCheckpoint()

    # Show checkpoint status
    summary = checkpoint.get_summary()
    if summary["indexed"] > 0 and not args.reset:
        print(f"\n📋 Checkpoint found:")
        print(f"   Already indexed: {summary['indexed']} papers")
        print(f"   Failed: {summary['failed']} papers")
        print(f"   Total chunks: {summary['total_chunks']}")
        print(f"   Last updated: {summary['last_updated']}")
        print("   (Use --reset to start fresh)")

    # Initialize components
    print("\n[1/4] Initializing components...")

    processor = EnhancedPDFProcessor(
        abstract_max_tokens=settings.abstract_max_tokens,
        section_max_tokens=settings.section_max_tokens,
        fine_chunk_tokens=settings.fine_chunk_tokens,
        fine_chunk_overlap=settings.fine_chunk_overlap,
    )
    print("  ✓ PDF Processor (MinerU backend)")

    embedder = VoyageEmbedder(api_key=settings.voyage_api_key)
    print(f"  ✓ Embedder (Voyage {settings.embedding_model})")

    store = QdrantStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection_name,
        embedding_dimension=settings.embedding_dimension,
    )
    print(f"  ✓ Vector Store (Qdrant at {settings.qdrant_host}:{settings.qdrant_port})")

    # Reset collection and checkpoint if requested
    if args.reset:
        print("\n[!] Resetting collection and checkpoint...")
        try:
            store.delete_collection()
            print("  ✓ Collection deleted")
        except Exception:
            print("  ✓ Collection didn't exist")
        checkpoint.reset()
        print("  ✓ Checkpoint reset")

    # Ensure collection exists
    store.ensure_collection()
    print(f"  ✓ Collection '{settings.qdrant_collection_name}' ready")

    # Find PDF files
    print(f"\n[2/4] Finding PDFs in {args.papers_dir}...")
    papers_dir = Path(args.papers_dir)

    if not papers_dir.exists():
        print(f"  ✗ Directory not found: {papers_dir}")
        return

    pdf_files = sorted(papers_dir.glob("*.pdf"))

    # If retry-failed, only process failed papers
    if args.retry_failed:
        failed_ids = set(checkpoint.failed_papers.keys())
        pdf_files = [p for p in pdf_files if generate_paper_id(p) in failed_ids]
        print(f"  ✓ Retrying {len(pdf_files)} previously failed papers")
    elif args.limit:
        pdf_files = pdf_files[:args.limit]

    print(f"  ✓ Found {len(pdf_files)} PDF files to process")

    # Count how many will be skipped
    already_indexed = sum(1 for p in pdf_files if checkpoint.is_indexed(generate_paper_id(p)))
    if already_indexed > 0:
        print(f"  ✓ {already_indexed} already indexed (will skip)")
        print(f"  ✓ {len(pdf_files) - already_indexed} remaining to index")

    if not pdf_files:
        print("  ✗ No PDFs to index")
        return

    # Process and index each PDF
    print(f"\n[3/4] Processing and indexing papers...")
    print("  (This may take a while - MinerU analyzes each page)")
    print("  (Progress is saved - safe to interrupt with Ctrl+C)")
    print()

    all_stats = []
    newly_indexed = 0
    skipped = 0
    failed = []

    try:
        for pdf_path in tqdm(pdf_files, desc="  Indexing"):
            stats = index_single_paper(pdf_path, processor, embedder, store, checkpoint)
            all_stats.append(stats)

            if stats.get("skipped"):
                skipped += 1
            elif stats["success"]:
                newly_indexed += 1
            else:
                failed.append(stats["file"])

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted! Progress saved to checkpoint.")
        print(f"   Run again to resume from where you left off.")

    # Print summary
    print("\n" + "="*60)
    print("[4/4] INDEXING COMPLETE")
    print("="*60)

    final_summary = checkpoint.get_summary()
    print(f"\nThis session:")
    print(f"  Newly indexed: {newly_indexed}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Failed: {len(failed)}")

    print(f"\nOverall progress:")
    print(f"  Total papers indexed: {final_summary['indexed']}")
    print(f"  Total chunks: {final_summary['total_chunks']}")
    print(f"  Failed papers: {final_summary['failed']}")

    # Aggregate chunk type counts from this session
    type_totals = {}
    for stats in all_stats:
        if "by_type" in stats:
            for t, count in stats["by_type"].items():
                type_totals[t] = type_totals.get(t, 0) + count

    if type_totals:
        print("\nChunks by type (this session):")
        for chunk_type in ["abstract", "section", "fine", "caption", "table", "full"]:
            if chunk_type in type_totals:
                print(f"  {chunk_type:10} {type_totals[chunk_type]:5}")

    if failed:
        print(f"\nFailed papers ({len(failed)}):")
        for name in failed[:5]:
            print(f"  - {name}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
        print("\n  Run with --retry-failed to retry these papers")

    # Verify collection
    try:
        collection_stats = store.get_collection_stats()
        print(f"\nQdrant collection stats:")
        print(f"  Total vectors: {collection_stats['total_points']}")
        print(f"  Status:        {collection_stats['status']}")
    except Exception as e:
        print(f"\nWarning: Could not get collection stats: {e}")

    print(f"\nCheckpoint saved to: {CHECKPOINT_FILE}")
    print("✓ Ready to run evaluation: python run_evaluation.py")
    print()


if __name__ == "__main__":
    main()
