"""Main script to process all PDFs and store in Qdrant."""

import sys
from pathlib import Path
import logging
from typing import List
import asyncio
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pdf_processor import PDFProcessor, PDFChunk
from config import settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import voyageai

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFPipeline:
    """Pipeline to process PDFs and store in vector database."""

    def __init__(self):
        self.processor = PDFProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )

        # Initialize Voyage AI client
        self.voyage_client = voyageai.Client(api_key=settings.voyage_api_key)

        # Initialize Qdrant client
        self.qdrant = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )

        self._ensure_collection()

    def _ensure_collection(self):
        """Create Qdrant collection if it doesn't exist."""
        collections = self.qdrant.get_collections().collections
        collection_names = [c.name for c in collections]

        if settings.qdrant_collection_name not in collection_names:
            logger.info(f"Creating collection: {settings.qdrant_collection_name}")
            self.qdrant.create_collection(
                collection_name=settings.qdrant_collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE
                )
            )
        else:
            logger.info(f"Collection {settings.qdrant_collection_name} already exists")

    def get_pdf_files(self, limit: int = None) -> List[Path]:
        """Get list of PDF files to process."""
        pdf_dir = Path(settings.pdf_source_dir)

        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

        pdf_files = list(pdf_dir.glob("*.pdf"))

        if limit:
            pdf_files = pdf_files[:limit]

        logger.info(f"Found {len(pdf_files)} PDF files to process")
        return pdf_files

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Voyage AI."""
        result = self.voyage_client.embed(
            texts=texts,
            model=settings.embedding_model,
            input_type="document"
        )
        return result.embeddings

    def process_and_store_pdf(self, pdf_path: Path, paper_id: str):
        """Process a single PDF and store chunks in Qdrant."""
        try:
            # Process PDF
            chunks = self.processor.process_pdf(pdf_path, paper_id)

            if not chunks:
                logger.warning(f"No chunks extracted from {pdf_path.name}")
                return

            # Prepare texts for embedding
            texts = [chunk.text for chunk in chunks]

            # Generate embeddings in batches
            batch_size = settings.batch_size
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.embed_texts(batch_texts)
                all_embeddings.extend(batch_embeddings)

            # Prepare points for Qdrant
            points = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
                point = PointStruct(
                    id=f"{paper_id}_{chunk.chunk_index}",
                    vector=embedding,
                    payload={
                        "text": chunk.text,
                        "paper_id": chunk.paper_id,
                        "paper_title": chunk.paper_title,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "metadata": chunk.metadata,
                        "file_name": pdf_path.name
                    }
                )
                points.append(point)

            # Upload to Qdrant
            self.qdrant.upsert(
                collection_name=settings.qdrant_collection_name,
                points=points
            )

            logger.info(f"Successfully processed and stored {len(points)} chunks from {pdf_path.name}")

        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {str(e)}")
            raise

    def process_all(self, limit: int = None):
        """Process all PDFs in the source directory."""
        pdf_files = self.get_pdf_files(limit=limit)

        logger.info(f"Starting processing of {len(pdf_files)} PDFs")

        for idx, pdf_path in enumerate(tqdm(pdf_files, desc="Processing PDFs")):
            paper_id = f"paper_{idx:06d}"

            try:
                self.process_and_store_pdf(pdf_path, paper_id)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                continue

        logger.info("Processing complete!")

        # Show collection stats
        collection_info = self.qdrant.get_collection(settings.qdrant_collection_name)
        logger.info(f"Total vectors in collection: {collection_info.points_count}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Process PDFs and store in Qdrant")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of PDFs to process (useful for testing)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only 5 PDFs"
    )

    args = parser.parse_args()

    limit = 5 if args.test else args.limit

    pipeline = PDFPipeline()
    pipeline.process_all(limit=limit)


if __name__ == "__main__":
    main()
