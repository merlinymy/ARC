"""Dependency injection container for shared instances.

This module provides centralized dependency management to ensure:
1. Single Qdrant client shared across all components
2. Single BM25Vectorizer instance with consistent IDF cache
3. Lazy initialization for testability
4. Clean shutdown of resources
"""

import logging
from typing import Optional
from functools import lru_cache

from qdrant_client import QdrantClient
from anthropic import Anthropic

from config import settings

logger = logging.getLogger(__name__)


class Dependencies:
    """Container for shared application dependencies.

    Use get_dependencies() to access the singleton instance.
    """

    _instance: Optional["Dependencies"] = None

    def __init__(self):
        self._qdrant_client: Optional[QdrantClient] = None
        self._anthropic_client: Optional[Anthropic] = None
        self._bm25_vectorizer: Optional["BM25Vectorizer"] = None
        self._embedder: Optional["VoyageEmbedder"] = None
        self._reranker: Optional["CohereReranker"] = None
        self._store: Optional["QdrantStore"] = None
        self._query_engine: Optional["QueryEngine"] = None
        self._paper_library_service: Optional["PaperLibraryService"] = None
        self._pdf_service: Optional["PDFService"] = None

    @property
    def qdrant_client(self) -> QdrantClient:
        """Get or create the shared Qdrant client."""
        if self._qdrant_client is None:
            self._qdrant_client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
            )
            logger.info(f"Created Qdrant client at {settings.qdrant_host}:{settings.qdrant_port}")
        return self._qdrant_client

    @property
    def anthropic_client(self) -> Anthropic:
        """Get or create the shared Anthropic client."""
        if self._anthropic_client is None:
            self._anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
            logger.info("Created Anthropic client")
        return self._anthropic_client

    @property
    def bm25_vectorizer(self) -> "BM25Vectorizer":
        """Get or create the shared BM25 vectorizer with loaded IDF cache."""
        if self._bm25_vectorizer is None:
            from retrieval.bm25 import BM25Vectorizer
            self._bm25_vectorizer = BM25Vectorizer()
            if self._bm25_vectorizer.load_idf_cache():
                logger.info(f"BM25 vectorizer loaded IDF cache with {len(self._bm25_vectorizer._idf_cache)} terms")
            else:
                logger.info("BM25 vectorizer initialized (no existing IDF cache)")
        return self._bm25_vectorizer

    @property
    def embedder(self) -> "VoyageEmbedder":
        """Get or create the shared Voyage embedder."""
        if self._embedder is None:
            from retrieval.embedder import VoyageEmbedder
            self._embedder = VoyageEmbedder(api_key=settings.voyage_api_key)
            logger.info(f"Created Voyage embedder ({settings.embedding_model})")
        return self._embedder

    @property
    def reranker(self) -> "CohereReranker":
        """Get or create the shared Cohere reranker."""
        if self._reranker is None:
            from retrieval.reranker import CohereReranker
            self._reranker = CohereReranker(api_key=settings.cohere_api_key)
            logger.info(f"Created Cohere reranker ({settings.reranker_model})")
        return self._reranker

    @property
    def store(self) -> "QdrantStore":
        """Get or create the shared QdrantStore."""
        if self._store is None:
            from retrieval.qdrant_store import QdrantStore
            self._store = QdrantStore(
                collection_name=settings.qdrant_collection_name,
                embedding_dimension=settings.embedding_dimension,
                client=self.qdrant_client,
                bm25_vectorizer=self.bm25_vectorizer,
            )
            logger.info(f"Created QdrantStore for collection '{settings.qdrant_collection_name}'")
        return self._store

    @property
    def query_engine(self) -> "QueryEngine":
        """Get or create the shared QueryEngine."""
        if self._query_engine is None:
            from retrieval.query_engine import QueryEngine
            self._query_engine = QueryEngine(
                embedder=self.embedder,
                reranker=self.reranker,
                store=self.store,
                anthropic_client=self.anthropic_client,
                claude_model=settings.claude_model,
                claude_model_fast=settings.claude_model_fast,
                claude_model_classifier=settings.claude_model_classifier,
                enable_classification=settings.enable_query_classification,
                enable_expansion=settings.enable_query_expansion,
                enable_caching=True,
                enable_hyde=True,  # Hypothetical document embeddings
                enable_query_rewriting=True,
                enable_entity_extraction=True,
                enable_citation_verification=True,  # Verify LLM citations
                enable_conversation_memory=True,
                pdf_service=self.pdf_service,  # Pass PDF service for full document support
            )
            logger.info("Created QueryEngine")
        return self._query_engine

    @property
    def paper_library_service(self) -> "PaperLibraryService":
        """Get or create the shared PaperLibraryService."""
        if self._paper_library_service is None:
            from services.paper_library import PaperLibraryService
            self._paper_library_service = PaperLibraryService(
                qdrant_store=self.store,
                upload_dir=settings.upload_dir,
                embedder=self.embedder,
                bm25_vectorizer=self.bm25_vectorizer,
            )
            logger.info("Created PaperLibraryService")
        return self._paper_library_service

    @property
    def pdf_service(self) -> "PDFService":
        """Get or create the shared PDFService."""
        if self._pdf_service is None:
            from services.pdf_service import PDFService
            self._pdf_service = PDFService(
                paper_library_service=self.paper_library_service,
            )
            logger.info("Created PDFService")
        return self._pdf_service

    def close(self) -> None:
        """Clean up resources."""
        if self._qdrant_client is not None:
            self._qdrant_client.close()
            self._qdrant_client = None
            logger.info("Closed Qdrant client")

        # Reset all cached instances
        self._anthropic_client = None
        self._bm25_vectorizer = None
        self._embedder = None
        self._reranker = None
        self._store = None
        self._query_engine = None
        self._paper_library_service = None

    def reset(self) -> None:
        """Reset all dependencies (useful for testing)."""
        self.close()

    @classmethod
    def get_instance(cls) -> "Dependencies":
        """Get the singleton Dependencies instance."""
        if cls._instance is None:
            cls._instance = Dependencies()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        if cls._instance is not None:
            cls._instance.close()
            cls._instance = None


def get_dependencies() -> Dependencies:
    """Get the shared Dependencies container.

    This is the main entry point for accessing shared dependencies.
    """
    return Dependencies.get_instance()


# FastAPI dependency functions
def get_qdrant_client() -> QdrantClient:
    """FastAPI dependency for Qdrant client."""
    return get_dependencies().qdrant_client


def get_anthropic_client() -> Anthropic:
    """FastAPI dependency for Anthropic client."""
    return get_dependencies().anthropic_client


def get_query_engine() -> "QueryEngine":
    """FastAPI dependency for QueryEngine."""
    return get_dependencies().query_engine


def get_store() -> "QdrantStore":
    """FastAPI dependency for QdrantStore."""
    return get_dependencies().store


def get_bm25_vectorizer() -> "BM25Vectorizer":
    """FastAPI dependency for BM25Vectorizer."""
    return get_dependencies().bm25_vectorizer


def get_paper_library_service() -> "PaperLibraryService":
    """FastAPI dependency for PaperLibraryService (singleton)."""
    return get_dependencies().paper_library_service
