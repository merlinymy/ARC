"""Retrieval module for query processing and vector search."""

from .domain_synonyms import DOMAIN_SYNONYMS, get_synonyms, find_entities_in_text
from .query_expander import QueryExpander, expand_query
from .query_classifier import (
    QueryClassifier,
    QueryType,
    QueryClassification,
    MultiQueryClassification,
    RETRIEVAL_STRATEGIES,
    classify_query_heuristic,
)
from .embedder import VoyageEmbedder
from .reranker import CohereReranker
from .qdrant_store import QdrantStore
from .query_engine import QueryEngine, QueryResult

# New modules for advanced RAG
from .cache import LRUCache, RAGCache, CacheEntry
from .hyde import HyDE, HyDEEmbedder
from .bm25 import BM25Vectorizer, SparseVector, HybridSearchMixer, vectorize_for_bm25
from .query_rewriter import QueryRewriter, RewrittenQuery, rewrite_query, correct_scientific_spelling
from .entity_extractor import EntityExtractor, LLMEntityExtractor, ExtractedEntities
from .citation_verifier import CitationVerifier, VerificationResult
from .conversation_memory import ConversationMemory, ConversationContext, Message

__all__ = [
    # Domain synonyms
    "DOMAIN_SYNONYMS",
    "get_synonyms",
    "find_entities_in_text",
    # Query expansion
    "QueryExpander",
    "expand_query",
    # Query classification
    "QueryClassifier",
    "QueryType",
    "QueryClassification",
    "MultiQueryClassification",
    "RETRIEVAL_STRATEGIES",
    "classify_query_heuristic",
    # Embedder
    "VoyageEmbedder",
    # Reranker
    "CohereReranker",
    # Vector store
    "QdrantStore",
    # Query engine
    "QueryEngine",
    "QueryResult",
    # Caching
    "LRUCache",
    "RAGCache",
    "CacheEntry",
    # HyDE
    "HyDE",
    "HyDEEmbedder",
    # BM25 / Hybrid Search
    "BM25Vectorizer",
    "SparseVector",
    "HybridSearchMixer",
    "vectorize_for_bm25",
    # Query Rewriting
    "QueryRewriter",
    "RewrittenQuery",
    "rewrite_query",
    "correct_scientific_spelling",
    # Entity Extraction
    "EntityExtractor",
    "LLMEntityExtractor",
    "ExtractedEntities",
    # Citation Verification
    "CitationVerifier",
    "VerificationResult",
    # Conversation Memory
    "ConversationMemory",
    "ConversationContext",
    "Message",
]
