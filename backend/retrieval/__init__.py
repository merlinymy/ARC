"""Retrieval module for query processing and vector search."""

from .domain_synonyms import DOMAIN_SYNONYMS, get_synonyms, find_entities_in_text
from .query_expander import QueryExpander, expand_query
from .query_classifier import (
    QueryClassifier,
    QueryType,
    QueryClassification,
    RETRIEVAL_STRATEGIES,
    classify_query_heuristic,
)
from .embedder import VoyageEmbedder
from .reranker import CohereReranker
from .qdrant_store import QdrantStore
from .query_engine import QueryEngine, QueryResult

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
]
