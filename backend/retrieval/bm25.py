"""BM25 sparse vector generation for hybrid search.

Provides BM25-based sparse vector representations for text,
enabling hybrid search combining dense semantic embeddings
with sparse lexical matching.
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default path for IDF cache persistence
DEFAULT_IDF_CACHE_PATH = Path("data/bm25_idf_cache.json")

# Scientific stopwords (common terms that add noise)
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "this", "that", "these", "those", "it", "its", "their", "they",
    "we", "our", "you", "your", "he", "she", "him", "her", "his",
    "which", "who", "whom", "what", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "not", "only", "same", "so", "than", "too",
    "very", "just", "also", "into", "through", "during", "before",
    "after", "above", "below", "between", "under", "over",
    # Common scientific but low-information words
    "study", "studies", "studied", "method", "methods", "result",
    "results", "used", "using", "use", "show", "shown", "showed",
    "found", "find", "finding", "analysis", "analyzed", "data",
}


@dataclass
class SparseVector:
    """Sparse vector representation with indices and values."""
    indices: List[int]
    values: List[float]

    def to_dict(self) -> Dict[str, List]:
        """Convert to Qdrant sparse vector format."""
        return {
            "indices": self.indices,
            "values": self.values,
        }


class BM25Vectorizer:
    """Generate BM25-based sparse vectors for hybrid search.

    Uses a vocabulary-based approach where each unique term
    gets assigned a consistent index (via hashing).
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        avg_doc_length: float = 500,
        min_term_freq: int = 1,
        max_vocab_size: int = 50000,
    ):
        """Initialize BM25 vectorizer.

        Args:
            k1: BM25 term frequency saturation parameter
            b: BM25 document length normalization parameter
            avg_doc_length: Average document length for normalization
            min_term_freq: Minimum term frequency to include
            max_vocab_size: Maximum vocabulary size (hash space)
        """
        self.k1 = k1
        self.b = b
        self.avg_doc_length = avg_doc_length
        self.min_term_freq = min_term_freq
        self.max_vocab_size = max_vocab_size

        # IDF approximations for common terms (can be updated with corpus stats)
        self._idf_cache: Dict[str, float] = {}
        self._doc_count = 0
        # Store document frequencies for accurate incremental updates
        self._doc_freq: Dict[str, int] = {}

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Lowercase and extract alphanumeric tokens
        text_lower = text.lower()

        # Keep scientific notation together (e.g., "1.5M", "IC50")
        tokens = re.findall(r'[a-z]+\d*|\d+\.?\d*[a-z]*', text_lower)

        # Filter stopwords and very short tokens
        tokens = [
            t for t in tokens
            if t not in STOPWORDS and len(t) >= 2
        ]

        return tokens

    def _term_to_index(self, term: str) -> int:
        """Convert term to consistent index via hashing."""
        return hash(term) % self.max_vocab_size

    def _get_idf(self, term: str, default_idf: float = 3.0) -> float:
        """Get IDF for a term.

        Uses cached IDF if available, otherwise returns default.
        For queries, higher default IDF emphasizes rare terms.
        """
        return self._idf_cache.get(term, default_idf)

    def update_idf(self, documents: List[str]) -> None:
        """Update IDF values from a corpus.

        Args:
            documents: List of document texts
        """
        self._doc_freq = Counter()
        self._doc_count = len(documents)

        for doc in documents:
            tokens = set(self.tokenize(doc))
            for token in tokens:
                self._doc_freq[token] += 1

        # Calculate IDF from document frequencies
        self._recalculate_idf()

        logger.info(f"Updated IDF cache with {len(self._idf_cache)} terms from {self._doc_count} documents")

    def _recalculate_idf(self) -> None:
        """Recalculate IDF values from stored document frequencies."""
        self._idf_cache.clear()
        for term, df in self._doc_freq.items():
            # BM25 IDF formula
            idf = math.log((self._doc_count - df + 0.5) / (df + 0.5) + 1)
            self._idf_cache[term] = max(idf, 0)  # Ensure non-negative

    def update_idf_incremental(self, new_documents: List[str]) -> None:
        """Incrementally update IDF values with new documents.

        Uses stored document frequencies for accurate updates.

        Args:
            new_documents: List of new document texts to add
        """
        if not new_documents:
            return

        # Track document frequencies for new documents
        new_doc_count = len(new_documents)
        self._doc_count += new_doc_count

        doc_freq_delta: Dict[str, int] = Counter()
        for doc in new_documents:
            tokens = set(self.tokenize(doc))
            for token in tokens:
                doc_freq_delta[token] += 1

        # Update stored document frequencies with new counts
        for term, new_df in doc_freq_delta.items():
            self._doc_freq[term] = self._doc_freq.get(term, 0) + new_df

        # Recalculate IDF values from accurate document frequencies
        self._recalculate_idf()

        logger.info(f"Incrementally updated IDF cache: +{new_doc_count} docs, {len(self._idf_cache)} terms total")

    def save_idf_cache(self, path: Optional[Path] = None) -> None:
        """Save IDF cache and document frequencies to disk for persistence.

        Args:
            path: Path to save cache (defaults to data/bm25_idf_cache.json)
        """
        save_path = path or DEFAULT_IDF_CACHE_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)

        cache_data = {
            "doc_count": self._doc_count,
            "idf_cache": self._idf_cache,
            "doc_freq": self._doc_freq,  # Store document frequencies for accurate incremental updates
            "avg_doc_length": self.avg_doc_length,
        }

        with open(save_path, "w") as f:
            json.dump(cache_data, f)

        logger.info(f"Saved IDF cache to {save_path}: {len(self._idf_cache)} terms, {self._doc_count} docs")

    def load_idf_cache(self, path: Optional[Path] = None) -> bool:
        """Load IDF cache and document frequencies from disk.

        Args:
            path: Path to load cache from (defaults to data/bm25_idf_cache.json)

        Returns:
            True if cache was loaded, False otherwise
        """
        load_path = path or DEFAULT_IDF_CACHE_PATH

        if not load_path.exists():
            logger.info(f"No IDF cache found at {load_path}")
            return False

        try:
            with open(load_path) as f:
                cache_data = json.load(f)

            self._doc_count = cache_data.get("doc_count", 0)
            self._idf_cache = cache_data.get("idf_cache", {})
            self._doc_freq = cache_data.get("doc_freq", {})
            self.avg_doc_length = cache_data.get("avg_doc_length", self.avg_doc_length)

            # If doc_freq is missing (old cache format), we can still use idf_cache
            # but incremental updates will be less accurate until next full rebuild
            if not self._doc_freq and self._idf_cache:
                logger.warning("Legacy cache format: doc_freq missing, incremental updates may be less accurate")

            logger.info(f"Loaded IDF cache from {load_path}: {len(self._idf_cache)} terms, {self._doc_count} docs")
            return True

        except Exception as e:
            logger.warning(f"Failed to load IDF cache: {e}")
            return False

    def vectorize(
        self,
        text: str,
        is_query: bool = False,
    ) -> SparseVector:
        """Generate sparse BM25 vector for text.

        Args:
            text: Input text
            is_query: If True, use query-specific scoring

        Returns:
            SparseVector with term indices and BM25 scores
        """
        tokens = self.tokenize(text)
        if not tokens:
            return SparseVector(indices=[], values=[])

        # Count term frequencies
        term_freq = Counter(tokens)
        doc_length = len(tokens)

        # Calculate BM25 scores
        scores: Dict[int, float] = {}

        for term, tf in term_freq.items():
            if tf < self.min_term_freq:
                continue

            idx = self._term_to_index(term)
            idf = self._get_idf(term)

            if is_query:
                # For queries, use simple TF*IDF with boost for emphasis
                score = tf * idf * 2.0
            else:
                # Standard BM25 formula for documents
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * (doc_length / self.avg_doc_length)
                )
                score = idf * (numerator / denominator)

            if idx in scores:
                scores[idx] = max(scores[idx], score)  # Keep highest
            else:
                scores[idx] = score

        # Sort by index for consistent ordering
        sorted_items = sorted(scores.items())
        indices = [idx for idx, _ in sorted_items]
        values = [val for _, val in sorted_items]

        return SparseVector(indices=indices, values=values)

    def vectorize_batch(
        self,
        texts: List[str],
        is_query: bool = False,
    ) -> List[SparseVector]:
        """Vectorize multiple texts.

        Args:
            texts: List of input texts
            is_query: If True, use query-specific scoring

        Returns:
            List of SparseVectors
        """
        return [self.vectorize(text, is_query) for text in texts]


class HybridSearchMixer:
    """Combine dense and sparse search results."""

    def __init__(
        self,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ):
        """Initialize hybrid mixer.

        Args:
            dense_weight: Weight for dense (semantic) scores
            sparse_weight: Weight for sparse (BM25) scores
        """
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def normalize_scores(
        self,
        results: List[Dict],
        score_key: str = "score",
    ) -> List[Dict]:
        """Normalize scores to [0, 1] range.

        Args:
            results: List of result dicts with scores
            score_key: Key for score field

        Returns:
            Results with normalized scores
        """
        if not results:
            return results

        scores = [r[score_key] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        if score_range == 0:
            return results

        for result in results:
            result[f"{score_key}_normalized"] = (
                (result[score_key] - min_score) / score_range
            )

        return results

    def merge_results(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        id_key: str = "_chunk_id",
        top_k: int = 50,
    ) -> List[Dict]:
        """Merge and re-rank dense and sparse results.

        Args:
            dense_results: Results from dense search
            sparse_results: Results from sparse search
            id_key: Key to identify unique documents
            top_k: Number of results to return

        Returns:
            Merged results sorted by combined score
        """
        # Normalize scores
        dense_results = self.normalize_scores(dense_results, "score")
        sparse_results = self.normalize_scores(sparse_results, "score")

        # Create lookup by ID
        results_by_id: Dict[str, Dict] = {}

        # Add dense results
        for result in dense_results:
            doc_id = result.get(id_key, id(result))
            results_by_id[doc_id] = {
                **result,
                "dense_score": result.get("score_normalized", result["score"]),
                "sparse_score": 0.0,
            }

        # Add/merge sparse results
        for result in sparse_results:
            doc_id = result.get(id_key, id(result))
            if doc_id in results_by_id:
                results_by_id[doc_id]["sparse_score"] = result.get(
                    "score_normalized", result["score"]
                )
            else:
                results_by_id[doc_id] = {
                    **result,
                    "dense_score": 0.0,
                    "sparse_score": result.get("score_normalized", result["score"]),
                }

        # Calculate combined scores
        merged = []
        for doc_id, result in results_by_id.items():
            combined_score = (
                self.dense_weight * result["dense_score"]
                + self.sparse_weight * result["sparse_score"]
            )
            result["hybrid_score"] = combined_score
            result["score"] = combined_score  # Override for compatibility
            merged.append(result)

        # Sort by combined score
        merged.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return merged[:top_k]


# Convenience instances
_default_vectorizer: Optional[BM25Vectorizer] = None


def get_bm25_vectorizer() -> BM25Vectorizer:
    """Get or create default BM25 vectorizer."""
    global _default_vectorizer
    if _default_vectorizer is None:
        _default_vectorizer = BM25Vectorizer()
    return _default_vectorizer


def vectorize_for_bm25(text: str, is_query: bool = False) -> SparseVector:
    """Convenience function to vectorize text for BM25.

    Args:
        text: Input text
        is_query: Whether this is a query

    Returns:
        SparseVector
    """
    vectorizer = get_bm25_vectorizer()
    return vectorizer.vectorize(text, is_query)
