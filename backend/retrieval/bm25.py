"""BM25 sparse vector generation for hybrid search.

Provides BM25-based sparse vector representations for text,
enabling hybrid search combining dense semantic embeddings
with sparse lexical matching.
"""

import hashlib
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

# Scheme version for stored sparse vectors.
#   1 = legacy: IDF baked into the *document* vector, term->index via builtin hash()
#       (hash() is salted per process, so stored indices were not reproducible).
#   2 = current: document vector holds only the TF/length-normalized BM25 component,
#       IDF is applied on the *query* side, term->index via blake2b (process-stable).
# Bump this whenever a change would invalidate already-stored document vectors.
SPARSE_SCHEME_VERSION = 2

# Hash space for term -> sparse index. The previous value (50,000) was far smaller than
# the corpus vocabulary (247,831 distinct tokens over 212,953 chunks), so roughly every
# bucket held several unrelated terms and the "lexical" arm matched on collisions.
# At 2^22 about 6% of terms share a bucket with any other; the cost is a larger
# dimension id space in Qdrant's in-memory sparse index, which is cheap.
DEFAULT_MAX_VOCAB_SIZE = 1 << 22  # 4,194,304

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
        max_vocab_size: int = DEFAULT_MAX_VOCAB_SIZE,
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
        # Running total of tokens seen, used to monitor avg_doc_length drift.
        # avg_doc_length is deliberately NOT recomputed on incremental updates: it is
        # baked into every stored document vector, so changing it would silently
        # invalidate the index. See _check_avg_doc_length_drift().
        self._total_doc_length = 0
        self._scheme_version = SPARSE_SCHEME_VERSION

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
        """Convert term to a consistent index via a process-stable hash.

        Must NOT use the builtin hash(): Python salts string hashing per process
        (PYTHONHASHSEED), so the same term mapped to a different sparse dimension in
        every process. Document vectors written by the indexer were therefore
        unreachable by query vectors built in the API process. blake2b is stable
        across processes, machines and Python versions.
        """
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.max_vocab_size

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
        self._total_doc_length = 0

        for doc in documents:
            tokens = self.tokenize(doc)
            self._total_doc_length += len(tokens)
            for token in set(tokens):
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
            tokens = self.tokenize(doc)
            self._total_doc_length += len(tokens)
            for token in set(tokens):
                doc_freq_delta[token] += 1

        # Update stored document frequencies with new counts
        for term, new_df in doc_freq_delta.items():
            self._doc_freq[term] = self._doc_freq.get(term, 0) + new_df

        # Recalculate IDF values from accurate document frequencies
        self._recalculate_idf()

        logger.info(f"Incrementally updated IDF cache: +{new_doc_count} docs, {len(self._idf_cache)} terms total")
        self._check_avg_doc_length_drift()

    def _check_avg_doc_length_drift(self, tolerance: float = 0.2) -> Optional[float]:
        """Warn if the corpus average document length has drifted from the frozen value.

        avg_doc_length is part of the *document* side of the BM25 formula and is baked
        into stored vectors, so it is held fixed rather than recomputed per upload.
        Small drift is harmless; large drift means the index should be re-vectorized.

        Returns:
            The observed average document length, or None if unknown.
        """
        if not self._doc_count or not self._total_doc_length:
            return None
        observed = self._total_doc_length / self._doc_count
        if self.avg_doc_length > 0:
            drift = abs(observed - self.avg_doc_length) / self.avg_doc_length
            if drift > tolerance:
                logger.warning(
                    "BM25 avg_doc_length drift %.0f%%: stored vectors were built with "
                    "avg_doc_length=%.1f but the corpus now averages %.1f tokens. "
                    "Consider re-vectorizing the sparse index.",
                    drift * 100, self.avg_doc_length, observed,
                )
        return observed

    def save_idf_cache(self, path: Optional[Path] = None) -> None:
        """Save IDF cache and document frequencies to disk for persistence.

        Args:
            path: Path to save cache (defaults to data/bm25_idf_cache.json)
        """
        save_path = path or DEFAULT_IDF_CACHE_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)

        cache_data = {
            "scheme_version": self._scheme_version,
            "doc_count": self._doc_count,
            "idf_cache": self._idf_cache,
            "doc_freq": self._doc_freq,  # Store document frequencies for accurate incremental updates
            "avg_doc_length": self.avg_doc_length,
            "total_doc_length": self._total_doc_length,
            # Persisted so that every process maps terms into the same sparse
            # dimension space the stored document vectors were built in.
            "max_vocab_size": self.max_vocab_size,
            "k1": self.k1,
            "b": self.b,
        }

        # Write atomically: a truncated cache file silently degrades every query.
        tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(cache_data, f)
        tmp_path.replace(save_path)

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
            self._total_doc_length = cache_data.get("total_doc_length", 0)
            self.k1 = cache_data.get("k1", self.k1)
            self.b = cache_data.get("b", self.b)

            # The hash space must match the one the stored document vectors were built
            # with, otherwise query terms land on dimensions no document occupies.
            self._scheme_version = cache_data.get("scheme_version", 1)
            if "max_vocab_size" in cache_data:
                self.max_vocab_size = cache_data["max_vocab_size"]

            if self._scheme_version != SPARSE_SCHEME_VERSION:
                logger.warning(
                    "BM25 IDF cache at %s was written under sparse scheme v%s but this "
                    "code expects v%s. Stored document vectors are incompatible with "
                    "query vectors until the sparse index is rebuilt "
                    "(scripts/rebuild_bm25_index.py).",
                    load_path, self._scheme_version, SPARSE_SCHEME_VERSION,
                )

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

        The BM25 score is split across the two sides of the dot product:

            document dimension t:  tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl/avgdl))
            query dimension t:     tf_q * idf(t)

        so that <doc, query> reproduces the BM25 sum over shared terms. IDF lives
        entirely on the query side; document vectors carry no corpus statistics beyond
        avg_doc_length. That is what lets the IDF table be refreshed at any time -- after
        every upload, forever -- without invalidating the 200k+ vectors already stored.
        Baking IDF into documents (the previous behaviour) meant chunks indexed at
        different times were scored against different corpus statistics.

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

            if is_query:
                # Query side carries the IDF weighting.
                score = tf * self._get_idf(term)
            else:
                # Document side carries only TF saturation + length normalization.
                # No IDF here: corpus statistics must not be frozen into the index.
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * (doc_length / self.avg_doc_length)
                )
                score = numerator / denominator

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
