"""Voyage AI embedding wrapper with mean pooling support.

Provides embedding functionality for documents and queries using
Voyage AI's voyage-3-large model, with support for mean pooling
to create full-paper embeddings.

Rate limits (with payment method):
- 2000 RPM (requests per minute)
- 3M TPM (tokens per minute)
"""

import logging
import time
from typing import Callable, List, Optional, Tuple
import numpy as np
import voyageai

logger = logging.getLogger(__name__)

# Rate limiting constants
DEFAULT_RPM = 2000  # requests per minute
DEFAULT_TPM = 3_000_000  # tokens per minute
MIN_REQUEST_INTERVAL = 60.0 / DEFAULT_RPM  # ~0.03 seconds between requests


class VoyageEmbedder:
    """Voyage AI embedding client with batch processing, rate limiting, and mean pooling."""

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-3-large",
        batch_size: int = 128,
        max_retries: int = 5,
        rpm_limit: int = DEFAULT_RPM,
    ):
        """Initialize embedder.

        Args:
            api_key: Voyage AI API key
            model: Embedding model name
            batch_size: Maximum texts per API call
            max_retries: Maximum retry attempts on rate limit errors
            rpm_limit: Requests per minute limit (default: 2000)
        """
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.min_interval = 60.0 / rpm_limit
        self.dimension = 1024  # voyage-3-large dimension
        self._last_request_time = 0.0

        logger.info(f"Initialized VoyageEmbedder with model {model}, rate limit {rpm_limit} RPM")

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _embed_with_retry(self, texts: List[str], input_type: str) -> List[List[float]]:
        """Embed texts with retry logic for rate limit errors.

        Args:
            texts: Texts to embed
            input_type: "document" or "query"

        Returns:
            List of embeddings
        """
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                result = self.client.embed(
                    texts=texts,
                    model=self.model,
                    input_type=input_type
                )
                return result.embeddings

            except Exception as e:
                error_msg = str(e).lower()

                # Check if it's a rate limit error
                if "rate" in error_msg or "limit" in error_msg or "429" in error_msg:
                    wait_time = (2 ** attempt) * 1.0  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                    logger.warning(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                else:
                    # Non-rate-limit error, raise immediately
                    raise

        # If we've exhausted retries, raise the last error
        raise Exception(f"Failed after {self.max_retries} retries due to rate limiting")

    def embed_documents(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[float]]:
        """Embed a list of documents with rate limiting and retry.

        Args:
            texts: List of document texts
            progress_callback: Optional callback(batch_index, total_batches) called after each batch

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        batch_index = 0

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = self._embed_with_retry(batch, input_type="document")
            all_embeddings.extend(embeddings)
            batch_index += 1

            if progress_callback:
                try:
                    progress_callback(batch_index, total_batches)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query with rate limiting and retry.

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        embeddings = self._embed_with_retry([query], input_type="query")
        return embeddings[0]

    def compute_mean_pooled_embedding(
        self,
        texts: List[str],
        weights: Optional[List[float]] = None
    ) -> List[float]:
        """Compute mean-pooled embedding from multiple texts.

        Used to create full-paper embeddings that preserve information
        from all sections (unlike truncation which loses end content).

        Args:
            texts: List of text chunks to pool
            weights: Optional weights for each text (default: equal weights)

        Returns:
            Mean-pooled and L2-normalized embedding vector
        """
        if not texts:
            return [0.0] * self.dimension

        # Get embeddings for all texts
        embeddings = self.embed_documents(texts)

        # Convert to numpy array
        embeddings_array = np.array(embeddings)

        # Apply weights if provided
        if weights:
            weights = np.array(weights).reshape(-1, 1)
            mean_embedding = np.average(embeddings_array, axis=0, weights=weights.flatten())
        else:
            mean_embedding = np.mean(embeddings_array, axis=0)

        # L2 normalize
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm

        return mean_embedding.tolist()

    def embed_paper_sections(
        self,
        sections: List[Tuple[str, str]],
        exclude_sections: Optional[List[str]] = None
    ) -> List[float]:
        """Create a full-paper embedding from sections.

        Args:
            sections: List of (section_name, section_text) tuples
            exclude_sections: Section names to exclude (e.g., ["references"])

        Returns:
            Mean-pooled embedding for the full paper
        """
        # Filter out excluded sections by name (case-insensitive)
        if exclude_sections:
            exclude_lower = [name.lower() for name in exclude_sections]
            valid_sections = [
                text for name, text in sections
                if name.lower() not in exclude_lower
            ]
        else:
            valid_sections = [text for _, text in sections]

        if not valid_sections:
            logger.warning("No valid sections for paper embedding")
            return [0.0] * self.dimension

        return self.compute_mean_pooled_embedding(valid_sections)
