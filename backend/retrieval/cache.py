"""Caching layer for embeddings and query results.

Provides LRU caching for:
- Query embeddings (avoid re-embedding same queries)
- Search results (avoid re-searching for same queries)
- HyDE generated answers

Includes cache invalidation support for when new papers are indexed.
"""

import hashlib
import logging
import time
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached entry with TTL support."""
    value: Any
    created_at: float
    ttl_seconds: float = 3600  # 1 hour default

    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds


class LRUCache:
    """Thread-safe LRU cache with TTL support."""

    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        """Initialize cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, *args, **kwargs) -> str:
        """Create a cache key from arguments."""
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check TTL
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        with self._lock:
            # Remove if exists (to update position)
            if key in self._cache:
                del self._cache[key]

            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl_seconds=ttl or self.default_ttl
            )

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.2%}",
            }


class RAGCache:
    """Unified caching for RAG pipeline components.

    Supports cache invalidation when new papers are indexed.
    """

    def __init__(
        self,
        embedding_cache_size: int = 500,
        search_cache_size: int = 200,
        hyde_cache_size: int = 100,
        embedding_ttl: float = 86400,  # 24 hours
        search_ttl: float = 600,       # 10 minutes (reduced from 1 hour for freshness)
        hyde_ttl: float = 1800,        # 30 minutes
    ):
        """Initialize RAG cache.

        Args:
            embedding_cache_size: Max cached embeddings
            search_cache_size: Max cached search results
            hyde_cache_size: Max cached HyDE results
            embedding_ttl: TTL for embeddings
            search_ttl: TTL for search results (short to avoid stale results)
            hyde_ttl: TTL for HyDE results
        """
        self.embedding_cache = LRUCache(embedding_cache_size, embedding_ttl)
        self.search_cache = LRUCache(search_cache_size, search_ttl)
        self.hyde_cache = LRUCache(hyde_cache_size, hyde_ttl)

        # Track invalidation state
        self._last_invalidation_time: float = 0
        self._invalidation_lock = threading.Lock()

        logger.info(
            f"Initialized RAGCache: embeddings={embedding_cache_size}, "
            f"search={search_cache_size}, hyde={hyde_cache_size}"
        )

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text."""
        key = hashlib.md5(text.encode()).hexdigest()
        return self.embedding_cache.get(key)

    def set_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache embedding for text."""
        key = hashlib.md5(text.encode()).hexdigest()
        self.embedding_cache.set(key, embedding)

    def get_search_results(
        self,
        query: str,
        chunk_types: Optional[List[str]] = None,
        section_filter: Optional[List[str]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results."""
        key = self._search_key(query, chunk_types, section_filter)
        return self.search_cache.get(key)

    def set_search_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        chunk_types: Optional[List[str]] = None,
        section_filter: Optional[List[str]] = None,
    ) -> None:
        """Cache search results."""
        key = self._search_key(query, chunk_types, section_filter)
        self.search_cache.set(key, results)

    def _search_key(
        self,
        query: str,
        chunk_types: Optional[List[str]],
        section_filter: Optional[List[str]],
    ) -> str:
        """Generate search cache key."""
        parts = [query]
        if chunk_types:
            parts.append(f"types:{','.join(sorted(chunk_types))}")
        if section_filter:
            parts.append(f"sections:{','.join(sorted(section_filter))}")
        return hashlib.md5("|".join(parts).encode()).hexdigest()

    def get_hyde_answer(self, query: str) -> Optional[str]:
        """Get cached HyDE hypothetical answer."""
        key = hashlib.md5(query.encode()).hexdigest()
        return self.hyde_cache.get(key)

    def set_hyde_answer(self, query: str, answer: str) -> None:
        """Cache HyDE hypothetical answer."""
        key = hashlib.md5(query.encode()).hexdigest()
        self.hyde_cache.set(key, answer)

    def clear_all(self) -> None:
        """Clear all caches."""
        self.embedding_cache.clear()
        self.search_cache.clear()
        self.hyde_cache.clear()
        logger.info("Cleared all RAG caches")

    def invalidate_search_cache(self) -> None:
        """Invalidate search cache when new papers are indexed.

        Call this after upserting new chunks to ensure fresh results.
        """
        with self._invalidation_lock:
            self.search_cache.clear()
            self._last_invalidation_time = time.time()
            logger.info("Invalidated search cache due to new indexing")

    def invalidate_if_needed(self, recently_indexed_papers: Set[str]) -> bool:
        """Conditionally invalidate cache if papers were recently indexed.

        Args:
            recently_indexed_papers: Set of paper IDs that were recently indexed

        Returns:
            True if cache was invalidated, False otherwise
        """
        if recently_indexed_papers:
            self.invalidate_search_cache()
            return True
        return False

    def get_last_invalidation_time(self) -> float:
        """Get timestamp of last cache invalidation."""
        return self._last_invalidation_time

    def stats(self) -> Dict[str, Any]:
        """Get all cache statistics."""
        return {
            "embedding": self.embedding_cache.stats(),
            "search": self.search_cache.stats(),
            "hyde": self.hyde_cache.stats(),
            "last_invalidation": self._last_invalidation_time,
        }
