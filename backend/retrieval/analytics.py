"""In-memory analytics tracking for the RAG pipeline.

Aggregates query metrics for the analytics dashboard:
- Query type distribution
- Citation verification scores
- Per-step latency
- Entity extraction frequency
"""

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Maximum number of queries to keep in rolling windows
MAX_ROLLING_WINDOW = 100
# Maximum number of top entities to return per category
MAX_TOP_ENTITIES = 5


@dataclass
class StepTimings:
    """Timing data for pipeline steps."""
    query_processing_ms: float = 0
    embedding_ms: float = 0
    retrieval_ms: float = 0
    reranking_ms: float = 0
    generation_ms: float = 0


@dataclass
class CitationResult:
    """Result from citation verification."""
    overall_score: float  # 0-1
    total_citations: int
    valid_citations: int
    partial_citations: int  # Citations with 0.3-0.6 confidence
    invalid_citations: int


class AnalyticsTracker:
    """Thread-safe in-memory analytics aggregator.

    Tracks rolling statistics from query executions for dashboard display.
    Data resets on server restart (intentionally ephemeral).
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._reset_stats()
        logger.info("AnalyticsTracker initialized")

    def _reset_stats(self):
        """Reset all statistics."""
        # Query type counts
        self._query_type_counts: Dict[str, int] = defaultdict(int)

        # Rolling window of citation scores
        self._citation_scores: List[CitationResult] = []

        # Rolling window of step latencies
        self._step_latencies: List[StepTimings] = []

        # Entity counts by category: {category: {entity: count}}
        self._entity_counts: Dict[str, Dict[str, int]] = {
            "chemicals": defaultdict(int),
            "proteins": defaultdict(int),
            "methods": defaultdict(int),
            "organisms": defaultdict(int),
            "metrics": defaultdict(int),
        }

        # Total query count
        self._total_queries = 0

    def record_query(
        self,
        query_type: str,
        step_timings: Optional[StepTimings] = None,
        citation_result: Optional[CitationResult] = None,
        entities: Optional[Dict[str, List[str]]] = None,
    ):
        """Record metrics from a completed query.

        Args:
            query_type: The classified query type (e.g., "FACTUAL", "METHODS")
            step_timings: Timing data for each pipeline step
            citation_result: Citation verification results
            entities: Extracted entities by category
        """
        with self._lock:
            self._total_queries += 1

            # Record query type
            self._query_type_counts[query_type.upper()] += 1

            # Record step timings (rolling window)
            if step_timings:
                self._step_latencies.append(step_timings)
                if len(self._step_latencies) > MAX_ROLLING_WINDOW:
                    self._step_latencies.pop(0)

            # Record citation result (rolling window)
            if citation_result:
                self._citation_scores.append(citation_result)
                if len(self._citation_scores) > MAX_ROLLING_WINDOW:
                    self._citation_scores.pop(0)

            # Record entity counts
            if entities:
                for category, entity_list in entities.items():
                    if category in self._entity_counts:
                        for entity in entity_list:
                            # Normalize entity name
                            normalized = entity.strip()
                            if normalized:
                                self._entity_counts[category][normalized] += 1

            logger.debug(f"Recorded query analytics: type={query_type}, total={self._total_queries}")

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics for the dashboard.

        Returns:
            Dictionary with analytics data for all dashboard sections.
        """
        with self._lock:
            return {
                "query_type_distribution": self._get_query_type_distribution(),
                "citation_stats": self._get_citation_stats(),
                "latency_stats": self._get_latency_stats(),
                "entity_stats": self._get_entity_stats(),
                "total_queries": self._total_queries,
            }

    def _get_query_type_distribution(self) -> Dict[str, int]:
        """Get query type counts."""
        # Return all types, defaulting to 0 for unseen types
        all_types = ["FACTUAL", "FRAMING", "METHODS", "SUMMARY",
                     "COMPARATIVE", "NOVELTY", "LIMITATIONS", "GENERAL"]
        return {t: self._query_type_counts.get(t, 0) for t in all_types}

    def _get_citation_stats(self) -> Dict[str, float]:
        """Calculate citation verification statistics."""
        if not self._citation_scores:
            return {
                "avg_score": 0,
                "verified_rate": 0,
                "partial_rate": 0,
                "failed_rate": 0,
                "total_checked": 0,
            }

        total_citations = sum(c.total_citations for c in self._citation_scores)
        valid = sum(c.valid_citations for c in self._citation_scores)
        partial = sum(c.partial_citations for c in self._citation_scores)
        invalid = sum(c.invalid_citations for c in self._citation_scores)

        avg_score = sum(c.overall_score for c in self._citation_scores) / len(self._citation_scores)

        if total_citations > 0:
            verified_rate = valid / total_citations
            partial_rate = partial / total_citations
            failed_rate = invalid / total_citations
        else:
            verified_rate = partial_rate = failed_rate = 0

        return {
            "avg_score": round(avg_score, 2),
            "verified_rate": round(verified_rate, 2),
            "partial_rate": round(partial_rate, 2),
            "failed_rate": round(failed_rate, 2),
            "total_checked": len(self._citation_scores),
        }

    def _get_latency_stats(self) -> Dict[str, float]:
        """Calculate average latencies for each pipeline step."""
        if not self._step_latencies:
            return {
                "query_processing_ms": 0,
                "embedding_ms": 0,
                "retrieval_ms": 0,
                "reranking_ms": 0,
                "generation_ms": 0,
                "total_avg_ms": 0,
            }

        n = len(self._step_latencies)

        query_processing = sum(s.query_processing_ms for s in self._step_latencies) / n
        embedding = sum(s.embedding_ms for s in self._step_latencies) / n
        retrieval = sum(s.retrieval_ms for s in self._step_latencies) / n
        reranking = sum(s.reranking_ms for s in self._step_latencies) / n
        generation = sum(s.generation_ms for s in self._step_latencies) / n

        total = query_processing + embedding + retrieval + reranking + generation

        return {
            "query_processing_ms": round(query_processing),
            "embedding_ms": round(embedding),
            "retrieval_ms": round(retrieval),
            "reranking_ms": round(reranking),
            "generation_ms": round(generation),
            "total_avg_ms": round(total),
        }

    def _get_entity_stats(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get top entities for each category."""
        result = {}

        for category, counts in self._entity_counts.items():
            # Sort by count descending, take top N
            sorted_entities = sorted(
                counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:MAX_TOP_ENTITIES]

            result[category] = [
                {"name": name, "count": count}
                for name, count in sorted_entities
            ]

        return result

    def reset(self):
        """Reset all statistics. Useful for testing."""
        with self._lock:
            self._reset_stats()
            logger.info("Analytics stats reset")


# Global singleton instance
_analytics_tracker: Optional[AnalyticsTracker] = None
_tracker_lock = threading.Lock()


def get_analytics_tracker() -> AnalyticsTracker:
    """Get the global analytics tracker singleton."""
    global _analytics_tracker
    if _analytics_tracker is None:
        with _tracker_lock:
            if _analytics_tracker is None:
                _analytics_tracker = AnalyticsTracker()
    return _analytics_tracker
