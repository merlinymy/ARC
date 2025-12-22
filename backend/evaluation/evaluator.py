"""Evaluate retrieval and answer quality with human evaluation support."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from .test_queries import TestQuery, QueryType, load_test_queries

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating a single query."""
    query_id: str
    query: str
    query_type: str

    # Classification results
    classified_type: str
    classification_correct: bool
    expanded_query: str

    # Retrieval metrics
    retrieval_count: int
    reranked_count: int
    unique_papers: int
    chunk_types_retrieved: List[str]

    # Expected vs actual
    expected_topics_found: List[str]
    expected_topics_missing: List[str]
    expected_entities_found: List[str]
    expected_entities_missing: List[str]
    expected_chunk_types_hit: List[str]
    expected_chunk_types_missed: List[str]

    # Answer
    answer: str
    answer_length: int
    has_citations: bool

    # Timing
    latency_ms: float

    # Human evaluation fields
    requires_human_eval: bool = False
    human_relevance_score: Optional[int] = None  # 1-5
    human_answer_quality: Optional[int] = None   # 1-5
    human_notes: str = ""


class Evaluator:
    """Evaluate retrieval quality on test queries."""

    def __init__(self, query_engine):
        """Initialize evaluator.

        Args:
            query_engine: QueryEngine instance for running queries
        """
        self.query_engine = query_engine

    def _check_coverage(
        self,
        text: str,
        expected: List[str]
    ) -> tuple[List[str], List[str]]:
        """Check which expected items are present in text."""
        text_lower = text.lower()
        found = []
        missing = []

        for item in expected:
            if item.lower() in text_lower:
                found.append(item)
            else:
                missing.append(item)

        return found, missing

    def evaluate_query(self, test_query: TestQuery) -> EvaluationResult:
        """Evaluate a single test query."""
        import time

        start_time = time.time()
        result = self.query_engine.query(test_query.query)
        latency_ms = (time.time() - start_time) * 1000

        # Analyze retrieved chunks
        chunk_types = list(set(s.get('chunk_type', 'unknown') for s in result.sources))
        unique_papers = len(set(s.get('paper_id', '') for s in result.sources))

        # Combine all text for topic/entity checking
        all_text = result.answer + " " + " ".join(s.get('text', '') for s in result.sources)

        # Check coverage
        topics_found, topics_missing = self._check_coverage(all_text, test_query.expected_topics)
        entities_found, entities_missing = self._check_coverage(all_text, test_query.expected_entities)

        # Check chunk type coverage
        chunk_types_hit = [ct for ct in test_query.expected_chunk_types if ct in chunk_types]
        chunk_types_missed = [ct for ct in test_query.expected_chunk_types if ct not in chunk_types]

        # Check citations
        has_citations = '[Source' in result.answer or '[source' in result.answer

        return EvaluationResult(
            query_id=test_query.query_id,
            query=test_query.query,
            query_type=test_query.query_type.value,
            classified_type=result.query_type.value,
            classification_correct=(test_query.query_type.value == result.query_type.value),
            expanded_query=result.expanded_query,
            retrieval_count=result.retrieval_count,
            reranked_count=result.reranked_count,
            unique_papers=unique_papers,
            chunk_types_retrieved=chunk_types,
            expected_topics_found=topics_found,
            expected_topics_missing=topics_missing,
            expected_entities_found=entities_found,
            expected_entities_missing=entities_missing,
            expected_chunk_types_hit=chunk_types_hit,
            expected_chunk_types_missed=chunk_types_missed,
            answer=result.answer,
            answer_length=len(result.answer),
            has_citations=has_citations,
            latency_ms=latency_ms,
            requires_human_eval=test_query.requires_human_eval,
        )

    def evaluate_all(
        self,
        queries: List[TestQuery],
        output_path: Optional[Path] = None
    ) -> List[EvaluationResult]:
        """Evaluate all test queries."""
        results = []

        for query in queries:
            logger.info(f"Evaluating: {query.query_id}")
            try:
                result = self.evaluate_query(query)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate {query.query_id}: {e}")

        if output_path:
            self._save_results(results, output_path)

        return results

    def _save_results(self, results: List[EvaluationResult], path: Path):
        """Save results to JSON."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(results),
            'results': [asdict(r) for r in results],
            'summary': self._compute_summary(results),
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved evaluation results to {path}")

    def _compute_summary(self, results: List[EvaluationResult]) -> Dict:
        """Compute summary statistics."""
        if not results:
            return {}

        total_topics = sum(len(r.expected_topics_found) + len(r.expected_topics_missing) for r in results)
        total_topics_found = sum(len(r.expected_topics_found) for r in results)

        total_entities = sum(len(r.expected_entities_found) + len(r.expected_entities_missing) for r in results)
        total_entities_found = sum(len(r.expected_entities_found) for r in results)

        classification_correct = sum(1 for r in results if r.classification_correct)

        return {
            'avg_latency_ms': sum(r.latency_ms for r in results) / len(results),
            'topic_coverage': total_topics_found / max(total_topics, 1),
            'entity_coverage': total_entities_found / max(total_entities, 1),
            'citation_rate': sum(1 for r in results if r.has_citations) / len(results),
            'classification_accuracy': classification_correct / len(results),
            'requires_human_eval': sum(1 for r in results if r.requires_human_eval),
            'by_query_type': self._summarize_by_type(results),
        }

    def _summarize_by_type(self, results: List[EvaluationResult]) -> Dict[str, Dict]:
        """Summarize results by query type."""
        by_type: Dict[str, Dict] = {}

        for r in results:
            if r.query_type not in by_type:
                by_type[r.query_type] = {
                    'count': 0,
                    'latencies': [],
                    'topic_hits': 0,
                    'topic_total': 0,
                    'classification_correct': 0,
                }

            by_type[r.query_type]['count'] += 1
            by_type[r.query_type]['latencies'].append(r.latency_ms)
            by_type[r.query_type]['topic_hits'] += len(r.expected_topics_found)
            by_type[r.query_type]['topic_total'] += len(r.expected_topics_found) + len(r.expected_topics_missing)
            if r.classification_correct:
                by_type[r.query_type]['classification_correct'] += 1

        # Compute averages
        for qt, data in by_type.items():
            data['avg_latency_ms'] = sum(data['latencies']) / len(data['latencies'])
            data['topic_coverage'] = data['topic_hits'] / max(data['topic_total'], 1)
            data['classification_accuracy'] = data['classification_correct'] / data['count']
            del data['latencies']

        return by_type


def generate_human_eval_template(results_path: Path, output_path: Path):
    """Generate a human evaluation template from results.

    Args:
        results_path: Path to evaluation results JSON
        output_path: Path for human evaluation template
    """
    with open(results_path) as f:
        data = json.load(f)

    template = []
    for r in data['results']:
        if r.get('requires_human_eval'):
            template.append({
                'query_id': r['query_id'],
                'query': r['query'],
                'query_type': r['query_type'],
                'answer': r['answer'][:1000] + '...' if len(r['answer']) > 1000 else r['answer'],
                'sources_count': r['reranked_count'],
                'human_relevance_score': None,  # 1-5: How relevant are the retrieved sources?
                'human_answer_quality': None,   # 1-5: How good is the answer?
                'human_notes': '',
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)

    print(f"Generated human evaluation template with {len(template)} queries")
    print(f"Please fill in human_relevance_score (1-5) and human_answer_quality (1-5)")


def merge_human_evaluations(results_path: Path, human_eval_path: Path, output_path: Path):
    """Merge human evaluations back into results.

    Args:
        results_path: Original evaluation results
        human_eval_path: Human evaluation template with scores filled in
        output_path: Output path for merged results
    """
    with open(results_path) as f:
        results = json.load(f)

    with open(human_eval_path) as f:
        human_evals = json.load(f)

    # Create lookup by query_id
    human_eval_lookup = {h['query_id']: h for h in human_evals}

    # Merge scores
    for r in results['results']:
        if r['query_id'] in human_eval_lookup:
            h = human_eval_lookup[r['query_id']]
            r['human_relevance_score'] = h.get('human_relevance_score')
            r['human_answer_quality'] = h.get('human_answer_quality')
            r['human_notes'] = h.get('human_notes', '')

    # Update summary with human eval metrics
    human_results = [r for r in results['results'] if r.get('human_relevance_score') is not None]
    if human_results:
        results['summary']['human_eval'] = {
            'count': len(human_results),
            'avg_relevance_score': sum(r['human_relevance_score'] for r in human_results) / len(human_results),
            'avg_answer_quality': sum(r['human_answer_quality'] for r in human_results) / len(human_results),
        }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Merged {len(human_results)} human evaluations into results")
