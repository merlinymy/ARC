"""Run Phase 1 evaluation.

This script runs the full evaluation pipeline on test queries,
generating metrics and identifying areas for improvement.

Usage:
    python run_evaluation.py [--limit N] [--query-type TYPE]
"""

import sys
import json
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from anthropic import Anthropic

from retrieval.embedder import VoyageEmbedder
from retrieval.reranker import CohereReranker
from retrieval.qdrant_store import QdrantStore
from retrieval.query_engine import QueryEngine
from evaluation.test_queries import load_test_queries, get_queries_by_type, QueryType
from evaluation.evaluator import Evaluator, generate_human_eval_template

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 evaluation")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to evaluate"
    )
    parser.add_argument(
        "--query-type",
        type=str,
        default=None,
        help="Only evaluate queries of this type (factual, framing, methods, etc.)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results (default: from config)"
    )

    args = parser.parse_args()

    # Initialize components
    logger.info("Initializing components...")

    embedder = VoyageEmbedder(api_key=settings.voyage_api_key)
    reranker = CohereReranker(api_key=settings.cohere_api_key)
    store = QdrantStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection_name
    )
    anthropic = Anthropic(api_key=settings.anthropic_api_key)

    query_engine = QueryEngine(
        embedder=embedder,
        reranker=reranker,
        store=store,
        anthropic_client=anthropic,
        claude_model=settings.claude_model,
        enable_classification=settings.enable_query_classification,
        enable_expansion=settings.enable_query_expansion,
    )

    # Load test queries
    queries = load_test_queries(settings.test_queries_path)

    # Filter by query type if specified
    if args.query_type:
        try:
            query_type = QueryType(args.query_type.lower())
            queries = get_queries_by_type(query_type)
            logger.info(f"Filtering to {query_type.value} queries: {len(queries)}")
        except ValueError:
            logger.error(f"Invalid query type: {args.query_type}")
            logger.info(f"Valid types: {[t.value for t in QueryType]}")
            return

    # Apply limit
    if args.limit:
        queries = queries[:args.limit]

    logger.info(f"Evaluating {len(queries)} queries...")

    # Determine output path
    output_path = Path(args.output) if args.output else settings.evaluation_results_path

    # Run evaluation
    evaluator = Evaluator(query_engine)
    results = evaluator.evaluate_all(queries, output_path=output_path)

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    with open(output_path) as f:
        data = json.load(f)
        summary = data['summary']

        print(f"Total queries: {data['total_queries']}")
        print(f"Avg latency: {summary['avg_latency_ms']:.0f}ms")
        print(f"Topic coverage: {summary['topic_coverage']:.1%}")
        print(f"Entity coverage: {summary['entity_coverage']:.1%}")
        print(f"Citation rate: {summary['citation_rate']:.1%}")
        print(f"Classification accuracy: {summary['classification_accuracy']:.1%}")
        print(f"Queries needing human eval: {summary['requires_human_eval']}")

        print("\nBy query type:")
        for qt, stats in summary['by_query_type'].items():
            print(f"  {qt}: {stats['topic_coverage']:.1%} coverage, "
                  f"{stats['classification_accuracy']:.1%} classification, "
                  f"{stats['avg_latency_ms']:.0f}ms")

    # Generate human evaluation template
    human_eval_path = output_path.parent / "human_evaluation_template.json"
    generate_human_eval_template(output_path, human_eval_path)

    print(f"\nResults saved to: {output_path}")
    print(f"Human eval template: {human_eval_path}")


if __name__ == "__main__":
    main()
