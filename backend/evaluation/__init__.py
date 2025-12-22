"""Evaluation module for testing retrieval quality."""

from .test_queries import (
    TestQuery,
    QueryType,
    TEST_QUERIES,
    load_test_queries,
    save_test_queries,
    get_queries_by_type,
    get_human_eval_queries,
)
from .evaluator import (
    Evaluator,
    EvaluationResult,
    generate_human_eval_template,
    merge_human_evaluations,
)

__all__ = [
    # Test queries
    "TestQuery",
    "QueryType",
    "TEST_QUERIES",
    "load_test_queries",
    "save_test_queries",
    "get_queries_by_type",
    "get_human_eval_queries",
    # Evaluator
    "Evaluator",
    "EvaluationResult",
    "generate_human_eval_template",
    "merge_human_evaluations",
]
