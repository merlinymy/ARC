"""W5 evaluation harness: a measured, reusable retrieval baseline for ARC.

Pieces, in the order they run:

    build_query_set  golden set sourced from real usage in backend/data/app.db
    query_set        loader + literal-term normalisation
    configs          named retrieval arms (dense / sparse / hybrid / routed / broad)
    runner           drives the real retrieval components, retrieval only
    judge            graded 0/1/2 relevance labels from claude-haiku-4-5, cached
    metrics          recall@k, nDCG@k, MRR, literal-term presence, paired tests
    run_w5           orchestrator: retrieve -> pool -> judge -> score

The durable assets are `golden_queries_v1.json` and `qrels_v1.json`. Everything
else is replaceable machinery around them.

Replaced 2026-09-15: `test_queries.py` (50 invented queries with invented
expected topics) and `evaluator.py` (metrics that required running full
`QueryEngine.query()`, i.e. answer generation, to measure retrieval).
"""

from .query_set import GoldenQuery, QuerySet, load_query_set, normalize
from .configs import CONFIGS, RetrievalConfig, get as get_config
from .judge import Judgment, RelevanceJudge, pair_key

__all__ = [
    "GoldenQuery",
    "QuerySet",
    "load_query_set",
    "normalize",
    "CONFIGS",
    "RetrievalConfig",
    "get_config",
    "Judgment",
    "RelevanceJudge",
    "pair_key",
]
