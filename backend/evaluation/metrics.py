"""Retrieval metrics for the W5 harness.

All metrics are computed per query and then macro-averaged, so a query with
many relevant chunks does not dominate one with few.

Relevance comes from the pooled qrels. Recall is therefore *pool-limited
recall*: the denominator is the set of relevant chunks any pooled arm found,
not the (unknowable) set of relevant chunks in the corpus. This is standard
TREC practice and is fair between arms only because every arm under comparison
contributed to the pool.

Two relevance granularities are kept:

- chunk-level, the usual RAG metric;
- paper-level, where a paper's grade is the max grade of its judged chunks.
  Paper-level numbers are the ones that survive the W1/W2 re-chunk, because
  `paper_id` is md5(filename)[:12] and is stable across a reindex while every
  chunk_id is regenerated.
"""

from __future__ import annotations

import math
import random
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from evaluation.query_set import GoldenQuery

KS = (5, 10, 20)
GAIN = {0: 0.0, 1: 1.0, 2: 3.0}  # 2**g - 1


# ---------------------------------------------------------------------------
# core metrics
# ---------------------------------------------------------------------------

def dcg(grades: Sequence[int]) -> float:
    return sum(GAIN[g] / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked_grades: Sequence[int], all_grades: Sequence[int], k: int) -> float:
    ideal = sorted(all_grades, reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg == 0:
        return 0.0
    return dcg(list(ranked_grades)[:k]) / idcg


def recall_at_k(ranked_grades: Sequence[int], n_relevant: int, k: int, threshold: int = 1) -> float:
    if n_relevant == 0:
        return 0.0
    hit = sum(1 for g in list(ranked_grades)[:k] if g >= threshold)
    return hit / n_relevant


def rr(ranked_grades: Sequence[int], threshold: int = 1) -> float:
    for i, g in enumerate(ranked_grades):
        if g >= threshold:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# per-query evaluation
# ---------------------------------------------------------------------------

def evaluate_query(
    q: GoldenQuery,
    results: List[Dict[str, Any]],
    chunk_qrels: Dict[str, int],
    paper_qrels: Dict[str, int],
) -> Dict[str, float]:
    """Metrics for one (query, config) run."""
    chunk_grades = [chunk_qrels.get(r["chunk_id"], 0) for r in results]

    # Paper-level: walk the ranking, keeping each paper's first appearance.
    seen: set = set()
    paper_grades: List[int] = []
    for r in results:
        pid = r.get("paper_id")
        if pid is None or pid in seen:
            continue
        seen.add(pid)
        paper_grades.append(paper_qrels.get(pid, 0))

    all_chunk_grades = list(chunk_qrels.values())
    all_paper_grades = list(paper_qrels.values())
    n_rel_chunks = sum(1 for g in all_chunk_grades if g >= 1)
    n_rel_chunks_2 = sum(1 for g in all_chunk_grades if g == 2)
    n_rel_papers = sum(1 for g in all_paper_grades if g >= 1)

    out: Dict[str, float] = {}
    for k in KS:
        out[f"recall@{k}"] = recall_at_k(chunk_grades, n_rel_chunks, k)
        out[f"recall2@{k}"] = recall_at_k(chunk_grades, n_rel_chunks_2, k, threshold=2)
        out[f"ndcg@{k}"] = ndcg_at_k(chunk_grades, all_chunk_grades, k)
        out[f"paper_recall@{k}"] = recall_at_k(paper_grades, n_rel_papers, k)
        out[f"paper_ndcg@{k}"] = ndcg_at_k(paper_grades, all_paper_grades, k)
        out[f"precision@{k}"] = (
            sum(1 for g in chunk_grades[:k] if g >= 1) / min(k, max(len(chunk_grades), 1))
            if chunk_grades else 0.0
        )
        out[f"judged@{k}"] = (
            sum(1 for r in results[:k] if r["chunk_id"] in chunk_qrels) / min(k, len(results))
            if results else 0.0
        )
        # -- literal term presence -------------------------------------
        if q.n_terms:
            top = results[:k]
            found: set = set()
            with_term = 0
            for r in top:
                hits = q.terms_present(r.get("text", ""))
                if hits:
                    with_term += 1
                found.update(hits)
            out[f"term_recall@{k}"] = len(found) / q.n_terms
            out[f"chunk_term_frac@{k}"] = with_term / len(top) if top else 0.0
            out[f"any_term@{k}"] = 1.0 if found else 0.0

    out["mrr"] = rr(chunk_grades)
    out["mrr2"] = rr(chunk_grades, threshold=2)
    out["paper_mrr"] = rr(paper_grades)
    out["n_results"] = float(len(results))
    out["n_rel_chunks"] = float(n_rel_chunks)
    out["n_rel_papers"] = float(n_rel_papers)
    return out


TERM_METRICS = tuple(
    f"{name}@{k}" for k in KS for name in ("term_recall", "chunk_term_frac", "any_term")
)


def aggregate(per_query: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Macro-average, skipping queries where a metric is undefined."""
    keys: List[str] = []
    for m in per_query.values():
        for k in m:
            if k not in keys:
                keys.append(k)
    out: Dict[str, float] = {}
    for k in keys:
        vals = [m[k] for m in per_query.values() if k in m]
        if vals:
            out[k] = mean(vals)
    out["n_queries"] = float(len(per_query))
    return out


# ---------------------------------------------------------------------------
# paired significance testing
# ---------------------------------------------------------------------------

def paired_test(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 10000,
    seed: int = 20260915,
) -> Dict[str, float]:
    """Two-sided paired randomisation test plus a bootstrap CI on the mean diff.

    `a` and `b` must be aligned per-query scores. The randomisation test flips
    the sign of each per-query difference at random; the p-value is the share of
    resamples whose mean |difference| is at least the observed one. With 50
    queries this is the right tool: it makes no normality assumption and it
    reports honestly when the arms are indistinguishable.
    """
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    if n == 0:
        return {"n": 0, "mean_diff": 0.0, "p_value": 1.0, "ci_low": 0.0, "ci_high": 0.0}
    observed = mean(diffs)

    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_resamples):
        m = mean(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(m) >= abs(observed) - 1e-12:
            extreme += 1
    p = (extreme + 1) / (n_resamples + 1)

    boots = []
    for _ in range(n_resamples):
        boots.append(mean(diffs[rng.randrange(n)] for _ in range(n)))
    boots.sort()
    lo = boots[int(0.025 * n_resamples)]
    hi = boots[int(0.975 * n_resamples)]

    n_win = sum(1 for d in diffs if d > 1e-12)
    n_loss = sum(1 for d in diffs if d < -1e-12)
    return {
        "n": n,
        "mean_diff": observed,
        "p_value": p,
        "ci_low": lo,
        "ci_high": hi,
        "wins": n_win,
        "losses": n_loss,
        "ties": n - n_win - n_loss,
    }
