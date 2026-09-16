"""W5 evaluation harness - retrieve, pool, judge, score.

Stages (run all by default; each is resumable and caches to disk):

  retrieve  run every named arm over the golden set, save the rankings
  pool      union the arms' top-N into a per-query judging pool
  judge     grade every pooled (query, chunk) pair with Claude Haiku 4.5
  score     compute recall@k / nDCG@k / MRR / literal-term presence and write
            the baseline artifact

Retrieval only. `QueryEngine.query()` is never called and no answer is ever
generated.

Usage:
    python -m evaluation.run_w5                       # everything
    python -m evaluation.run_w5 --stages retrieve
    python -m evaluation.run_w5 --stages score --quiet
    python -m evaluation.run_w5 --estimate-only       # judging cost, no spend
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation import metrics as M  # noqa: E402
from evaluation.configs import CONFIGS, DEFAULT_ARMS, RetrievalConfig  # noqa: E402
from evaluation.judge import RelevanceJudge, estimate_cost, pair_key  # noqa: E402
from evaluation.query_set import QuerySet, load_query_set  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
CACHE_DIR = EVAL_DIR / "cache"

RUN_PATH = RESULTS_DIR / "retrieval_runs.json"
TEXTS_PATH = RESULTS_DIR / "chunk_texts.json"
POOL_PATH = RESULTS_DIR / "pool.json"
QRELS_PATH = EVAL_DIR / "qrels_v1.json"
JUDGE_CACHE = CACHE_DIR / "judgments.json"

logger = logging.getLogger("w5")


# ---------------------------------------------------------------------------
# stage 1 - retrieve
# ---------------------------------------------------------------------------

def stage_retrieve(qs: QuerySet, arms: List[str], force: bool) -> Tuple[dict, dict]:
    from evaluation.runner import RetrievalRunner

    runs: Dict[str, Dict[str, Any]] = {}
    texts: Dict[str, str] = {}
    if RUN_PATH.exists() and not force:
        runs = json.loads(RUN_PATH.read_text())
        texts = json.loads(TEXTS_PATH.read_text()) if TEXTS_PATH.exists() else {}

    missing = [
        (arm, q) for arm in arms for q in qs if f"{arm}|{q.query_id}" not in runs
    ]
    if not missing:
        logger.info("retrieve: all %d arm-query runs already on disk", len(runs))
        return runs, texts

    runner = RetrievalRunner()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for i, (arm, q) in enumerate(missing, 1):
        cfg = CONFIGS[arm]
        out = runner.run(cfg, q)
        for r in out["results"]:
            if r["chunk_id"]:
                texts[r["chunk_id"]] = r.pop("text")
            else:
                r.pop("text", None)
        runs[f"{arm}|{q.query_id}"] = out
        if i % 10 == 0 or i == len(missing):
            RUN_PATH.write_text(json.dumps(runs))
            TEXTS_PATH.write_text(json.dumps(texts))
            logger.info(
                "retrieve: %d/%d  (%s / %s, %d hits)",
                i, len(missing), arm, q.query_id, out["n_results"],
            )

    RUN_PATH.write_text(json.dumps(runs))
    TEXTS_PATH.write_text(json.dumps(texts))
    logger.info(
        "retrieve: done. rerank calls=%d docs=%d llm=%s",
        runner.rerank_calls, runner.rerank_docs, runner.llm_calls,
    )
    return runs, texts


# ---------------------------------------------------------------------------
# stage 2 - pool
# ---------------------------------------------------------------------------

def stage_pool(
    qs: QuerySet,
    runs: dict,
    arms: List[str],
    depth: int,
    cap: int,
) -> dict:
    """Round-robin union of each arm's top-`depth`, capped at `cap` per query.

    Round-robin by rank rather than arm-by-arm concatenation: if the cap binds,
    every arm loses its deepest documents equally instead of the last-listed arm
    losing everything.
    """
    pool: Dict[str, List[str]] = {}
    contrib: Dict[str, Dict[str, int]] = {}
    for q in qs:
        chosen: List[str] = []
        seen: set = set()
        by_arm = {
            arm: [r["chunk_id"] for r in runs[f"{arm}|{q.query_id}"]["results"][:depth]]
            for arm in arms
            if f"{arm}|{q.query_id}" in runs
        }
        c: Dict[str, int] = defaultdict(int)
        for rank in range(depth):
            for arm in arms:
                ids = by_arm.get(arm, [])
                if rank >= len(ids):
                    continue
                cid = ids[rank]
                if cid in seen or not cid:
                    continue
                seen.add(cid)
                chosen.append(cid)
                c[arm] += 1
                if len(chosen) >= cap:
                    break
            if len(chosen) >= cap:
                break
        pool[q.query_id] = chosen
        contrib[q.query_id] = dict(c)

    sizes = [len(v) for v in pool.values()]
    meta = {
        "arms": arms,
        "depth_per_arm": depth,
        "cap_per_query": cap,
        "total_pairs": sum(sizes),
        "mean_pool_size": round(sum(sizes) / max(len(sizes), 1), 1),
        "min_pool_size": min(sizes) if sizes else 0,
        "max_pool_size": max(sizes) if sizes else 0,
        "unique_contributions": contrib,
        "pool": pool,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    POOL_PATH.write_text(json.dumps(meta))
    logger.info(
        "pool: %d pairs across %d queries (mean %.1f, max %d)",
        meta["total_pairs"], len(pool), meta["mean_pool_size"], meta["max_pool_size"],
    )
    return meta


# ---------------------------------------------------------------------------
# stage 3 - judge
# ---------------------------------------------------------------------------

def _chunk_meta(runs: dict, texts: dict) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    for run in runs.values():
        for r in run["results"]:
            cid = r["chunk_id"]
            if cid and cid not in meta:
                meta[cid] = {
                    "chunk_id": cid,
                    "paper_id": r.get("paper_id"),
                    "chunk_type": r.get("chunk_type"),
                    "section_name": r.get("section_name"),
                    "title": r.get("title"),
                    "text": texts.get(cid, ""),
                }
    return meta


def stage_judge(
    qs: QuerySet,
    pool_meta: dict,
    runs: dict,
    texts: dict,
    max_spend: float,
    estimate_only: bool,
) -> Optional[dict]:
    from anthropic import Anthropic

    from config import settings

    meta = _chunk_meta(runs, texts)
    pairs = [
        (q.query_id, q.text, meta[cid])
        for q in qs
        for cid in pool_meta["pool"][q.query_id]
        if cid in meta
    ]

    est = estimate_cost(pairs)
    logger.info("judge: %d pairs, estimated $%.2f", est["pairs"], est["est_usd"])
    if estimate_only:
        print(json.dumps(est, indent=2))
        return None

    judge = RelevanceJudge(
        client=Anthropic(api_key=settings.anthropic_api_key),
        cache_path=JUDGE_CACHE,
        max_spend_usd=max_spend,
    )
    judged = judge.judge_pairs(pairs)

    qrels: Dict[str, Any] = {}
    for q in qs:
        chunks: Dict[str, Any] = {}
        papers: Dict[str, int] = {}
        for cid in pool_meta["pool"][q.query_id]:
            key = pair_key(q.text, cid)
            j = judged.get(key)
            if j is None:
                continue
            cm = meta[cid]
            chunks[cid] = {
                "grade": j.grade,
                "reason": j.reason,
                "paper_id": cm["paper_id"],
                "chunk_type": cm["chunk_type"],
                "section_name": cm["section_name"],
                "title": cm["title"],
            }
            pid = cm["paper_id"]
            if pid:
                papers[pid] = max(papers.get(pid, 0), j.grade)
        qrels[q.query_id] = {
            "query_text": q.text,
            "kind": q.kind,
            "chunks": chunks,
            "papers": papers,
            "n_judged": len(chunks),
            "n_rel_chunks": sum(1 for c in chunks.values() if c["grade"] >= 1),
            "n_rel2_chunks": sum(1 for c in chunks.values() if c["grade"] == 2),
            "n_rel_papers": sum(1 for g in papers.values() if g >= 1),
        }

    doc = {
        "version": "v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query_set_version": qs.version,
        "method": (
            "TREC-style pooling. Each arm contributed its top-N; the de-duplicated "
            "union was graded 0/1/2 by claude-haiku-4-5, one (query, chunk) pair per "
            "call, blind to which arm retrieved it. Paper grades are the max grade "
            "over that paper's judged chunks and survive a re-chunk, because "
            "paper_id = md5(filename)[:12] is stable while chunk_ids are not."
        ),
        "pool": {k: v for k, v in pool_meta.items() if k not in ("pool", "unique_contributions")},
        "judge": judge.stats(),
        "queries": qrels,
    }
    QRELS_PATH.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    logger.info(
        "judge: wrote %s  spend=$%.3f  failures=%d",
        QRELS_PATH.name, judge.spend_usd, judge.failures,
    )
    return doc


def stage_reliability(qs: QuerySet, pool_meta: dict, runs: dict, texts: dict,
                      n: int, seed: int = 7) -> dict:
    """Re-judge a random sample of pooled pairs to bound judge noise.

    This measures self-consistency, not correctness: it tells you how much of
    the label set is coin-flip, and therefore how large a metric difference has
    to be before it can be attributed to retrieval rather than to the judge. It
    cannot detect a bias the judge holds consistently.
    """
    import random

    from anthropic import Anthropic

    from config import settings

    meta = _chunk_meta(runs, texts)
    cache = json.loads(JUDGE_CACHE.read_text()) if JUDGE_CACHE.exists() else {}
    candidates = [
        (q.query_id, q.text, meta[cid])
        for q in qs
        for cid in pool_meta["pool"][q.query_id]
        if cid in meta and pair_key(q.text, cid) in cache
    ]
    random.Random(seed).shuffle(candidates)
    sample = candidates[:n]

    judge = RelevanceJudge(
        client=Anthropic(api_key=settings.anthropic_api_key),
        cache_path=CACHE_DIR / "reliability_probe.json",
    )
    import concurrent.futures as cf

    pairs_out: List[Tuple[int, int]] = []
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(judge._call, qt, ch): (qt, ch) for _, qt, ch in sample}
        for fut in cf.as_completed(futs):
            qt, ch = futs[fut]
            try:
                second = fut.result()
            except Exception:  # noqa: BLE001
                continue
            first = cache[pair_key(qt, ch["chunk_id"])]["grade"]
            pairs_out.append((first, second.grade))

    n_pairs = len(pairs_out)
    exact = sum(1 for a, b in pairs_out if a == b) / max(n_pairs, 1)
    within1 = sum(1 for a, b in pairs_out if abs(a - b) <= 1) / max(n_pairs, 1)
    binary = sum(1 for a, b in pairs_out if (a >= 1) == (b >= 1)) / max(n_pairs, 1)

    # linear-weighted Cohen's kappa over the 3-point scale
    obs_dis = sum(abs(a - b) for a, b in pairs_out) / max(n_pairs, 1)
    ma = [sum(1 for a, _ in pairs_out if a == g) / max(n_pairs, 1) for g in (0, 1, 2)]
    mb = [sum(1 for _, b in pairs_out if b == g) / max(n_pairs, 1) for g in (0, 1, 2)]
    exp_dis = sum(ma[i] * mb[j] * abs(i - j) for i in range(3) for j in range(3))
    kappa = 1 - obs_dis / exp_dis if exp_dis else 1.0

    out = {
        "n_resampled": n_pairs,
        "exact_agreement": round(exact, 3),
        "within_one_grade": round(within1, 3),
        "binary_agreement_rel_vs_not": round(binary, 3),
        "linear_weighted_kappa": round(kappa, 3),
        "probe_spend_usd": round(judge.spend_usd, 4),
    }
    (RESULTS_DIR / "judge_reliability.json").write_text(json.dumps(out, indent=1))
    logger.info("reliability: %s", out)
    return out


# ---------------------------------------------------------------------------
# stage 4 - score
# ---------------------------------------------------------------------------

HEADLINE = ["recall@10", "ndcg@10", "recall@20", "ndcg@20", "mrr",
            "paper_recall@10", "term_recall@10", "chunk_term_frac@10"]


def stage_score(qs: QuerySet, runs: dict, texts: dict, qrels: dict, arms: List[str]) -> dict:
    per_config: Dict[str, Dict[str, Any]] = {}

    for arm in arms:
        per_query: Dict[str, Dict[str, float]] = {}
        for q in qs:
            key = f"{arm}|{q.query_id}"
            if key not in runs:
                continue
            results = [
                {**r, "text": texts.get(r["chunk_id"], "")}
                for r in runs[key]["results"]
            ]
            qr = qrels["queries"][q.query_id]
            chunk_qrels = {cid: c["grade"] for cid, c in qr["chunks"].items()}
            per_query[q.query_id] = M.evaluate_query(q, results, chunk_qrels, qr["papers"])

        overall = M.aggregate(per_query)
        by_kind = {
            kind: M.aggregate({q.query_id: per_query[q.query_id]
                               for q in qs.of_kind(kind) if q.query_id in per_query})
            for kind in ("factual", "methods", "library_filtered", "synthesis")
        }
        routes = defaultdict(int)
        for q in qs:
            key = f"{arm}|{q.query_id}"
            if key in runs and runs[key].get("route_type"):
                routes[runs[key]["route_type"]] += 1

        per_config[arm] = {
            "config": {
                "name": arm,
                "description": CONFIGS[arm].description,
                **{f: getattr(CONFIGS[arm], f) for f in (
                    "search_mode", "route", "apply_filters", "use_strategy_sizes",
                    "top_k", "rerank_top_n", "max_per_paper", "use_hyde", "use_expansion")},
            },
            "overall": overall,
            "by_kind": by_kind,
            "route_distribution": dict(routes),
            "per_query": per_query,
        }
    return per_config



def corpus_snapshot() -> dict:
    """Live collection composition and the reach of each routed strategy.

    Recorded in the baseline so the post-W1/W2 comparison can tell a retrieval
    change apart from a corpus change. `searchable` is the number of chunks a
    query routed to that QueryType can even be scored against - everything else
    is excluded by the hard `chunk_types` / `section_filter` conditions before
    any similarity is computed.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    from config import settings
    from retrieval.query_classifier import RETRIEVAL_STRATEGIES

    c = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    name = settings.qdrant_collection_name
    total = c.count(name, exact=True).count
    types = {h.value: h.count for h in c.facet(collection_name=name, key="chunk_type", limit=25).hits}
    sections = {h.value: h.count for h in c.facet(collection_name=name, key="section_name", limit=40).hits}

    reach = {}
    for qt, strat in RETRIEVAL_STRATEGIES.items():
        must = [FieldCondition(key="chunk_type", match=MatchAny(any=strat["chunk_types"]))]
        if strat.get("section_filter"):
            must.append(FieldCondition(key="section_name", match=MatchAny(any=strat["section_filter"])))
        n = c.count(name, count_filter=Filter(must=must), exact=True).count
        reach[qt.value] = {
            "chunk_types": strat["chunk_types"],
            "section_filter": strat.get("section_filter"),
            "searchable_chunks": n,
            "fraction_of_corpus": round(n / total, 4),
        }
    return {
        "collection": name,
        "total_chunks": total,
        "chunk_types": types,
        "section_names": sections,
        "strategy_reach": reach,
    }


COMPARISONS = [
    ("classifier_routed", "broad_hybrid_hyde",
     "SS4b as written: LLM-classifier routing vs one broad hybrid path (HyDE held equal)"),
    ("keyword_routed", "broad_hybrid_hyde",
     "Live production routing vs one broad hybrid path (HyDE held equal)"),
    ("classifier_filters", "hybrid",
     "The hard chunk_type / section_filter exclusions, single variable"),
    ("classifier_routed", "broad_hybrid",
     "Production-exact routed arm vs the SS4b proposal as specified (HyDE differs)"),
    ("classifier_routed", "keyword_routed",
     "Does the Sonnet classifier route better than the keyword matcher that is live today?"),
    ("hybrid", "dense_only", "Sparse fusion on top of dense"),
    ("hybrid", "sparse_only", "Dense fusion on top of sparse"),
    ("broad_hybrid", "hybrid", "Retrieval depth 200 vs 100"),
]


def run_comparisons(qs: QuerySet, per_config: dict, metric: str = "ndcg@10") -> List[dict]:
    out = []
    for a, b, why in COMPARISONS:
        if a not in per_config or b not in per_config:
            continue
        ids = [q.query_id for q in qs
               if q.query_id in per_config[a]["per_query"]
               and q.query_id in per_config[b]["per_query"]]
        va = [per_config[a]["per_query"][i].get(metric, 0.0) for i in ids]
        vb = [per_config[b]["per_query"][i].get(metric, 0.0) for i in ids]
        row = {"a": a, "b": b, "why": why, "metric": metric, **M.paired_test(va, vb)}
        row["by_kind"] = {}
        for kind in ("factual", "methods", "library_filtered", "synthesis"):
            kids = [q.query_id for q in qs.of_kind(kind) if q.query_id in ids]
            if not kids:
                continue
            row["by_kind"][kind] = M.paired_test(
                [per_config[a]["per_query"][i].get(metric, 0.0) for i in kids],
                [per_config[b]["per_query"][i].get(metric, 0.0) for i in kids],
                n_resamples=5000,
            )
        out.append(row)
    return out


def print_table(per_config: dict, arms: List[str]) -> None:
    cols = HEADLINE
    w = max(len(a) for a in arms) + 2
    print("\n" + "=" * (w + 12 * len(cols)))
    print("CURRENT-STATE BASELINE (macro-averaged over the golden set)")
    print("=" * (w + 12 * len(cols)))
    print(f"{'config':<{w}}" + "".join(f"{c:>12}" for c in cols))
    for arm in arms:
        o = per_config[arm]["overall"]
        print(f"{arm:<{w}}" + "".join(f"{o.get(c, 0.0):>12.3f}" for c in cols))


def print_by_kind(qs: QuerySet, per_config: dict, arms: List[str], metric: str) -> None:
    kinds = ("factual", "methods", "library_filtered", "synthesis")
    w = max(len(a) for a in arms) + 2
    print(f"\n{metric} by query kind")
    print("-" * (w + 20 * len(kinds)))
    header = "".join(f"{k + f' (n={len(qs.of_kind(k))})':>20}" for k in kinds)
    print(f"{'config':<{w}}" + header)
    for arm in arms:
        row = "".join(f"{per_config[arm]['by_kind'][k].get(metric, 0.0):>20.3f}" for k in kinds)
        print(f"{arm:<{w}}" + row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=["retrieve", "pool", "judge", "score"])
    ap.add_argument("--arms", nargs="*", default=DEFAULT_ARMS)
    ap.add_argument("--pool-depth", type=int, default=20)
    ap.add_argument("--pool-cap", type=int, default=70)
    ap.add_argument("--max-spend", type=float, default=5.0)
    ap.add_argument("--metric", default="ndcg@10")
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--reliability", type=int, default=0,
                    help="re-judge N random pooled pairs to measure judge self-consistency")
    ap.add_argument("--force-retrieve", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    for noisy in ("httpx", "retrieval", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    qs = load_query_set()
    logger.info("query set %s: %d queries", qs.version, len(qs))

    runs, texts = {}, {}
    if "retrieve" in args.stages:
        runs, texts = stage_retrieve(qs, args.arms, args.force_retrieve)
    else:
        runs = json.loads(RUN_PATH.read_text()) if RUN_PATH.exists() else {}
        texts = json.loads(TEXTS_PATH.read_text()) if TEXTS_PATH.exists() else {}

    pool_meta = None
    if "pool" in args.stages:
        pool_meta = stage_pool(qs, runs, args.arms, args.pool_depth, args.pool_cap)
    elif POOL_PATH.exists():
        pool_meta = json.loads(POOL_PATH.read_text())

    qrels = None
    if "judge" in args.stages:
        qrels = stage_judge(qs, pool_meta, runs, texts, args.max_spend, args.estimate_only)
        if args.estimate_only:
            return
    elif QRELS_PATH.exists():
        qrels = json.loads(QRELS_PATH.read_text())

    reliability = None
    if args.reliability:
        reliability = stage_reliability(qs, pool_meta, runs, texts, args.reliability)
    elif (RESULTS_DIR / "judge_reliability.json").exists():
        reliability = json.loads((RESULTS_DIR / "judge_reliability.json").read_text())

    if "score" not in args.stages:
        return

    per_config = stage_score(qs, runs, texts, qrels, args.arms)
    comparisons = run_comparisons(qs, per_config, args.metric)
    all_comparisons = {
        m: run_comparisons(qs, per_config, m)
        for m in ("ndcg@10", "ndcg@20", "recall@10", "recall@20", "mrr",
                  "paper_recall@10", "term_recall@10")
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = args.out or RESULTS_DIR / f"baseline_{stamp}.json"
    artifact = {
        "artifact": "ARC W5 retrieval baseline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Current-state retrieval baseline captured BEFORE the W1/W2 re-extract "
            "and re-chunk. Re-run `python -m evaluation.run_w5` after the reindex and "
            "compare. Chunk-level numbers are not comparable across a re-chunk "
            "(chunk_ids are regenerated); paper-level numbers are."
        ),
        "query_set": {
            "version": qs.version,
            "n": len(qs),
            "by_kind": qs.composition["by_kind"],
            "source": qs.source,
        },
        "qrels": {
            "version": qrels["version"],
            "created_at": qrels["created_at"],
            "pool": qrels["pool"],
            "judge": qrels["judge"],
            "totals": {
                "judged_pairs": sum(v["n_judged"] for v in qrels["queries"].values()),
                "relevant_chunks": sum(v["n_rel_chunks"] for v in qrels["queries"].values()),
                "directly_relevant_chunks": sum(v["n_rel2_chunks"] for v in qrels["queries"].values()),
                "relevant_papers": sum(v["n_rel_papers"] for v in qrels["queries"].values()),
            },
            "judge_reliability": reliability,
        },
        "corpus": corpus_snapshot(),
        "configs": {a: per_config[a]["config"] for a in args.arms},
        "results": {
            a: {
                "overall": per_config[a]["overall"],
                "by_kind": per_config[a]["by_kind"],
                "route_distribution": per_config[a]["route_distribution"],
            }
            for a in args.arms
        },
        "comparisons": comparisons,
        "comparisons_by_metric": all_comparisons,
        "per_query": {a: per_config[a]["per_query"] for a in args.arms},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=1))

    print_table(per_config, args.arms)
    print_by_kind(qs, per_config, args.arms, args.metric)

    print(f"\nPAIRED COMPARISONS on {args.metric} "
          f"(two-sided randomisation test, 95% bootstrap CI)")
    print("-" * 110)
    for c in comparisons:
        sig = "SIGNIFICANT" if c["p_value"] < 0.05 else "n.s."
        print(f"{c['a']:>20} - {c['b']:<20} "
              f"diff={c['mean_diff']:+.4f} "
              f"CI[{c['ci_low']:+.4f},{c['ci_high']:+.4f}] "
              f"p={c['p_value']:.4f} {sig:<12} "
              f"W/L/T={c['wins']}/{c['losses']}/{c['ties']}")
        print(f"{'':>20}   {c['why']}")

    print(f"\nBaseline artifact: {out_path}")
    print(f"Qrels: {QRELS_PATH}")


if __name__ == "__main__":
    main()
