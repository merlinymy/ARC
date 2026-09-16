"""Retrieval runner for the W5 harness.

Executes a named `RetrievalConfig` over the golden query set and returns ranked
chunk lists. Retrieval only - `QueryEngine.query()` is never called, so no
answer is ever generated.

Fidelity: the runner builds one real `QueryEngine` and drives its own
sub-components (`query_rewriter`, `expander`, `classifier`, `hyde_embedder`,
`embedder`, `store`, `reranker`, `_detect_targeted_query_type`). Nothing in
`retrieval/` is modified or re-implemented, except sparse-only search, which
`QdrantStore` has no method for - that one path issues the equivalent
`query_points(using="bm25")` call directly.

Deterministic-by-cache: every LLM and embedding call is cached to disk keyed by
its input, so a re-run costs nothing and produces identical rankings even though
HyDE and the classifier are non-deterministic.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client.models import (  # noqa: E402
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    SparseVector as QdrantSparseVector,
)

from config import settings  # noqa: E402
from evaluation.configs import RetrievalConfig  # noqa: E402
from evaluation.query_set import GoldenQuery  # noqa: E402

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent / "cache"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _DiskCache:
    """Tiny JSON/pickle-backed dict that flushes on write."""

    def __init__(self, path: Path, binary: bool = False):
        self.path = path
        self.binary = binary
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if binary:
                with open(path, "rb") as fh:
                    self.data = pickle.load(fh)
            else:
                self.data = json.loads(path.read_text())
        else:
            self.data = {}
        self._dirty = False

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        if self.binary:
            with open(tmp, "wb") as fh:
                pickle.dump(self.data, fh)
        else:
            tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False))
        tmp.replace(self.path)
        self._dirty = False


class _RateLimiter:
    """Token bucket for the Cohere rerank endpoint.

    The key in backend/.env is a Cohere **Trial** key, capped at 40 rerank calls
    per minute. Without this the harness spends most of its wall time in 429
    backoff, and a long enough burst exhausts the retries - at which point
    `CohereReranker.rerank` returns the *unreranked* candidate list, silently
    substituting raw retrieval order for a reranked ranking. That would be
    invalid data, so the limiter is a correctness measure, not an optimisation.
    """

    def __init__(self, calls: int, per_seconds: float):
        self.calls = calls
        self.per = per_seconds
        self._times: List[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._times = [t for t in self._times if now - t < self.per]
                if len(self._times) < self.calls:
                    self._times.append(now)
                    return
                wait = self.per - (now - self._times[0]) + 0.05
            time.sleep(max(wait, 0.05))


def _build_filter(
    chunk_types: Optional[List[str]],
    section_names: Optional[List[str]],
) -> Optional[Filter]:
    """Mirror of QdrantStore.search's filter construction, for the sparse path."""
    conditions = []
    for key, values in (("chunk_type", chunk_types), ("section_name", section_names)):
        if not values:
            continue
        if len(values) == 1:
            conditions.append(FieldCondition(key=key, match=MatchValue(value=values[0])))
        else:
            conditions.append(FieldCondition(key=key, match=MatchAny(any=values)))
    return Filter(must=conditions) if conditions else None


class RetrievalRunner:
    """Runs retrieval arms over the golden set."""

    def __init__(self, cache_dir: Path = CACHE_DIR, verbose: bool = True):
        from anthropic import Anthropic
        from retrieval.bm25 import BM25Vectorizer
        from retrieval.embedder import VoyageEmbedder
        from retrieval.qdrant_store import QdrantStore
        from retrieval.query_engine import QueryEngine
        from retrieval.reranker import CohereReranker

        self.verbose = verbose
        self.cache_dir = cache_dir

        bm25 = BM25Vectorizer()
        if not bm25.load_idf_cache():
            raise SystemExit("BM25 IDF cache failed to load - sparse arms would be meaningless")

        self.embedder = VoyageEmbedder(api_key=settings.voyage_api_key)
        self.reranker = CohereReranker(api_key=settings.cohere_api_key)
        self.store = QdrantStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection_name,
            embedding_dimension=settings.embedding_dimension,
            bm25_vectorizer=bm25,
        )
        self.bm25 = bm25
        self.anthropic = Anthropic(api_key=settings.anthropic_api_key)

        # One real QueryEngine, used purely as a component container. Classification
        # is forced on so `self.engine.classifier` exists; the keyword route calls
        # `_detect_targeted_query_type` directly, which is what the live server
        # (ENABLE_QUERY_CLASSIFICATION=false) does.
        self.engine = QueryEngine(
            embedder=self.embedder,
            reranker=self.reranker,
            store=self.store,
            anthropic_client=self.anthropic,
            claude_model=settings.claude_model,
            claude_model_fast=settings.claude_model_fast,
            claude_model_classifier=settings.claude_model_classifier,
            enable_classification=True,
            enable_expansion=settings.enable_query_expansion,
            enable_caching=False,
            enable_hyde=True,
            enable_query_rewriting=True,
            enable_entity_extraction=False,
            enable_citation_verification=False,
            enable_conversation_memory=False,
            enable_hybrid_search=settings.enable_hybrid_search,
        )

        self.c_query = _DiskCache(cache_dir / "query_artifacts.json")
        self.c_hyde = _DiskCache(cache_dir / "hyde_docs.json")
        self.c_embed = _DiskCache(cache_dir / "embeddings.pkl", binary=True)

        self.llm_calls: Dict[str, int] = {"classifier": 0, "hyde": 0}
        self.rerank_calls = 0
        self.rerank_docs = 0
        self.rerank_retries = 0
        # Cohere Trial keys allow 40 rerank calls/min; leave headroom for the
        # live backend, which shares the same key.
        self.limiter = _RateLimiter(calls=30, per_seconds=60.0)

    # -- cached primitives ------------------------------------------------

    def _embed(self, text: str) -> List[float]:
        key = _sha(text)
        hit = self.c_embed.get(key)
        if hit is not None:
            return hit
        vec = self.embedder.embed_query(text)
        self.c_embed.set(key, vec)
        self.c_embed.flush()
        return vec

    def _artifacts(self, q: GoldenQuery) -> Dict[str, Any]:
        """Rewrite, expansion, classifier route and keyword route for one query."""
        hit = self.c_query.get(q.query_id)
        if hit is not None:
            return hit

        rewritten = self.engine.query_rewriter.rewrite(q.text).rewritten
        expanded, added = self.engine.expander.expand_query(rewritten)

        cls = self.engine.classifier.classify(rewritten)
        self.llm_calls["classifier"] += 1

        kw = self.engine._detect_targeted_query_type(rewritten)

        art = {
            "rewritten": rewritten,
            "expanded": expanded,
            "expansion_terms": added,
            "classifier_type": cls.query_type.value,
            "classifier_confidence": cls.confidence,
            "classifier_reasoning": cls.reasoning,
            "keyword_type": kw.value,
        }
        self.c_query.set(q.query_id, art)
        self.c_query.flush()
        return art

    def _hyde_embed(self, query_text: str, query_type: Optional[str], key: str) -> Tuple[List[float], str]:
        cached = self.c_hyde.get(key)
        if cached is None:
            hypothetical, _ = self.engine.hyde_embedder.hyde.generate_hypothetical(
                query_text, query_type
            )
            self.llm_calls["hyde"] += 1
            cached = hypothetical
            self.c_hyde.set(key, cached)
            self.c_hyde.flush()
        combined = f"Query: {query_text}\n\nRelevant excerpt: {cached}"
        return self._embed(combined), combined

    # -- search paths -----------------------------------------------------

    def _sparse_search(
        self,
        text: str,
        limit: int,
        chunk_types: Optional[List[str]],
        section_names: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        vec = self.bm25.vectorize(text, is_query=True)
        if not vec.indices:
            return []
        res = self.store.client.query_points(
            collection_name=self.store.collection_name,
            query=QdrantSparseVector(indices=vec.indices, values=vec.values),
            using="bm25",
            limit=limit,
            query_filter=_build_filter(chunk_types, section_names),
        )
        return [{"score": p.score, **p.payload} for p in res.points]

    def _rerank(self, query: str, docs: List[Dict[str, Any]], top_n: int, max_per_paper: int):
        last = None
        for attempt in range(8):
            self.limiter.acquire()
            res = self.reranker.rerank_with_metadata(
                query=query, documents=docs, top_n=top_n, max_per_paper=max_per_paper
            )
            self.rerank_calls += 1
            self.rerank_docs += len(docs)
            if res.success:
                return res
            last = res
            self.rerank_retries += 1
            time.sleep(min(5 * (attempt + 1), 35))
        logger.error("rerank failed after 8 attempts: %s", last.error if last else "?")
        return last

    # -- main entry point -------------------------------------------------

    def run(self, cfg: RetrievalConfig, q: GoldenQuery) -> Dict[str, Any]:
        from retrieval.query_classifier import RETRIEVAL_STRATEGIES, QueryType

        t0 = time.perf_counter()
        art = self._artifacts(q)

        route_type: Optional[str] = None
        chunk_types: Optional[List[str]] = None
        section_filter: Optional[List[str]] = None
        top_k = cfg.top_k
        rerank_top_n = cfg.rerank_top_n
        max_per_paper = cfg.max_per_paper

        if cfg.route == "classifier":
            route_type = art["classifier_type"]
        elif cfg.route == "keyword":
            route_type = art["keyword_type"]

        if route_type is not None:
            strategy = RETRIEVAL_STRATEGIES[QueryType(route_type)]
            if cfg.apply_filters:
                chunk_types = strategy["chunk_types"]
                section_filter = strategy.get("section_filter")
            if cfg.use_strategy_sizes:
                top_k = strategy["top_k"]
                rerank_top_n = strategy["rerank_top_n"]
                max_per_paper = strategy.get("max_per_paper", 3)

        search_text = art["expanded"] if cfg.use_expansion else (
            art["rewritten"] if cfg.use_rewrite else q.text
        )

        # Embedding: HyDE substitutes a generated excerpt, exactly as production does.
        if cfg.use_hyde:
            hyde_key = f"{q.query_id}|{route_type or 'none'}"
            embedding, embedded_text = self._hyde_embed(search_text, route_type, hyde_key)
        else:
            embedding = self._embed(search_text)
            embedded_text = search_text

        t_search = time.perf_counter()
        if cfg.search_mode == "dense":
            candidates = self.store.search(
                query_embedding=embedding,
                limit=top_k,
                chunk_types=chunk_types,
                section_names=section_filter,
            )
        elif cfg.search_mode == "sparse":
            candidates = self._sparse_search(search_text, top_k, chunk_types, section_filter)
        elif cfg.search_mode == "hybrid":
            candidates = self.store.hybrid_search(
                query=search_text,
                query_embedding=embedding,
                limit=top_k,
                chunk_types=chunk_types,
                section_names=section_filter,
            )
        else:
            raise ValueError(f"unknown search_mode {cfg.search_mode}")
        search_ms = (time.perf_counter() - t_search) * 1000

        n_candidates = len(candidates)
        if candidates:
            res = self._rerank(q.text, candidates, rerank_top_n, max_per_paper)
            ranked = res.documents
            rerank_ok = res.success
        else:
            ranked, rerank_ok = [], True

        results = [
            {
                "rank": i + 1,
                "chunk_id": d.get("chunk_id") or d.get("_chunk_id"),
                "paper_id": d.get("paper_id"),
                "chunk_type": d.get("chunk_type"),
                "section_name": d.get("section_name"),
                "title": d.get("title"),
                "score": d.get("score"),
                "rerank_score": d.get("rerank_score"),
                "text": d.get("text", ""),
            }
            for i, d in enumerate(ranked)
        ]

        return {
            "query_id": q.query_id,
            "config": cfg.name,
            "route_type": route_type,
            "chunk_types": chunk_types,
            "section_filter": section_filter,
            "top_k": top_k,
            "rerank_top_n": rerank_top_n,
            "max_per_paper": max_per_paper,
            "used_hyde": cfg.use_hyde,
            "embedded_text_chars": len(embedded_text),
            "n_candidates": n_candidates,
            "n_results": len(results),
            "rerank_ok": rerank_ok,
            "search_ms": round(search_ms, 1),
            "total_ms": round((time.perf_counter() - t0) * 1000, 1),
            "results": results,
        }
