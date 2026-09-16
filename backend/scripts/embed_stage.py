#!/usr/bin/env python
"""Stage 2 of the reindex: paper record -> chunks -> vectors -> Qdrant.

    ~1 s/paper of local work + the embedding API bill, and **free to redo**.

It reads only ``processed_data/{id}.json``.  No PDF is opened, MinerU never
runs, so re-chunking or changing embedding model costs minutes and single-digit
dollars instead of another four days of GPU time.  That is the whole reason
stage 1 exists separately.

Safety, in the order it is enforced (all at startup, never at hour three)
------------------------------------------------------------------------
1. The model and the collection arrive together as a ``config.EmbeddingProfile``
   (§3b.8).  A collection belonging to a *different* profile is refused: a
   query vector from one model against document vectors from another is
   silently wrong retrieval, not an error.
2. A target collection that already holds points is refused unless ``--append``.
   That is what keeps the live 212,953-point index serving while its
   replacement is built next to it; the cutover is then a profile switch.
3. ``voyage-context-*`` requires the contextualized embedder API.  Each paper's
   chunks are sent **nested, per paper, in order**, which is what makes the
   vectors document-aware; flattening every paper's chunks into one list would
   throw away the entire benefit of the model.
4. The BM25 IDF cache is loaded read-only and never written.  Document sparse
   vectors carry no corpus statistics (IDF is query-side since the v2 scheme),
   so building a new collection cannot disturb the live one's scoring.  Refresh
   IDF after the cutover with ``scripts/rebuild_bm25_index.py pass1``.

Usage
-----
    # what it would cost, with real chunking and no API calls
    python scripts/embed_stage.py --papers-from data/cohorts/cited_papers.txt --estimate

    # build the new collection for the active profile
    python scripts/embed_stage.py --papers-from data/cohorts/cited_papers.txt

    # everything that has a record but no vectors yet
    python scripts/embed_stage.py --pending
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

BACKEND = Path(__file__).resolve().parents[1]
for p in (str(BACKEND), str(BACKEND / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                          # noqa: E402

import stage_lib as SL                                      # noqa: E402
from config import get_embedding_profile, profile_for_collection, settings  # noqa: E402
from preprocessing import paper_record as pr                # noqa: E402
from preprocessing.chunker import PaperChunker              # noqa: E402
from preprocessing.models import ChunkType                  # noqa: E402

#: $/M tokens, 2026-09-16 (§3b.8).  Override with --price-per-mtok.
PRICE_PER_MTOK = {"voyage-3-large": 0.06, "voyage-context-4": 0.12,
                  "voyage-context-3": 0.12}

#: Corpus means from the plan, used only until this manifest has its own.
PLAN_CHUNKS_PER_PAPER = 229_656 / 4_797      # 47.9
PLAN_TOKENS_PER_PAPER = 104_322_997 / 4_797  # 21,748


# ---------------------------------------------------------------------------
# token counting
# ---------------------------------------------------------------------------

class TokenCounter:
    """Token counts for the money line.

    Prefers the embedder's ``count_tokens`` -- the *model's own* tokenizer,
    which reproduced Voyage's billed count exactly in their measurement -- and
    falls back to cl100k_base inflated by ``TIKTOKEN_INFLATION`` when no
    embedder is around (``--estimate`` on a machine with no API key).
    """

    def __init__(self, embedder=None) -> None:
        self.embedder = embedder if hasattr(embedder, "count_tokens") else None
        self.exact = self.embedder is not None
        self.enc = None
        try:
            from retrieval.embedder import TIKTOKEN_INFLATION
            self.inflation = TIKTOKEN_INFLATION
        except ImportError:
            self.inflation = 1.20

    def count(self, texts: Sequence[str]) -> int:
        if self.embedder is not None:
            try:
                return int(sum(self.embedder.count_tokens(list(texts))))
            except Exception as exc:                        # noqa: BLE001
                print(f"  note              count_tokens failed ({exc}); "
                      f"falling back to tiktoken")
                self.embedder = None
                self.exact = False
        if self.enc is None:
            import tiktoken
            self.enc = tiktoken.get_encoding("cl100k_base")
        return int(sum(len(self.enc.encode(t, disallowed_special=()))
                       for t in texts) * self.inflation)


# ---------------------------------------------------------------------------
# the embedder seam (§3b.8, implemented by the embedder work)
# ---------------------------------------------------------------------------

class PaperEmbedder:
    """One call per paper, whichever API the bound profile calls for.

    §3b.8's contract: ``embed_document_chunks(chunk_texts)`` returns one vector
    per chunk, in input order, with the 32K per-document window handled inside;
    ``embed_paper_batch(papers)`` does several papers in one request and is
    preferred for throughput.  Chunks are always grouped **by paper** -- that
    grouping is what makes the vectors document-aware, and flattening the corpus
    into one list would throw away the entire reason for the model switch.

    Under a non-contextualized profile the embedder's contextual methods raise
    on purpose, so this branches on ``profile.contextualized`` rather than on
    method presence alone.
    """

    def __init__(self, embedder, profile, *, dimension: int):
        self.embedder = embedder
        self.profile = profile
        self.dimension = dimension
        self.batch = getattr(embedder, "embed_paper_batch", None)
        self.per_paper = getattr(embedder, "embed_document_chunks", None)

        if profile.contextualized:
            if not (self.batch or self.per_paper):
                raise SystemExit(
                    f"profile {profile.name!r} is contextualized, but "
                    f"retrieval.embedder.VoyageEmbedder exposes neither "
                    f"embed_document_chunks() nor embed_paper_batch().\n"
                    f"TODO(seam, §3b.8): that method is the embedder work's "
                    f"half of this stage. Until it lands, build with "
                    f"--profile voyage-3-large into a throwaway --collection "
                    f"to exercise the plumbing; do NOT fall back to "
                    f"embed_documents for a context model -- client.embed() "
                    f"400s on it, and a flat vector in a contextualized "
                    f"collection is the silent mismatch this profile exists "
                    f"to prevent."
                )
            self.mode = "paper_batch" if self.batch else "document_chunks"
        else:
            # voyage-3-large and friends: contextualized_embed refuses them, so
            # the flat endpoint is correct here, not a degradation.
            self.mode = "flat"

    def embed_papers(self, papers: List[List[str]],
                     paper_ids: Optional[Sequence[str]] = None
                     ) -> List[List[List[float]]]:
        """Embed several papers; each paper's chunks stay together and in order."""
        if self.mode == "paper_batch":
            out = self.batch(papers, paper_ids=list(paper_ids) if paper_ids else None)
        elif self.mode == "document_chunks":
            out = [self.per_paper(texts, paper_id=(paper_ids[i] if paper_ids else None))
                   for i, texts in enumerate(papers)]
        else:
            out = [self.embedder.embed_documents(texts) for texts in papers]
        for texts, vectors in zip(papers, out):
            if len(vectors) != len(texts):
                raise RuntimeError(
                    f"embedder returned {len(vectors)} vectors for "
                    f"{len(texts)} chunks ({self.mode})")
            if vectors and len(vectors[0]) != self.dimension:
                raise RuntimeError(
                    f"embedder returned dim {len(vectors[0])}, collection "
                    f"expects {self.dimension} ({self.mode})")
        return list(out)


def pooled_full_vector(chunks, vectors, pool=None) -> Optional[List[float]]:
    """Mean-pool the abstract/section vectors we already paid for.

    ``index_papers.py`` called ``compute_mean_pooled_embedding`` here, which
    re-embedded every abstract and section chunk a second time -- roughly a
    quarter of the corpus embedded twice, for a vector that is the mean of
    vectors already in hand.  Pooling locally is identical in meaning, free,
    and keeps the paper-level point consistent with its own chunks (including
    their document-aware context).
    """
    keep = [i for i, c in enumerate(chunks)
            if c.chunk_type in (ChunkType.ABSTRACT, ChunkType.SECTION)]
    if not keep:
        return None
    picked = [vectors[i] for i in keep]
    if pool is not None:
        return pool(picked)
    stacked = np.asarray(picked, dtype=np.float64)
    pooled = stacked.mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    if norm > 0:
        pooled = pooled / norm
    return pooled.tolist()


# ---------------------------------------------------------------------------
# selection and planning
# ---------------------------------------------------------------------------

def ids_from_tokens(tokens: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for token in tokens:
        if SL.looks_like_paper_id(token):
            pid = token
        else:
            name = Path(token).name
            name = name if name.lower().endswith(".pdf") else f"{name}.pdf"
            pid = pr.paper_id_for_filename(name)
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def records_on_disk(records_base: Optional[Path]) -> List[str]:
    directory = pr.records_dir(records_base)
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.json"))


def embed_state(paper_id: str, manifest: SL.StageManifest,
                records_base: Optional[Path], *, allow_stale_records: bool
                ) -> Tuple[str, str, Dict[str, Any]]:
    """(action, reason, record_facts) for one paper, using stats not reads.

    action: embed | skip | no_record | stale_record
    """
    record_path = pr.record_path(paper_id, records_base)
    if not record_path.exists():
        return "no_record", "no record on disk", {}

    facts = SL.record_file_facts(record_path)
    entry = manifest.get(paper_id) or {}
    prev = entry.get("record_fp") or {}
    if entry.get("status") == "ok" and prev:
        if (prev.get("record_bytes") == facts.get("record_bytes")
                and prev.get("record_mtime") == facts.get("record_mtime")):
            return "skip", "already embedded from this record", facts
        return "embed", "record changed since it was embedded", facts

    if not allow_stale_records:
        # Cheap guard against embedding a schema-1 record into a new collection:
        # read only the head of the file rather than the whole thing.
        record = pr.load_record(paper_id, records_base)
        if record is None:
            return "no_record", "record unreadable", facts
        if record.get("schema_version") != SL.CURRENT_SCHEMA:
            return ("stale_record",
                    f"schema_version {record.get('schema_version')} != {SL.CURRENT_SCHEMA}",
                    facts)
        if not (record.get("blocks") or []):
            return "stale_record", "record has no blocks", facts
    prev_status = entry.get("status")
    if not prev_status:
        reason = "no vectors yet"
    elif prev_status == "estimated":
        reason = "estimated only, never embedded"
    else:
        reason = f"previous attempt ended as {prev_status}"
    return "embed", reason, facts


# ---------------------------------------------------------------------------
# qdrant target
# ---------------------------------------------------------------------------

def open_target(args, profile) -> Tuple[Any, str, int]:
    from retrieval.bm25 import BM25Vectorizer
    from retrieval.qdrant_store import QdrantStore

    collection = args.collection or profile.collection
    owner = profile_for_collection(collection)
    if owner is not None and owner.name != profile.name:
        raise SystemExit(
            f"collection {collection!r} belongs to profile {owner.name!r} "
            f"(model {owner.model!r}), but this run embeds with "
            f"{profile.model!r}. Mixing models in one collection is silently "
            f"wrong retrieval; refusing."
        )

    bm25 = None
    if not args.no_hybrid:
        bm25 = BM25Vectorizer()
        loaded = bm25.load_idf_cache()      # read-only, never saved back
        print(f"  bm25              sparse vectors on "
              f"(IDF cache {'loaded read-only' if loaded else 'absent'}; "
              f"document vectors carry no IDF)")

    store = QdrantStore(host=settings.qdrant_host, port=settings.qdrant_port,
                        collection_name=collection,
                        embedding_dimension=profile.dimension,
                        enable_hybrid=not args.no_hybrid,
                        bm25_vectorizer=bm25)

    existing = {c.name for c in store.client.get_collections().collections}
    points = 0
    if collection in existing:
        info = store.client.get_collection(collection)
        points = info.points_count or 0
        params = info.config.params.vectors
        if isinstance(params, dict):            # named-vector collections
            params = params.get("") or next(iter(params.values()), None)
        size = getattr(params, "size", None)
        if size and size != profile.dimension:
            raise SystemExit(f"collection {collection!r} has dim {size}, profile "
                             f"{profile.name!r} produces {profile.dimension}")
        if points and not args.append:
            raise SystemExit(
                f"collection {collection!r} already holds {points:,} points.\n"
                f"Refusing to write into a populated collection -- this is the "
                f"guard that keeps the live index serving while its replacement "
                f"is built. Pass --append if adding to it is what you meant, or "
                f"--collection <new name> to build beside it."
            )
    return store, collection, points


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Stage 2: chunk, embed and upsert from persisted records.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage\n-----\n")[-1],
    )
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--papers-from", metavar="FILE")
    sel.add_argument("--paper", action="append", metavar="ID|NAME")
    sel.add_argument("--all-records", action="store_true",
                     help="every record in the records dir")
    sel.add_argument("--pending", action="store_true",
                     help="same as --all-records; current ones are skipped anyway")
    sel.add_argument("--retry-failed", action="store_true")

    ap.add_argument("--profile", default=settings.embedding_profile,
                    help=f"embedding profile: model + dimension + collection as one "
                         f"unit (default: {settings.embedding_profile})")
    ap.add_argument("--collection", default=None,
                    help="override the profile's collection. Only for throwaway "
                         "test collections; another profile's collection is refused.")
    ap.add_argument("--records-dir", default=None)
    ap.add_argument("--manifest", default=None,
                    help="default: data/stage_manifests/embed__<collection>.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--abort-after-errors", type=int, default=5,
                    help="stop after this many consecutive failures (an expired "
                         "key would otherwise 'fail' the whole cohort in a minute)")
    ap.add_argument("--papers-per-request", type=int, default=4,
                    help="papers packed into one embedding call under a "
                         "contextualized profile (their 120K-token per-request "
                         "ceiling is handled inside the embedder); chunks always "
                         "stay grouped by paper, which is what makes the vectors "
                         "document-aware")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan only: no chunking, no API calls")
    ap.add_argument("--estimate", action="store_true",
                    help="chunk for real and count tokens, but do not embed or "
                         "upsert. $0, and gives a measured cost projection.")
    ap.add_argument("--append", action="store_true",
                    help="allow writing into a collection that already has points")
    ap.add_argument("--no-hybrid", action="store_true",
                    help="skip the bm25 sparse vector")
    ap.add_argument("--allow-stale-records", action="store_true",
                    help="embed records whose schema_version is not current")
    ap.add_argument("--no-verify-spans", action="store_true",
                    help="skip W2's full_text[char_start:char_end] == text check")
    ap.add_argument("--price-per-mtok", type=float, default=None)
    ap.add_argument("--verbose", action="store_true",
                    help="keep the library loggers at INFO")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--expect-cohort", type=int, default=0)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    SL.configure_logging(args.verbose)
    profile = get_embedding_profile(args.profile)
    records_base = Path(args.records_dir).resolve() if args.records_dir else None
    collection_guess = args.collection or profile.collection
    manifest_path = Path(args.manifest) if args.manifest else \
        SL.MANIFEST_DIR / f"embed__{collection_guess}.json"

    price = args.price_per_mtok if args.price_per_mtok is not None \
        else PRICE_PER_MTOK.get(profile.model, 0.12)

    print(SL.rule("EMBED STAGE (stage 2 of 2: record -> chunks -> Qdrant)"))
    print(f"  profile           {profile.name}  model={profile.model} "
          f"dim={profile.dimension} contextualized={profile.contextualized}")
    print(f"  records dir       {pr.records_dir(records_base)}")

    manifest, soft_changes = SL.open_manifest(
        manifest_path,
        signature={"stage": "embed", "collection": collection_guess,
                   "embedding_model": profile.model,
                   "embedding_dimension": profile.dimension,
                   "records_dir": str(pr.records_dir(records_base)),
                   "record_schema_version": SL.CURRENT_SCHEMA,
                   "chunk_params": {
                       "abstract_max_tokens": settings.abstract_max_tokens,
                       "section_max_tokens": settings.section_max_tokens,
                       "fine_chunk_tokens": settings.fine_chunk_tokens,
                       "fine_chunk_overlap": settings.fine_chunk_overlap}},
        # A manifest claiming "done" for another collection, model or dimension
        # is a lie that would skip real work; stop at startup instead.
        # records_dir and chunk_params are deliberately soft: moving the records
        # or retuning the chunker does not un-embed what is already in the
        # collection, it just means the next run will differ -- reported, not fatal.
        hard_keys=("stage", "collection", "embedding_model", "embedding_dimension"),
        save_every=args.save_every,
    )
    for change in soft_changes:
        print(f"  note              manifest signature changed -- {change}")
    if manifest.replayed or manifest.truncated_lines:
        print(f"  recovered {manifest.replayed} journal entries"
              f"{f', discarded {manifest.truncated_lines} truncated line(s)' if manifest.truncated_lines else ''}")

    # ---- selection ----------------------------------------------------
    if args.papers_from:
        ids = ids_from_tokens(SL.read_selection_file(Path(args.papers_from)))
    elif args.paper:
        ids = ids_from_tokens(args.paper)
    elif args.retry_failed:
        ids = sorted(set(manifest.ids_with_status("error", "failed", "stale_record",
                                                  "no_record")))
    else:
        ids = records_on_disk(records_base)

    if args.expect_cohort and len(ids) != args.expect_cohort:
        raise SystemExit(f"selection produced {len(ids)} papers, expected "
                         f"{args.expect_cohort}; refusing to continue")

    todo: List[str] = []
    buckets: Dict[str, int] = {}
    reasons: Dict[str, int] = {}
    to_record: List[Tuple[str, str, str, Dict[str, Any]]] = []
    for pid in ids:
        action, reason, facts = embed_state(pid, manifest, records_base,
                                            allow_stale_records=args.allow_stale_records)
        buckets[action] = buckets.get(action, 0) + 1
        if action == "embed":
            todo.append(pid)
            reasons[reason] = reasons.get(reason, 0) + 1
        elif action in ("no_record", "stale_record"):
            # Written to the manifest only if this run is real -- "what is stale
            # and why" is worth persisting, a dry run's opinion is not.
            to_record.append((pid, action, reason, facts))
    if args.limit:
        todo = todo[:args.limit]

    # An --estimate run measures chunks and tokens without spending anything, so
    # its numbers are as good as a real run's for projecting the next cohort.
    done = ("ok", "estimated")
    n_done = len(manifest.ids_with_status(*done))
    measured = n_done > 0
    mean_chunks = (manifest.sum_field("chunks", done) / n_done) if measured \
        else PLAN_CHUNKS_PER_PAPER
    mean_tokens = (manifest.sum_field("tokens", done) / n_done) if measured \
        else PLAN_TOKENS_PER_PAPER

    print(SL.rule("PLAN"))
    print(f"  selected          {len(ids)}   ({SL.counter_line(buckets)})")
    print(f"  to embed          {len(todo)}")
    if reasons:
        print(f"      why           {SL.counter_line(reasons)}")
    proj_tokens = mean_tokens * len(todo)
    print(f"  projection        ~{mean_chunks:.1f} chunks/paper, "
          f"~{mean_tokens:,.0f} tokens/paper "
          f"({f'measured, n={n_done}' if measured else 'plan corpus mean'})")
    print(f"                    ~{proj_tokens / 1e6:.2f}M tokens "
          f"-> ${proj_tokens / 1e6 * price:.2f} at ${price}/M ({profile.model})")
    if len(ids) != len(todo):
        whole = mean_tokens * len(ids)
        print(f"  whole selection   {len(ids)} papers would be "
              f"~{whole / 1e6:.2f}M tokens -> ${whole / 1e6 * price:.2f} "
              f"once they all have records")
    print(SL.rule())

    if args.dry_run:
        print("dry run: nothing chunked, nothing embedded")
        return 0
    for pid, action, reason, facts in to_record:
        manifest.mark(pid, action, reason=reason, **facts)
    if not todo:
        manifest.snapshot()
        print("nothing to do")
        return 0

    # ---- components ---------------------------------------------------
    chunker = PaperChunker(
        abstract_max_tokens=settings.abstract_max_tokens,
        section_max_tokens=settings.section_max_tokens,
        fine_chunk_tokens=settings.fine_chunk_tokens,
        fine_chunk_overlap=settings.fine_chunk_overlap,
    )
    store = collection = None
    paper_embedder = None
    embedder = None
    if not args.estimate:
        store, collection, existing_points = open_target(args, profile)
        print(f"  target            {collection} "
              f"({existing_points:,} points now{', appending' if existing_points else ''})")
        from retrieval.embedder import VoyageEmbedder
        embedder = VoyageEmbedder(api_key=settings.voyage_api_key, profile=profile)
        paper_embedder = PaperEmbedder(embedder, profile, dimension=profile.dimension)
        print(f"  embedder          {paper_embedder.mode}")
        store.ensure_collection()
    else:
        print("  estimate mode     chunking only, no API calls, no Qdrant writes")
        try:
            from retrieval.embedder import VoyageEmbedder
            # Constructed only to count tokens: `count_tokens` runs a local
            # tokenizer, so this spends nothing.
            embedder = VoyageEmbedder(api_key=settings.voyage_api_key, profile=profile)
        except Exception as exc:                            # noqa: BLE001
            print(f"  note              no embedder for token counts ({exc})")

    counter = TokenCounter(embedder)
    print(f"  token counts      {'exact (model tokenizer)' if counter.exact else 'tiktoken x' + str(counter.inflation)}")

    # ---- run ----------------------------------------------------------
    counts: Dict[str, int] = {}
    consecutive_errors = 0
    tot = {"chunks": 0, "tokens": 0, "verbatim": 0, "verbatim_ok": 0}
    t_start = time.time()
    interrupted = False

    def _on_term(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, _on_term)

    group = max(1, args.papers_per_request)
    try:
        for start in range(0, len(todo), group):
            batch_ids = todo[start:start + group]
            prepared: List[Dict[str, Any]] = []
            for pid in batch_ids:
                t0 = time.time()
                record = pr.load_record(pid, records_base)
                if record is None:
                    manifest.mark(pid, "no_record", reason="disappeared mid-run")
                    counts["no_record"] = counts.get("no_record", 0) + 1
                    continue
                chunks = chunker.chunk_record(record)
                if not chunks:
                    manifest.mark(pid, "no_chunks",
                                  record_fp=SL.fingerprint(record, None,
                                                           pr.record_path(pid, records_base)))
                    counts["no_chunks"] = counts.get("no_chunks", 0) + 1
                    continue
                texts = [c.embed_text for c in chunks]
                spans = {"verbatim": 0, "verbatim_ok": 0}
                if not args.no_verify_spans:
                    doc = pr.full_text(record)
                    for c in chunks:
                        if c.text_is_verbatim and c.char_start is not None:
                            spans["verbatim"] += 1
                            if doc[c.char_start:c.char_end] == c.text:
                                spans["verbatim_ok"] += 1
                prepared.append({"paper_id": pid, "record": record, "chunks": chunks,
                                 "texts": texts, "spans": spans, "t0": t0,
                                 "tokens": counter.count(texts)})

            if not prepared:
                continue

            vectors_per_paper: List[Optional[List[List[float]]]] = [None] * len(prepared)
            if not args.estimate:
                try:
                    vectors_per_paper = paper_embedder.embed_papers(
                        [p["texts"] for p in prepared],
                        [p["paper_id"] for p in prepared])
                except KeyboardInterrupt:
                    raise
                except Exception as exc:                    # noqa: BLE001
                    # One bad paper must not fail the papers riding with it, so
                    # a failed group is retried one paper at a time.
                    if len(prepared) == 1:
                        raise
                    print(f"  retry     group of {len(prepared)} failed "
                          f"({type(exc).__name__}: {exc}); retrying per paper",
                          flush=True)
                    vectors_per_paper = []
                    for item in prepared:
                        try:
                            vectors_per_paper.append(paper_embedder.embed_papers(
                                [item["texts"]], [item["paper_id"]])[0])
                        except KeyboardInterrupt:
                            raise
                        except Exception as inner:          # noqa: BLE001
                            vectors_per_paper.append(None)
                            manifest.mark(item["paper_id"], "error",
                                          error=f"{type(inner).__name__}: {inner}")
                            counts["error"] = counts.get("error", 0) + 1
                            consecutive_errors += 1
                            print(f"  error     {item['paper_id']} "
                                  f"{type(inner).__name__}: {inner}", flush=True)

            for item, vectors in zip(prepared, vectors_per_paper):
                pid = item["paper_id"]
                chunks = item["chunks"]
                if vectors is None and not args.estimate:
                    continue
                try:
                    n_vectors = 0
                    points = chunks
                    if not args.estimate:
                        full = chunker.make_full_chunk(chunks)
                        pooled = pooled_full_vector(
                            chunks, vectors,
                            pool=getattr(embedder, "pool_vectors", None))
                        if full is not None and pooled is not None:
                            # The paper-level point is mean-pooled from vectors we
                            # already have, so it costs no tokens -- which is why
                            # `chunks` (billed) and `points` (stored) differ by one.
                            points = list(chunks) + [full]
                            vectors = list(vectors) + [pooled]
                        store.upsert_chunks([c.chunk_id for c in points], vectors,
                                            [c.to_payload() for c in points])
                        n_vectors = len(vectors)

                    row = {
                        "chunks": len(chunks),
                        "points": len(points) if not args.estimate else 0,
                        "vectors": n_vectors,
                        "tokens": item["tokens"],
                        "seconds": round(time.time() - item["t0"], 2),
                        "verbatim_chunks": item["spans"]["verbatim"],
                        "verbatim_ok": item["spans"]["verbatim_ok"],
                        "by_type": {},
                        "collection": collection,
                        "model": profile.model,
                        "record_fp": SL.fingerprint(
                            item["record"], None, pr.record_path(pid, records_base)),
                    }
                    for c in points:
                        key = c.chunk_type.value
                        row["by_type"][key] = row["by_type"].get(key, 0) + 1
                    status = "estimated" if args.estimate else "ok"
                    manifest.mark(pid, status, **row)
                    counts[status] = counts.get(status, 0) + 1
                    tot["chunks"] += row["chunks"]
                    tot["tokens"] += row["tokens"]
                    tot["verbatim"] += row["verbatim_chunks"]
                    tot["verbatim_ok"] += row["verbatim_ok"]
                    consecutive_errors = 0
                    bad = row["verbatim_chunks"] - row["verbatim_ok"]
                    print(f"  {status:9s} {pid} chunks={row['chunks']:4d} "
                          f"tok={row['tokens'] / 1000:6.1f}k "
                          f"spans={row['verbatim_ok']}/{row['verbatim_chunks']}"
                          f"{'  SPAN MISMATCH' if bad else ''} "
                          f"{row['seconds']:5.2f}s", flush=True)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:                    # noqa: BLE001
                    manifest.mark(pid, "error", error=f"{type(exc).__name__}: {exc}",
                                  traceback=traceback.format_exc(limit=4))
                    counts["error"] = counts.get("error", 0) + 1
                    consecutive_errors += 1
                    print(f"  error     {pid} {type(exc).__name__}: {exc}", flush=True)

            if consecutive_errors >= args.abort_after_errors:
                # An expired key or a depleted balance fails every paper in
                # milliseconds; without this the whole cohort would be marked
                # failed in a minute and the real cause buried in the middle.
                print(f"\n{consecutive_errors} consecutive failures -- stopping. "
                      f"Fix the cause and re-run the same command; finished "
                      f"papers are skipped.", flush=True)
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted -- snapshotting manifest", flush=True)
    finally:
        manifest.stats.setdefault("embed", {})
        manifest.stats["embed"].update({
            "last_run_at": SL.utc_now(),
            "last_run_seconds": round(time.time() - t_start, 1),
            "last_run_counts": counts,
            "collection": collection,
            "model": profile.model,
        })
        manifest.close()

    elapsed = time.time() - t_start
    done = counts.get("ok", 0) + counts.get("estimated", 0)
    spend = tot["tokens"] / 1e6 * price
    print(SL.rule("EMBED DONE"))
    print(f"  this run          {SL.counter_line(counts) or 'nothing'}"
          f"{'  (INTERRUPTED)' if interrupted else ''}")
    print(f"  chunks            {tot['chunks']:,}"
          + (f"  ({tot['chunks'] / done:.1f}/paper)" if done else ""))
    print(f"  tokens            {tot['tokens']:,}"
          f"  ({'model tokenizer, billable' if counter.exact else 'tiktoken x' + str(counter.inflation) + ', estimate'})")
    print(f"  cost{'  (would be)  ' if args.estimate else '              '}"
          f"${spend:.2f} at ${price}/M {profile.model}")
    print(f"  W2 span check     {tot['verbatim_ok']}/{tot['verbatim']} verbatim chunks "
          f"byte-exact against full_text")
    print(f"  wall time         {SL.fmt_duration(elapsed)}"
          + (f"  ({elapsed / done:.2f} s/paper)" if done else ""))
    if store is not None:
        info = store.client.get_collection(collection)
        print(f"  collection        {collection}: {info.points_count:,} points")
    print(f"  manifest          {manifest.path}")
    print(SL.rule())
    return 1 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
