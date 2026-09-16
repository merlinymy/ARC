#!/usr/bin/env python3
"""Rebuild the BM25 sparse index over the whole Qdrant collection.

Why this exists
---------------
The sparse (``bm25``) vectors stored in Qdrant were written under sparse scheme v1,
which had two defects:

1. ``BM25Vectorizer._term_to_index`` used the builtin ``hash()``. Python salts string
   hashing per process, so the same term mapped to a different sparse dimension in
   every process -- stored document vectors were unreachable from query vectors.
2. IDF was baked into the *document* vector. The IDF table was also built from a
   fraction of the corpus (doc_count=13,979 vs 212,953 points) and was refreshed
   incrementally after each indexing run, so chunks indexed at different times were
   scored against different corpus statistics.

Scheme v2 fixes both: a process-stable blake2b hash, and IDF applied on the query
side only. Document vectors then depend on nothing but their own text (plus the
frozen ``avg_doc_length``), so the IDF table can be refreshed forever without
invalidating the index.

What it does
------------
pass1  Scroll every point (payload only, no vectors), tokenize ``text``, accumulate
       ``doc_freq`` / ``doc_count`` / ``avg_doc_length``, and stream per-point term
       frequencies to a work file.
pass2  Replay the work file, compute the v2 document vector for each point, and write
       back *only* the ``bm25`` named vector via Qdrant's ``update_vectors``. The dense
       vector and the payload are never sent and are left untouched.

Both passes checkpoint and resume. pass2 records exactly how many points have been
updated, so an interruption leaves a known -- not a mysterious -- mixed state.

Usage
-----
    python scripts/rebuild_bm25_index.py pass1
    python scripts/rebuild_bm25_index.py pass2
    python scripts/rebuild_bm25_index.py status
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import PointVectors, SparseVector as QdrantSparseVector  # noqa: E402

from preprocessing.models import BM25_PAYLOAD_FIELDS, embed_text_from_payload  # noqa: E402
from retrieval.bm25 import (  # noqa: E402
    BM25Vectorizer,
    DEFAULT_MAX_VOCAB_SIZE,
    SPARSE_SCHEME_VERSION,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("rebuild_bm25")

WORK_DIR = BACKEND_DIR / "data" / "bm25_rebuild"
TF_FILE = WORK_DIR / "chunk_tf.jsonl"
STATS_FILE = WORK_DIR / "corpus_stats.json"
CHECKPOINT_FILE = WORK_DIR / "checkpoint.json"
IDF_CACHE_PATH = BACKEND_DIR / "data" / "bm25_idf_cache.json"

SCROLL_BATCH = 1000
UPDATE_BATCH = 1000


# --------------------------------------------------------------------------- utils


def load_checkpoint() -> Dict:
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Unreadable checkpoint (%s); starting fresh", exc)
    return {}


def save_checkpoint(data: Dict) -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(CHECKPOINT_FILE)


def make_client() -> QdrantClient:
    from config import settings

    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        timeout=600,
    )


def collection_name() -> str:
    from config import settings

    return settings.qdrant_collection_name


def _truncate_to_line_count(path: Path, keep_lines: int) -> int:
    """Truncate a JSONL file to exactly ``keep_lines`` complete lines.

    Guards against a torn final line from an interrupted write.
    """
    if not path.exists():
        return 0
    kept = 0
    byte_offset = 0
    with open(path, "rb") as fh:
        for line in fh:
            if kept >= keep_lines:
                break
            if not line.endswith(b"\n"):
                break  # torn final line - drop it
            kept += 1
            byte_offset += len(line)
    with open(path, "r+b") as fh:
        fh.truncate(byte_offset)
    return kept


# ----------------------------------------------------------------------- pass 1


def pass1(args: argparse.Namespace) -> int:
    """Scroll the collection, tokenize, accumulate corpus statistics."""
    client = make_client()
    coll = collection_name()
    total_points = client.count(coll, exact=True).count
    logger.info("Collection '%s': %d points", coll, total_points)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = load_checkpoint()
    p1 = ckpt.get("pass1", {})

    vectorizer = BM25Vectorizer(max_vocab_size=args.max_vocab_size)

    doc_freq: Counter = Counter()
    doc_count = 0
    total_length = 0
    empty_text = 0
    offset = None
    mode = "w"

    if args.resume and p1.get("lines_written"):
        kept = _truncate_to_line_count(TF_FILE, p1["lines_written"])
        logger.info("Resuming pass1: replaying %d already-written rows", kept)
        with open(TF_FILE) as fh:
            for line in fh:
                row = json.loads(line)
                tf = row["tf"]
                doc_count += 1
                total_length += row["dl"]
                if not tf:
                    empty_text += 1
                for term in tf:
                    doc_freq[term] += 1
        offset = p1.get("next_offset")
        mode = "a"
        logger.info("Replayed %d rows, %d vocab terms; resuming at offset %s",
                    doc_count, len(doc_freq), offset)

    started = time.time()
    lines_written = doc_count

    with open(TF_FILE, mode) as out:
        while True:
            points, next_offset = client.scroll(
                collection_name=coll,
                limit=SCROLL_BATCH,
                offset=offset,
                with_payload=BM25_PAYLOAD_FIELDS,
                with_vectors=False,  # never pull the 1024-dim dense vectors
            )
            if not points:
                break

            for pt in points:
                # W2: BM25 indexes `embed_text`, recomposed from the payload
                # parts, so the sparse vectors describe the same document the
                # dense vectors were built from.
                text = embed_text_from_payload(pt.payload or {})
                tokens = vectorizer.tokenize(text)
                tf = Counter(tokens)
                doc_count += 1
                total_length += len(tokens)
                if not tokens:
                    empty_text += 1
                for term in tf:
                    doc_freq[term] += 1
                out.write(json.dumps(
                    {"id": str(pt.id), "tf": tf, "dl": len(tokens)},
                    separators=(",", ":"),
                ) + "\n")
                lines_written += 1

            out.flush()
            os.fsync(out.fileno())
            offset = next_offset
            save_checkpoint({
                **load_checkpoint(),
                "pass1": {
                    "next_offset": str(offset) if offset is not None else None,
                    "lines_written": lines_written,
                    "complete": offset is None,
                },
            })

            if lines_written % 20000 < SCROLL_BATCH:
                rate = lines_written / max(time.time() - started, 1e-9)
                logger.info("pass1 %d/%d points (%.0f/s), vocab=%d",
                            lines_written, total_points, rate, len(doc_freq))

            if offset is None:
                break

    avg_doc_length = total_length / doc_count if doc_count else 0.0
    stats = {
        "scheme_version": SPARSE_SCHEME_VERSION,
        "doc_count": doc_count,
        "total_doc_length": total_length,
        "avg_doc_length": avg_doc_length,
        "vocab_size": len(doc_freq),
        "empty_text_chunks": empty_text,
        "max_vocab_size": args.max_vocab_size,
        "collection_points": total_points,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    STATS_FILE.write_text(json.dumps(stats, indent=2))

    # Persist the IDF table (query side) with the corpus-wide statistics.
    vectorizer._doc_freq = dict(doc_freq)
    vectorizer._doc_count = doc_count
    vectorizer._total_doc_length = total_length
    vectorizer.avg_doc_length = avg_doc_length
    vectorizer._recalculate_idf()
    vectorizer.save_idf_cache(IDF_CACHE_PATH)

    logger.info("pass1 complete in %.1fs", time.time() - started)
    logger.info("  doc_count       = %d", doc_count)
    logger.info("  avg_doc_length  = %.2f", avg_doc_length)
    logger.info("  vocabulary      = %d terms", len(doc_freq))
    logger.info("  empty chunks    = %d", empty_text)
    logger.info("  IDF cache saved to %s", IDF_CACHE_PATH)
    return 0


# ----------------------------------------------------------------------- pass 2


def _iter_rows(path: Path, skip: int) -> Iterator[Tuple[int, Dict]]:
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i < skip:
                continue
            yield i, json.loads(line)


def pass2(args: argparse.Namespace) -> int:
    """Re-vectorize every chunk and write back only the ``bm25`` named vector."""
    if not STATS_FILE.exists():
        logger.error("Missing %s - run pass1 first", STATS_FILE)
        return 1

    stats = json.loads(STATS_FILE.read_text())
    client = make_client()
    coll = collection_name()

    vectorizer = BM25Vectorizer(
        avg_doc_length=stats["avg_doc_length"],
        max_vocab_size=stats["max_vocab_size"],
    )
    logger.info(
        "pass2 using avg_doc_length=%.2f max_vocab_size=%d (scheme v%d)",
        vectorizer.avg_doc_length, vectorizer.max_vocab_size, SPARSE_SCHEME_VERSION,
    )

    ckpt = load_checkpoint()
    done = ckpt.get("pass2", {}).get("points_updated", 0) if args.resume else 0
    if done:
        logger.info("Resuming pass2: %d points already updated", done)

    total_rows = stats["doc_count"]
    k1, b = vectorizer.k1, vectorizer.b
    avgdl = vectorizer.avg_doc_length

    batch: list = []
    started = time.time()
    updated = done
    nnz_total = 0
    nnz_docs = 0

    def flush() -> None:
        nonlocal batch, updated
        if not batch:
            return
        client.update_vectors(collection_name=coll, points=batch, wait=True)
        updated += len(batch)
        batch = []
        save_checkpoint({
            **load_checkpoint(),
            "pass2": {
                "points_updated": updated,
                "total": total_rows,
                "complete": updated >= total_rows,
                "scheme_version": SPARSE_SCHEME_VERSION,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        })

    for i, row in _iter_rows(TF_FILE, done):
        dl = row["dl"]
        scores: Dict[int, float] = {}
        if dl:
            norm = k1 * (1 - b + b * (dl / avgdl))
            for term, tf in row["tf"].items():
                idx = vectorizer._term_to_index(term)
                score = (tf * (k1 + 1)) / (tf + norm)
                # Hash collisions keep the strongest signal, matching vectorize().
                if score > scores.get(idx, 0.0):
                    scores[idx] = score

        items = sorted(scores.items())
        nnz_total += len(items)
        if items:
            nnz_docs += 1

        batch.append(PointVectors(
            id=row["id"],
            vector={"bm25": QdrantSparseVector(
                indices=[idx for idx, _ in items],
                values=[val for _, val in items],
            )},
        ))

        if len(batch) >= UPDATE_BATCH:
            flush()
            if updated % 20000 < UPDATE_BATCH:
                rate = (updated - done) / max(time.time() - started, 1e-9)
                eta = (total_rows - updated) / max(rate, 1e-9)
                logger.info("pass2 %d/%d points (%.0f/s, ETA %.0fs)",
                            updated, total_rows, rate, eta)

    flush()

    logger.info("pass2 complete in %.1fs", time.time() - started)
    logger.info("  points updated  = %d / %d", updated, total_rows)
    logger.info("  mean nnz/vector = %.1f", nnz_total / max(nnz_docs, 1))
    return 0 if updated >= total_rows else 1


# ----------------------------------------------------------------------- status


def status(args: argparse.Namespace) -> int:
    ckpt = load_checkpoint()
    print(json.dumps(ckpt, indent=2))
    if STATS_FILE.exists():
        print(STATS_FILE.read_text())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["pass1", "pass2", "status"])
    ap.add_argument("--max-vocab-size", type=int, default=DEFAULT_MAX_VOCAB_SIZE,
                    dest="max_vocab_size")
    ap.add_argument("--no-resume", action="store_false", dest="resume")
    args = ap.parse_args()
    return {"pass1": pass1, "pass2": pass2, "status": status}[args.phase](args)


if __name__ == "__main__":
    raise SystemExit(main())
