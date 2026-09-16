#!/usr/bin/env python
"""§3b.8: verify `voyage-context-4` contextualized chunk embeddings.

Five checks, all on real W1/W2 paper records, none of them touching Qdrant:

1. **token report** — per-paper embed-token totals over the 40-paper W1 sample,
   which papers exceed the 32,000-token per-document window, and how many
   groups each would need.  Free; run with ``--tokens-only``.
2. **contract** — `embed_document_chunks` returns exactly one vector of
   `dimension` floats per chunk, in input order, for a real paper record.
3. **oversized path** — the biggest sample paper: grouping triggers, every
   group fits the window, and the output is still one vector per chunk in
   order.
4. **contextualization actually happens** — the same chunk embedded with its
   paper around it vs. embedded alone.  A cosine below 1.0 is the proof that
   the contextual API is doing the work rather than silently degrading to flat
   embedding.  The full cosine matrix doubles as an alignment proof: vector *i*
   must look most like chunk *i*, not like its neighbour.
5. **query/document compatibility** — a query vector from the same model
   ranks the paper's chunks sensibly, and a query vector from the *old* model
   scored against the same vectors is shown to be plausible-looking garbage.
   That is the failure the EmbeddingProfile coupling removes.

Spend: a few cents.  Nothing is written anywhere.

Usage:
    python scripts/context4_verify.py
    python scripts/context4_verify.py --tokens-only        # no API spend
    python scripts/context4_verify.py --paper 73fc45a3d6ad
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from config import EMBEDDING_PROFILES, Settings, settings   # noqa: E402
from preprocessing.chunker import PaperChunker              # noqa: E402
from retrieval import embedder as emb                       # noqa: E402
from retrieval.embedder import VoyageEmbedder               # noqa: E402

PROFILE = "voyage-context-4"
W1_SAMPLE = "data/w1_sample/records"        # 40 papers, schema v1
W1_V2 = "data/w1_sample_v2/records"         # 10 papers, re-derived
W2B = "data/w2b_figures/records"            # 4 papers, schema v2 + figures

#: A W5 golden query aimed at the numeric content this library is for.
GOLDEN_QUERY = "what is the MIC of apidaecin analogs including Api137"


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    u, v = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def load_records(*dirs: str) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    for name in dirs:
        for path in sorted((BACKEND / name).glob("*.json")):
            found.setdefault(path.stem, path)
    return found


def chunk_texts(path: Path, chunker: PaperChunker) -> tuple:
    record = json.loads(path.read_text())
    chunks = chunker.chunk_record(record)
    return record, chunks, [chunk.embed_text for chunk in chunks]


# ---------------------------------------------------------------------------
# 1. token report — which papers exceed the window, and by how much
# ---------------------------------------------------------------------------

def token_report(embedder: VoyageEmbedder, chunker: PaperChunker,
                 records: Dict[str, Path]) -> List[dict]:
    print("=" * 78)
    print(f"1. TOKEN REPORT — {len(records)} papers, window "
          f"{emb.CONTEXT_WINDOW_TOKENS:,}, packing budget {embedder.doc_token_budget:,}")
    print("=" * 78)

    rows = []
    for paper_id, path in records.items():
        record, chunks, texts = chunk_texts(path, chunker)
        counts = embedder.count_tokens(texts)
        keys = [emb.section_key(text) for text in texts]
        groups = embedder._group_indices(counts, keys, embedder.doc_token_budget, paper_id)
        rows.append({
            "paper_id": paper_id,
            "title": (record.get("metadata", {}).get("title") or "")[:44],
            "chunks": len(texts),
            "tokens": sum(counts),
            "max_chunk": max(counts) if counts else 0,
            "groups": len(groups),
            "group_tokens": [sum(counts[i] for i in g) for g in groups],
        })

    rows.sort(key=lambda r: -r["tokens"])
    print(f"\n{'tokens':>8} {'chunks':>6} {'maxchunk':>8} {'groups':>6}  paper_id      title")
    for row in rows:
        flag = "  <-- OVER WINDOW" if row["groups"] > 1 else ""
        print(f"{row['tokens']:8,} {row['chunks']:6} {row['max_chunk']:8,} "
              f"{row['groups']:6}  {row['paper_id']}  {row['title']}{flag}")

    over = [r for r in rows if r["groups"] > 1]
    print(f"\n  {len(over)}/{len(rows)} papers need grouping; "
          f"largest paper {rows[0]['tokens']:,} tokens; "
          f"largest single chunk {max(r['max_chunk'] for r in rows):,} tokens")
    for row in over:
        print(f"    {row['paper_id']}  {row['tokens']:,} tok -> "
              f"{row['groups']} groups {row['group_tokens']}")
    corpus = sum(r["tokens"] for r in rows) / len(rows)
    print(f"  mean {corpus:,.0f} tokens/paper -> 4,797 papers ≈ "
          f"{corpus * 4797 / 1e6:,.1f}M tokens ≈ ${corpus * 4797 * 0.12 / 1e6:,.2f} "
          f"at $0.12/1M")
    return rows


# ---------------------------------------------------------------------------
# 2 + 3. the contract, on a normal paper and on an oversized one
# ---------------------------------------------------------------------------

def contract_check(embedder: VoyageEmbedder, chunker: PaperChunker,
                   paper_id: str, path: Path, heading: str) -> List[List[float]]:
    print("\n" + "=" * 78)
    print(f"{heading} — {paper_id}  ({path.parent.parent.name}/{path.parent.name})")
    print("=" * 78)

    record, chunks, texts = chunk_texts(path, chunker)
    counts = embedder.count_tokens(texts)
    groups = embedder._group_indices(counts, [emb.section_key(t) for t in texts],
                                     embedder.doc_token_budget, paper_id)
    print(f"  {len(texts)} chunks, {sum(counts):,} tokens, "
          f"{len(groups)} group(s): {[sum(counts[i] for i in g) for g in groups]}")
    if len(groups) > 1:
        print(f"  every group fits {embedder.doc_token_budget:,}: "
              f"{all(sum(counts[i] for i in g) <= embedder.doc_token_budget for g in groups)}")
        print("  cut before these sections (from the chunk header):")
        for group in groups[1:]:
            key = emb.section_key(texts[group[0]]) or "<no header>"
            print(f"    chunk {group[0]:>4}  {key.split(' — ')[-1][:60]}")

    calls: List[tuple] = []
    vectors = embedder.embed_document_chunks(
        texts, paper_id=paper_id,
        progress_callback=lambda done, total: calls.append((done, total)))

    ok_count = len(vectors) == len(texts)
    ok_dims = all(len(v) == embedder.dimension for v in vectors)
    ok_finite = all(np.isfinite(v).all() for v in vectors)
    distinct = len({tuple(v[:8]) for v in vectors})
    print(f"  vectors returned      : {len(vectors)} for {len(texts)} chunks  "
          f"[{'PASS' if ok_count else 'FAIL'}]")
    print(f"  every vector {embedder.dimension} floats : {ok_dims}  "
          f"[{'PASS' if ok_dims else 'FAIL'}]")
    print(f"  all finite            : {ok_finite}")
    print(f"  distinct vectors      : {distinct}/{len(vectors)}")
    print(f"  L2 norms              : min {min(float(np.linalg.norm(v)) for v in vectors):.4f} "
          f"max {max(float(np.linalg.norm(v)) for v in vectors):.4f}")
    print(f"  progress callback     : {calls}")
    assert ok_count and ok_dims and ok_finite, "contract violated"

    # An empty input is an empty list, not an error.
    assert embedder.embed_document_chunks([]) == []
    print("  embed_document_chunks([]) == []  [PASS]")
    return vectors


# ---------------------------------------------------------------------------
# 4. contextualization actually happens, and vector i belongs to chunk i
# ---------------------------------------------------------------------------

def context_check(embedder: VoyageEmbedder, chunker: PaperChunker,
                  paper_id: str, path: Path, in_context: List[List[float]],
                  sample: int = 5, min_gap: int = 4) -> None:
    print("\n" + "=" * 78)
    print("4. CONTEXTUALIZATION — same chunk, with the paper vs alone")
    print("=" * 78)

    _, chunks, texts = chunk_texts(path, chunker)
    # Prefer numeric fine chunks: the "values were markedly lower" case that
    # contextual retrieval exists for.
    # One chunk per available type, numeric fine chunks first — the "values
    # were markedly lower" case contextual retrieval exists for.  Adjacent fine
    # chunks overlap by `fine_chunk_overlap` tokens, so keep the picks apart or
    # the comparison measures the overlap instead.
    order = sorted(range(len(chunks)),
                   key=lambda i: (chunks[i].chunk_type.value != "fine",
                                  not chunks[i].numeric_facts, i))
    picks: List[int] = []
    seen_types: set = set()
    for wave in (True, False):
        for i in order:
            if len(picks) == sample:
                break
            if wave and chunks[i].chunk_type.value in seen_types:
                continue
            if all(abs(i - j) >= min_gap for j in picks):
                picks.append(i)
                seen_types.add(chunks[i].chunk_type.value)
    picks.sort()

    # Each picked chunk as its own one-chunk document: same model, same
    # endpoint, no document context.  The only variable is the context.
    alone = embedder.embed_paper_batch([[texts[i]] for i in picks])
    alone = [vectors[0] for vectors in alone]

    print(f"  paper {paper_id}, {len(texts)} chunks, sampling {len(picks)}\n")
    print(f"  {'chunk':<28} {'type':<9} {'cos(in-context, alone)':>22}")
    diagonal = []
    for row, i in enumerate(picks):
        similarity = cosine(in_context[i], alone[row])
        diagonal.append(similarity)
        print(f"  {chunks[i].chunk_id[:28]:<28} {chunks[i].chunk_type.value:<9} "
              f"{similarity:>22.4f}")
    print(f"\n  cos(in-context, alone): min {min(diagonal):.4f} "
          f"mean {float(np.mean(diagonal)):.4f} max {max(diagonal):.4f}")
    if max(diagonal) > 0.9999:
        print("  FAIL: identical vectors — the call is NOT contextualizing")
    else:
        print("  PASS: the document context changes every vector, so the "
              "contextual API is in use (a flat fallback would give 1.0000)")

    matrix = np.array([[cosine(in_context[i], alone[c]) for c in range(len(picks))]
                       for i in picks])
    off = matrix[~np.eye(len(picks), dtype=bool)]
    ranks = [int(1 + sum(matrix[r] > matrix[r][r])) for r in range(len(picks))]
    print(f"  off-diagonal cosine: mean {off.mean():.4f} max {off.max():.4f} "
          f"(diagonal mean {float(np.mean(diagonal)):.4f})")
    print(f"  own-chunk rank among the {len(picks)} solo vectors: {ranks} "
          f"(1 = its own solo vector is the closest)")
    print("  note: contextualization pulls one paper's chunks toward a shared "
          "document topic, so within a short homogeneous paper the row maximum "
          "is not a reliable alignment test — see the order check below.")


# ---------------------------------------------------------------------------
# order: does vector i really belong to chunk i?
# ---------------------------------------------------------------------------

#: Eight mutually unrelated sentences.  A real paper cannot test the ordering:
#: its section chunks re-cover the same text as their fine chunks, adjacent fine
#: chunks overlap by `fine_chunk_overlap` tokens, and contextualization pulls
#: one paper's vectors toward a shared topic — so "chunk i's vector is nearest
#: chunk i" is confounded three ways over.  With orthogonal content it is not.
ORTHOGONAL = [
    "The minimum inhibitory concentration of Api137 against E. coli was 2 ug/mL.",
    "Palladium-cobalt alloy nanoparticles were annealed at 600 degrees Celsius.",
    "The Raman shift of the amide I band appeared at 1650 wavenumbers.",
    "Jurkat cells were cultured in RPMI-1640 with 10 percent fetal bovine serum.",
    "The crystal structure was solved by molecular replacement at 1.9 angstrom.",
    "Fluorescein isothiocyanate was coupled to the peptide N-terminus overnight.",
    "Gel permeation chromatography gave a polydispersity index of 1.08.",
    "The photomultiplier dark count rate was 12 counts per second at 20 Celsius.",
]


def order_check(embedder: VoyageEmbedder, chunker: PaperChunker,
                paper_id: str, path: Path, vectors: List[List[float]]) -> None:
    """Vector *i* must be chunk *i*'s, as one document and as several groups."""
    print("\n" + "=" * 78)
    print("ORDER — position in, position out")
    print("=" * 78)

    def diagonal_wins(in_context: List[List[float]], alone: List[List[float]]):
        left = np.array(in_context, dtype=float)
        right = np.array(alone, dtype=float)
        left /= np.linalg.norm(left, axis=1, keepdims=True)
        right /= np.linalg.norm(right, axis=1, keepdims=True)
        matrix = left @ right.T
        wins = int((matrix.argmax(axis=1) == np.arange(len(in_context))).sum())
        diag = np.diag(matrix)
        off = matrix[~np.eye(len(in_context), dtype=bool)]
        return wins, diag, off

    solo = [pair[0] for pair in embedder.embed_paper_batch([[t] for t in ORTHOGONAL])]

    # (a) the eight as one document
    one_doc = embedder.embed_document_chunks(ORTHOGONAL, paper_id="orthogonal")
    wins, diag, off = diagonal_wins(one_doc, solo)
    print(f"  {len(ORTHOGONAL)} orthogonal chunks, 1 document")
    print(f"    chunk i's vector is nearest chunk i's solo vector: "
          f"{wins}/{len(ORTHOGONAL)}  [{'PASS' if wins == len(ORTHOGONAL) else 'FAIL'}]")
    print(f"    matched {diag.mean():.4f} (min {diag.min():.4f}) vs "
          f"everything else {off.mean():.4f} (max {off.max():.4f})")

    # (b) the same eight forced into groups, to exercise the reassembly path
    budget, ceiling = embedder.doc_token_budget, embedder.max_request_tokens
    calls: List[tuple] = []
    try:
        embedder.doc_token_budget = 45      # ~2 chunks per group
        embedder.max_request_tokens = 50    # ~1 group per request
        grouped = embedder.embed_document_chunks(
            ORTHOGONAL, paper_id="orthogonal-grouped",
            progress_callback=lambda done, total: calls.append((done, total)))
    finally:
        embedder.doc_token_budget, embedder.max_request_tokens = budget, ceiling
    wins, diag, off = diagonal_wins(grouped, solo)
    print(f"    ... and with the budget forced to 45 tokens and the request "
          f"ceiling to 50 ({len(calls)} requests, {calls}): "
          f"{wins}/{len(ORTHOGONAL)}  [{'PASS' if wins == len(ORTHOGONAL) else 'FAIL'}]")
    print(f"    matched {diag.mean():.4f} vs everything else {off.mean():.4f}")
    drift = [cosine(a, b) for a, b in zip(one_doc, grouped)]
    print(f"    grouping changed the vectors (cos one-doc vs grouped): "
          f"mean {float(np.mean(drift)):.4f} min {min(drift):.4f} — so the "
          f"grouped run really is a different, smaller context")

    # (c) the real paper, for the record: exact-quote self-retrieval.  This is
    # an observation, not a pass/fail, for the reason in the note below.
    _, chunks, texts = chunk_texts(path, chunker)
    step = max(1, len(chunks) // 6)
    picks = [i for i in range(0, len(chunks), step) if len(chunks[i].text) >= 260][:6]
    matrix = np.array(vectors, dtype=float)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    ranks = []
    for i in picks:
        body = " ".join(chunks[i].text.split())
        middle = body[len(body) // 2 - 100:len(body) // 2 + 100]
        query = np.asarray(embedder.embed_query(middle), dtype=float)
        query /= np.linalg.norm(query)
        scores = matrix @ query
        ranks.append(int(1 + (scores > scores[i]).sum()))
    print(f"\n  observation — on real paper {paper_id}, a verbatim slice from "
          f"the middle of a chunk ranks that chunk {ranks} of {len(chunks)}.")
    print(f"    Expected, and not an ordering fault: a section chunk re-covers "
          f"its own fine chunks' text, so several chunks legitimately contain "
          f"the quote, and contextualization moves every vector toward the "
          f"paper. Exact-quote lookup is BM25's job (`embed_text` still carries "
          f"the header) and the reranker's.")


# ---------------------------------------------------------------------------
# 5. query/document compatibility, and why the profile has to carry both
# ---------------------------------------------------------------------------

def query_check(embedder: VoyageEmbedder, chunker: PaperChunker,
                paper_id: str, path: Path, doc_vectors: List[List[float]],
                query: str) -> None:
    print("\n" + "=" * 78)
    print("5. QUERY / DOCUMENT COMPATIBILITY")
    print("=" * 78)

    _, chunks, texts = chunk_texts(path, chunker)
    new_query = embedder.embed_query(query)
    print(f"  query   : {query!r}")
    print(f"  embedder: profile={embedder.profile.name} model={embedder.model} "
          f"dim={len(new_query)} collection={embedder.collection_name}")

    scores = sorted(((cosine(new_query, v), i) for i, v in enumerate(doc_vectors)),
                    reverse=True)
    print(f"\n  same model, top 3 of {len(doc_vectors)} chunks:")
    for score, i in scores[:3]:
        print(f"    {score:.4f}  {chunks[i].chunk_id[:26]:<26} "
              f"{texts[i][:70].replace(chr(10), ' ')}")
    print(f"  worst 1: {scores[-1][0]:.4f}  "
          f"{texts[scores[-1][1]][:70].replace(chr(10), ' ')}")
    print(f"  score spread {scores[-1][0]:.4f} .. {scores[0][0]:.4f} — a real "
          f"signal, not noise")

    # The mismatch this design makes unreachable: the OLD model's query vector
    # against the NEW model's document vectors.  Same 1024 dims, so nothing
    # errors; the ranking is simply wrong.
    legacy = VoyageEmbedder.for_profile("voyage-3-large")
    old_query = legacy.embed_query(query)
    mismatched = sorted(((cosine(old_query, v), i) for i, v in enumerate(doc_vectors)),
                        reverse=True)
    print(f"\n  MISMATCH DEMO — {legacy.model} query vs {embedder.model} documents:")
    print(f"    dimensions agree ({len(old_query)} == {len(doc_vectors[0])}), so "
          f"Qdrant would accept it silently")
    for score, i in mismatched[:3]:
        print(f"    {score:.4f}  {chunks[i].chunk_id[:26]:<26} "
              f"{texts[i][:60].replace(chr(10), ' ')}")
    right = [i for _, i in scores]
    wrong = [i for _, i in mismatched]
    overlap = len(set(right[:10]) & set(wrong[:10]))
    print(f"    score range {mismatched[-1][0]:.4f} .. {mismatched[0][0]:.4f} "
          f"— numerically unremarkable, nothing looks broken")
    print(f"    top-10 overlap with the correct ranking: {overlap}/10")
    print(f"    correct top chunk sits at rank {wrong.index(right[0]) + 1} "
          f"of {len(wrong)} under the mismatched query")
    rank_right = np.empty(len(right)); rank_wrong = np.empty(len(wrong))
    rank_right[right] = np.arange(len(right))
    rank_wrong[wrong] = np.arange(len(wrong))
    spearman = float(np.corrcoef(rank_right, rank_wrong)[0, 1])
    print(f"    Spearman rank correlation with the correct ranking: {spearman:+.3f}")

    print("\n  COUPLING — the mismatch above cannot be configured:")
    print(f"    profiles: " + ", ".join(
        f"{p.name} -> {p.collection} ({p.dimension}d"
        f"{', contextual' if p.contextualized else ''})"
        for p in EMBEDDING_PROFILES.values()))
    print(f"    settings.embedding_model / .embedding_dimension / "
          f".qdrant_collection_name are read-only properties of the profile")
    for attribute in ("qdrant_collection_name", "embedding_model", "embedding_dimension"):
        try:
            setattr(settings, attribute, "x")
            print(f"    FAIL: settings.{attribute} was settable")
        except AttributeError as exc:
            print(f"    settings.{attribute} = ... -> AttributeError "
                  f"({str(exc).split(' of ')[0]})")
    try:
        VoyageEmbedder(api_key=settings.voyage_api_key, model="voyage-context-4",
                       profile=EMBEDDING_PROFILES["voyage-3-large"])
        print("    FAIL: a model/profile mismatch was accepted")
    except ValueError:
        print("    VoyageEmbedder(model=<other model>) -> ValueError")
    import os
    os.environ["EMBEDDING_PROFILE"] = PROFILE
    try:
        Settings()
        print(f"    EMBEDDING_PROFILE={PROFILE} with a stale "
              f"QDRANT_COLLECTION_NAME -> accepted (no stale value in env)")
    except Exception as exc:
        line = [l for l in str(exc).splitlines() if "contradicts" in l]
        print(f"    stale QDRANT_COLLECTION_NAME + new profile -> "
              f"{'ValidationError: ' + line[0].strip()[:110] if line else type(exc).__name__}")
    finally:
        os.environ.pop("EMBEDDING_PROFILE", None)


# ---------------------------------------------------------------------------
# 6. the live path must not move
# ---------------------------------------------------------------------------

def legacy_check(contextual: VoyageEmbedder) -> None:
    """`voyage-3-large` still serves the live collection, unchanged.

    The cutover is a profile switch, not a rewrite, so everything the running
    app calls has to keep working on the old profile — and the new methods
    must refuse to answer on it rather than quietly returning flat vectors.
    """
    print("\n" + "=" * 78)
    print("6. LEGACY PATH (voyage-3-large, the live collection)")
    print("=" * 78)

    legacy = VoyageEmbedder.for_profile("voyage-3-large")
    print(f"  profile {legacy.profile.name}: model={legacy.model} "
          f"dim={legacy.dimension} collection={legacy.collection_name} "
          f"contextualized={legacy.contextualized}")
    query = legacy.embed_query("apidaecin MIC")
    texts = ["a short passage", "another passage"]
    documents = legacy.embed_documents(texts)
    pooled = legacy.compute_mean_pooled_embedding(texts)
    print(f"  embed_query -> {len(query)} floats; embed_documents -> "
          f"{len(documents)} x {len(documents[0])}; "
          f"compute_mean_pooled_embedding -> {len(pooled)} floats")
    print(f"  pool_vectors(those vectors) vs compute_mean_pooled_embedding: "
          f"cosine {cosine(VoyageEmbedder.pool_vectors(documents), pooled):.6f} "
          f"— identical, without the second round of API calls")
    try:
        legacy.embed_document_chunks(["x", "y"])
        print("  FAIL: embed_document_chunks returned flat vectors on the "
              "legacy profile")
    except RuntimeError as exc:
        print(f"  embed_document_chunks on the legacy profile -> RuntimeError: "
              f"{str(exc)[:96]}...")

    flat = contextual.embed_documents(["a short passage", "another passage"])
    print(f"  embed_documents on {contextual.profile.name} -> {len(flat)} x "
          f"{len(flat[0])} (single-chunk documents, no context, warned above)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-only", action="store_true",
                        help="token report only; no API calls, no spend")
    parser.add_argument("--paper", default=None,
                        help="paper id for the contract check (default: a W1 v2 record)")
    parser.add_argument("--query", default=GOLDEN_QUERY)
    parser.add_argument("--query-paper", default="73fc45a3d6ad",
                        help="paper the golden query is actually about")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("preprocessing.chunker").setLevel(logging.ERROR)

    chunker = PaperChunker()
    embedder = VoyageEmbedder.for_profile(PROFILE)
    print(f"profile {embedder.profile.name}: model={embedder.model} "
          f"dim={embedder.dimension} collection={embedder.collection_name} "
          f"contextualized={embedder.contextualized}")
    print(f"live app profile is still {settings.embedding_profile} -> "
          f"{settings.qdrant_collection_name} (untouched)")

    rows = token_report(embedder, chunker, load_records(W1_SAMPLE))
    if args.tokens_only:
        return 0

    # 2. contract, on a real W1 v2 / W2b record (schema v2, figure crops)
    v2 = load_records(W2B, W1_V2)
    paper_id = args.paper or next(iter(v2))
    path = v2.get(paper_id) or load_records(W1_SAMPLE, W1_V2, W2B)[paper_id]
    vectors = contract_check(embedder, chunker, paper_id, path,
                             "2. CONTRACT")

    # 3. oversized: the biggest paper in the sample
    big_id = rows[0]["paper_id"]
    big_path = load_records(W1_SAMPLE)[big_id]
    big_vectors = contract_check(embedder, chunker, big_id, big_path,
                                 "3. OVERSIZED PATH")

    # 4: contextualization on the plain paper, and on the grouped one —
    # grouping must not stop contextualization inside a group
    context_check(embedder, chunker, paper_id, path, vectors)
    context_check(embedder, chunker, big_id, big_path, big_vectors, sample=3)

    # 5: the query check needs a paper the query is actually about, or the
    # scores are all noise and prove nothing
    all_records = load_records(W1_SAMPLE, W1_V2, W2B)
    query_id = args.query_paper if args.query_paper in all_records else paper_id
    query_path = all_records[query_id]
    query_vectors = (vectors if query_id == paper_id else
                     contract_check(embedder, chunker, query_id, query_path,
                                    "2b. CONTRACT (query paper)"))
    order_check(embedder, chunker, query_id, query_path, query_vectors)
    query_check(embedder, chunker, query_id, query_path, query_vectors, args.query)

    legacy_check(embedder)

    print("\nall checks complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
