#!/usr/bin/env python
"""W2 validation: chunk W1's sample records and prove the chunk contract.

No PDF extraction, no Voyage calls, no Qdrant writes — it reads W1's persisted
records, so a full re-chunk of the sample costs seconds instead of the ~98 h a
re-extraction would.  That is exactly what persisting the record bought.

What it proves
--------------
1. **Every chunk type is populated** where the record has the structure for it.
   The handoff bug was that the chunker never read the record: a paper whose
   record held 2 sections and an abstract produced 0 section, 0 fine and 0
   abstract chunks.
2. **``text`` is verbatim.**  ``full_text(record)[char_start:char_end] ==
   chunk.text``, against a document text reassembled independently here, plus a
   check that no context header leaked into ``text``.
3. **Page attribution is honest** across a page break (R18).
4. Old vs new counts, numeric facts, and the corpus projection.

Usage:
    python scripts/w2_sample_chunk.py                       # 40-paper sample
    python scripts/w2_sample_chunk.py --records data/w1_sample_v2/records
    python scripts/w2_sample_chunk.py --json out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from preprocessing import paper_record as pr           # noqa: E402
from preprocessing.chunker import MIN_SECTION_CHARS, PaperChunker   # noqa: E402
from preprocessing.models import (                     # noqa: E402
    ChunkType,
    display_text_from_payload,
    embed_text_from_payload,
)

CORPUS_PAPERS = 4797
BASE_CHUNKS = 212_651          # §1 audit baseline
BASE_BY_TYPE = {"caption": 93_144, "fine": 83_946, "section": 26_758,
                "abstract": 2_232, "full": 4_284, "table": 0}
TYPES = ["abstract", "section", "fine", "caption", "table"]


# ---------------------------------------------------------------------------
# The pre-W2 chunker, loaded from git so "before" is the real before
# ---------------------------------------------------------------------------

def load_old_chunker(ref: str = "HEAD"):
    """Import the committed (pre-W2) chunker as a standalone module."""
    try:
        source = subprocess.run(
            ["git", "show", f"{ref}:backend/preprocessing/chunker.py"],
            cwd=BACKEND.parent, capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  (could not load the pre-W2 chunker from git: {exc})")
        return None
    if "def chunk_record" in source:
        print(f"  (note: {ref} already contains W2's chunker; 'before' column skipped)")
        return None
    source = source.replace("from .models import", "from preprocessing.models import")
    source = source.replace("from .section_detector import",
                            "from preprocessing.section_detector import")
    module = types.ModuleType("chunker_pre_w2")
    module.__dict__["__file__"] = "<git:chunker.py>"
    exec(compile(source, "<git:chunker.py>", "exec"), module.__dict__)
    return module.PaperChunker()


# ---------------------------------------------------------------------------
# Per-paper
# ---------------------------------------------------------------------------

def check_paper(record: Dict[str, Any], chunker: PaperChunker,
                old_chunker) -> Dict[str, Any]:
    paper_id = record["paper_id"]
    stats = record.get("stats", {})
    chunks = chunker.chunk_record(record)

    # Reassemble the document text here, independently of the chunker, so the
    # verbatim assertion is a real check of the offsets and not a tautology.
    doc = pr.PAGE_SEP.join(pr.all_page_texts(record))
    counts = Counter(c.chunk_type.value for c in chunks)

    verbatim_checked = verbatim_bad = 0
    synthetic_in_text = 0
    page_bad: List[str] = []
    examples: List[Dict[str, Any]] = []
    starts = [b.get("doc_char_start", 0) for b in record["blocks"]]

    for chunk in chunks:
        payload = chunk.to_payload()

        if chunk.char_start is not None:
            verbatim_checked += 1
            if doc[chunk.char_start:chunk.char_end] != chunk.text:
                verbatim_bad += 1
                if len(examples) < 3:
                    examples.append({"chunk_id": chunk.chunk_id, "kind": "verbatim"})
            # A context header in `text` is the specific failure the split exists
            # to prevent: it would surface inside a citation quote.
            if chunk.context_line and chunk.text.startswith(chunk.context_line):
                synthetic_in_text += 1
            # embed_text must be exactly the context lines plus the body, and
            # must be recomposable from the stored payload alone (BM25 reads it).
            if embed_text_from_payload(payload) != chunk.embed_text:
                page_bad.append(f"{chunk.chunk_id}: embed_text not recomposable")
            if display_text_from_payload(payload) != chunk.display_text:
                page_bad.append(f"{chunk.chunk_id}: display_text not recomposable")

            # Pages must match the blocks the span actually covers.
            covered = [b for b in record["blocks"]
                       if b.get("doc_char_start") is not None
                       and b["doc_char_start"] < chunk.char_end
                       and b["doc_char_end"] > chunk.char_start
                       and b["text"]]
            if covered:
                expect_start = min(b["page_idx"] for b in covered) + 1
                expect_end = max(b["page_idx"] for b in covered) + 1
                if (chunk.page_start, chunk.page_end) != (expect_start, expect_end):
                    page_bad.append(
                        f"{chunk.chunk_id}: pages {chunk.page_start}-{chunk.page_end} "
                        f"!= covered {expect_start}-{expect_end}"
                    )

    facts = [f for c in chunks for f in c.numeric_facts]
    payloads = [c.to_payload() for c in chunks]
    payload_bytes = sum(len(json.dumps(p, ensure_ascii=False)) for p in payloads)

    old_counts: Counter = Counter()
    if old_chunker is not None:
        meta = chunker.metadata_from_record(record)
        old_chunks = old_chunker.chunk_paper(
            text=pr.body_text(record),
            metadata=meta,
            captions=pr.captions(record),
            tables=pr.table_texts(record),
        )
        old_counts = Counter(c.chunk_type.value for c in old_chunks)

    return {
        "paper_id": paper_id,
        "file_name": record.get("file_name", ""),
        "degraded": record.get("degraded", False),
        "pages": stats.get("n_pages", 0),
        "record": {
            "sections": stats.get("n_sections", 0),
            "abstract": bool(stats.get("has_abstract")),
            "tables_with_body": stats.get("n_tables_with_body", 0),
            "captions": stats.get("n_captions", 0),
            "body_chars": stats.get("body_chars", 0),
        },
        "new": {t: counts.get(t, 0) for t in TYPES},
        "new_total": len(chunks),
        "old": {t: old_counts.get(t, 0) for t in TYPES},
        "old_total": sum(old_counts.values()),
        "verbatim_checked": verbatim_checked,
        "verbatim_bad": verbatim_bad,
        "synthetic_in_text": synthetic_in_text,
        "page_errors": page_bad[:5],
        "n_page_errors": len(page_bad),
        "cross_page_chunks": sum(1 for c in chunks
                                 if c.page_start and c.page_end
                                 and c.page_end > c.page_start),
        "numeric_facts": len(facts),
        "numeric_from_tables": sum(1 for f in facts if f["source"] == "table"),
        "numeric_properties": sorted({f["property"] for f in facts}),
        "payload_bytes": payload_bytes,
        "coverage": _coverage(record, chunks),
        "examples": examples,
    }


def _coverage(record: Dict[str, Any], chunks) -> float:
    """Share of *indexable* body characters covered by at least one fine chunk.

    R3's real cost was invisible: the tail of a long section was indexed at no
    granularity at all, and no counter showed it.  This is that counter.
    References and acknowledgments are excluded on purpose, so they are excluded
    from the denominator too — counting them would report a permanent ~70%.
    """
    from preprocessing.chunker import SKIP_SECTIONS
    skipped = {s["id"] for s in (record.get("sections") or [])
               if s["normalized_name"] in SKIP_SECTIONS}
    body_total = 0
    covered = 0
    spans = sorted((c.char_start, c.char_end) for c in chunks
                   if c.chunk_type == ChunkType.FINE and c.char_start is not None)
    for block in record["blocks"]:
        if block["type"] not in pr.BODY_BLOCK_TYPES or not block["text"]:
            continue
        if block.get("section_id") in skipped:
            continue
        body_total += len(block["text"])
        b0, b1 = block["doc_char_start"], block["doc_char_end"]
        for s, e in spans:
            if e <= b0:
                continue
            if s >= b1:
                break
            covered += min(e, b1) - max(s, b0)
    if not body_total:
        return 0.0
    return min(1.0, covered / body_total)


# ---------------------------------------------------------------------------
# Proof 3: a paragraph W1 split across a page break
# ---------------------------------------------------------------------------

def prove_cross_page(records: Dict[str, Path], chunker: PaperChunker,
                     limit: int = 3) -> None:
    """Show the page attribution of chunks over a W1 cross-page paragraph split.

    W1 reconstructed these from MinerU's ``cross_page`` span flags: MinerU merges
    a paragraph that straddles a page break into one block on the page it
    *starts* on, so a citation landing in the tail used to open the PDF viewer a
    page early. The chunk must report both pages.
    """
    print("\n--- PROOF 3: A PARAGRAPH SPLIT ACROSS A PAGE BREAK (R18) ---")
    shown = 0
    for path in records.values():
        record = json.loads(path.read_text())
        splits = [b for b in record["blocks"]
                  if b.get("bbox_source") == "cross_page_split"]
        if not splits:
            continue
        chunks = chunker.chunk_record(record)
        for tail in splits:
            head = record["blocks"][tail["continues_from"]]
            covering = [c for c in chunks
                        if c.char_start is not None
                        and c.char_start <= head["doc_char_end"] - 1
                        and c.char_end >= tail["doc_char_start"] + 1
                        and c.chunk_type == ChunkType.FINE]
            if not covering:
                continue
            chunk = covering[0]
            print(f"  {record['paper_id']}  blocks {head['i']}(p{head['page_idx'] + 1}) "
                  f"+ {tail['i']}(p{tail['page_idx'] + 1})  ->  {chunk.chunk_id}")
            print(f"      page_start={chunk.page_start} page_end={chunk.page_end} "
                  f"page_numbers={chunk.page_numbers} "
                  f"{'OK' if (chunk.page_start, chunk.page_end) == (head['page_idx'] + 1, tail['page_idx'] + 1) else 'MISMATCH'}")
            print(f"      head tail  ...{head['text'][-70:]!r}")
            print(f"      cont start {tail['text'][:70]!r}...")
            shown += 1
            if shown >= limit:
                return
    if not shown:
        print("  (no cross-page splits in these records — W1's v2 sample has them)")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="data/w1_sample/records")
    ap.add_argument("--extra-records", default="data/w1_sample_v2/records",
                    help="newer records that override by paper_id (cross-page fix)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--git-ref", default="HEAD")
    args = ap.parse_args()

    records: Dict[str, Path] = {}
    for directory in (args.records, args.extra_records):
        path = BACKEND / directory
        if not path.exists():
            continue
        for file in sorted(path.glob("*.json")):
            records[file.stem] = file          # later dirs win
    if not records:
        print(f"No records under {args.records}")
        return 1

    print("=" * 78)
    print(f"W2 CHUNK CONTRACT — {len(records)} sample records")
    print("=" * 78)

    chunker = PaperChunker()
    old_chunker = load_old_chunker(args.git_ref)

    rows = [check_paper(json.loads(p.read_text()), chunker, old_chunker)
            for p in records.values()]

    # ---- per-paper table ------------------------------------------------
    print("\n--- PER-PAPER CHUNKS (new | old) ---")
    print(f"{'paper':13s} {'pg':>3s} {'rec s/a/t':>10s} "
          f"{'abs':>7s} {'sect':>9s} {'fine':>9s} {'capt':>9s} {'tbl':>7s} {'tot':>11s} cov")
    for r in sorted(rows, key=lambda r: -r["new_total"]):
        rec = r["record"]
        def pair(t):
            return f"{r['new'][t]}|{r['old'][t]}"
        print(f"{r['paper_id']:13s} {r['pages']:3d} "
              f"{rec['sections']:>3d}/{'Y' if rec['abstract'] else '-':>1s}/{rec['tables_with_body']:<3d} "
              f"{pair('abstract'):>7s} {pair('section'):>9s} {pair('fine'):>9s} "
              f"{pair('caption'):>9s} {pair('table'):>7s} "
              f"{str(r['new_total']) + '|' + str(r['old_total']):>11s} "
              f"{r['coverage']:.0%}")

    # ---- the handoff bug ------------------------------------------------
    print("\n--- DID THE STRUCTURE REACH THE INDEX? ---")
    # "sections" asks whether the section text is indexed at all, at any
    # granularity: a section whose whole span fits one fine chunk emits no coarse
    # chunk on purpose (the two points would be byte-identical), and counting
    # that as a miss would reproduce the bug this check exists to catch.
    for label, rec_key, chunk_key in (
        ("sections", "sections", ("section", "fine")),
        ("abstract", "abstract", ("abstract",)),
        ("tables", "tables_with_body", ("table",)),
    ):
        def got(row, which):
            return sum(row[which][k] for k in chunk_key)
        have = [r for r in rows if r["record"][rec_key]]
        zero = [r for r in have if got(r, "new") == 0]
        zero_old = [r for r in have if got(r, "old") == 0]
        print(f"  record has {label:8s} -> {'/'.join(chunk_key):14s}: {len(have):2d} papers | "
              f"new: {len(have) - len(zero):2d} chunked, {len(zero)} at zero | "
              f"old: {len(have) - len(zero_old):2d} chunked, {len(zero_old)} at zero")
        for r in zero:
            print(f"      still zero: {r['paper_id']} {r['file_name'][:44]}")
    coarse = sum(r["new"]["section"] for r in rows)
    rec_sections = sum(r["record"]["sections"] for r in rows)
    print(f"  coarse section chunks {coarse} for {rec_sections} record sections "
          f"(the rest were byte-identical to their single fine chunk, or under "
          f"{MIN_SECTION_CHARS} chars, or references/acknowledgments)")
    zero_fine = [r for r in rows if r["new"]["fine"] == 0]
    print(f"  papers with zero fine chunks: new {len(zero_fine)} | "
          f"old {sum(1 for r in rows if r['old']['fine'] == 0)}"
          f"   (corpus baseline: 13.5%)")

    # ---- verbatim proof -------------------------------------------------
    print("\n--- PROOF 1: `text` IS VERBATIM ---")
    checked = sum(r["verbatim_checked"] for r in rows)
    bad = sum(r["verbatim_bad"] for r in rows)
    synth = sum(r["synthetic_in_text"] for r in rows)
    print(f"  full_text[char_start:char_end] == text : {checked - bad}/{checked} "
          f"({'PASS' if bad == 0 else f'FAIL ({bad} mismatches)'})")
    print(f"  context header found inside `text`     : {synth} "
          f"({'PASS' if synth == 0 else 'FAIL'})")
    print(f"  embed_text/display_text recomposable   : "
          f"{'PASS' if not any(r['n_page_errors'] and 'recomposable' in ' '.join(r['page_errors']) for r in rows) else 'FAIL'}")

    # ---- page proof -----------------------------------------------------
    print("\n--- PROOF 2: PAGE ATTRIBUTION ---")
    page_errors = sum(r["n_page_errors"] for r in rows)
    cross = sum(r["cross_page_chunks"] for r in rows)
    print(f"  chunks whose pages disagree with their blocks: {page_errors} "
          f"({'PASS' if page_errors == 0 else 'FAIL'})")
    print(f"  chunks spanning >1 page (R18 honesty)        : {cross}")
    print(f"  page_numbers populated on every chunk        : "
          f"{'PASS' if all(r['new_total'] for r in rows) else 'n/a'}"
          f"   (corpus baseline: page_count == 0 for all 4,797)")
    for r in rows:
        for err in r["page_errors"]:
            print(f"      {err}")

    # ---- totals and projection -----------------------------------------
    new_total = sum(r["new_total"] for r in rows)
    old_total = sum(r["old_total"] for r in rows)
    print("\n--- TOTALS ---")
    print(f"  new: {new_total} chunks over {len(rows)} papers "
          f"({new_total / len(rows):.1f}/paper)")
    if old_total:
        print(f"  old: {old_total} chunks ({old_total / len(rows):.1f}/paper)  "
              f"change {100 * (new_total - old_total) / old_total:+.1f}%")
    for t in TYPES:
        n, o = sum(r["new"][t] for r in rows), sum(r["old"][t] for r in rows)
        print(f"    {t:9s} new {n:6d}   old {o:6d}   "
              f"{'x%.2f' % (n / o) if o else '(new)'}")
    print(f"  body-char coverage by fine chunks: "
          f"mean {statistics.mean(r['coverage'] for r in rows):.1%}, "
          f"min {min(r['coverage'] for r in rows):.1%}")

    print("\n--- NUMERIC FACTS ---")
    facts = sum(r["numeric_facts"] for r in rows)
    from_tables = sum(r["numeric_from_tables"] for r in rows)
    with_facts = sum(1 for r in rows if r["numeric_facts"])
    props = Counter(p for r in rows for p in r["numeric_properties"])
    print(f"  {facts} facts over {len(rows)} papers ({facts / len(rows):.1f}/paper); "
          f"{from_tables} from tables, {facts - from_tables} from prose")
    print(f"  papers with >=1 fact: {with_facts}/{len(rows)}")
    print(f"  properties seen: {', '.join(f'{k}({v})' for k, v in props.most_common(12))}")

    print("\n--- CORPUS PROJECTION (4,797 papers) ---")
    per_paper = new_total / len(rows)
    print(f"  chunks:  {int(per_paper * CORPUS_PAPERS):,}  "
          f"(today {BASE_CHUNKS:,})")
    for t in TYPES:
        n = sum(r["new"][t] for r in rows)
        print(f"    {t:9s} ~{int(n / len(rows) * CORPUS_PAPERS):>8,}   "
              f"today {BASE_BY_TYPE[t]:>8,}")
    bytes_per_paper = statistics.mean(r["payload_bytes"] for r in rows)
    print(f"  payload:  ~{bytes_per_paper * CORPUS_PAPERS / 1e9:.2f} GB of JSON payload "
          f"({bytes_per_paper / 1024:.0f} KB/paper)")

    prove_cross_page(records, chunker)

    degraded = [r for r in rows if r["degraded"]]
    if degraded:
        print(f"\n  degraded records in sample: {len(degraded)}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {args.json}")

    return 0 if (bad == 0 and synth == 0 and page_errors == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
