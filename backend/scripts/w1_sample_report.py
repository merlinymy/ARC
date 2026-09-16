#!/usr/bin/env python
"""Summarise the W1 sample run: old vs new, offsets, storage projection."""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing import paper_record as pr  # noqa: E402

CORPUS_PAPERS = 4797
BASE_SECTION_COVERAGE = 0.854   # 14.6% of papers get zero section chunks
BASE_ABSTRACT_COVERAGE = 0.46   # only 46% of papers get an abstract chunk


def pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def main(out_dir="data/w1_sample"):
    out = Path(out_dir)
    rows = json.loads((out / "results.json").read_text())
    ok = [r for r in rows if r.get("ok")]
    bad = [r for r in rows if not r.get("ok")]
    mineru = [r for r in ok if not r["degraded"]]
    degraded = [r for r in ok if r["degraded"]]

    print("=" * 78)
    print(f"W1 SAMPLE — {len(rows)} papers | {len(ok)} extracted | "
          f"{len(mineru)} mineru | {len(degraded)} degraded | {len(bad)} failed")
    print("=" * 78)

    if bad:
        print("\nFAILED:")
        for r in bad:
            print(f"  {r['file_name'][:60]:60s} {r.get('error')}")

    # ---- tables -----------------------------------------------------------
    print("\n--- TABLES (R1) ---")
    tot_new = sum(r["new_tables"] for r in ok)
    tot_body = sum(r["new_tables_with_body"] for r in ok)
    tot_old = sum(r["old_tables"] for r in ok)
    with_t = sum(1 for r in ok if r["new_tables_with_body"] > 0)
    print(f"  new: {tot_new} table blocks, {tot_body} with an HTML body")
    print(f"  old: {tot_old} (item['latex'] / item['text'] — keys MinerU never emits)")
    print(f"  papers with >=1 table body: {with_t}/{len(ok)} ({pct(with_t, len(ok))})")
    if mineru:
        counts = sorted((r["new_tables_with_body"], r["file_name"]) for r in mineru)
        print(f"  per-paper: min {counts[0][0]} / median "
              f"{statistics.median([c[0] for c in counts])} / max {counts[-1][0]} "
              f"({counts[-1][1][:50]})")

    # ---- captions ---------------------------------------------------------
    print("\n--- CAPTIONS (R2) ---")
    new_c = sum(r["new_captions"] for r in ok)
    old_c = sum(r["old_captions"] for r in ok)
    print(f"  new (MinerU image_caption/table_caption): {new_c}"
          f"  (figure {sum(r['new_captions_figure'] for r in ok)},"
          f" table {sum(r['new_captions_table'] for r in ok)})")
    print(f"  old (regex over whole markdown):          {old_c}")
    if old_c:
        print(f"  reduction: {pct(old_c - new_c, old_c)}  "
              f"(x{old_c / max(new_c, 1):.1f} fewer)")
    print(f"  mean per paper: new {new_c / max(len(ok),1):.1f}  old {old_c / max(len(ok),1):.1f}")
    print(f"  corpus projection: 93,144 -> ~{int(93144 * new_c / max(old_c,1)):,}")

    # ---- sections / abstract ---------------------------------------------
    print("\n--- SECTIONS + ABSTRACT (R16) ---")
    new_s = sum(1 for r in ok if r["new_sections"] > 0)
    new_sn = sum(1 for r in ok if r["new_sections_named"] > 0)
    old_s = sum(1 for r in ok if r["old_sections"] > 0)
    print(f"  papers with >=1 section:  new {new_s}/{len(ok)} ({pct(new_s, len(ok))})"
          f"   old {old_s}/{len(ok)} ({pct(old_s, len(ok))})"
          f"   corpus baseline {BASE_SECTION_COVERAGE:.1%}")
    print(f"  papers with >=1 *named* section (methods/results/...): "
          f"{new_sn}/{len(ok)} ({pct(new_sn, len(ok))})")
    print(f"  mean sections/paper: new {statistics.mean([r['new_sections'] for r in ok]):.1f}"
          f"   old {statistics.mean([r['old_sections'] for r in ok]):.1f}")
    new_a = sum(1 for r in ok if r["new_abstract"])
    old_a = sum(1 for r in ok if r["old_abstract"])
    print(f"  papers with an abstract:  new {new_a}/{len(ok)} ({pct(new_a, len(ok))})"
          f"   old {old_a}/{len(ok)} ({pct(old_a, len(ok))})"
          f"   corpus baseline {BASE_ABSTRACT_COVERAGE:.0%}")
    srcs = {}
    for r in ok:
        if r["new_abstract"]:
            srcs[r["new_abstract_source"]] = srcs.get(r["new_abstract_source"], 0) + 1
    print(f"  abstract route: {srcs}")
    lens = [r["new_abstract_len"] for r in ok if r["new_abstract"]]
    oldlens = [r["old_abstract_len"] for r in ok if r["old_abstract"]]
    if lens:
        print(f"  abstract length: new median {int(statistics.median(lens))} chars"
              + (f"   old median {int(statistics.median(oldlens))} chars" if oldlens else ""))

    # ---- equations / figures / offsets ------------------------------------
    print("\n--- EQUATIONS, FIGURES, PAGES (R6 + dropped content) ---")
    print(f"  equation blocks: {sum(r['new_equations'] for r in ok)}"
          f" ({sum(r['new_equations_latex'] for r in ok)} with LaTeX) — previously all dropped")
    print(f"  figure blocks:   {sum(r['new_figures'] for r in ok)}")
    print(f"  pages recorded:  {sum(r['pages'] for r in ok)} across {len(ok)} papers"
          f" (page_count was 0 for all 4,797)")
    bb = [r["bbox_coverage"] for r in mineru]
    if bb:
        print(f"  blocks with a bbox: mean {statistics.mean(bb):.1%} (mineru papers)")

    print("\n--- OFFSETS (W3 prerequisite) ---")
    bad_off = [r for r in ok if not r["offsets_ok"]]
    print(f"  papers verified: {sum(1 for r in ok if r['offsets_ok'])}/{len(ok)}")
    print(f"  blocks verified: {sum(r['blocks_verified'] for r in ok):,} "
          f"(page slice AND doc slice byte-exact, plus page md5)")
    if bad_off:
        for r in bad_off[:5]:
            print(f"  FAIL {r['file_name'][:50]}: {r['offset_errors']} errors "
                  f"{r['offset_error_sample']}")

    # ---- storage ----------------------------------------------------------
    print("\n--- STORAGE ---")
    sizes = [r["record_bytes"] for r in ok]
    total = sum(sizes)
    mean = total / max(len(sizes), 1)
    print(f"  sample: {total/1e6:.1f} MB over {len(sizes)} records")
    print(f"  per record: min {min(sizes)/1024:.0f} KB / median "
          f"{statistics.median(sizes)/1024:.0f} KB / mean {mean/1024:.0f} KB / "
          f"max {max(sizes)/1024:.0f} KB")
    print(f"  projected for {CORPUS_PAPERS:,} papers: {mean * CORPUS_PAPERS / 1e9:.2f} GB")
    print(f"  ratio to extracted text: "
          f"{total / max(sum(r['total_chars'] for r in ok), 1):.2f} bytes/char")

    # ---- timing -----------------------------------------------------------
    secs = [r["seconds"] for r in ok]
    if secs:
        print(f"\n--- TIMING ---")
        print(f"  mean {statistics.mean(secs):.0f}s / median {statistics.median(secs):.0f}s "
              f"/ max {max(secs):.0f}s per paper")
        print(f"  projected full corpus: {statistics.mean(secs) * CORPUS_PAPERS / 3600:.0f} h")

    # ---- MinerU schema reality check --------------------------------------
    print("\n--- MINERU SCHEMA OBSERVED ---")
    union = {}
    for r in ok:
        for t, keys in (r.get("mineru_item_keys") or {}).items():
            union.setdefault(t, set()).update(keys)
    for t in sorted(union):
        print(f"  {t:12s} {sorted(union[t])}")

    # ---- per-paper table --------------------------------------------------
    print("\n--- PER PAPER ---")
    print(f"  {'file':<44}{'pg':>3}{'tblB':>5}{'capN':>5}{'capO':>6}"
          f"{'secN':>5}{'secO':>5}{'absN':>5}{'absO':>5}{'eq':>4}{'KB':>6}")
    for r in rows:
        if not r.get("ok"):
            print(f"  {r['file_name'][:43]:<44}  FAILED {r.get('error','')[:40]}")
            continue
        mark = "*" if r["degraded"] else " "
        print(f" {mark}{r['file_name'][:43]:<44}{r['pages']:>3}{r['new_tables_with_body']:>5}"
              f"{r['new_captions']:>5}{r['old_captions']:>6}{r['new_sections']:>5}"
              f"{r['old_sections']:>5}{'Y' if r['new_abstract'] else '-':>5}"
              f"{'Y' if r['old_abstract'] else '-':>5}{r['new_equations']:>4}"
              f"{r['record_bytes']/1024:>6.0f}")
    print("  (* = degraded / pypdfium2 fallback)")


if __name__ == "__main__":
    main(*sys.argv[1:])
