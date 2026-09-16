#!/usr/bin/env python
"""Show concrete artifacts from the W1 sample: tables, captions, sections, abstracts.

Re-runs the pre-W1 caption regex over the record's own text so the old and new
caption lists come from exactly the same extraction.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing import paper_record as pr  # noqa: E402

OLD_PATTERNS = [
    r'(?:Figure|Fig\.?)\s*(\d+[a-zA-Z]?)\s*[\.:\-–—]?\s*([^\n]+)',
    r'Table\s*(\d+[a-zA-Z]?)\s*[\.:\-–—]?\s*([^\n]+)',
    r'Scheme\s*(\d+[a-zA-Z]?)\s*[\.:\-–—]?\s*([^\n]+)',
]


def old_captions(text):
    out = []
    for pattern in OLD_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            s = m.group(0).strip()
            if len(s) >= 20:
                out.append(s[:1000])
    return list(dict.fromkeys(out))


def load(rec_dir):
    for p in sorted(Path(rec_dir).glob("*.json")):
        yield json.loads(p.read_text())


def main(rec_dir="data/w1_sample/records"):
    records = list(load(rec_dir))

    print("#" * 78)
    print("# TABLES — a chunk type that had zero members across all 4,797 papers")
    print("#" * 78)
    shown = 0
    for r in sorted(records, key=lambda r: -r["stats"]["n_tables_with_body"]):
        for t in r["tables"]:
            if not t["has_body"] or shown >= 3:
                continue
            body = r["blocks"][t["block"]]["text"]
            print(f"\n{r['file_name']}  (page {t['page_idx']}, {t['n_rows']}x{t['n_cols']}, "
                  f"bbox {t['bbox']} per-mille)")
            for c in t["caption_blocks"]:
                print(f"  caption : {r['blocks'][c]['text'].strip()[:220]}")
            for f in t["footnote_blocks"]:
                print(f"  footnote: {r['blocks'][f]['text'].strip()[:220]}")
            print(f"  body    : {body[:600]}")
            shown += 1

    print("\n" + "#" * 78)
    print("# CAPTIONS — MinerU caption lists vs the old regex over body prose")
    print("#" * 78)
    picked = [r for r in records if not r["degraded"] and r["stats"]["n_captions"] >= 3][:2]
    for r in picked:
        text = pr.full_text(r)
        old = old_captions(text)
        new = pr.captions(r)
        print(f"\n=== {r['file_name']}  new {len(new)} / old {len(old)}")
        print("  NEW (MinerU image_caption / table_caption):")
        for c in new[:4]:
            print(f"    + {c.strip()[:190]}")
        real = {re.sub(r'\s+', ' ', c).strip().lower()[:60] for c in new}
        spurious = [c for c in old
                    if re.sub(r'\s+', ' ', c).strip().lower()[:60] not in real]
        print(f"  OLD regex matches that are NOT captions ({len(spurious)} of {len(old)}):")
        for c in spurious[:6]:
            print(f"    - {re.sub(chr(10), ' ', c)[:190]}")

    print("\n" + "#" * 78)
    print("# SECTIONS — structural, from MinerU heading blocks")
    print("#" * 78)
    for r in sorted(records, key=lambda r: -r["stats"]["n_sections"])[:2]:
        print(f"\n=== {r['file_name']}  ({r['stats']['n_sections']} sections)")
        for s in r["sections"][:18]:
            parent = f" <- {s['parent_id']}" if s["parent_id"] is not None else ""
            print(f"  [{s['id']:2d}] p{s['page_start']}-{s['page_end']} "
                  f"{s['normalized_name']:<20}{parent:<8} {s['char_len']:>6}c  {s['name'][:52]!r}")

    print("\n" + "#" * 78)
    print("# ABSTRACTS — from the leading block sequence")
    print("#" * 78)
    seen = set()
    for r in records:
        a = r["abstract"]
        if not a or a["source"] in seen:
            continue
        seen.add(a["source"])
        print(f"\n=== {r['file_name']}  route={a['source']} "
              f"page={a['page_idx']} chars={len(a['text'])} verbatim={a.get('verbatim_span')}")
        print(f"  {a['text'][:400]}")


if __name__ == "__main__":
    main(*sys.argv[1:])
