#!/usr/bin/env python
"""Measure the in-text figure/table reference resolver on the W1 sample.

Reads the persisted records (no extraction, no API calls, no Qdrant) and reports:

  * how many objects get a label, and from where (caption / OCR-repaired
    caption / reading-order sequence)
  * how many in-text references exist and how many resolve
  * the same reference set scored with *naive* label parsing, so the delta is
    attributable to label parsing rather than to a changed denominator
  * a breakdown of what still misses, bucketed by cause

    python scripts/w2b_figure_refs_report.py --records data/w1_sample/records
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing import figure_refs as fr          # noqa: E402
from preprocessing import paper_record as pr         # noqa: E402

# --- the naive baseline, reproduced so the comparison is like-for-like -------
# One label per caption, only when the caption opens with "<Kind> <digits>";
# one reference per "<Kind> <digits>" in body prose; exact string match.
NAIVE_CAP = re.compile(r"^\s*(Fig(?:ure)?s?\.?|Tables?|Schemes?|Charts?)\s*(\d+)", re.I)
NAIVE_REF = re.compile(r"\b(Fig(?:ure)?s?\.?|Tables?|Schemes?|Charts?)\s*(\d+)", re.I)


def naive_kind(word):
    w = word.lower().rstrip(".").rstrip("s")
    for prefix, kind in (("fig", "figure"), ("tab", "table"),
                         ("scheme", "scheme"), ("chart", "chart")):
        if w.startswith(prefix):
            return kind
    return None


def naive_label_index(record):
    index = {}
    for i in record.get("caption_blocks") or []:
        match = NAIVE_CAP.match(record["blocks"][i]["text"])
        if not match:
            continue
        kind = naive_kind(match.group(1))
        if kind:
            index.setdefault((kind, match.group(2)), i)
    return index


def body_blocks(record):
    for block in record.get("blocks") or []:
        if block["type"] in pr.BODY_BLOCK_TYPES and block["text"]:
            yield block


def section_name(record, block):
    sections = {s["id"]: s for s in (record.get("sections") or [])}
    section = sections.get(block.get("section_id"))
    return (section or {}).get("normalized_name")


SUPP_SECTIONS = {"references", "acknowledgments", "abbreviations"}


def miss_bucket(record, reference, block, index_all):
    """Why did this reference not resolve?"""
    kind, number = reference["kind"], reference["number"]
    tail = block["text"][reference["end"]:reference["end"] + 60].lower()
    if number.startswith("S"):
        return "supplementary label (Figure S1 etc.)"
    if re.search(r"support(ing|ary)\s+(information|material)|\bsi\b|supplement",
                 tail):
        return "explicitly marked Supporting Information"
    name = section_name(record, block)
    if name in SUPP_SECTIONS:
        return f"inside a {name} section (another paper's figure)"
    same_kind = sorted({n for k, n in index_all if k == kind},
                       key=lambda v: (len(v), v))
    if not same_kind:
        return f"paper has no labelled {kind} at all"
    numeric = [v for v in same_kind if v.replace(".", "").isdigit()]
    if numeric and number.replace(".", "").isdigit():
        try:
            top = max(float(v) for v in numeric)
            if float(number) > top:
                return f"number above the paper's highest {kind} ({number} > {top:g})"
        except ValueError:
            pass
    if "." in number:
        return "chapter-style number with no matching caption"
    return f"no {kind} labelled {number} in this paper"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="data/w1_sample/records")
    ap.add_argument("--examples", type=int, default=12)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    paths = sorted(Path(args.records).glob("*.json"))
    if not paths:
        sys.exit(f"no records in {args.records}")

    label_src = collections.Counter()
    totals = collections.Counter()
    buckets = collections.Counter()
    examples = []
    per_paper = []

    for path in paths:
        record = json.loads(path.read_text())
        summary = fr.label_record_objects(record)
        for key in ("caption", "caption_fuzzy", "sequence"):
            label_src[key] += summary[key]
        label_src["unlabeled"] += summary["unlabeled"]
        label_src["objects"] += summary["objects"]
        totals["duplicate_labels"] += summary["duplicates"]

        index_all = fr.label_index(record, include_inferred=True)
        index_caption = fr.label_index(record, include_inferred=False)
        naive_index = naive_label_index(record)

        refs = hits = hits_caption_only = naive_hits = 0
        for block in body_blocks(record):
            for reference in fr.find_references(block["text"]):
                refs += 1
                target = fr.resolve_one(reference, index_all)
                if target is not None:
                    hits += 1
                    totals[f"match_{target['match']}"] += 1
                    if target["inferred"]:
                        totals["resolved_via_inferred_label"] += 1
                else:
                    bucket = miss_bucket(record, reference, block, index_all)
                    buckets[bucket] += 1
                    if len(examples) < args.examples:
                        pos = reference["start"]
                        examples.append({
                            "file": record["file_name"][:48],
                            "ref": reference["text"],
                            "bucket": bucket,
                            "context": block["text"][max(0, pos - 45):pos + 40]
                                       .replace("\n", " "),
                        })
                if fr.resolve_one(reference, index_caption) is not None:
                    hits_caption_only += 1
                key = (reference["kind"], reference["number"])
                if key in naive_index:
                    naive_hits += 1

        # the naive detector's own reference set, scored naively
        naive_refs = naive_naive = 0
        for block in body_blocks(record):
            for match in NAIVE_REF.finditer(block["text"]):
                kind = naive_kind(match.group(1))
                if not kind:
                    continue
                naive_refs += 1
                if (kind, match.group(2)) in naive_index:
                    naive_naive += 1

        totals["refs"] += refs
        totals["resolved"] += hits
        totals["resolved_caption_labels_only"] += hits_caption_only
        totals["resolved_naive_labels"] += naive_hits
        totals["naive_refs"] += naive_refs
        totals["naive_resolved"] += naive_naive
        per_paper.append({
            "file_name": record["file_name"],
            "objects": summary["objects"],
            "labeled": summary["objects"] - summary["unlabeled"],
            "refs": refs, "resolved": hits,
        })

    def pct(a, b):
        return f"{a}/{b} = {100.0 * a / b:.1f}%" if b else "n/a"

    print(f"records: {len(paths)}  from {args.records}\n")
    print("OBJECT LABELLING")
    print(f"  objects (figures + tables)      {label_src['objects']}")
    print(f"  labelled from caption           {pct(label_src['caption'], label_src['objects'])}")
    print(f"  labelled from OCR-repaired cap. {pct(label_src['caption_fuzzy'], label_src['objects'])}")
    print(f"  labelled from reading sequence  {pct(label_src['sequence'], label_src['objects'])}")
    print(f"  still unlabelled                {pct(label_src['unlabeled'], label_src['objects'])}")
    print(f"  duplicate labels (continuations) {totals['duplicate_labels']}")

    print("\nREFERENCE RESOLUTION")
    print(f"  naive detector, naive labels    {pct(totals['naive_resolved'], totals['naive_refs'])}")
    print(f"  this detector, naive labels     {pct(totals['resolved_naive_labels'], totals['refs'])}")
    print(f"  this detector, caption labels   {pct(totals['resolved_caption_labels_only'], totals['refs'])}")
    print(f"  this detector, all labels       {pct(totals['resolved'], totals['refs'])}")
    print(f"    of which exact (kind, number) {totals['match_exact']}")
    print(f"    of which cross-kind family    {totals['match_family']}")
    print(f"    of which via inferred label   {totals['resolved_via_inferred_label']}")

    print("\nMISSES BY CAUSE")
    total_miss = totals["refs"] - totals["resolved"]
    for bucket, n in buckets.most_common():
        print(f"  {n:4d} ({100.0 * n / max(total_miss, 1):5.1f}%)  {bucket}")

    print("\nMISS EXAMPLES")
    for ex in examples:
        print(f"  [{ex['bucket']}] {ex['ref']!r}")
        print(f"      ...{ex['context']}...")
        print(f"      {ex['file']}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "records": len(paths),
            "labels": dict(label_src),
            "totals": dict(totals),
            "misses": dict(buckets),
            "per_paper": per_paper,
        }, indent=1))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
