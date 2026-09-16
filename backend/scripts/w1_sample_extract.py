#!/usr/bin/env python
"""W1 validation harness: extract a sample of papers and compare old vs new.

Runs the real production extraction path (`MinerUExtractor.extract`, subprocess
isolated), writes each paper record to a sample directory, and measures:

  tables     new (MinerU `table_body`) vs old (`item["latex"] or item["text"]`)
  captions   new (MinerU caption lists) vs old (regex over the whole markdown)
  sections   new (heading blocks) vs old (`SectionDetector` over raw lines)
  abstract   new (leading block sequence) vs old (`extract_abstract` patterns)
  offsets    every block round-trips byte-exactly through page and doc text
  size       record bytes on disk

It does NOT touch Qdrant, the database, or the production processed_data dir.

    python scripts/w1_sample_extract.py --out data/w1_sample [--limit N]
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing import paper_record as pr  # noqa: E402
from preprocessing.pdf_processor import MinerUExtractor  # noqa: E402
from preprocessing.section_detector import SectionDetector  # noqa: E402


# --- the pre-W1 behaviour, reproduced here so the comparison is like-for-like ---

OLD_CAPTION_PATTERNS = [
    r'(?:Figure|Fig\.?)\s*(\d+[a-zA-Z]?)\s*[\.:\-–—]?\s*([^\n]+)',
    r'Table\s*(\d+[a-zA-Z]?)\s*[\.:\-–—]?\s*([^\n]+)',
    r'Scheme\s*(\d+[a-zA-Z]?)\s*[\.:\-–—]?\s*([^\n]+)',
]


def old_captions_from_markdown(markdown: str):
    captions = []
    for pattern in OLD_CAPTION_PATTERNS:
        for match in re.finditer(pattern, markdown, re.IGNORECASE):
            text = match.group(0).strip()
            if len(text) >= 20:
                captions.append(text[:1000])
    return list(set(captions))


def old_full_text(record):
    """What `_extract_text_from_content` produced.

    Note MinerU reports headings as `type: "text"`, so they fell through the
    `item_type == "text"` branch and the `elif item_type == "title"` branch was
    unreachable; every text item, heading or not, was joined with a single "\\n".
    """
    parts = [b["text"] for b in record["blocks"]
             if b["type"] in (pr.BLOCK_TEXT, pr.BLOCK_TITLE)]
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default="/Volumes/ARC/ARC/papers")
    ap.add_argument("--out", default="data/w1_sample")
    ap.add_argument("--list", default=None, help="file with one filename per line")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    papers_dir = Path(args.papers)
    out_dir = Path(args.out).resolve()
    rec_dir = out_dir / "records"
    rec_dir.mkdir(parents=True, exist_ok=True)

    if args.list:
        names = [l.strip() for l in open(args.list) if l.strip() and not l.startswith("#")]
    else:
        names = sorted(p.name for p in papers_dir.glob("*.pdf") if not p.name.startswith("._"))
    if args.limit:
        names = names[:args.limit]

    detector = SectionDetector()
    extractor = MinerUExtractor(timeout=args.timeout)
    results = []
    results_path = out_dir / "results.json"

    for n, name in enumerate(names, 1):
        pdf_path = papers_dir / name
        paper_id = pr.paper_id_for_filename(name)
        row = {"n": n, "file_name": name, "paper_id": paper_id}
        t0 = time.time()
        print(f"[{n}/{len(names)}] {name}", flush=True)
        try:
            content = extractor.extract(pdf_path, paper_id=paper_id)
            record = content.record
            if record is None:
                raise RuntimeError("no record returned")

            path = pr.save_record(record, base=rec_dir)
            old_text = old_full_text(record)
            old_sections = detector.detect_sections(old_text)
            old_abstract = detector.extract_abstract(old_text)
            old_caps = old_captions_from_markdown(content.markdown or "")

            verify = pr.verify_record_offsets(record)
            stats = record["stats"]

            row.update({
                "ok": True,
                "degraded": record["degraded"],
                "degraded_reason": record["degraded_reason"],
                "extractor": record["extractor"],
                "seconds": round(time.time() - t0, 1),
                "pages": stats["n_pages"],
                "blocks": stats["n_blocks"],
                "blocks_by_type": stats["blocks_by_type"],
                "mineru_item_keys": stats.get("mineru_item_keys", {}),
                # tables
                "new_tables": stats["n_tables"],
                "new_tables_with_body": stats["n_tables_with_body"],
                "old_tables": 0,  # recomputed below from observed keys
                # captions
                "new_captions": stats["n_captions"],
                "new_captions_figure": len(pr.captions(record, "figure")),
                "new_captions_table": len(pr.captions(record, "table")),
                "old_captions": len(old_caps),
                # sections / abstract
                "new_sections": stats["n_sections"],
                "new_sections_named": sum(
                    1 for s in record["sections"]
                    if s["normalized_name"] not in ("body", "frontmatter")),
                "old_sections": len(old_sections),
                "new_abstract": stats["has_abstract"],
                "new_abstract_source": stats["abstract_source"],
                "new_abstract_len": len((record["abstract"] or {}).get("text", "")),
                "old_abstract": old_abstract is not None,
                "old_abstract_len": len(old_abstract or ""),
                # equations / figures
                "new_equations": stats["n_equations"],
                "new_equations_latex": sum(1 for e in record["equations"] if e["has_latex"]),
                "new_figures": stats["n_figures"],
                # offsets and size
                "offsets_ok": verify["ok"],
                "offset_errors": verify["n_errors"],
                "offset_error_sample": verify["errors"][:3],
                "blocks_verified": verify["blocks_checked"],
                "record_bytes": path.stat().st_size,
                "body_chars": stats["body_chars"],
                "total_chars": stats["total_chars"],
                "bbox_coverage": round(
                    sum(1 for b in record["blocks"] if b["bbox"]) / max(len(record["blocks"]), 1), 3),
            })
            # The old table reader asked for keys MinerU never emits; confirm per paper.
            tkeys = set(stats.get("mineru_item_keys", {}).get("table", []))
            row["old_tables"] = stats["n_tables"] if (tkeys & {"latex", "text"}) else 0
            row["old_table_keys_present"] = sorted(tkeys & {"latex", "text"})
        except Exception as exc:
            row.update({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                        "seconds": round(time.time() - t0, 1)})
            traceback.print_exc()

        results.append(row)
        results_path.write_text(json.dumps(results, indent=1, ensure_ascii=False))

    print(f"\nwrote {results_path} ({len(results)} papers)")


if __name__ == "__main__":
    main()
