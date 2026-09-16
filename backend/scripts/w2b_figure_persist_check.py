#!/usr/bin/env python
"""Prove the figure crops survive extraction, and measure what they cost.

Re-extracts a handful of papers through the real production path
(``MinerUExtractor.extract``, subprocess isolated) into a **scratch** directory,
then checks, per paper:

  crops on disk        every ``img_path`` in the record resolves to a real file
  crops decode         each one opens as an image with non-zero dimensions
  idempotent           a second extraction of the same paper leaves the same
                       file set -- no orphan accumulation
  offsets intact       ``verify_record_offsets`` still passes (the crop work
                       must not perturb the text coordinate system)
  chunk contract       ``full_text(record)[c.char_start:c.char_end] == c.text``
                       on every chunk, figure-bearing ones included
  figure coordinates   caption and table chunks carry the *figure's* page and
                       bbox, not the caption's own rectangle

Writes nothing to ``processed_data/``, Qdrant, or the database.

    python scripts/w2b_figure_persist_check.py --limit 4
"""

import argparse
import json
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing import figure_refs as fx            # noqa: E402
from preprocessing import paper_record as pr           # noqa: E402
from preprocessing.chunker import PaperChunker         # noqa: E402
from preprocessing.pdf_processor import MinerUExtractor  # noqa: E402


def crop_facts(record, base):
    """Resolve every asset reference and decode each file."""
    from PIL import Image

    paper_id = record["paper_id"]
    facts = {"referenced": 0, "resolved": 0, "decoded": 0, "bytes": 0,
             "dangling": [], "by_block_type": {}, "sizes": []}
    for block in record["blocks"]:
        ref = block.get("img_path")
        if not ref:
            continue
        facts["referenced"] += 1
        facts["by_block_type"][block["type"]] = \
            facts["by_block_type"].get(block["type"], 0) + 1
        path = pr.asset_path(paper_id, ref, base)
        if path is None:
            facts["dangling"].append(ref)
            continue
        facts["resolved"] += 1
        facts["bytes"] += path.stat().st_size
        try:
            with Image.open(path) as img:
                width, height = img.size
            if width > 0 and height > 0:
                facts["decoded"] += 1
                facts["sizes"].append((width, height))
        except Exception as exc:                        # noqa: BLE001
            facts["dangling"].append(f"{ref}: undecodable ({exc})")
    return facts


def chunk_facts(record):
    """The W2 invariant, plus what the figure chunks now carry."""
    chunker = PaperChunker()
    doc = pr.full_text(record)
    chunks = chunker.chunk_record(record)
    out = {"chunks": len(chunks), "verbatim_violations": 0,
           "caption_chunks": 0, "table_chunks": 0,
           "with_figure_page_and_bbox": 0, "with_figure_image": 0,
           "figure_bbox_equals_object": 0, "resolved_refs": 0,
           "chunks_with_refs": 0, "example": None}
    objects = {e["figure_id"]: e
               for e in (record.get("figures") or []) + (record.get("tables") or [])
               if e.get("figure_id")}
    for chunk in chunks:
        if chunk.char_start is not None and chunk.text_is_verbatim:
            if doc[chunk.char_start:chunk.char_end] != chunk.text:
                out["verbatim_violations"] += 1
        if chunk.chunk_type.value == "caption":
            out["caption_chunks"] += 1
        if chunk.chunk_type.value == "table":
            out["table_chunks"] += 1
        if chunk.figure_page is not None and chunk.figure_bbox:
            out["with_figure_page_and_bbox"] += 1
            entry = objects.get(chunk.figure_id or "")
            if entry is not None:
                target = entry.get("image_bbox") or entry.get("bbox")
                if (chunk.figure_bbox == target
                        and chunk.figure_page == (entry["page_idx"] + 1)):
                    out["figure_bbox_equals_object"] += 1
        if chunk.figure_image:
            out["with_figure_image"] += 1
        if chunk.figure_refs:
            out["chunks_with_refs"] += 1
            out["resolved_refs"] += len(chunk.figure_refs)
        if out["example"] is None and chunk.chunk_type.value == "caption" \
                and chunk.figure_bbox:
            payload = chunk.to_payload()
            out["example"] = {k: payload[k] for k in (
                "chunk_id", "chunk_type", "page_start", "page_end", "bbox",
                "char_start", "char_end", "figure_id", "figure_kind",
                "figure_label", "figure_page", "figure_bbox", "figure_image",
                "has_figure_image")}
            out["example"]["text"] = chunk.text[:110]
            out["example"]["verbatim_slice_matches"] = (
                doc[chunk.char_start:chunk.char_end] == chunk.text)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default="/Volumes/ARC/ARC/papers")
    ap.add_argument("--out", default="data/w2b_figures")
    ap.add_argument("--list", default="data/w1_sample/sample_list.txt")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--keep", action="store_true",
                    help="keep the scratch directory afterwards")
    args = ap.parse_args()

    papers_dir = Path(args.papers)
    out_dir = Path(args.out).resolve()
    rec_dir = out_dir / "records"
    if rec_dir.exists():
        shutil.rmtree(rec_dir)
    rec_dir.mkdir(parents=True, exist_ok=True)

    listing = Path(args.list)
    if listing.exists():
        names = [l.strip() for l in listing.read_text().splitlines()
                 if l.strip() and not l.startswith("#")]
    else:
        names = sorted(p.name for p in papers_dir.glob("*.pdf")
                       if not p.name.startswith("._"))
    random.Random(args.seed).shuffle(names)
    names = [n for n in names if (papers_dir / n).exists()][:args.limit]

    extractor = MinerUExtractor(timeout=args.timeout, assets_base=rec_dir)
    rows = []

    for n, name in enumerate(names, 1):
        pdf_path = papers_dir / name
        paper_id = pr.paper_id_for_filename(name)
        print(f"\n[{n}/{len(names)}] {name}  (paper_id {paper_id})", flush=True)
        row = {"file_name": name, "paper_id": paper_id}
        t0 = time.time()
        try:
            content = extractor.extract(pdf_path, paper_id=paper_id,
                                        assets_base=rec_dir)
            record = content.record
            if record is None:
                raise RuntimeError("no record returned")
            record_file = pr.save_record(record, base=rec_dir)

            crops = crop_facts(record, rec_dir)
            verify = pr.verify_record_offsets(record)
            labels = fx.label_record_objects(record)
            chunks = chunk_facts(record)
            crop_dir = pr.figures_dir(paper_id, rec_dir)
            on_disk = sorted(p.name for p in crop_dir.glob("*")
                             if p.is_file()) if crop_dir.is_dir() else []

            row.update({
                "ok": True,
                "seconds": round(time.time() - t0, 1),
                "schema_version": record["schema_version"],
                "degraded": record["degraded"],
                "pages": record["stats"]["n_pages"],
                "figures": record["stats"]["n_figures"],
                "tables": record["stats"]["n_tables"],
                "equations": record["stats"]["n_equations"],
                "crops": crops,
                "files_on_disk": len(on_disk),
                "orphans": sorted(set(on_disk) -
                                  {Path(r).name for r in pr.asset_refs(record)}),
                "record_assets": record["assets"],
                "pruned_first_pass": record["stats"].get("assets_pruned"),
                "offsets_ok": verify["ok"],
                "offset_errors": verify["n_errors"],
                "labels": labels,
                "chunks": chunks,
                "record_bytes": record_file.stat().st_size,
                "crop_dir": str(crop_dir),
            })
            print(f"    {row['figures']} figures, {row['tables']} tables, "
                  f"{crops['resolved']}/{crops['referenced']} crops resolve, "
                  f"{crops['decoded']} decode, {crops['bytes'] / 1e6:.2f} MB, "
                  f"offsets_ok={verify['ok']}, "
                  f"verbatim_violations={chunks['verbatim_violations']}")
        except Exception as exc:                        # noqa: BLE001
            row.update({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                        "seconds": round(time.time() - t0, 1)})
            traceback.print_exc()
        rows.append(row)
        (out_dir / "results.json").write_text(json.dumps(rows, indent=1))

    # --- re-extract the first paper to prove nothing accumulates ------------
    rerun = None
    good = [r for r in rows if r.get("ok") and r.get("figures")]
    if good:
        target = good[0]
        name, paper_id = target["file_name"], target["paper_id"]
        crop_dir = pr.figures_dir(paper_id, rec_dir)
        before = sorted(p.name for p in crop_dir.glob("*") if p.is_file())
        # Plant an orphan the way a MinerU upgrade would: a crop the new record
        # will not reference.
        planted = crop_dir / "0000_orphan_from_a_previous_extraction.jpg"
        planted.write_bytes(b"\xff\xd8\xff\xd9")
        print(f"\n[re-extract] {name} (planted 1 orphan)", flush=True)
        content = extractor.extract(papers_dir / name, paper_id=paper_id,
                                    assets_base=rec_dir)
        after = sorted(p.name for p in crop_dir.glob("*") if p.is_file())
        rerun = {
            "file_name": name,
            "files_before": len(before),
            "files_after": len(after),
            "identical_file_set": before == after,
            "planted_orphan_removed": planted.name not in after,
            "pruned": content.record["stats"].get("assets_pruned"),
        }
        print(f"    before={len(before)} after={len(after)} "
              f"identical={rerun['identical_file_set']} "
              f"orphan_removed={rerun['planted_orphan_removed']}")

    # --- storage projection -------------------------------------------------
    ok = [r for r in rows if r.get("ok")]
    total_bytes = sum(r["crops"]["bytes"] for r in ok)
    total_crops = sum(r["crops"]["resolved"] for r in ok)
    per_paper = total_bytes / max(len(ok), 1)
    projection = {
        "papers_measured": len(ok),
        "crops": total_crops,
        "bytes": total_bytes,
        "mean_bytes_per_paper": round(per_paper),
        "mean_crops_per_paper": round(total_crops / max(len(ok), 1), 1),
        "projected_gb_591": round(per_paper * 591 / 1e9, 2),
        "projected_gb_4797": round(per_paper * 4797 / 1e9, 2),
    }

    summary = {"rows": rows, "rerun": rerun, "projection": projection}
    (out_dir / "results.json").write_text(json.dumps(summary, indent=1))

    print("\n=== SUMMARY ===")
    print(f"papers extracted           {len(ok)}/{len(rows)}")
    print(f"crops resolved / referenced "
          f"{sum(r['crops']['resolved'] for r in ok)}/"
          f"{sum(r['crops']['referenced'] for r in ok)}")
    print(f"crops decoded as images    {sum(r['crops']['decoded'] for r in ok)}")
    print(f"dangling refs              "
          f"{sum(len(r['crops']['dangling']) for r in ok)}")
    print(f"orphan files               {sum(len(r['orphans']) for r in ok)}")
    print(f"offsets ok                 {all(r['offsets_ok'] for r in ok)}")
    print(f"verbatim violations        "
          f"{sum(r['chunks']['verbatim_violations'] for r in ok)}")
    print(f"chunks w/ figure page+bbox "
          f"{sum(r['chunks']['with_figure_page_and_bbox'] for r in ok)} "
          f"(matching the object exactly: "
          f"{sum(r['chunks']['figure_bbox_equals_object'] for r in ok)})")
    print(f"resolved in-text refs      "
          f"{sum(r['chunks']['resolved_refs'] for r in ok)}")
    print(f"\nstorage: {projection}")
    print(f"\nwrote {out_dir / 'results.json'}")
    if not args.keep:
        print(f"(scratch kept at {out_dir}; delete when done)")


if __name__ == "__main__":
    main()
