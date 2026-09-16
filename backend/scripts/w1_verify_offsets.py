#!/usr/bin/env python
"""Prove the offset chain W3's PDF highlight depends on.

For each sampled record:

  1. reassemble page text from the record and check every block's
     ``page_char_start/end`` and ``doc_char_start/end`` slice back byte-exactly;
  2. take a span from the middle of a real block -- what the Citations API would
     hand back as ``cited_text`` -- convert its document offset with ``locate()``
     and check the returned ``page_idx`` / ``bbox``;
  3. check the located page against the **actual PDF**, by pulling that page's
     text with pypdfium2 and looking for the span. This is the step that proves
     ``page_idx`` is the real page number and not merely self-consistent.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing import paper_record as pr  # noqa: E402

PAPERS = Path("/Volumes/ARC/ARC/papers")


def norm(text):
    """Alphanumerics only.

    MinerU's text is not pypdfium2's text: MinerU re-renders inline maths as LaTeX
    (``$Z = 8$``, ``\\mathrm{Cu}``), de-hyphenates line breaks and normalises
    spacing. Comparing on letters and digits alone asks the question we actually
    care about -- is this passage on this page -- without asking two different
    extractors to agree on punctuation.
    """
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def pdf_page_text(pdf_path, page_idx):
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(pdf_path)
    try:
        if page_idx >= len(doc):
            return None
        page = doc[page_idx]
        tp = page.get_textpage()
        text = tp.get_text_bounded()
        tp.close()
        return text
    finally:
        doc.close()


def check(record, verbose=False):
    rep = pr.verify_record_offsets(record)
    pdf_path = PAPERS / record["file_name"]
    doc = pr.full_text(record)

    # A mid-document body block with enough text to quote from.
    # Prefer prose without inline maths: MinerU rewrites it as LaTeX, so a span
    # containing it can never string-match pypdfium2's rendering of the same page.
    candidates = [b for b in record["blocks"]
                  if b["type"] == pr.BLOCK_TEXT and len(b["text"]) > 400
                  and "$" not in b["text"] and "\\" not in b["text"]]
    if not candidates:
        candidates = [b for b in record["blocks"]
                      if b["type"] == pr.BLOCK_TEXT and len(b["text"]) > 400]
    if not candidates:
        return {"file": record["file_name"], "offsets_ok": rep["ok"],
                "page_match": None, "note": "no long text block"}
    block = candidates[len(candidates) // 2]

    # Pretend the Citations API cited chars 100..260 of this block.
    q_start = block["doc_char_start"] + 100
    q_end = q_start + 160
    quote = doc[q_start:q_end]

    hit = pr.locate(record, q_start)
    page_txt = pr.page_text(record, hit["page_idx"])
    from_page = page_txt[hit["page_char_offset"]:hit["page_char_offset"] + len(quote)]

    real = pdf_page_text(pdf_path, hit["page_idx"])
    needle = norm(quote)[:60]
    page_match = bool(needle) and needle in norm(real or "")
    # Where the quote is not found, say which page of the PDF it is actually on.
    found_on = None
    if not page_match and real is not None:
        import pypdfium2 as pdfium
        d = pdfium.PdfDocument(pdf_path)
        try:
            for i in range(len(d)):
                tp = d[i].get_textpage()
                t = tp.get_text_bounded()
                tp.close()
                if needle and needle in norm(t):
                    found_on = i
                    break
        finally:
            d.close()

    page = record["pages"][hit["page_idx"]]
    bbox = hit["bbox"]
    bbox_pt = None
    if bbox and page["width"]:
        bbox_pt = [round(bbox[0] / 1000 * page["width"], 1),
                   round(bbox[1] / 1000 * page["height"], 1),
                   round(bbox[2] / 1000 * page["width"], 1),
                   round(bbox[3] / 1000 * page["height"], 1)]

    out = {
        "file": record["file_name"],
        "offsets_ok": rep["ok"],
        "blocks": rep["blocks_checked"],
        "block": block["i"],
        "doc_offset": q_start,
        "page_idx": hit["page_idx"],
        "page_char_offset": hit["page_char_offset"],
        "page_slice_matches_doc_slice": from_page == quote,
        "page_match": page_match,
        "actually_on_page": found_on,
        "bbox_per_mille": bbox,
        "bbox_pdf_points": bbox_pt,
        "page_size": [page["width"], page["height"]],
        "quote": quote,
    }
    if verbose:
        print(json.dumps({k: v for k, v in out.items() if k != "quote"}, indent=1))
        print(f"  quote: {quote!r}")
    return out


def main(rec_dir="data/w1_sample/records", show=3):
    paths = sorted(Path(rec_dir).glob("*.json"))
    rows = []
    for p in paths:
        record = json.loads(p.read_text())
        if record["degraded"]:
            continue
        try:
            rows.append(check(record))
        except Exception as exc:
            rows.append({"file": record["file_name"], "error": f"{type(exc).__name__}: {exc}"})

    good = [r for r in rows if r.get("offsets_ok")]
    matched = [r for r in rows if r.get("page_match")]
    consistent = [r for r in rows if r.get("page_slice_matches_doc_slice")]
    print(f"records checked          : {len(rows)}")
    print(f"offset identities hold   : {len(good)}/{len(rows)}"
          f"  ({sum(r.get('blocks',0) for r in rows):,} blocks)")
    print(f"page slice == doc slice  : {len(consistent)}/{len(rows)}")
    print(f"quote found on the page  : {len(matched)}/{len(rows)}  "
          f"(checked against the real PDF with pypdfium2)")
    off = [r for r in rows if r.get("page_match") is False]
    if off:
        print("\nmismatches:")
        for r in off:
            print(f"  {r['file'][:55]:55s} record says p{r['page_idx']}, "
                  f"pypdfium2 finds it on p{r['actually_on_page']}")

    print("\n--- worked examples ---")
    for r in rows[:int(show)]:
        if "error" in r:
            continue
        print(f"\n{r['file']}")
        print(f"  block {r['block']}  doc offset {r['doc_offset']}"
              f" -> page_idx {r['page_idx']}, page offset {r['page_char_offset']}")
        print(f"  page size {r['page_size']} pt   bbox {r['bbox_per_mille']} per-mille"
              f" -> {r['bbox_pdf_points']} pt (top-left origin)")
        print(f"  page slice == doc slice: {r['page_slice_matches_doc_slice']}")
        print(f"  span present on PDF page {r['page_idx']}: {r['page_match']}")
        print(f"  span: {r['quote'][:150]!r}")


if __name__ == "__main__":
    main(*sys.argv[1:])
