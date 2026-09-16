#!/usr/bin/env python
"""Derive the first reindex cohort: the papers ARC has actually cited (§3b.10).

Distinct ``sources[].paper_id`` across every message in ``data/app.db``, resolved
to PDFs on disk and written as a selection file the two stages can consume.
Derived every time rather than hardcoded, so it stays honest as the library
grows -- with ``--expect`` asserts so it fails loudly if the number moves for a
reason nobody noticed.

    python scripts/cohort_cited_papers.py                    # writes data/cohorts/cited_papers.txt
    python scripts/cohort_cited_papers.py --expect 591       # and asserts the count
    python scripts/cohort_cited_papers.py --show-unresolved
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

BACKEND = Path(__file__).resolve().parents[1]
for p in (str(BACKEND), str(BACKEND / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import stage_lib as SL                                      # noqa: E402
from config import settings                                 # noqa: E402
from preprocessing import paper_record as pr                # noqa: E402

DEFAULT_OUT = SL.COHORT_DIR / "cited_papers.txt"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=str(BACKEND / "data" / "app.db"))
    ap.add_argument("--papers-dir", default=None)
    ap.add_argument("--records-dir", default=None)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--expect", type=int, default=0,
                    help="fail unless exactly this many distinct cited paper ids")
    ap.add_argument("--expect-resolved", type=int, default=0,
                    help="fail unless exactly this many resolve to a PDF")
    ap.add_argument("--show-unresolved", action="store_true")
    ap.add_argument("--show-aliases", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    papers_dir = Path(args.papers_dir) if args.papers_dir else settings.pdf_source_dir
    records_base = Path(args.records_dir).resolve() if args.records_dir else None
    checkpoint = BACKEND / "data" / "indexing_checkpoint.json"

    ids = SL.cited_paper_ids(Path(args.db))
    resolver = SL.PaperResolver(papers_dir, legacy_checkpoint=checkpoint)
    hits = resolver.resolve_all(ids)

    resolved = [h for h in hits if h.ok]
    unresolved = [h for h in hits if not h.ok]
    via: dict = {}
    for h in resolved:
        via[h.via] = via.get(h.via, 0) + 1

    in_checkpoint = sum(1 for pid in ids if pid in resolver.legacy_names)
    have_record = sum(1 for h in resolved
                      if pr.record_path(h.paper_id, records_base).exists())
    aliased = [h for h in resolved if h.alias_ids]

    print(SL.rule("CITED COHORT"))
    print(f"  db                {args.db}")
    print(f"  papers dir        {resolver.papers_dir}  ({resolver.n_files} PDFs)")
    print(f"  distinct cited    {len(ids)}")
    print(f"  in old checkpoint {in_checkpoint} / {len(ids)}  (paper_metadata)")
    print(f"  resolved to a PDF {len(resolved)}  ({SL.counter_line(via)})")
    print(f"  unresolved        {len(unresolved)}")
    print(f"  already have a record  {have_record}  "
          f"(in {pr.records_dir(records_base)})")
    if aliased:
        print(f"  unicode aliases   {len(aliased)} papers whose on-disk filename is "
              f"NFD, so md5(name) differs from the cited id; the cited id is kept")
    if unresolved and args.show_unresolved:
        for h in unresolved:
            print(f"      - {h.paper_id}  {h.note}")
    if aliased and args.show_aliases:
        for h in aliased:
            print(f"      ~ {h.paper_id} <-> {','.join(h.alias_ids)}  {h.path.name}")
    print(SL.rule())

    if args.expect and len(ids) != args.expect:
        raise SystemExit(f"FAIL: {len(ids)} distinct cited papers, expected "
                         f"{args.expect}. Something moved underfoot.")
    if args.expect_resolved and len(resolved) != args.expect_resolved:
        raise SystemExit(f"FAIL: {len(resolved)} resolved, expected "
                         f"{args.expect_resolved}.")

    if args.no_write:
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# The papers ARC has actually cited -- first reindex cohort (plan §3b.10).",
        f"# Derived {SL.utc_now()} from {args.db}: distinct sources[].paper_id.",
        f"# {len(ids)} cited, {len(resolved)} resolved to a PDF, "
        f"{len(unresolved)} unresolved.",
        "# One paper_id per line; the filename after '#' is a comment.",
    ]
    for h in resolved:
        lines.append(f"{h.paper_id}  # {h.path.name}")
    for h in unresolved:
        lines.append(f"# UNRESOLVED {h.paper_id}  # {h.note}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}  ({len(resolved)} selectable papers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
