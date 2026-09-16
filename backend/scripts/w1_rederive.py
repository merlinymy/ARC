#!/usr/bin/env python
"""Re-derive sections/abstract/stats for persisted records, without re-extracting.

This is the capability the persisted record buys: a change to heading
classification or abstract detection costs milliseconds per paper instead of a
60-80 h corpus re-extraction. W2 uses the same entry point for re-chunking.

    python scripts/w1_rederive.py data/w1_sample/records [--results data/w1_sample/results.json]
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing import paper_record as pr  # noqa: E402


def main(rec_dir="data/w1_sample/records", results=None):
    paths = sorted(Path(rec_dir).glob("*.json"))
    t0 = time.time()
    changed = 0
    updates = {}
    for path in paths:
        record = json.loads(path.read_text())
        before = (record["stats"]["n_sections"], record["stats"]["abstract_source"],
                  sum(1 for s in record["sections"]
                      if s["normalized_name"] not in ("body", "frontmatter")))
        # `base` so the asset summary is recomputed against the sample's own
        # crop directory rather than the production library.
        pr.rebuild_derived(record, base=Path(rec_dir))
        after = (record["stats"]["n_sections"], record["stats"]["abstract_source"],
                 sum(1 for s in record["sections"]
                     if s["normalized_name"] not in ("body", "frontmatter")))
        verify = pr.verify_record_offsets(record)
        assert verify["ok"], f"{path.name}: offsets broke during re-derivation"
        pr.save_record(record, base=Path(rec_dir))
        if before != after:
            changed += 1
            print(f"  {record['file_name'][:52]:54s} sections {before[0]}->{after[0]} "
                  f"named {before[2]}->{after[2]}  abstract {before[1]}->{after[1]}")
        updates[record["paper_id"]] = {
            "new_sections": after[0],
            "new_sections_named": after[2],
            "new_abstract": record["abstract"] is not None,
            "new_abstract_source": after[1],
            "new_abstract_len": len((record["abstract"] or {}).get("text", "")),
            "record_bytes": pr.record_path(record["paper_id"], Path(rec_dir)).stat().st_size,
        }

    elapsed = time.time() - t0
    print(f"\nre-derived {len(paths)} records in {elapsed:.2f}s "
          f"({elapsed / max(len(paths),1) * 1000:.0f} ms/paper); {changed} changed")
    print(f"projection for 4,797 papers: {elapsed / max(len(paths),1) * 4797 / 60:.1f} min "
          f"(vs a full re-extraction)")

    if results:
        rows = json.loads(Path(results).read_text())
        for row in rows:
            if row.get("ok") and row["paper_id"] in updates:
                row.update(updates[row["paper_id"]])
        Path(results).write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        print(f"updated {results}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    res = None
    for i, a in enumerate(sys.argv):
        if a == "--results":
            res = sys.argv[i + 1]
    main(args[0] if args else "data/w1_sample/records", res)
