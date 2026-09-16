#!/usr/bin/env python
"""Verify the record-reuse check in both directions (§3b.10 item 1).

Extraction is idempotent only if a valid record is skipped *and* an invalid one
is re-extracted.  Both halves matter: skipping too eagerly silently indexes
stale text, skipping too rarely costs 74 s of GPU time per paper.

This harness copies the existing sample records into a scratch directory, breaks
one fingerprint field at a time, and asserts the decision.  It runs no
extraction, opens no PDF, writes nothing outside the scratch dir, and touches
neither the library nor Qdrant.

    python scripts/stage_reuse_check.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BACKEND = Path(__file__).resolve().parents[1]
for p in (str(BACKEND), str(BACKEND / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import stage_lib as SL                                      # noqa: E402
from config import settings                                 # noqa: E402
from preprocessing import paper_record as pr                # noqa: E402

SCHEMA2_RECORDS = BACKEND / "data" / "w2b_figures" / "records"
SCHEMA1_RECORDS = BACKEND / "data" / "w1_sample_v2" / "records"


def check(label: str, got: Tuple[str, str], want_state: str,
          want_reason_contains: str = "") -> bool:
    state, reason = got
    ok = state == want_state and (want_reason_contains in reason)
    print(f"  {'PASS' if ok else 'FAIL'}  {label:38s} -> {state:8s} {reason}")
    return ok


def state_of(record: Dict[str, Any], pdf: Optional[Path], **kw) -> Tuple[str, str]:
    st = SL.record_state(record, pdf, **kw)
    reason = st.reason or (st.warnings[0] if st.warnings else "")
    return st.state, reason


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--papers-dir", default=None)
    args = ap.parse_args(argv)
    papers_dir = Path(args.papers_dir) if args.papers_dir else settings.pdf_source_dir

    failures = 0
    print(SL.rule("RECORD REUSE CHECK"))
    print(f"  current schema v{SL.CURRENT_SCHEMA}, MinerU {SL.current_extractor_version()}")

    # ---- unmodified schema-2 records: reuse --------------------------------
    print("\nschema-2 records, untouched (must be reused):")
    schema2: List[Tuple[str, Dict[str, Any], Path]] = []
    for path in sorted(SCHEMA2_RECORDS.glob("*.json")):
        record = json.loads(path.read_text())
        pdf = papers_dir / record["file_name"]
        schema2.append((record["paper_id"], record, pdf))
        failures += not check(f"{record['paper_id']} untouched",
                              state_of(record, pdf), "current")

    # ---- schema-1 records: stale ------------------------------------------
    print("\nschema-1 records (must be re-extracted -- no figure crops):")
    for path in sorted(SCHEMA1_RECORDS.glob("*.json")):
        record = json.loads(path.read_text())
        pdf = papers_dir / record["file_name"]
        failures += not check(f"{record['paper_id']} schema 1",
                              state_of(record, pdf), "stale", "schema_version")

    if not schema2:
        print("no schema-2 sample records found; cannot test invalidation")
        return 1

    # ---- one broken fingerprint field at a time ---------------------------
    print("\none fingerprint field invalidated at a time:")
    pid, base_record, pdf = schema2[0]
    sha = SL.sha256_file(pdf)

    mutations = [
        ("file_size + 1", {"file_size": base_record["file_size"] + 1},
         "stale", "file_size"),
        ("extractor_version bumped", {"extractor_version": "0.0.0-old"},
         "stale", "extractor_version"),
        ("schema_version -> 1", {"schema_version": 1}, "stale", "schema_version"),
        ("file_mtime moved (default policy)", {"file_mtime": base_record["file_mtime"] - 86400},
         "current", "mtime moved"),
    ]
    for label, patch, want, contains in mutations:
        record = dict(base_record)
        record.update(patch)
        failures += not check(label, state_of(record, pdf), want, contains)

    record = dict(base_record)
    record["file_mtime"] = base_record["file_mtime"] - 86400
    failures += not check("file_mtime moved, --mtime-policy strict",
                          state_of(record, pdf, mtime_policy="strict"),
                          "stale", "file_mtime")
    failures += not check("file_mtime moved, sha256 known and equal",
                          state_of(record, pdf, known_sha256=sha),
                          "current", "sha256 identical")
    failures += not check("file_mtime moved, sha256 known and different",
                          state_of(record, pdf, known_sha256="0" * 64),
                          "stale", "pdf_sha256 changed")

    # ---- missing / failed records ------------------------------------------
    print("\nmissing and failed records:")
    failures += not check("no record at all", state_of(None, pdf), "missing", "no_record")
    failed = pr.build_failed_record(pid, pdf.name, pdf_path=pdf)
    failures += not check("record says extraction failed",
                          state_of(failed, pdf), "failed", "")
    degraded = dict(base_record)
    degraded["degraded"] = True
    degraded["degraded_reason"] = pr.DEGRADED_FALLBACK
    failures += not check("pypdfium2 fallback record (kept by default)",
                          state_of(degraded, pdf), "current", "degraded")
    failures += not check("pypdfium2 fallback record, --redo-degraded",
                          state_of(degraded, pdf, redo_degraded=True),
                          "stale", "degraded")

    # ---- the manifest fast path -------------------------------------------
    print("\nmanifest fast path (decision without reading the record):")
    import extract_stage as ES

    scratch = Path(tempfile.mkdtemp(prefix="arc_reuse_"))
    try:
        records_dir = scratch / "records"
        records_dir.mkdir(parents=True)
        shutil.copy2(SCHEMA2_RECORDS / f"{pid}.json", records_dir / f"{pid}.json")
        record_path = records_dir / f"{pid}.json"

        manifest = SL.StageManifest(scratch / "m.json",
                                    signature={"stage": "extract",
                                               "records_dir": str(records_dir)},
                                    hard_keys=("stage", "records_dir"))
        fp = SL.fingerprint(base_record, pdf, record_path, pdf_sha256=sha)
        manifest.mark(pid, "ok", fingerprint=fp)
        hit = SL.Resolved(pid, pid, pdf, via="exact")

        st = ES.state_for(hit, manifest, records_dir, mtime_policy="warn",
                          redo_degraded=False,
                          extractor_version=base_record["extractor_version"])
        failures += not check("fingerprint matches", (st.state, st.warnings[0] if st.warnings else ""),
                              "current", "manifest fast path")

        # A record edited behind the manifest's back must not be trusted: its
        # size and mtime no longer match what the manifest wrote.
        edited = json.loads(record_path.read_text())
        edited["schema_version"] = 1
        record_path.write_text(json.dumps(edited) + "   ")
        st = ES.state_for(hit, manifest, records_dir, mtime_policy="warn",
                          redo_degraded=False,
                          extractor_version=base_record["extractor_version"])
        failures += not check("record changed under the manifest",
                              (st.state, st.reason), "stale", "schema_version")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    print(SL.rule())
    print("ALL PASS" if not failures else f"{failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
