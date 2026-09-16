#!/usr/bin/env python
"""Stage 1 of the reindex: PDF -> paper record (+ figure crops).

    ~74 s/paper, $0, GPU-bound, idempotent.

This is the expensive, repeat-never stage.  It writes ``processed_data/{id}.json``
and ``processed_data/{id}/figures/*.jpg`` and nothing else: it never touches
Qdrant, never touches ``data/indexing_checkpoint.json``, and never embeds.
Stage 2 (``scripts/embed_stage.py``) reads only the records, which is what makes
every future chunking or embedding change cost minutes instead of four days.

What makes it re-runnable
-------------------------
Before extracting, the record already on disk is compared against the PDF
(``schema_version``, ``extractor_version``, ``file_size``, ``file_mtime`` -- the
four fields W1 persists) and skipped if it still describes that file.  So:
run the 591-paper cohort today, the remaining 4,206 next week, and only the new
ones are processed.

Extraction stays in its isolated subprocess (``MinerUExtractor.extract``, see
docs/BUG_REPORT_pdfium_deadlock_2026-05-04.md).  The PDF-metadata/DOI pass runs
in a subprocess too -- see ``_metadata_worker`` -- because it is pypdfium2 in the
parent, which is exactly what bricked the backend after a few hundred papers.

Usage
-----
    # what would happen, without doing it
    python scripts/extract_stage.py --papers-from data/cohorts/cited_papers.txt --dry-run

    # the 591-paper cohort
    python scripts/extract_stage.py --papers-from data/cohorts/cited_papers.txt

    # later: everything that has no current record yet (the remaining 4,206)
    python scripts/extract_stage.py --pending
"""

from __future__ import annotations

import os

# MinerU reads this at import time inside the extraction subprocess; the child
# inherits the environment, so it has to be set before preprocessing is imported.
os.environ.setdefault("MINERU_VIRTUAL_VRAM_SIZE", "8")

import argparse
import multiprocessing
import queue as queue_mod
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BACKEND = Path(__file__).resolve().parents[1]
for p in (str(BACKEND), str(BACKEND / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import stage_lib as SL                                      # noqa: E402
from config import settings                                 # noqa: E402
from preprocessing import paper_record as pr                # noqa: E402

DEFAULT_MANIFEST = SL.MANIFEST_DIR / "extract.json"

#: Measured on the 40-paper W1 sample (hard-weighted); only used until this
#: manifest has timings of its own.
FALLBACK_SECONDS_PER_PAPER = 74.0
FALLBACK_BYTES_PER_PAPER = 0.45 * 1024 * 1024


# ---------------------------------------------------------------------------
# PDF-metadata enrichment, in its own process
# ---------------------------------------------------------------------------

def _metadata_worker(pdf_path_str: str, do_crossref: bool, result_queue) -> None:
    """Title/authors/year/DOI (+ Crossref), isolated from the parent.

    These three helpers open the PDF with pypdfium2.  Production calls them in
    the parent, which is the accumulation pattern that made every upload fail
    with "PDFium: Data format error" after three days of uptime -- and a
    12-hour, 591-paper loop is a long-running process by any definition.  So
    they run here and the parent never loads pdfium.

    ``object.__new__`` and not ``EnhancedPDFProcessor()``: all six helpers were
    checked to use no instance state, so there is no reason for a process whose
    whole job is three pdfium reads to build a chunker and a tokenizer.
    """
    out: Dict[str, Any] = {}
    try:
        from preprocessing.pdf_processor import EnhancedPDFProcessor

        pdf_path = Path(pdf_path_str)
        proc = object.__new__(EnhancedPDFProcessor)

        title = proc._extract_title_from_pdf_metadata(pdf_path)
        authors = proc._extract_authors_from_pdf_metadata(pdf_path)
        year = proc._extract_year(pdf_path)
        doi = proc._extract_doi_from_pdf(pdf_path)
        crossref: Dict[str, Any] = {}
        if doi and do_crossref:
            crossref = proc._fetch_metadata_from_doi(doi) or {}

        out = {"title": title, "authors": authors, "year": year, "doi": doi,
               "crossref": crossref, "ok": True}
    except Exception as exc:                                # noqa: BLE001
        out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    result_queue.put(out)


def enrich_metadata(pdf_path: Path, *, do_crossref: bool, timeout: float = 120.0
                    ) -> Dict[str, Any]:
    """Run ``_metadata_worker`` and return {} rather than fail the paper."""
    result_queue: Any = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_metadata_worker,
                                   args=(str(pdf_path), do_crossref, result_queue))
    proc.start()
    try:
        # Queue before join: the deadlock in the bug report, avoided the same way.
        result = result_queue.get(timeout=timeout)
        proc.join(10)
        return result if result.get("ok") else {}
    except queue_mod.Empty:
        return {}
    except Exception:                                       # noqa: BLE001
        return {}
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            if proc.is_alive():
                proc.kill()


def apply_metadata(record: Dict[str, Any], meta: Dict[str, Any], pdf_path: Path) -> None:
    """Merge enrichment into the record exactly as ``process_pdf`` does.

    Same precedence -- PDF metadata > MinerU > filename, Crossref overriding a
    DOI hit -- so a record written by this stage is byte-comparable with one
    written by an upload.
    """
    rec_meta = record.setdefault("metadata", {})
    title = meta.get("title") or rec_meta.get("title") or pdf_path.stem
    authors = meta.get("authors") or rec_meta.get("authors") or []
    year = meta.get("year") or rec_meta.get("year")
    doi = meta.get("doi") or rec_meta.get("doi")
    journal = rec_meta.get("journal")

    crossref = meta.get("crossref") or {}
    if crossref.get("title") and len(crossref["title"]) > 10:
        title = crossref["title"]
    if crossref.get("authors"):
        authors = crossref["authors"]
    if crossref.get("year"):
        year = crossref["year"]
    if crossref.get("journal"):
        journal = crossref["journal"]

    rec_meta.update({"title": title, "authors": authors, "year": year,
                     "journal": journal, "doi": doi,
                     "file_name": rec_meta.get("file_name") or pdf_path.name})


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

class Plan:
    """What the stage would do, decided before anything is extracted."""

    def __init__(self) -> None:
        self.todo: List[Tuple[SL.Resolved, SL.RecordState]] = []
        self.skip: List[Tuple[SL.Resolved, SL.RecordState]] = []
        self.unresolved: List[SL.Resolved] = []
        self.reasons: Dict[str, int] = {}
        self.warnings: Dict[str, int] = {}
        self.aliases: List[SL.Resolved] = []
        #: PDFs selected under two paper ids at once (NFC/NFD split).  Both are
        #: extracted, because both ids are already cited in stored answers and
        #: choosing a winner is a library-dedupe decision (W4), not this stage's.
        self.dup_paths: Dict[str, List[str]] = {}

    def add_reason(self, reason: str) -> None:
        key = reason.split(" ")[0] or reason
        self.reasons[key] = self.reasons.get(key, 0) + 1


def state_for(
    hit: SL.Resolved,
    manifest: SL.StageManifest,
    records_base: Optional[Path],
    *,
    mtime_policy: str,
    redo_degraded: bool,
    extractor_version: str,
) -> SL.RecordState:
    """Record state for one paper, reading the record only when it has to.

    The manifest carries the fingerprint of what it wrote, so the common case
    ("nothing moved since last run") is answered from two ``stat`` calls -- which
    is what keeps a 4,797-paper plan instant instead of a 2.5 GB read.
    """
    record_path = pr.record_path(hit.paper_id, records_base)
    entry = manifest.get(hit.paper_id) or {}
    fp = entry.get("fingerprint") or {}

    if (entry.get("status") in ("ok", "degraded") and fp
            and not (redo_degraded and fp.get("degraded"))
            and fp.get("schema_version") == SL.CURRENT_SCHEMA
            and (fp.get("extractor") != "mineru"
                 or not extractor_version
                 or fp.get("extractor_version") == extractor_version)
            and record_path.exists()):
        rec_facts = SL.record_file_facts(record_path)
        if (rec_facts.get("record_bytes") == fp.get("record_bytes")
                and rec_facts.get("record_mtime") == fp.get("record_mtime")):
            try:
                disk = SL.file_facts(hit.path) if hit.path else {}
            except OSError:
                disk = {}
            if disk.get("file_size") == fp.get("file_size"):
                if disk.get("file_mtime") == fp.get("file_mtime"):
                    return SL.RecordState("current", "", ["manifest fast path"])
                if mtime_policy != "strict" and not fp.get("pdf_sha256"):
                    return SL.RecordState("current", "",
                                          ["mtime moved, size identical (manifest)"])

    record = pr.load_record(hit.paper_id, records_base)
    return SL.record_state(record, hit.path, mtime_policy=mtime_policy,
                           known_sha256=fp.get("pdf_sha256"),
                           extractor_version=extractor_version,
                           redo_degraded=redo_degraded)


def build_plan(args, resolver: SL.PaperResolver, manifest: SL.StageManifest,
               records_base: Optional[Path], extractor_version: str) -> Plan:
    plan = Plan()

    if args.papers_from:
        tokens = SL.read_selection_file(Path(args.papers_from))
        hits = resolver.resolve_all(tokens)
    elif args.paper:
        hits = resolver.resolve_all(args.paper)
    elif args.retry_failed:
        ids = sorted(set(manifest.ids_with_status("failed", "error", "missing_file")))
        hits = resolver.resolve_all(ids)
    else:                                   # --all / --pending
        hits = resolver.all_on_disk()

    if args.expect_cohort and len(hits) != args.expect_cohort:
        raise SystemExit(
            f"selection produced {len(hits)} papers, expected {args.expect_cohort}. "
            f"Something moved underfoot -- refusing to continue. Re-run with "
            f"--expect-cohort {len(hits)} once you know why."
        )

    for hit in hits:
        if hit.alias_ids:
            plan.aliases.append(hit)
        if not hit.ok:
            plan.unresolved.append(hit)
            continue
        state = state_for(hit, manifest, records_base,
                          mtime_policy=args.mtime_policy,
                          redo_degraded=args.redo_degraded,
                          extractor_version=extractor_version)
        for warn in state.warnings:
            key = warn.split(",")[0]
            plan.warnings[key] = plan.warnings.get(key, 0) + 1
        if state.state == "current":
            plan.skip.append((hit, state))
            continue
        if state.state == "failed" and not (args.retry_failed or args.include_failed):
            plan.skip.append((hit, state))
            plan.add_reason("failed_record(skipped)")
            continue
        plan.todo.append((hit, state))
        plan.add_reason(state.reason or state.state)

    if args.skip_alias_duplicates:
        # A paper whose filename is NFD hashes to two ids. Extracting it twice
        # produces two records and two sets of crops for one PDF (measured: 20
        # such files, ~24 min of GPU time and ~14 MB). Opt-in, because *which*
        # id should own the paper is a library-dedupe decision (W4) -- this only
        # declines to do the work twice once one of them already has a record.
        kept: List[Tuple[SL.Resolved, SL.RecordState]] = []
        for hit, state in plan.todo:
            owner = next(
                (alias for alias in hit.alias_ids
                 if pr.record_path(alias, records_base).exists()
                 and not state_for(SL.Resolved(alias, alias, hit.path, via="alias"),
                                   manifest, records_base,
                                   mtime_policy=args.mtime_policy,
                                   redo_degraded=args.redo_degraded,
                                   extractor_version=extractor_version).needs_extract),
                None)
            if owner:
                key = (state.reason or state.state).split(" ")[0]
                if plan.reasons.get(key):
                    plan.reasons[key] -= 1
                    if not plan.reasons[key]:
                        del plan.reasons[key]
                plan.skip.append((hit, SL.RecordState(
                    "current", "", [f"alias_duplicate: same PDF already extracted "
                                    f"as {owner}"])))
                plan.reasons["alias_duplicate(skipped)"] = \
                    plan.reasons.get("alias_duplicate(skipped)", 0) + 1
            else:
                kept.append((hit, state))
        plan.todo = kept

    by_path: Dict[str, List[str]] = {}
    for hit, _ in plan.todo + plan.skip:
        by_path.setdefault(str(hit.path), []).append(hit.paper_id)
    plan.dup_paths = {path: ids for path, ids in by_path.items() if len(ids) > 1}

    if args.limit:
        plan.todo = plan.todo[:args.limit]
    return plan


def print_plan(plan: Plan, manifest: SL.StageManifest, resolver: SL.PaperResolver,
               args, records_base: Optional[Path]) -> None:
    n_sel = len(plan.todo) + len(plan.skip) + len(plan.unresolved)
    via: Dict[str, int] = {}
    for hit, _ in plan.todo + plan.skip:
        via[hit.via] = via.get(hit.via, 0) + 1

    print(SL.rule("PLAN"))
    print(f"  papers dir        {resolver.papers_dir}  ({resolver.n_files} PDFs, "
          f"{resolver.n_nfd_names} with NFD names)")
    print(f"  records dir       {pr.records_dir(records_base)}")
    print(f"  manifest          {manifest.path}")
    print(f"  selected          {n_sel}")
    print(f"  resolved to a PDF {len(plan.todo) + len(plan.skip)}  ({SL.counter_line(via)})")
    print(f"  unresolved        {len(plan.unresolved)}"
          + ("  (reported and skipped, not a crash)" if plan.unresolved else ""))
    for hit in plan.unresolved[:10]:
        print(f"      - {hit.paper_id}  {hit.note}")
    print(f"  skip (current)    {len(plan.skip)}")
    print(f"  to extract        {len(plan.todo)}")
    if plan.reasons:
        print(f"      why           {SL.counter_line(plan.reasons)}")
    if plan.warnings:
        print(f"      warnings      {SL.counter_line(plan.warnings)}")
    if plan.aliases:
        print(f"  unicode aliases   {len(plan.aliases)} papers whose filename hashes to "
              f"a second paper_id (NFC vs NFD); ids kept as selected")
    if plan.dup_paths:
        extra = sum(len(ids) - 1 for ids in plan.dup_paths.values())
        print(f"  same PDF twice    {len(plan.dup_paths)} file(s) selected under "
              f"{extra} extra id(s) -- extracted once per id, see --help on aliases")
        for path, ids in list(plan.dup_paths.items())[:5]:
            print(f"      ~ {' + '.join(ids)}  {Path(path).name}")

    n_done = len(manifest.ids_with_status("ok", "degraded"))
    measured = manifest.mean_seconds()
    rate = measured or FALLBACK_SECONDS_PER_PAPER
    source = f"measured, n={n_done}" if measured else "W1 sample estimate"
    total = rate * len(plan.todo)
    print(f"  projection        {len(plan.todo)} x {rate:.1f} s = {SL.fmt_duration(total)} "
          f"({source})")
    bytes_each = FALLBACK_BYTES_PER_PAPER
    disk_source = "plan estimate"
    if n_done:
        crops = manifest.sum_field("assets_bytes")
        records = sum((row.get("fingerprint") or {}).get("record_bytes") or 0
                      for row in manifest.entries.values()
                      if row.get("status") in ("ok", "degraded"))
        if crops or records:
            bytes_each = (crops + records) / n_done
            disk_source = f"measured, n={n_done}"
    print(f"  disk              ~{SL.human_bytes(bytes_each * len(plan.todo))} "
          f"of records + crops ({disk_source})")
    print(SL.rule())


# ---------------------------------------------------------------------------
# the stage itself
# ---------------------------------------------------------------------------

def extract_one(hit: SL.Resolved, extractor, records_base: Optional[Path],
                *, do_metadata: bool, do_crossref: bool, verify: bool
                ) -> Dict[str, Any]:
    """Extract one paper and persist its record. Returns a manifest row."""
    t0 = time.time()
    pdf_path = hit.path
    assert pdf_path is not None

    content = extractor.extract(pdf_path, paper_id=hit.paper_id,
                                assets_base=records_base)
    record = content.record
    if record is None:
        raise RuntimeError("extractor returned no record")

    if do_metadata and not record.get("degraded"):
        meta = enrich_metadata(pdf_path, do_crossref=do_crossref)
        apply_metadata(record, meta, pdf_path)

    record_path = pr.save_record(record, base=records_base)

    row: Dict[str, Any] = {
        "file_name": pdf_path.name,
        "seconds": round(time.time() - t0, 1),
        "extractor": record.get("extractor"),
        "degraded": bool(record.get("degraded")),
        "degraded_reason": record.get("degraded_reason"),
        "pages": record["stats"].get("n_pages"),
        "blocks": record["stats"].get("n_blocks"),
        "sections": record["stats"].get("n_sections"),
        "tables": record["stats"].get("n_tables"),
        "captions": record["stats"].get("n_captions"),
        "figures": record["stats"].get("n_figures"),
        "has_abstract": record["stats"].get("has_abstract"),
        "assets_present": (record.get("assets") or {}).get("n_present"),
        "assets_bytes": (record.get("assets") or {}).get("bytes"),
        "resolved_via": hit.via,
        "alias_ids": hit.alias_ids or None,
    }
    if verify:
        check = pr.verify_record_offsets(record)
        row["offsets_ok"] = check["ok"]
        row["offset_errors"] = check["n_errors"]
        row["blocks_verified"] = check["blocks_checked"]

    sha = SL.sha256_file(pdf_path)
    row["fingerprint"] = SL.fingerprint(record, pdf_path, record_path, pdf_sha256=sha)
    if record.get("degraded_reason") == pr.DEGRADED_FAILED:
        row["status"] = "failed"
    elif record.get("degraded"):
        row["status"] = "degraded"
    else:
        row["status"] = "ok"
    return row


def run(plan: Plan, manifest: SL.StageManifest, args, records_base: Optional[Path]) -> int:
    from preprocessing.pdf_processor import MinerUExtractor

    extractor = MinerUExtractor(timeout=args.timeout, assets_base=records_base)

    for hit in plan.unresolved:
        manifest.mark(hit.paper_id, "missing_file", note=hit.note, token=hit.token)

    total = len(plan.todo)
    counts: Dict[str, int] = {}
    t_start = time.time()
    interrupted = False

    def _on_term(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, _on_term)

    try:
        for i, (hit, state) in enumerate(plan.todo, 1):
            prefix = f"[{i}/{total}]"
            try:
                row = extract_one(hit, extractor, records_base,
                                  do_metadata=args.metadata != "off",
                                  do_crossref=args.metadata == "full",
                                  verify=not args.no_verify_offsets)
                row["stale_reason"] = state.reason or state.state
                status = row.pop("status")
                manifest.mark(hit.paper_id, status, **row)
                counts[status] = counts.get(status, 0) + 1
                print(f"{prefix} {status:8s} {hit.paper_id} {row['seconds']:6.1f}s "
                      f"blocks={row['blocks']} figs={row['figures']} "
                      f"offs={'ok' if row.get('offsets_ok', True) else 'FAIL'} "
                      f"{hit.path.name[:60]}", flush=True)
            except KeyboardInterrupt:
                raise
            except Exception as exc:                        # noqa: BLE001
                manifest.mark(hit.paper_id, "error",
                              file_name=hit.path.name if hit.path else None,
                              error=f"{type(exc).__name__}: {exc}",
                              traceback=traceback.format_exc(limit=4),
                              stale_reason=state.reason or state.state)
                counts["error"] = counts.get("error", 0) + 1
                print(f"{prefix} error    {hit.paper_id} {type(exc).__name__}: {exc}",
                      flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted -- snapshotting manifest", flush=True)
    finally:
        manifest.stats.setdefault("extract", {})
        manifest.stats["extract"].update({
            "last_run_at": SL.utc_now(),
            "last_run_seconds": round(time.time() - t_start, 1),
            "last_run_counts": counts,
        })
        manifest.close()

    elapsed = time.time() - t_start
    done = sum(counts.values())
    print(SL.rule("EXTRACT DONE"))
    print(f"  this run          {SL.counter_line(counts) or 'nothing'}"
          f"{'  (INTERRUPTED)' if interrupted else ''}")
    print(f"  wall time         {SL.fmt_duration(elapsed)}"
          + (f"  ({elapsed / done:.1f} s/paper)" if done else ""))
    print(f"  manifest totals   {SL.counter_line(manifest.counts())}")
    print(f"  manifest          {manifest.path}")
    remaining = total - done
    if remaining > 0:
        mean = manifest.mean_seconds() or FALLBACK_SECONDS_PER_PAPER
        print(f"  remaining         {remaining} papers ~ {SL.fmt_duration(remaining * mean)} "
              f"-- re-run the same command to resume")
    print(SL.rule())
    return 1 if interrupted else 0


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Stage 1: extract paper records (no embedding, no Qdrant).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage\n-----\n")[-1],
    )
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--papers-from", metavar="FILE",
                     help="file of paper ids / filenames / paths ('#' comments ok)")
    sel.add_argument("--paper", action="append", metavar="ID|NAME",
                     help="a single paper id, filename or path (repeatable)")
    sel.add_argument("--all", action="store_true",
                     help="every PDF in the papers dir (current records still skipped)")
    sel.add_argument("--pending", action="store_true",
                     help="same as --all; named for what it does after the reuse check")
    sel.add_argument("--retry-failed", action="store_true",
                     help="only papers this manifest recorded as failed/error/missing")

    ap.add_argument("--papers-dir", default=None,
                    help=f"default: settings.pdf_source_dir ({settings.pdf_source_dir})")
    ap.add_argument("--records-dir", default=None,
                    help="root for records and crops (default: PROCESSED_DATA_DIR). "
                         "A test harness MUST set this or it writes into the library.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=settings.pdf_extraction_timeout)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mtime-policy", choices=("warn", "strict"), default="warn",
                    help="warn (default): same size + moved mtime is reused, because "
                         "copying a volume is not a content change. strict: re-extract.")
    ap.add_argument("--redo-degraded", action="store_true",
                    help="re-extract papers whose record came from the pypdfium2 fallback")
    ap.add_argument("--skip-alias-duplicates", action="store_true",
                    help="skip a PDF that already has a record under its other "
                         "Unicode-normalisation paper_id (20 such files here)")
    ap.add_argument("--include-failed", action="store_true",
                    help="also retry papers with a failed record in a --all/--pending run")
    ap.add_argument("--metadata", choices=("full", "pdf-only", "off"), default="full",
                    help="full: PDF metadata + DOI + Crossref (production parity). "
                         "pdf-only: no network. off: keep MinerU's own metadata.")
    ap.add_argument("--no-verify-offsets", action="store_true",
                    help="skip the per-block byte-exactness check (it is cheap; don't)")
    ap.add_argument("--verbose", action="store_true",
                    help="keep the library loggers at INFO")
    ap.add_argument("--save-every", type=int, default=10,
                    help="papers between manifest snapshots (the journal is per paper)")
    ap.add_argument("--expect-cohort", type=int, default=0,
                    help="fail unless the selection resolves to exactly this many papers")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    SL.configure_logging(args.verbose)

    papers_dir = Path(args.papers_dir) if args.papers_dir else settings.pdf_source_dir
    if not papers_dir or not Path(papers_dir).exists():
        raise SystemExit(f"papers dir not found: {papers_dir}")
    records_base = Path(args.records_dir).resolve() if args.records_dir else None
    extractor_version = SL.current_extractor_version()

    resolver = SL.PaperResolver(papers_dir,
                                legacy_checkpoint=BACKEND / "data" / "indexing_checkpoint.json")

    manifest, soft_changes = SL.open_manifest(
        Path(args.manifest),
        signature={
            "stage": "extract",
            "records_dir": str(pr.records_dir(records_base)),
            "papers_dir": str(papers_dir),
            "record_schema_version": SL.CURRENT_SCHEMA,
            "extractor_version": extractor_version,
        },
        # Only the artifact roots are hard: an extractor upgrade must *not* void
        # the manifest, it must make the affected papers stale, which the
        # per-paper fingerprints already do.
        hard_keys=("stage", "records_dir"),
        save_every=args.save_every,
    )
    for change in soft_changes:
        print(f"  note              manifest signature changed -- {change}")
    if manifest.replayed or manifest.truncated_lines:
        print(f"  recovered {manifest.replayed} journal entries"
              f"{f', discarded {manifest.truncated_lines} truncated line(s)' if manifest.truncated_lines else ''}")

    print(SL.rule("EXTRACT STAGE (stage 1 of 2: PDF -> paper record)"))
    print(f"  MinerU {extractor_version}, record schema v{SL.CURRENT_SCHEMA}, "
          f"timeout {args.timeout}s, metadata={args.metadata}")

    plan = build_plan(args, resolver, manifest, records_base, extractor_version)
    print_plan(plan, manifest, resolver, args, records_base)

    if args.dry_run:
        # Deliberately no snapshot: a dry run writes nothing at all, so it can
        # be run against a production manifest without changing its mtime.
        print("dry run: nothing extracted")
        return 0
    if not plan.todo and not plan.unresolved:
        print("nothing to do")
        return 0
    return run(plan, manifest, args, records_base)


if __name__ == "__main__":
    sys.exit(main())
