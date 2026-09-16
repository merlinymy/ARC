#!/usr/bin/env python
"""Indexing entry point -- two stages, kept separable on purpose (plan §3b.10).

The reindex was assumed to be one operation.  It is two, with wildly different
economics, and conflating them is what made the old tooling unusable:

    stage 1  extract  PDF -> paper record (+ figure crops)
             ~74 s/paper, $0, GPU-bound, expensive to repeat
    stage 2  embed    record -> chunks -> vectors -> Qdrant
             ~1 s/paper + the API bill, **free to redo**

Each stage is idempotent, selectable by cohort, and has its own manifest, so
"indexed" no longer has to mean two different things at once.

    python index_papers.py cohort                     # derive the cited cohort
    python index_papers.py extract --papers-from ...  # stage 1
    python index_papers.py embed   --papers-from ...  # stage 2
    python index_papers.py plan    --papers-from ...  # dry-run both stages
    python index_papers.py status                     # progress of both stages

Every flag of the stage is accepted after its name; ``--help`` works per stage:

    python index_papers.py extract --help

What happened to the old behaviour
----------------------------------
``index_papers.py`` with no arguments resumed from ``data/indexing_checkpoint.json``,
where all 4,792 papers were already marked "indexed" by the *old* pipeline, so it
extracted nothing; ``--reset`` cleared that checkpoint **and deleted the live
Qdrant collection**, which is 212,953 points the running server is serving.
Neither could run the incremental plan, so both are gone.  The legacy flags are
still recognised -- and refused with the command to use instead, at startup
rather than three hours into a run.  ``data/indexing_checkpoint.json`` itself is
untouched: it is real history, and ``services/paper_library.py`` still reads it.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
for _p in (str(BACKEND), str(BACKEND / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Imported at startup, not lazily: a broken stage module must fail here, in the
# first second, rather than at hour three of a 12-hour extraction.
import cohort_cited_papers                                  # noqa: E402
import embed_stage                                          # noqa: E402
import extract_stage                                        # noqa: E402
import stage_lib as SL                                      # noqa: E402
from config import get_embedding_profile, settings          # noqa: E402

def status(argv) -> int:
    """Where both stages stand, without reading a single record or PDF."""
    import argparse

    ap = argparse.ArgumentParser(prog="index_papers.py status")
    ap.add_argument("--profile", default=settings.embedding_profile)
    ap.add_argument("--extract-manifest", default=str(extract_stage.DEFAULT_MANIFEST))
    ap.add_argument("--embed-manifest", default=None)
    args = ap.parse_args(argv)
    SL.configure_logging(False)
    profile = get_embedding_profile(args.profile)
    embed_manifest = Path(args.embed_manifest) if args.embed_manifest else \
        SL.MANIFEST_DIR / f"embed__{profile.collection}.json"

    print(SL.rule("STAGE STATUS"))
    for label, path in (("extract", Path(args.extract_manifest)),
                        ("embed", embed_manifest)):
        if not path.exists() and not path.with_suffix(".jsonl").exists():
            print(f"  {label:8s} no manifest at {path}")
            continue
        manifest = SL.StageManifest(path, signature={}, hard_keys=())
        counts = SL.counter_line(manifest.counts()) or "empty"
        print(f"  {label:8s} {counts}")
        if label == "extract":
            mean = manifest.mean_seconds()
            if mean:
                print(f"           {mean:.1f} s/paper measured over "
                      f"{len(manifest.ids_with_status('ok', 'degraded'))} papers")
        else:
            done = manifest.ids_with_status("ok")
            if done:
                print(f"           {manifest.sum_field('chunks', ('ok',)):,.0f} chunks, "
                      f"{manifest.sum_field('tokens', ('ok',)):,.0f} tokens over "
                      f"{len(done)} papers")
        if manifest.replayed:
            print(f"           {manifest.replayed} entries recovered from the journal")
        if manifest.truncated_lines:
            print(f"           {manifest.truncated_lines} truncated journal line(s) "
                  f"discarded (a kill mid-append)")

    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        names = sorted(c.name for c in client.get_collections().collections)
        for name in names:
            info = client.get_collection(name)
            mark = "  <- active profile" if name == profile.collection else ""
            print(f"  qdrant   {name}: {info.points_count:,} points{mark}")
    except Exception as exc:                                # noqa: BLE001
        print(f"  qdrant   unreachable ({exc})")
    print(SL.rule())
    return 0


STAGES = {
    "cohort": cohort_cited_papers.main,
    "extract": extract_stage.main,
    "embed": embed_stage.main,
    "status": status,
}

LEGACY_FLAGS = {
    "--reset": (
        "--reset deleted the live Qdrant collection (212,953 points, currently "
        "serving) and cleared data/indexing_checkpoint.json.\n"
        "Build the replacement beside it instead -- the collection comes with the "
        "embedding profile, and the cutover is a profile switch:\n"
        "    python index_papers.py extract --papers-from data/cohorts/cited_papers.txt\n"
        "    python index_papers.py embed   --papers-from data/cohorts/cited_papers.txt"
    ),
    "--retry-failed": (
        "--retry-failed now belongs to a stage, because extraction and embedding "
        "fail for different reasons:\n"
        "    python index_papers.py extract --retry-failed\n"
        "    python index_papers.py embed   --retry-failed"
    ),
    "--limit": (
        "--limit is a stage flag now:\n"
        "    python index_papers.py extract --pending --limit 5"
    ),
    "--papers-dir": (
        "--papers-dir is a stage flag now:\n"
        "    python index_papers.py extract --all --papers-dir PATH"
    ),
}

USAGE = f"""usage: index_papers.py {{{'|'.join(STAGES)}|plan}} [stage options]

  cohort   derive the cited-papers cohort from data/app.db  (scripts/cohort_cited_papers.py)
  extract  stage 1: PDF -> paper record, no Qdrant, no API  (scripts/extract_stage.py)
  embed    stage 2: record -> chunks -> vectors -> Qdrant   (scripts/embed_stage.py)
  plan     dry-run both stages over the same selection
  status   what each stage has done so far, and the Qdrant collections

  python index_papers.py <stage> --help   for a stage's own options
"""


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        print(USAGE)
        return 0 if argv else 2

    for flag, message in LEGACY_FLAGS.items():
        if flag in argv or any(a.startswith(flag + "=") for a in argv):
            print(f"index_papers.py: {flag} is gone (plan §3b.10).\n\n{message}\n",
                  file=sys.stderr)
            return 2

    stage = argv[0]
    rest = argv[1:]

    if stage == "plan":
        code = extract_stage.main(rest + ["--dry-run"])
        if code:
            return code
        return embed_stage.main(rest + ["--dry-run"])

    if stage not in STAGES:
        print(f"index_papers.py: unknown stage {stage!r}\n\n{USAGE}", file=sys.stderr)
        return 2
    return STAGES[stage](rest)


if __name__ == "__main__":
    sys.exit(main())
