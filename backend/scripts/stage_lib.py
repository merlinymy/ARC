"""Shared plumbing for the two-stage, non-destructive reindex (plan §3b.10).

The reindex is two stages with wildly different economics:

    extract -> paper record (+ figure crops)    ~74 s/paper, $0,   GPU bound
    chunk -> embed -> upsert                    ~1 s/paper, $$,    API bound

Keeping them separable is the whole point: a chunking or embedding change then
costs minutes and single-digit dollars instead of another four days of GPU time.
This module holds only what both stages need:

* ``PaperResolver``   -- paper_id <-> PDF on disk, including the Unicode
  normalisation trap that silently splits one paper into two ids on macOS.
* ``record_state``    -- is a persisted record still valid for this PDF, and if
  not, why.  This is what makes extraction idempotent.
* ``StageManifest``   -- per-stage progress, append-only journal plus atomic
  snapshot, so a ``kill -9`` mid-run can never truncate it into garbage.

Nothing here touches ``data/indexing_checkpoint.json`` (2.5 MB of real history
that the running server reads) or the live Qdrant collection.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from preprocessing import paper_record as pr  # noqa: E402

#: Where stage manifests live.  Deliberately *not* ``indexing_checkpoint.json``:
#: "indexed" there means extracted *and* embedded, and one flag cannot mean both
#: once the stages are separable.
MANIFEST_DIR = BACKEND / "data" / "stage_manifests"

#: Cohort selection files (one paper per line, ``#`` comments allowed).
COHORT_DIR = BACKEND / "data" / "cohorts"

ID_LEN = 12


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def fmt_duration(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def looks_like_paper_id(token: str) -> bool:
    return len(token) == ID_LEN and all(c in "0123456789abcdef" for c in token.lower())


def read_selection_file(path: Path) -> List[str]:
    """Read a selection file: one paper id, filename or path per line.

    ``#`` starts a comment, so a cohort file can carry the filename next to the
    id and still be machine-readable.
    """
    tokens: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if line:
                tokens.append(line)
    return tokens


# ---------------------------------------------------------------------------
# paper_id <-> file resolution
# ---------------------------------------------------------------------------

@dataclass
class Resolved:
    """One selection token resolved (or not) to a PDF on disk."""
    token: str
    paper_id: str
    path: Optional[Path] = None
    #: how the file was found: exact | nfc | nfd | legacy_name | path | none
    via: str = "none"
    note: Optional[str] = None
    #: other ids the same file hashes to under a different Unicode form
    alias_ids: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.path is not None


def pdf_files(papers_dir: Path) -> List[Path]:
    """Every real PDF in ``papers_dir``, sorted, matched case-insensitively.

    ``Path.glob`` is case-sensitive on APFS, which silently hid **7 papers with
    an uppercase extension** from the corpus enumeration -- and none of them
    were in the live index either, so a ``--pending`` run would have skipped
    them permanently rather than merely once.  Measured 2026-09-16: 4,854
    lowercase vs 4,861 any-case.

    ``._`` entries are macOS AppleDouble resource sidecars, not documents; there
    are 4,860 of them beside the real files, and MinerU would spend a timeout on
    each one.
    """
    return sorted(
        (f for f in papers_dir.iterdir()
         if f.suffix.lower() == ".pdf" and not f.name.startswith("._")),
        key=lambda f: f.name,
    )


class PaperResolver:
    """Maps paper ids and filenames onto the PDFs in ``papers_dir``.

    Why this is more than ``md5(filename)[:12]``
    --------------------------------------------
    ``paper_id`` is ``md5(filename)[:12]`` over the filename *bytes*.  APFS
    stores some names decomposed (NFD: ``e`` + combining acute) and returns them
    that way from ``glob``, while every other source in this project -- the
    SQLite message metadata, the old checkpoint, hand-written lists -- carries
    the composed (NFC) form.  The two forms hash to different ids, so the same
    paper can and does exist under two ids: measured on this corpus, **20 of
    4,854 PDFs have NFD names and 8 of them are already in the live Qdrant
    collection twice, under both ids** (363 duplicated chunks).

    The filesystem hides this, because APFS lookup is normalisation-insensitive:
    ``(dir / nfc_name).exists()`` is True for a file stored as NFD.  Only the
    hash notices.  So the disk index is keyed by all three forms and every hit
    reports which form matched.
    """

    def __init__(self, papers_dir: Path, legacy_checkpoint: Optional[Path] = None):
        self.papers_dir = Path(papers_dir)
        self.by_id: Dict[str, Path] = {}
        self.id_form: Dict[str, str] = {}
        self.aliases: Dict[str, List[str]] = {}
        self.n_files = 0
        self.n_nfd_names = 0

        for path in pdf_files(self.papers_dir):
            self.n_files += 1
            raw = path.name
            nfc = unicodedata.normalize("NFC", raw)
            nfd = unicodedata.normalize("NFD", raw)
            if nfc != raw:
                self.n_nfd_names += 1
            ids: List[Tuple[str, str]] = [(pr.paper_id_for_filename(raw), "exact")]
            if nfc != raw:
                ids.append((pr.paper_id_for_filename(nfc), "nfc"))
            if nfd != raw and nfd != nfc:
                ids.append((pr.paper_id_for_filename(nfd), "nfd"))
            all_ids = [i for i, _ in ids]
            for pid, form in ids:
                # First writer wins; a genuine md5-12 collision would show up as
                # a differing path, which we surface rather than silently drop.
                if pid in self.by_id and self.by_id[pid] != path:
                    continue
                self.by_id[pid] = path
                self.id_form[pid] = form
                self.aliases[pid] = [i for i in all_ids if i != pid]

        # The old checkpoint's `paper_metadata[id]['filename']` is a *display*
        # name (it is rewritten by the metadata layer), so it is only a fallback
        # -- for 1 of the 591 cited papers it disagrees with the id entirely.
        self.legacy_names: Dict[str, str] = {}
        if legacy_checkpoint and Path(legacy_checkpoint).exists():
            try:
                with open(legacy_checkpoint, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                for pid, meta in (data.get("paper_metadata") or {}).items():
                    name = (meta or {}).get("filename")
                    if name:
                        self.legacy_names[pid] = name
            except (json.JSONDecodeError, OSError):
                pass

    # -- single token ------------------------------------------------------

    def resolve(self, token: str) -> Resolved:
        token = token.strip()

        if token.lower().endswith(".pdf") and ("/" in token or os.sep in token):
            path = Path(token).expanduser()
            pid = pr.paper_id_for_filename(path.name)
            if path.exists():
                return Resolved(token, pid, path, via="path",
                                alias_ids=self.aliases.get(pid, []))
            return Resolved(token, pid, None, via="none", note="path does not exist")

        if looks_like_paper_id(token):
            return self._resolve_id(token)

        name = token if token.lower().endswith(".pdf") else f"{token}.pdf"
        pid = pr.paper_id_for_filename(name)
        hit = self._resolve_id(pid)
        if hit.ok:
            return Resolved(token, pid, hit.path, via=hit.via, note=hit.note,
                            alias_ids=hit.alias_ids)
        # Name given in the other normalisation than the disk holds.
        for form in ("NFC", "NFD"):
            alt = pr.paper_id_for_filename(unicodedata.normalize(form, name))
            if alt in self.by_id:
                return Resolved(token, pid, self.by_id[alt], via=form.lower(),
                                note=f"file found via {form} of the given name; "
                                     f"id kept as {pid}",
                                alias_ids=[alt])
        return Resolved(token, pid, None, via="none", note="no PDF with that name")

    def _resolve_id(self, paper_id: str) -> Resolved:
        path = self.by_id.get(paper_id)
        if path is not None:
            return Resolved(paper_id, paper_id, path,
                            via=self.id_form.get(paper_id, "exact"),
                            alias_ids=self.aliases.get(paper_id, []))
        # Fall back to the old checkpoint's display filename.  The id is kept as
        # given -- every citation already stored in SQLite points at it.
        name = self.legacy_names.get(paper_id)
        if name:
            for cand in {name, unicodedata.normalize("NFC", name),
                         unicodedata.normalize("NFD", name)}:
                p = self.papers_dir / cand
                if p.exists():
                    return Resolved(paper_id, paper_id, p, via="legacy_name",
                                    note=f"resolved via checkpoint filename {cand!r}")
        return Resolved(paper_id, paper_id, None, via="none",
                        note="no PDF hashes to this id"
                             + (f" (checkpoint filename {name!r} also absent)" if name else ""))

    # -- bulk --------------------------------------------------------------

    def resolve_all(self, tokens: Iterable[str]) -> List[Resolved]:
        seen: set = set()
        out: List[Resolved] = []
        for token in tokens:
            hit = self.resolve(token)
            if hit.paper_id in seen:
                continue
            seen.add(hit.paper_id)
            out.append(hit)
        return out

    def all_on_disk(self) -> List[Resolved]:
        """Every PDF in the directory, keyed by its on-disk-name id.

        This is the derivation production's upload path uses
        (``paper_library._generate_paper_id``), so a corpus-wide run and an
        upload agree.  Papers whose name is NFD carry their NFC id in
        ``alias_ids`` so the caller can see the split rather than discover it in
        the frontend.
        """
        out: List[Resolved] = []
        for path in pdf_files(self.papers_dir):
            pid = pr.paper_id_for_filename(path.name)
            out.append(Resolved(path.name, pid, path, via="exact",
                                alias_ids=self.aliases.get(pid, [])))
        return out


# ---------------------------------------------------------------------------
# cited cohort (the 591)
# ---------------------------------------------------------------------------

def cited_paper_ids(db_path: Path) -> List[str]:
    """Distinct ``sources[].paper_id`` across every assistant message.

    Derived, never hardcoded: this is the cohort ARC has actually cited, so it
    is the cohort whose retrieval quality is measurable against W5's baseline.
    """
    ids: set = set()
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = con.execute(
            "SELECT message_metadata FROM messages WHERE message_metadata IS NOT NULL"
        )
        for (blob,) in cur:
            if not blob:
                continue
            try:
                meta = json.loads(blob)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(meta, dict):
                continue
            for src in meta.get("sources") or []:
                if isinstance(src, dict) and src.get("paper_id"):
                    ids.add(str(src["paper_id"]))
    finally:
        con.close()
    return sorted(ids)


# ---------------------------------------------------------------------------
# record validity -- the record-reuse check
# ---------------------------------------------------------------------------

CURRENT_SCHEMA = pr.SCHEMA_VERSION


def current_extractor_version() -> str:
    from preprocessing.pdf_processor import _mineru_version
    return _mineru_version()


def file_facts(path: Path) -> Dict[str, Any]:
    st = os.stat(path)
    return {"file_size": st.st_size, "file_mtime": int(st.st_mtime)}


def record_file_facts(path: Path) -> Dict[str, Any]:
    try:
        st = os.stat(path)
        return {"record_bytes": st.st_size, "record_mtime": int(st.st_mtime)}
    except OSError:
        return {}


def fingerprint(record: Dict[str, Any], pdf_path: Optional[Path] = None,
                record_path: Optional[Path] = None,
                pdf_sha256: Optional[str] = None) -> Dict[str, Any]:
    """The four W1 fields the reuse check compares, plus what makes it cheap.

    ``pdf_sha256`` is ours, not the record's: it is the only way to tell "this
    file was copied across volumes so its mtime moved" from "this file changed".
    """
    fp = {
        "schema_version": record.get("schema_version"),
        "extractor": record.get("extractor"),
        "extractor_version": record.get("extractor_version"),
        "file_size": record.get("file_size"),
        "file_mtime": record.get("file_mtime"),
        "extracted_at": record.get("extracted_at"),
        "degraded": bool(record.get("degraded")),
    }
    if pdf_sha256:
        fp["pdf_sha256"] = pdf_sha256
    if record_path is not None:
        fp.update(record_file_facts(record_path))
    return fp


@dataclass
class RecordState:
    #: current  -- reuse it, skip extraction
    #: missing  -- no record on disk
    #: stale    -- record exists but does not describe this PDF / this schema
    #: failed   -- a record exists saying no extractor could read the PDF.
    #:             Distinct from stale on purpose: reaching it again costs a full
    #:             extraction timeout and lands in the same place, so a plain
    #:             "extract everything pending" run must not keep retrying it.
    state: str
    reason: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def needs_extract(self) -> bool:
        return self.state != "current"


def record_state(
    record: Optional[Dict[str, Any]],
    pdf_path: Optional[Path],
    *,
    mtime_policy: str = "warn",
    known_sha256: Optional[str] = None,
    extractor_version: Optional[str] = None,
    redo_degraded: bool = False,
) -> RecordState:
    """Decide whether a persisted record can be reused for this PDF.

    Compares exactly the four fields W1 already stores -- ``schema_version``,
    ``extractor_version``, ``file_size``, ``file_mtime`` -- with one deliberate
    softening documented below.

    mtime policy
    ------------
    Copying a file between volumes rewrites ``st_mtime`` without changing a
    byte.  Treating that as a change would re-run 98 h of GPU time for a file
    move, so the default (``warn``) is: **size identical + mtime moved => reuse,
    and say so.**  If we recorded the PDF's sha256 when we extracted it, the
    ambiguity is resolved properly: hash the file and only re-extract if the
    bytes really differ.  ``--mtime-policy strict`` restores the pedantic
    behaviour for anyone who wants it.
    """
    if record is None:
        return RecordState("missing", "no_record")

    if record.get("degraded_reason") == pr.DEGRADED_FAILED or not (record.get("blocks") or []):
        return RecordState("failed",
                           record.get("degraded_reason") or "no_blocks")

    if record.get("schema_version") != CURRENT_SCHEMA:
        return RecordState("stale", f"schema_version {record.get('schema_version')} "
                                    f"!= {CURRENT_SCHEMA}")

    if record.get("degraded"):
        if redo_degraded:
            return RecordState("stale", f"degraded:{record.get('degraded_reason')}")
        # The pypdfium2 fallback (~3.1% of papers) is an accepted outcome, not a
        # failure to retry on every run -- it costs a full extraction timeout to
        # reach it again and lands in the same place.
        return RecordState("current", "", [f"degraded:{record.get('degraded_reason')}"])

    want_ev = extractor_version if extractor_version is not None else current_extractor_version()
    if record.get("extractor") == "mineru" and want_ev and record.get("extractor_version") != want_ev:
        return RecordState("stale", f"extractor_version {record.get('extractor_version')!r} "
                                    f"!= {want_ev!r}")

    if pdf_path is None:
        return RecordState("current", "", ["source PDF not resolved; record kept"])

    try:
        facts = file_facts(pdf_path)
    except OSError as exc:
        return RecordState("current", "", [f"could not stat PDF ({exc}); record kept"])

    if record.get("file_size") != facts["file_size"]:
        return RecordState("stale", f"file_size {record.get('file_size')} "
                                    f"!= {facts['file_size']}")

    if record.get("file_mtime") != facts["file_mtime"]:
        if mtime_policy == "strict":
            return RecordState("stale", f"file_mtime {record.get('file_mtime')} "
                                        f"!= {facts['file_mtime']} (strict)")
        if known_sha256:
            actual = sha256_file(pdf_path)
            if actual != known_sha256:
                return RecordState("stale", "pdf_sha256 changed")
            return RecordState("current", "", ["mtime moved, sha256 identical"])
        return RecordState("current", "", ["mtime moved, size identical, no stored sha256"])

    return RecordState("current", "")


# ---------------------------------------------------------------------------
# stage manifest
# ---------------------------------------------------------------------------

class ManifestSignatureError(RuntimeError):
    """Raised at startup, never mid-run, when a manifest describes other work."""


class StageManifest:
    """Per-stage progress: append-only journal + atomic snapshot.

    Why both
    --------
    A checkpoint written by ``json.dump(open(path, "w"))`` is truncated garbage
    if the process is killed during the write, and re-dumping a 600-entry file
    after every paper is what made the old checkpoint 2.5 MB of rewrite churn.
    So: every outcome is appended to a flushed ``.jsonl`` journal the instant it
    happens, and the snapshot is written to a temp file and ``os.replace``d
    every ``save_every`` papers.  Load = snapshot + replay of the journal, with
    a truncated final line tolerated and counted.
    """

    def __init__(self, path: Path, signature: Dict[str, Any],
                 hard_keys: Sequence[str] = (), save_every: int = 25):
        self.path = Path(path)
        self.journal_path = self.path.with_suffix(".jsonl")
        self.signature = dict(signature)
        self.hard_keys = tuple(hard_keys)
        self.save_every = max(1, save_every)
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.header: Dict[str, Any] = {}
        self.stats: Dict[str, Any] = {}
        self.replayed = 0
        self.truncated_lines = 0
        self._since_snapshot = 0
        self._journal = None
        self._load()

    # -- load / signature --------------------------------------------------

    def _load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self.header = data.get("header", {})
                self.entries = data.get("papers", {}) or {}
                self.stats = data.get("stats", {}) or {}
            except (json.JSONDecodeError, OSError) as exc:
                raise ManifestSignatureError(
                    f"{self.path} exists but is unreadable ({exc}). Move it aside "
                    f"or pass --manifest <other path>; refusing to guess."
                ) from exc
        if self.journal_path.exists():
            with open(self.journal_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        # Only ever the last line, from a kill mid-append.
                        self.truncated_lines += 1
                        continue
                    pid = row.get("paper_id")
                    if pid:
                        self.entries[pid] = row
                        self.replayed += 1

    def check_signature(self) -> List[str]:
        """Refuse a manifest that describes different work. Startup, not hour 3.

        Returns the *soft* differences (things that changed but do not
        invalidate the recorded work) so the caller can print them.
        """
        old = self.header.get("signature")
        if not old:
            return []
        diffs = [(k, old.get(k), self.signature.get(k))
                 for k in self.hard_keys if old.get(k) != self.signature.get(k)]
        if diffs:
            detail = "; ".join(f"{k}: manifest={o!r} now={n!r}" for k, o, n in diffs)
            raise ManifestSignatureError(
                f"{self.path} was written for different work ({detail}).\n"
                f"A manifest that says 'done' for another collection or model is a "
                f"lie, so this stops here. Use --manifest <new path> for the new "
                f"configuration (the old one stays valid for the old one)."
            )
        return [f"{k}: was {old.get(k)!r}, now {self.signature.get(k)!r}"
                for k in self.signature
                if k not in self.hard_keys and k in old
                and old.get(k) != self.signature.get(k)]

    # -- writes ------------------------------------------------------------

    def _open_journal(self):
        if self._journal is None:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            self._journal = open(self.journal_path, "a", encoding="utf-8")
        return self._journal

    def mark(self, paper_id: str, status: str, **fields: Any) -> Dict[str, Any]:
        row = {"paper_id": paper_id, "status": status, "at": utc_now()}
        row.update(fields)
        self.entries[paper_id] = row
        journal = self._open_journal()
        journal.write(json.dumps(row, ensure_ascii=False) + "\n")
        journal.flush()
        os.fsync(journal.fileno())
        self._since_snapshot += 1
        if self._since_snapshot >= self.save_every:
            self.snapshot()
        return row

    def snapshot(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.header.setdefault("created_at", utc_now())
        self.header["signature"] = self.signature
        self.header["updated_at"] = utc_now()
        self.header["n_papers"] = len(self.entries)
        payload = {"header": self.header, "stats": self.stats, "papers": self.entries}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)
        # The journal's contents are now inside the snapshot.
        if self._journal is not None:
            self._journal.close()
            self._journal = None
        if self.journal_path.exists():
            self.journal_path.unlink()
        self._since_snapshot = 0

    def close(self) -> None:
        self.snapshot()

    # -- reads -------------------------------------------------------------

    def get(self, paper_id: str) -> Optional[Dict[str, Any]]:
        return self.entries.get(paper_id)

    def status_of(self, paper_id: str) -> Optional[str]:
        entry = self.entries.get(paper_id)
        return entry.get("status") if entry else None

    def ids_with_status(self, *statuses: str) -> List[str]:
        want = set(statuses)
        return [pid for pid, row in self.entries.items() if row.get("status") in want]

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for row in self.entries.values():
            out[row.get("status", "?")] = out.get(row.get("status", "?"), 0) + 1
        return dict(sorted(out.items()))

    def mean_seconds(self, statuses: Sequence[str] = ("ok", "degraded")) -> Optional[float]:
        vals = [row["seconds"] for row in self.entries.values()
                if row.get("status") in statuses and isinstance(row.get("seconds"), (int, float))]
        if not vals:
            return None
        return sum(vals) / len(vals)

    def sum_field(self, name: str, statuses: Sequence[str] = ("ok", "degraded")) -> float:
        return float(sum(row.get(name) or 0 for row in self.entries.values()
                         if row.get("status") in statuses))


def open_manifest(path: Path, signature: Dict[str, Any],
                  hard_keys: Sequence[str] = (), save_every: int = 25
                  ) -> Tuple["StageManifest", List[str]]:
    """Load a manifest, turning a signature clash into a clean startup exit.

    A traceback is the wrong shape for "this manifest describes other work":
    the operator needs the sentence, not the stack.
    """
    try:
        manifest = StageManifest(path, signature, hard_keys=hard_keys,
                                 save_every=save_every)
        return manifest, manifest.check_signature()
    except ManifestSignatureError as exc:
        raise SystemExit(f"\nmanifest refused:\n  {exc}\n") from None


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------

def configure_logging(verbose: bool = False) -> None:
    """Keep the stage's own per-paper lines readable.

    The chunker logs one INFO line per paper and MinerU/httpx several, which
    buries the progress lines these stages print.  ``--verbose`` puts them back.
    """
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if not verbose:
        for noisy in ("httpx", "mineru", "preprocessing.chunker",
                      "preprocessing.pdf_processor", "preprocessing.paper_record",
                      "retrieval.qdrant_store", "retrieval.bm25", "retrieval.embedder"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def rule(title: str = "", width: int = 78) -> str:
    if not title:
        return "=" * width
    return f"{title} " + "=" * max(0, width - len(title) - 1)


def counter_line(counts: Dict[str, Any]) -> str:
    return "  ".join(f"{k}={v}" for k, v in counts.items())
