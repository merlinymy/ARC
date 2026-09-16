"""Figure and table labels, and the in-text references that point at them.

Two jobs, one vocabulary:

1. **Label an object.**  Given a figure or table in the paper record, work out
   what the paper *calls* it -- ``Figure 3``, ``Table II``, ``Scheme 1``,
   ``Figure S1`` -- from its caption, and where the caption does not say, from
   the object's position in the reading-order sequence.
2. **Resolve a reference.**  Given body prose, find every in-text mention
   ("as shown in Figure 3", "see Table 2", "Fig. 2a") and map it back to an
   object, or to nothing.

Why both live here
------------------
These are the same problem read from two directions, and they have to agree on
one normalization or the join silently fails.  A paper that prints ``TABLE II``
in the caption and writes "see Table II" in the text, next to one that prints
``Table 2`` and writes "Table 2", must produce the same key from both sides.

Why references are not chunks
-----------------------------
R2: the pre-W1 caption regex ran ``finditer`` over the whole markdown, so every
one of these mentions became a "caption" chunk -- 93,144 of them, 44% of the
index, and 58.8% of them mid-sentence body fragments.  They are worthless as
retrievable text and useful as *links*.  This module produces links.  Nothing
here creates a chunk.

The failure mode is invisible, so partial resolution is safe
------------------------------------------------------------
:func:`resolve_references` returns a target of ``None`` for a mention it cannot
place, and the contract with the UI is that an unresolved mention renders as
ordinary text.  There is no dead-link state.  That is what makes it acceptable
to ship at less than 100%: a miss costs nothing, while a *wrong* link costs
trust -- which is why cross-kind guessing is deliberately restricted below.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

#: Every space-like codepoint a typesetter puts between "Figure" and "3".  A
#: non-breaking space is the norm in Elsevier and Wiley PDFs, and `\s` in a
#: non-Unicode-aware pattern would not match it.
_SPACE_CHARS = "     ⁠﻿~"
_SPACE_RE = re.compile(f"[{re.escape(_SPACE_CHARS)}]")

#: En dash, em dash, minus sign, figure dash -- all used for ranges ("Figs 2-4").
_DASH_CHARS = "‐‑‒–—―−"
_DASH_RE = re.compile(f"[{re.escape(_DASH_CHARS)}]")


def normalize(text: str) -> str:
    """Fold exotic spaces to ``" "`` and exotic dashes to ``"-"``.

    Length-preserving on purpose: every replacement is one character for one
    character, so an offset into the normalized string is also an offset into
    the original.  That is what lets a match found here be reported in *record*
    coordinates without a second search.
    """
    return _DASH_RE.sub("-", _SPACE_RE.sub(" ", text or ""))


assert len(normalize("a b–c")) == 5  # the offset guarantee, pinned


# ---------------------------------------------------------------------------
# Object kinds
# ---------------------------------------------------------------------------

#: Printed label word -> canonical label kind.
_KIND_WORDS = {
    "fig": "figure", "figs": "figure", "figure": "figure", "figures": "figure",
    "figur": "figure",
    "tab": "table", "tabs": "table", "table": "table", "tables": "table",
    "scheme": "scheme", "schemes": "scheme",
    "chart": "chart", "charts": "chart",
    "plate": "plate", "plates": "plate",
    "box": "box", "boxes": "box",
    "panel": "figure", "panels": "figure",
    "exhibit": "figure", "exhibits": "figure",
    "graph": "figure", "graphs": "figure",
}

#: Kinds that may stand in for one another when the exact kind misses and the
#: substitute is unambiguous.  A "Scheme 1" reference in a paper that captions
#: the object "Figure 1" is a real and common mismatch; a figure standing in for
#: a *table* is not, so the two families never cross.
_KIND_FAMILY = {
    "figure": ("figure", "scheme", "chart", "plate", "box"),
    "scheme": ("scheme", "figure", "chart"),
    "chart": ("chart", "figure", "scheme"),
    "plate": ("plate", "figure"),
    "box": ("box", "figure"),
    "table": ("table",),
}

_KIND_ALT = "|".join(sorted(_KIND_WORDS, key=len, reverse=True))

#: Numbers restricted to I/V/X: I..XXXIX covers every real "Table XII", while
#: excluding C/D/L/M keeps a stray panel letter ("Figure C") from being read as
#: figure 100.
_ROMAN_RE = re.compile(r"^(?=[IVX]{1,6}$)X{0,3}(IX|IV|V?I{0,3})$")
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10}

#: A label number: optional supplementary/electronic prefix, digits with an
#: optional chapter part ("13.1" in a book chapter), or a roman numeral.
_NUMBER = r"(?:(?:S|ESI|SI)[ .\-]?)?(?:\d{1,3}(?:\.\d{1,3})?|[IVX]{1,6})"
#: A panel suffix: "3a", "3(a)", "3A", "3 (a)".
_PANEL = r"(?:[ ]?\(\s*[a-zA-Z]\s*\)|(?<=\d)[a-hA-H](?![a-zA-Z0-9]))"


def _roman_to_int(token: str) -> Optional[int]:
    if not _ROMAN_RE.match(token):
        return None
    total = 0
    prev = 0
    for char in reversed(token):
        value = _ROMAN_VALUES[char]
        total += -value if value < prev else value
        prev = max(prev, value)
    return total or None


def normalize_number(raw: str) -> Optional[str]:
    """Canonical form of a label number: ``"3"``, ``"13.1"``, ``"S2"``.

    Roman numerals fold to arabic so that a paper captioning ``TABLE II`` and
    writing "see Table 2" -- or the reverse -- joins up.  The supplementary
    prefix is kept, because ``Figure S1`` and ``Figure 1`` are different objects.
    """
    if not raw:
        return None
    token = normalize(raw).strip().upper().replace(" ", "").replace("-", "")
    supplementary = ""
    for prefix in ("ESI", "SI", "S"):
        if token.startswith(prefix) and len(token) > len(prefix):
            supplementary = "S"
            token = token[len(prefix):]
            break
    token = token.lstrip(".")
    if not token:
        return None
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3})?", token):
        number = token.rstrip(".0") if token.endswith(".0") else token
        return f"{supplementary}{number}"
    value = _roman_to_int(token)
    if value is not None:
        return f"{supplementary}{value}"
    return None


def kind_of(word: str) -> Optional[str]:
    return _KIND_WORDS.get(normalize(word).strip().rstrip(".").lower())


def label_text(kind: str, number: str) -> str:
    return f"{kind.title()} {number}"


# ---------------------------------------------------------------------------
# Labelling an object from its caption
# ---------------------------------------------------------------------------

#: Anchored at the caption start, tolerating any short run of non-word
#: characters before the label: markdown emphasis, a leading bracket, an OCR'd
#: bullet, a stray pipe, or a journal's decorative glyph -- the American Journal
#: of Clinical Pathology sets its table captions as "❚Table 1❚ ...", which no
#: hand-listed character class was going to anticipate.  Still anchored, and
#: still excluding word characters, so "(a) Figure 3" or "50 nm. Figure 3" is
#: not read as a caption opening.
_CAPTION_LABEL_RE = re.compile(
    r"^[^\w\n]{0,8}"
    rf"(?P<kind>{_KIND_ALT})"
    r"\s*[.:\-|]?\s*"
    rf"(?P<num>{_NUMBER})"
    rf"(?P<panel>{_PANEL})?"
    r"(?![\d.]*\s*(?:[A-Za-z]{4,}\s+){0,0}\w*\d{4}\b)",   # not a year run-on
    re.IGNORECASE,
)

#: "Figure 3 (continued)", "Table 2, cont." -- a second caption block for an
#: object that was already labelled, or for the object immediately before it.
_CONTINUATION_RE = re.compile(
    r"^[^\w\n]{0,8}(?:continu|cont\b|contd\b)", re.IGNORECASE
)

#: OCR mangles the label word itself often enough to be worth one fuzzy pass:
#: "FIURE 3", "Fi 3.", "Fgure 2".  Only fires when a number follows, and only
#: for a token that is *nearly* one of the kind words.
_FUZZY_HEAD_RE = re.compile(
    r"^[^\w\n]{0,8}(?P<word>[A-Za-z]{2,9})\s*\.?\s*"
    rf"(?P<num>{_NUMBER})(?P<panel>{_PANEL})?\b",
)

_FUZZY_TARGETS = ("figure", "table", "scheme", "chart")

#: Two-letter stubs worth accepting. A ratio rule would also accept "Fe", which
#: starts a great many real chemistry captions ("Fe 3 O 4 nanoparticles...").
_FUZZY_STUBS = {"fi": "figure", "fg": "figure", "ta": "table", "tb": "table",
                "sc": "scheme"}


def _fuzzy_kind(word: str) -> Optional[str]:
    """Is ``word`` a damaged spelling of a kind word?

    Requires the first letter to survive and at least half the letters to appear
    in order -- enough for "FIURE"/"Fgure"/"Fi", not enough for "From" or
    "Where".
    """
    low = word.lower()
    if low in _KIND_WORDS:
        return _KIND_WORDS[low]
    if len(low) < 3:
        return _FUZZY_STUBS.get(low)
    for target in _FUZZY_TARGETS:
        if low[0] != target[0]:
            continue
        # subsequence check
        it = iter(target)
        if not all(ch in it for ch in low):
            continue
        if len(low) * 2 >= len(target):
            return target
        # A long stub of a short word: "tabl" of "table".
        if len(low) >= 4:
            return target
    return None


def parse_label(text: str, fuzzy: bool = True) -> Optional[Dict[str, Any]]:
    """Parse an object label out of the *start* of a caption.

    Returns ``{"kind", "number", "panel", "label", "fuzzy"}`` or ``None``.
    Anchored deliberately: a caption that merely *mentions* another figure
    mid-sentence must not be read as that figure's caption.
    """
    if not text:
        return None
    candidate = normalize(text)[:120]
    match = _CAPTION_LABEL_RE.match(candidate)
    if match:
        kind = kind_of(match.group("kind"))
        number = normalize_number(match.group("num"))
        if kind and number:
            return {
                "kind": kind,
                "number": number,
                "panel": _panel_letter(match.group("panel")),
                "label": label_text(kind, number),
                "fuzzy": False,
            }
    if not fuzzy:
        return None
    match = _FUZZY_HEAD_RE.match(candidate)
    if match:
        kind = _fuzzy_kind(match.group("word"))
        number = normalize_number(match.group("num"))
        if kind and number:
            return {
                "kind": kind,
                "number": number,
                "panel": _panel_letter(match.group("panel")),
                "label": label_text(kind, number),
                "fuzzy": True,
            }
    return None


def _panel_letter(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    letters = re.findall(r"[A-Za-z]", raw)
    return letters[0].lower() if letters else None


def is_continuation(text: str) -> bool:
    return bool(_CONTINUATION_RE.match(normalize(text or "")))


# ---------------------------------------------------------------------------
# Labelling every object in a record
# ---------------------------------------------------------------------------

def ensure_object_ids(record: Dict[str, Any]) -> None:
    """Backfill ``figure_id`` / ``kind`` / ``index`` on a schema-1 record.

    Schema 1 stored figures and tables without a public handle, so an older
    record read back today would have nothing for a chunk payload or an API
    route to point at.  The handle is purely positional, so it can be recomputed
    exactly -- which is what keeps the 40 existing sample records usable instead
    of forcing a re-extraction to measure anything.
    """
    for group, kind in (("figures", "figure"), ("tables", "table")):
        for n, entry in enumerate(record.get(group) or []):
            entry.setdefault("kind", kind)
            entry.setdefault("index", n)
            entry.setdefault("figure_id", f"{kind}_{n}")


def _object_entries(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """``figures[]`` then ``tables[]``, each in reading order."""
    ensure_object_ids(record)
    return list(record.get("figures") or []) + list(record.get("tables") or [])


def _caption_texts(record: Dict[str, Any], entry: Dict[str, Any]) -> List[str]:
    blocks = record.get("blocks") or []
    return [blocks[i]["text"] for i in entry.get("caption_blocks", [])
            if 0 <= i < len(blocks)]


def label_record_objects(record: Dict[str, Any]) -> Dict[str, Any]:
    """Attach ``label`` / ``label_kind`` / ``label_number`` / ``label_source``.

    Three passes, in falling order of confidence:

    ``caption``       the object's own caption opens with a parseable label.
    ``caption_fuzzy`` the same, after repairing an OCR-damaged label word.
    ``sequence``      the caption says nothing, so the number is read off the
                      object's position among its labelled siblings.

    The third pass is what recovers the two categories that made up most of the
    measured misses: 44 objects with an unlabelled caption and 33 with no
    caption at all.  It is marked as inferred so a consumer can drop it.
    """
    summary = {"objects": 0, "caption": 0, "caption_fuzzy": 0,
               "sequence": 0, "unlabeled": 0, "duplicates": 0}
    ensure_object_ids(record)
    entries = _object_entries(record)
    if not entries:
        return summary
    summary["objects"] = len(entries)

    for entry in entries:
        entry.pop("label", None)
        entry.pop("label_kind", None)
        entry.pop("label_number", None)
        entry.pop("label_source", None)
        entry.pop("label_panel", None)

        parsed = None
        for text in _caption_texts(record, entry):
            parsed = parse_label(text)
            if parsed:
                break
        if parsed is None:
            continue
        entry["label_kind"] = parsed["kind"]
        entry["label_number"] = parsed["number"]
        entry["label"] = parsed["label"]
        entry["label_panel"] = parsed["panel"]
        entry["label_source"] = "caption_fuzzy" if parsed["fuzzy"] else "caption"
        summary[entry["label_source"]] += 1

    _label_by_sequence(record, entries, summary)

    # A label claimed by two objects (a figure continued across a page gets a
    # second `image` item whose caption repeats the label). Keep the first for
    # resolution and say so, rather than letting dict insertion order decide.
    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for entry in entries:
        key = (entry.get("label_kind"), entry.get("label_number"))
        if key == (None, None):
            continue
        if key in seen:
            entry["label_duplicate"] = True
            summary["duplicates"] += 1
        else:
            seen[key] = entry
    summary["unlabeled"] = sum(1 for e in entries if not e.get("label"))
    return summary


_MINOR_RE = re.compile(r"^(?P<prefix>\d{1,3}\.)?(?P<minor>\d{1,3})$")


def _split_number(number: Optional[str]) -> Optional[Tuple[str, int]]:
    """``"29.6"`` -> ``("29.", 6)``; ``"3"`` -> ``("", 3)``; else ``None``.

    Book chapters number figures ``29.1 .. 29.12``, so the sequence arithmetic
    has to run on the minor part while holding the chapter prefix fixed.  Doing
    it on the float would make 29.9 + 1 = 30.9.
    """
    if not number:
        return None
    match = _MINOR_RE.match(number)
    if not match:
        return None
    return (match.group("prefix") or "", int(match.group("minor")))


def _label_by_sequence(record, entries, summary) -> None:
    """Fill in missing numbers from position among labelled siblings.

    Runs per *group* (the record's ``figures`` list, then ``tables``), because an
    interleaved sequence numbers independently, and only against siblings that
    share the dominant label kind and numbering prefix.

    A gap is filled only when the arithmetic works out exactly: two unlabelled
    objects between Figure 2 and Figure 5 is a gap of two, so they become 3 and
    4.  Between Figure 2 and Figure 4 it is a gap of one for two objects, which
    means an assumption is wrong -- so neither is labelled.  That asymmetry is
    the point: a missing link is invisible, a wrong link is not.
    """
    groups: List[List[Dict[str, Any]]] = [
        list(record.get("figures") or []), list(record.get("tables") or [])]

    for group_n, group in enumerate(groups):
        if not group:
            continue
        # The dominant kind in the group decides what an unlabelled member is
        # presumed to be: a `figures` list of eight schemes and one unlabelled
        # object means a scheme, not a figure.
        kinds = [e.get("label_kind") for e in group if e.get("label_kind")]
        main_kind = max(set(kinds), key=kinds.count) if kinds else (
            "table" if group_n == 1 else "figure")

        # Supplementary objects number separately; never interpolate across.
        members = [e for e in group
                   if not (e.get("label_number") or "").startswith("S")]
        split = [(i, _split_number(e.get("label_number")))
                 for i, e in enumerate(members)
                 if e.get("label_kind") == main_kind]
        prefixes = [v[0] for _, v in split if v]
        prefix = max(set(prefixes), key=prefixes.count) if prefixes else ""
        known = [(i, v[1]) for i, v in split if v and v[0] == prefix]
        blanks = [i for i, e in enumerate(members) if not e.get("label")]
        if not blanks:
            continue

        def assign(index: int, minor: int) -> None:
            entry = members[index]
            number = f"{prefix}{minor}"
            entry["label_kind"] = main_kind
            entry["label_number"] = number
            entry["label"] = label_text(main_kind, number)
            entry["label_source"] = "sequence"
            summary["sequence"] += 1

        if not known:
            # Nothing labelled anywhere: fall back to plain reading order, which
            # is right whenever the paper numbers from 1 and MinerU found every
            # object. Only safe because every member is unlabelled -- there is
            # no evidence to contradict.
            if len(blanks) == len(members):
                for position, index in enumerate(blanks, start=1):
                    assign(index, position)
            continue

        for index in blanks:
            before = [(i, n) for i, n in known if i < index]
            after = [(i, n) for i, n in known if i > index]
            if before and after:
                lo_i, lo_n = before[-1]
                hi_i, hi_n = after[0]
                gap = [b for b in blanks if lo_i < b < hi_i]
                if hi_n - lo_n - 1 == len(gap):
                    assign(index, lo_n + 1 + gap.index(index))
            elif before:
                lo_i, lo_n = before[-1]
                offset = len([b for b in blanks if lo_i < b <= index])
                assign(index, lo_n + offset)
            elif after:
                hi_i, hi_n = after[0]
                offset = len([b for b in blanks if index <= b < hi_i])
                if hi_n - offset >= 1:
                    assign(index, hi_n - offset)


def label_index(record: Dict[str, Any],
                include_inferred: bool = True) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """``{(label_kind, label_number): object}`` -- the resolution table."""
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for entry in _object_entries(record):
        kind, number = entry.get("label_kind"), entry.get("label_number")
        if not kind or not number:
            continue
        if not include_inferred and entry.get("label_source") == "sequence":
            continue
        index.setdefault((kind, number), entry)
    return index


# ---------------------------------------------------------------------------
# Finding in-text references
# ---------------------------------------------------------------------------

#: One mention.  The kind word, then a first number, then any number of
#: continuations joined by ",", "and", "&", "to" or a dash -- which is what
#: "Figures 3 and 4", "Figs. 2-4" and "Fig. 3a,b" all are.
_REFERENCE_RE = re.compile(
    r"(?<![A-Za-z])"
    rf"(?P<kind>{_KIND_ALT})"
    r"(?P<dot>\.)?"
    r"[ ]{0,2}"
    rf"(?P<num>{_NUMBER})"
    rf"(?P<panel>{_PANEL})?"
    r"(?P<more>(?:[ ]{0,2}(?:,|and|&|to|-|/|or)[ ]{0,2}"
    rf"(?:{_NUMBER}|[a-hA-H](?![A-Za-z0-9]))(?:{_PANEL})?)*)",
    re.IGNORECASE,
)

#: Continuation items inside the `more` group.
_MORE_RE = re.compile(
    rf"(?P<sep>,|and|&|to|-|/|or)[ ]{{0,2}}"
    rf"(?P<num>{_NUMBER}|[a-hA-H](?![A-Za-z0-9]))(?P<panel>{_PANEL})?",
    re.IGNORECASE,
)

#: A label number is a whole token.  Anything glued to its right end means the
#: match clipped something else: "Fig1abc" is an identifier, and ``\d{1,3}``
#: against "schemes.114116" or "Fig. 1568" stops three digits in.
_TRAILING_WORD = re.compile(r"^[A-Za-z0-9]")

#: Only for continuation items.  "...for the sample of Fig. 3(A) and 50:50 for
#: the other" parses " and 50" as a second reference; a ratio or a fraction
#: immediately after the number says it is not one.  Not applied to the head,
#: because "Fig. 3/4" legitimately uses "/" as the separator.
_RATIO_TAIL = re.compile(r"^[:/]\s*\d")


def find_references(text: str) -> List[Dict[str, Any]]:
    """Every in-text mention in ``text``, with offsets into ``text``.

    One entry per *number*, not per mention: "Figures 3 and 4" yields two, which
    is what a reader wants -- two separate links -- and what makes the
    resolution rate mean "fraction of pointers that land".
    """
    if not text:
        return []
    source = normalize(text)          # length-preserving, so offsets transfer
    out: List[Dict[str, Any]] = []
    for match in _REFERENCE_RE.finditer(source):
        kind = kind_of(match.group("kind"))
        if kind is None:
            continue
        number = normalize_number(match.group("num"))
        if not number:
            continue
        head_end = match.end("panel") if match.group("panel") else match.end("num")
        # "Fig1abc" / "Table 1S": a letter glued to the number means this is an
        # identifier, not a pointer. Checked at the *head* so that a malformed
        # continuation cannot discard a well-formed first reference.
        if _TRAILING_WORD.match(source[head_end:head_end + 1]):
            continue
        out.append({
            "kind": kind,
            "number": number,
            "panel": _panel_letter(match.group("panel")),
            "start": match.start(),
            "end": head_end,
            "text": text[match.start():head_end],
            "mention_start": match.start(),
        })

        more = match.group("more") or ""
        if not more:
            continue
        base = match.end("panel") if match.group("panel") else match.end("num")
        last_number = number
        for item in _MORE_RE.finditer(more):
            raw = item.group("num")
            if re.fullmatch(r"[a-hA-H]", raw):
                # "Fig. 3a,b" -- a bare panel letter of the same object, not a
                # new object. Emitted so the whole "3a,b" span can be a link,
                # but it points at the same number.
                number_i, panel_i = last_number, raw.lower()
            else:
                number_i = normalize_number(raw)
                panel_i = _panel_letter(item.group("panel"))
                if not number_i:
                    continue
                last_number = number_i
            start = base + item.start("num")
            end = base + (item.end("panel") if item.group("panel") else item.end("num"))
            tail = source[end:end + 3]
            if _TRAILING_WORD.match(tail) or _RATIO_TAIL.match(tail):
                continue                       # "or 2nd", "and 50:50"
            out.append({
                "kind": kind,
                "number": number_i,
                "panel": panel_i,
                "start": start,
                "end": end,
                "text": text[start:end],
                "mention_start": match.start(),
                "continuation": True,
            })
    return out


# ---------------------------------------------------------------------------
# Resolving references against a record
# ---------------------------------------------------------------------------

def _target(entry: Dict[str, Any], panel: Optional[str],
            how: str) -> Dict[str, Any]:
    return {
        "figure_id": entry.get("figure_id"),
        "kind": entry.get("kind"),
        "block": entry.get("block"),
        "page_idx": entry.get("page_idx"),
        "page": (entry["page_idx"] + 1) if entry.get("page_idx") is not None else None,
        "bbox": entry.get("image_bbox") or entry.get("bbox"),
        "label": entry.get("label"),
        "label_source": entry.get("label_source"),
        "panel": panel,
        "match": how,
        "inferred": entry.get("label_source") == "sequence",
    }


def resolve_one(reference: Dict[str, Any],
                index: Dict[Tuple[str, str], Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Map one reference onto an object, or ``None``.

    Exact ``(kind, number)`` first.  Then, only within the same family and only
    when the substitute is unique, a cross-kind match -- a paper that writes
    "Scheme 2" about an object it captioned "Figure 2" is common; a figure
    standing in for a table is not, and is never tried.
    """
    kind, number = reference["kind"], reference["number"]
    entry = index.get((kind, number))
    if entry is not None:
        return _target(entry, reference.get("panel"), "exact")

    family = [k for k in _KIND_FAMILY.get(kind, (kind,)) if k != kind]
    hits = [index[(k, number)] for k in family if (k, number) in index]
    if len(hits) == 1:
        return _target(hits[0], reference.get("panel"), "family")
    return None


def resolve_references(record: Dict[str, Any], text: str,
                       include_inferred: bool = True,
                       index: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """Find and resolve every reference in ``text`` against ``record``.

    Each entry keeps its ``start``/``end`` into ``text`` and a ``target`` that
    is ``None`` when nothing matched.  The caller renders a ``None`` target as
    ordinary text.
    """
    table = index if index is not None else label_index(record, include_inferred)
    out = []
    for reference in find_references(text):
        reference = dict(reference)
        reference["target"] = resolve_one(reference, table)
        out.append(reference)
    return out


def refs_for_span(record: Dict[str, Any], text: str,
                  index: Optional[Dict] = None,
                  max_refs: int = 12) -> List[Dict[str, Any]]:
    """Compact, payload-sized resolved references for one chunk's text.

    Offsets are relative to ``text`` -- the chunk's own verbatim span -- so the
    UI can slice the string it already has.  Unresolved references are dropped
    *here*, not in the UI: an entry that reaches the frontend is always a live
    link, and anything absent is simply prose.
    """
    out: List[Dict[str, Any]] = []
    for reference in resolve_references(record, text, index=index):
        target = reference.get("target")
        if target is None or not target.get("figure_id"):
            continue
        out.append({
            "start": reference["start"],
            "end": reference["end"],
            "text": reference["text"],
            "figure_id": target["figure_id"],
            "label": target.get("label"),
            "page": target.get("page"),
            "inferred": bool(target.get("inferred")),
        })
        if len(out) >= max_refs:
            break
    return out
