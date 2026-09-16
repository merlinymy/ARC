"""The paper record — W1's keystone artifact.

One structured JSON document per paper, persisted to ``processed_data/{paper_id}.json``.

Why it exists
-------------
``pdf_processor`` used to flatten MinerU's ``content_list`` into four flat strings
(full text, markdown, a list of table strings, a list of regex-scraped captions) and
throw the rest away.  Everything downstream then had to re-derive structure with
regexes over that flattened text.  This module keeps the structure instead:

* **R1** — tables are read from ``table_body`` (HTML), with ``table_caption`` /
  ``table_footnote`` attached.
* **R2** — captions come from MinerU's ``image_caption`` / ``table_caption`` lists,
  never from a regex over body prose.
* **R16** — sections come from MinerU's heading blocks, and the abstract from the
  leading block sequence, instead of line-anchored regexes.
* **R6** — every block keeps ``page_idx`` and ``bbox``.

Coordinate systems
------------------
There is exactly one text coordinate system, and it is built here so that offsets
are *derived from* the text rather than searched for afterwards:

* ``page_text(i)`` = the texts of every block on page ``i``, in MinerU reading order,
  joined by ``BLOCK_SEP``.  Blocks with empty text occupy a zero-length span.
* ``full_text``    = every page text in order, joined by ``PAGE_SEP``.

Each block therefore satisfies, byte-exactly::

    page_text(b["page_idx"])[b["page_char_start"]:b["page_char_end"]] == b["text"]
    full_text[b["doc_char_start"]:b["doc_char_end"]]                  == b["text"]

W3 maps a Citations API ``char_location`` through a chunk's stored offsets into
``doc_char_*``, then to ``page_idx`` + ``bbox`` for the PDF highlight.  If those two
identities do not hold the highlight cannot be built, so
:func:`verify_record_offsets` checks them and ``pages[i]["md5"]`` pins the exact
page text the offsets were computed against.

Bounding boxes
--------------
MinerU emits ``bbox`` as integers scaled to 0..1000 of the page width/height, with a
**top-left** origin (it detects layout on a rendered raster, not in PDF user space).
Both facts are recorded on the document (``bbox_units`` / ``bbox_origin``) alongside
each page's true ``width`` / ``height`` in PDF points, so a consumer can convert::

    x_pt = bbox[0] / 1000 * page["width"]
    y_from_top_pt = bbox[1] / 1000 * page["height"]

Storage
-------
Block text is stored once.  Page text, the document text and section text are all
*derived* by joining block text, never stored, so the record is ~1x the paper's
characters rather than 4x.

Figure and table crops (schema 2)
---------------------------------
MinerU crops every figure and table it detects to a JPEG.  Schema 1 handed the
crop writer a ``tempfile.TemporaryDirectory()``, so the bytes were deleted when
the extraction subprocess exited and every ``img_path`` in every record was
dangling.  Schema 2 writes them to a durable per-paper directory instead::

    processed_data/{paper_id}.json          the record
    processed_data/{paper_id}/figures/*.jpg the crops

``img_path`` is now stored **relative to** ``processed_data/{paper_id}/`` --
``"figures/<sha256>.jpg"`` -- so a record stays valid if the library moves.
Resolve it with :func:`asset_path` / :func:`figure_image`, never by hand.

Filenames are MinerU's content hash, so re-extracting an unchanged paper
rewrites the same names.  :func:`prune_assets` then deletes anything in the
directory the new record does not reference, so a re-extraction that *does*
change the crops (a MinerU upgrade, a different page raster) cannot accumulate
orphans.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: 1 -> 2: figure/table crops persisted; ``img_path`` became a relative asset
#: reference and image blocks gained ``image_bbox``.  Schema-1 records stay
#: readable -- every new field is optional and every accessor degrades to
#: "no image available" rather than raising.
SCHEMA_VERSION = 2

#: Subdirectory of ``processed_data/{paper_id}/`` holding the crops.
FIGURES_SUBDIR = "figures"

#: Extensions MinerU can write as a crop.  Used to bound :func:`prune_assets`
#: so it can never delete anything else that ends up in the directory.
ASSET_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})

#: Separator inserted between two blocks when a page's text is reassembled.
BLOCK_SEP = "\n\n"
#: Separator inserted between two pages when the document text is reassembled.
PAGE_SEP = "\n\n"

# Record-level block types.  These are *normalized*: MinerU's content_list nests
# captions and footnotes inside their image/table item and reports headings as
# `type: "text"` carrying a `text_level`.  Both are flattened into real blocks here.
BLOCK_TEXT = "text"
BLOCK_TITLE = "title"
BLOCK_CAPTION = "caption"
BLOCK_FOOTNOTE = "footnote"
BLOCK_TABLE = "table"
BLOCK_IMAGE = "image"
BLOCK_EQUATION = "equation"
BLOCK_DISCARDED = "discarded"

#: Block types whose text is running prose and should feed chunking / embedding.
BODY_BLOCK_TYPES = frozenset({BLOCK_TEXT, BLOCK_TITLE, BLOCK_EQUATION})

DEGRADED_FALLBACK = "pypdfium2_fallback"
DEGRADED_FAILED = "extraction_failed"


# ---------------------------------------------------------------------------
# Heading normalization
# ---------------------------------------------------------------------------

def _heading_patterns():
    from .section_detector import SECTION_PATTERNS

    return [
        (re.compile(pattern, re.IGNORECASE), name, level)
        for pattern, name, level in SECTION_PATTERNS
    ]


_HEADING_PATTERNS = None

# Headings MinerU sometimes emits that are not real sections.
_NON_SECTION_HEADING = re.compile(
    r"^\s*(?:figure|fig\.?|table|scheme|chart|eq(?:uation)?)\s*\d", re.IGNORECASE
)


def normalize_heading(text: str) -> Optional[str]:
    """Map a heading string onto a canonical section name, or ``None``.

    Applied to a *heading block* rather than to raw lines, which is the whole point
    of R16: the string handed in is already known to be a heading, so the patterns
    only have to classify it, not find it.
    """
    global _HEADING_PATTERNS
    if _HEADING_PATTERNS is None:
        _HEADING_PATTERNS = _heading_patterns()

    cleaned = re.sub(r"[#*_`]", "", text or "").strip()
    cleaned = re.sub(r"^[IVXLC]+\.\s*", "", cleaned)  # roman numerals: "III. Results"
    if not cleaned or len(cleaned) > 160:
        return None
    if _NON_SECTION_HEADING.match(cleaned):
        return None

    # Elsevier sets its headings letter-spaced ("a b s t r a c t", "a r t i c l e
    # i n f o"), which no pattern in the list can match. Try a de-spaced variant
    # as well, so those papers get a real abstract section.
    candidates = [cleaned]

    # SECTION_PATTERNS hardcode the *expected* number for each section
    # ("^(?:3\\.?\\s*)?results?"), so a paper that numbers Results as 2 does not
    # match. Matching the heading with its own numbering stripped fixes that
    # without changing the pattern list the legacy line-scanner still uses.
    unnumbered = re.sub(r"^\d{1,2}(?:\.\d{1,2})*\.?[\s ]+", "", cleaned)
    if unnumbered != cleaned:
        candidates.append(unnumbered)

    despaced = re.sub(r"(?<=\b\w) (?=\w\b)", "", cleaned)
    if despaced != cleaned:
        candidates.append(despaced)

    for candidate in candidates:
        for pattern, name, level in _HEADING_PATTERNS:
            if level != 1:
                continue
            if pattern.match(candidate):
                return name
    return None


# ---------------------------------------------------------------------------
# Block assembly
# ---------------------------------------------------------------------------

def paper_id_for_filename(filename: str) -> str:
    """``md5(filename)[:12]`` -- the corpus-stable paper identifier.

    Must not change: it is the only identifier that survives a reindex, so W5's
    paper-level baseline for the post-reindex comparison is keyed on it.  The same
    derivation appears in ``index_papers.generate_paper_id`` and
    ``paper_library._generate_paper_id``; all three must agree.
    """
    return hashlib.md5(filename.encode()).hexdigest()[:12]


def _sanitize(text: Any) -> str:
    """Drop Unicode that cannot survive a JSON round-trip (lone surrogates)."""
    if not isinstance(text, str):
        return ""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def _image_ref(img_path: Any) -> Optional[str]:
    """Normalize MinerU's ``img_path`` into a record-relative asset reference.

    ``pdf_processor`` passes ``FIGURES_SUBDIR`` as MinerU's ``img_buket_path``,
    so ``content_list`` already carries ``"figures/<sha256>.jpg"`` -- relative to
    ``processed_data/{paper_id}/``, which is exactly what we want to store.

    An *absolute* path means the crop went somewhere outside the record's own
    directory (schema-1 behaviour: a temp dir that no longer exists).  Those are
    reduced to ``figures/<basename>`` so the reference has the schema-2 shape;
    :func:`asset_path` existence-checks it, so a schema-1 record reports "no
    image" instead of handing out a dangling path.
    """
    if not img_path:
        return None
    # MinerU writes flat, content-hashed filenames into the bucket directory, so
    # the basename is the whole identity and normalizing to one shape means
    # asset_path only ever has to resolve `figures/<name>`.
    name = str(img_path).strip().replace("\\", "/").rsplit("/", 1)[-1]
    if not name or name in (".", ".."):
        return None
    return f"{FIGURES_SUBDIR}/{name}"


def _clean_bbox(bbox: Any) -> Optional[List[int]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        return [int(v) for v in bbox]
    except (TypeError, ValueError):
        return None


def _flatten_content_list(content_list: List[Any],
                          cross_page_map: Optional[Dict] = None,
                          image_bbox_map: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """Turn MinerU's ``content_list`` into a flat list of record blocks.

    One content item can yield several blocks: a table item becomes its caption
    blocks, a table-body block and its footnote blocks, each individually
    addressable and each carrying the item's page and bbox.

    ``cross_page_map`` (built by ``pdf_processor._cross_page_map`` from
    ``middle_json``) marks paragraphs MinerU merged across a page break and says
    how many characters belong to the starting page. Those are split back into
    two blocks so that a page's text really is the text on that page -- without
    it, ~15% of long body paragraphs would resolve a citation in their tail to
    the previous page.
    """
    blocks: List[Dict[str, Any]] = []
    cross_page_map = cross_page_map or {}

    def emit(btype: str, text: str, page_idx: int, bbox, **extra) -> int:
        block = {
            "i": len(blocks),
            "seq": len(blocks),
            "type": btype,
            "text": _sanitize(text),
            "page_idx": int(page_idx),
            "bbox": _clean_bbox(bbox),
        }
        block.update({k: v for k, v in extra.items() if v not in (None, [], "")})
        blocks.append(block)
        return block["i"]

    for item in content_list or []:
        if not isinstance(item, dict):
            if isinstance(item, str) and item.strip():
                emit(BLOCK_TEXT, item, 0, None)
            continue

        itype = item.get("type", "")
        page_idx = item.get("page_idx", 0) or 0
        bbox = item.get("bbox")

        if itype == "text":
            # MinerU's pipeline backend reports headings as type "text" with a
            # `text_level` key -- there is no `type: "title"` in content_list.
            level = item.get("text_level")
            text = item.get("text", "") or ""
            if level:
                emit(BLOCK_TITLE, text, page_idx, bbox, level=int(level))
            elif text.strip():
                split_at = cross_page_map.get((page_idx, tuple(_clean_bbox(bbox) or ())))
                if split_at and 0 < split_at < len(text):
                    head = emit(BLOCK_TEXT, text[:split_at], page_idx, bbox)
                    emit(BLOCK_TEXT, text[split_at:], page_idx + 1, None,
                         continues_from=head, bbox_source="cross_page_split")
                else:
                    emit(BLOCK_TEXT, text, page_idx, bbox)

        elif itype == "discarded":
            text = item.get("text", "") or ""
            if text.strip():
                emit(BLOCK_DISCARDED, text, page_idx, bbox)

        elif itype == "equation":
            emit(
                BLOCK_EQUATION,
                item.get("text", "") or "",
                page_idx,
                bbox,
                text_format=item.get("text_format"),
                img_path=_image_ref(item.get("img_path")),
            )

        elif itype in ("image", "table"):
            is_table = itype == "table"
            kind = "table" if is_table else "figure"
            cap_key = "table_caption" if is_table else "image_caption"
            foot_key = "table_footnote" if is_table else "image_footnote"

            caption_ids: List[int] = []
            footnote_ids: List[int] = []

            # MinerU renders caption first for tables, body first for images; the
            # record keeps caption-first for both so the reassembled page text
            # reads the way a caption/body pair does on the page.
            for cap in item.get(cap_key, []) or []:
                if str(cap).strip():
                    caption_ids.append(
                        emit(BLOCK_CAPTION, cap, page_idx, bbox, kind=kind)
                    )

            body_text = item.get("table_body", "") if is_table else ""
            body_id = emit(
                BLOCK_TABLE if is_table else BLOCK_IMAGE,
                body_text,
                page_idx,
                bbox,
                img_path=_image_ref(item.get("img_path")),
                # MinerU's content_list bbox for an image/table item is the
                # *group* rectangle (picture + caption + footnote). The tighter
                # rectangle of the picture itself only exists in middle_json, so
                # pdf_processor passes it in; it is what the figure highlight and
                # the crop actually correspond to.
                image_bbox=_clean_bbox((image_bbox_map or {}).get(
                    (int(page_idx), tuple(_clean_bbox(bbox) or ())))),
            )

            for foot in item.get(foot_key, []) or []:
                if str(foot).strip():
                    footnote_ids.append(
                        emit(BLOCK_FOOTNOTE, foot, page_idx, bbox, kind=kind, parent=body_id)
                    )

            for cid in caption_ids:
                blocks[cid]["parent"] = body_id
            if caption_ids:
                blocks[body_id]["caption_blocks"] = caption_ids
            if footnote_ids:
                blocks[body_id]["footnote_blocks"] = footnote_ids

    return blocks


_BLOCK_REF_KEYS = ("parent", "continues_from")
_BLOCK_REF_LIST_KEYS = ("caption_blocks", "footnote_blocks")


def _reorder_by_page(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Put blocks back in document order after cross-page splits, and reindex.

    A split moves a paragraph's continuation onto the following page, so emission
    order is no longer page order. Everything downstream -- ``_assign_offsets``,
    section ranges, and ``locate``'s bisect -- assumes ``blocks[i]["doc_char_start"]``
    is non-decreasing in ``i``, so restore that here and remap the index
    cross-references in one pass.
    """
    order = sorted(blocks, key=lambda b: (b["page_idx"], b["seq"]))
    remap = {}
    for new_i, block in enumerate(order):
        remap[block["i"]] = new_i
    for new_i, block in enumerate(order):
        block["i"] = new_i
        block.pop("seq", None)
        for key in _BLOCK_REF_KEYS:
            if key in block:
                block[key] = remap[block[key]]
        for key in _BLOCK_REF_LIST_KEYS:
            if key in block:
                block[key] = [remap[v] for v in block[key]]
    return order


def _assign_offsets(blocks: List[Dict[str, Any]], page_sizes: Dict[int, Tuple[float, float]]):
    """Compute page/document character offsets and build the page index.

    Offsets are produced *while* the text is assembled, so they cannot disagree
    with it.  Returns ``(pages, page_texts)``.
    """
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for block in blocks:
        by_page.setdefault(block["page_idx"], []).append(block)

    max_page = max(by_page) if by_page else -1
    max_sized = max(page_sizes) if page_sizes else -1
    n_pages = max(max_page + 1, max_sized + 1)

    pages: List[Dict[str, Any]] = []
    page_texts: List[str] = []
    doc_cursor = 0

    for page_idx in range(n_pages):
        page_blocks = by_page.get(page_idx, [])
        parts: List[str] = []
        cursor = 0
        for block in page_blocks:
            text = block["text"]
            if not text:
                # Empty blocks (an image with no OCR'd text, an equation kept only
                # as a picture) still get a well-defined zero-length span.
                block["page_char_start"] = cursor
                block["page_char_end"] = cursor
                continue
            if parts:
                cursor += len(BLOCK_SEP)
            block["page_char_start"] = cursor
            cursor += len(text)
            block["page_char_end"] = cursor
            parts.append(text)

        page_text = BLOCK_SEP.join(parts)
        for block in page_blocks:
            block["doc_char_start"] = doc_cursor + block["page_char_start"]
            block["doc_char_end"] = doc_cursor + block["page_char_end"]

        width, height = page_sizes.get(page_idx, (0.0, 0.0))
        pages.append({
            "page_idx": page_idx,
            "width": round(float(width), 2),
            "height": round(float(height), 2),
            "char_start": doc_cursor,
            "char_len": len(page_text),
            "n_blocks": len(page_blocks),
            "md5": hashlib.md5(page_text.encode("utf-8", "replace")).hexdigest()[:8],
        })
        page_texts.append(page_text)
        doc_cursor += len(page_text) + len(PAGE_SEP)

    return pages, page_texts


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

_NUMBER_PREFIX = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2})*)\.?[\s ]+\S")


def _heading_depth(text: str) -> Optional[int]:
    """Depth implied by a heading's own numbering: "3.1.4 ..." -> 3."""
    match = _NUMBER_PREFIX.match(text or "")
    if not match:
        return None
    return len(match.group(1).split("."))


def _build_sections(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Derive sections from heading blocks (R16).

    A section runs from its heading block to the next heading block of any level.
    Level tracking gives subsections a parent, matching the shape
    ``section_detector.Section`` already produced, but without the regex having to
    find the boundaries.
    """
    heading_idx = [b["i"] for b in blocks if b["type"] == BLOCK_TITLE]
    sections: List[Dict[str, Any]] = []

    def add(name, normalized, level, title_block, block_start, block_end,
            parent_id, subsection_name):
        body = [b for b in blocks[block_start:block_end] if b["type"] in BODY_BLOCK_TYPES]
        span_blocks = blocks[block_start:block_end]
        if title_block is not None:
            span_blocks = [blocks[title_block]] + span_blocks
        if not span_blocks:
            return
        sections.append({
            "id": len(sections),
            "name": name,
            "normalized_name": normalized,
            "level": level,
            "title_block": title_block,
            "block_start": block_start,
            "block_end": block_end,
            "parent_id": parent_id,
            "subsection_name": subsection_name,
            "doc_char_start": span_blocks[0]["doc_char_start"],
            "doc_char_end": span_blocks[-1]["doc_char_end"],
            "page_start": span_blocks[0]["page_idx"],
            "page_end": span_blocks[-1]["page_idx"],
            "n_body_blocks": len(body),
            "char_len": sum(len(b["text"]) for b in body),
        })

    if not heading_idx:
        if blocks:
            add("Document", "body", 1, None, 0, len(blocks), None, None)
        return sections

    # Anything before the first heading is front matter (title, authors, often the
    # abstract) and would otherwise be unreachable.
    if heading_idx[0] > 0:
        add("Front matter", "frontmatter", 1, None, 0, heading_idx[0], None, None)

    # MinerU's pipeline backend almost always emits every heading at text_level 1,
    # so a pure level stack would make every section a root and lose the
    # "Overview of Immunoassays belongs to Results" relationship.  Track the last
    # *recognized* section separately and let unrecognized headings inherit it.
    stack: List[Tuple[int, int, str]] = []       # (level, section_id, normalized)
    last_known: Optional[Tuple[int, str]] = None  # (section_id, normalized)

    for n, title_i in enumerate(heading_idx):
        title_block = blocks[title_i]
        heading_text = title_block["text"].strip()
        # MinerU's `text_level` is 1 for essentially every heading, so a paper's
        # own section numbering ("3.1.4 Pancreas peptides") is the only real depth
        # signal available. Prefer it where present.
        depth = _heading_depth(heading_text)
        level = depth if depth is not None else int(title_block.get("level", 1) or 1)
        normalized = normalize_heading(heading_text)

        while stack and stack[-1][0] >= level:
            stack.pop()

        parent_id = stack[-1][1] if stack else None
        subsection_name = None
        if normalized is None:
            if stack:
                normalized, parent_id = stack[-1][2], stack[-1][1]
                subsection_name = heading_text
            elif depth == 1:
                # A numbered top-level heading is its own section, not a
                # continuation of whatever was last recognised.
                normalized, parent_id = "body", None
            elif last_known is not None:
                parent_id, normalized = last_known
                subsection_name = heading_text
            else:
                normalized = "body"
        elif level > 1 and stack:
            subsection_name = heading_text

        block_start = title_i + 1
        block_end = heading_idx[n + 1] if n + 1 < len(heading_idx) else len(blocks)
        section_id = len(sections)
        add(heading_text or "Untitled", normalized, level, title_i,
            block_start, block_end, parent_id, subsection_name)
        if len(sections) > section_id:
            stack.append((level, section_id, normalized))
            if subsection_name is None and normalized not in ("body", "frontmatter"):
                last_known = (section_id, normalized)

    return sections


def _tag_blocks_with_sections(blocks, sections):
    for section in sections:
        if section["title_block"] is not None:
            blocks[section["title_block"]]["section_id"] = section["id"]
        for block in blocks[section["block_start"]:section["block_end"]]:
            block["section_id"] = section["id"]


# ---------------------------------------------------------------------------
# Abstract
# ---------------------------------------------------------------------------

_ABSTRACT_MARKER = re.compile(r"^\W{0,4}abstract\b[\s:.\-–—]*", re.IGNORECASE)
_ABSTRACT_INLINE = re.compile(r"\babstract\b[\s:.\-–—]+", re.IGNORECASE)
_AFFILIATION_HINT = re.compile(
    r"(university|department|institute|laborator|@|e-?mail|correspond|"
    r"received|accepted|revised|\bdoi\b|copyright|all rights reserved|issn)",
    re.IGNORECASE,
)
_KEYWORDS_TAIL = re.compile(r"\s*(?:key\s*words?|keywords?)\s*[:.\-].*$", re.IGNORECASE | re.DOTALL)

# Elsevier prints a left-hand column of article history and keywords *inside* the
# abstract band. Joining it in and then trimming from "Keywords:" to the end
# deletes the abstract itself, so drop these blocks before joining.
_FRONT_MATTER_BLOCK = re.compile(
    r"^\s*(?:article\s+history|key\s*words?|keywords?|received|accepted|available\s+online"
    r"|contents\s+lists|journal\s+homepage|©|\(c\)\s*\d{4})",
    re.IGNORECASE,
)

ABSTRACT_MIN = 150
ABSTRACT_MAX = 6000


def _looks_like_prose(text: str) -> bool:
    if len(text) < ABSTRACT_MIN:
        return False
    letters = sum(c.isalpha() for c in text)
    if letters < len(text) * 0.55:
        return False
    words = text.split()
    if len(words) < 30:
        return False
    return text.count(".") >= 2


def _clean_abstract(text: str) -> str:
    text = _ABSTRACT_MARKER.sub("", text.strip())
    text = _KEYWORDS_TAIL.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _extract_abstract(blocks: List[Dict[str, Any]], sections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Take the abstract from the leading block sequence (R16).

    Four routes, tried in order of how much structure they rely on.  The
    ``extract_abstract`` pattern list this replaces only ever found route 2, and
    only when the word "Abstract" survived extraction on its own line.
    """

    def build(source, block_ids, text):
        text = _clean_abstract(text)
        if not (ABSTRACT_MIN <= len(text) <= ABSTRACT_MAX):
            return None
        first, last = blocks[block_ids[0]], blocks[block_ids[-1]]
        start, end = first["doc_char_start"], last["doc_char_end"]
        # Where the cleaned abstract is still a contiguous run of the source text,
        # narrow the offsets onto it so W3 can cite the abstract by offset. When it
        # is stitched from several blocks it cannot be, and says so.
        verbatim = False
        if len(block_ids) == 1:
            found = first["text"].find(text)
            if found >= 0:
                start = first["doc_char_start"] + found
                end = start + len(text)
                verbatim = True
        return {
            "text": text,
            "source": source,
            "block_start": block_ids[0],
            "block_end": block_ids[-1] + 1,
            "page_idx": first["page_idx"],
            "doc_char_start": start,
            "doc_char_end": end,
            "verbatim_span": verbatim,
        }

    # Route 1: an explicit "Abstract" heading block -- take the body under it.
    for section in sections:
        if section["normalized_name"] != "abstract":
            continue
        ids = [b["i"] for b in blocks[section["block_start"]:section["block_end"]]
               if b["type"] == BLOCK_TEXT and b["text"].strip()]
        keep = [i for i in ids if not _FRONT_MATTER_BLOCK.match(blocks[i]["text"])]
        for candidate_ids in ([keep, ids] if keep and keep != ids else [ids]):
            if not candidate_ids:
                continue
            joined = " ".join(blocks[i]["text"] for i in candidate_ids)
            built = build("heading", candidate_ids, joined)
            if built:
                return built

    # Leading region: everything before the first heading that starts real body
    # text, capped so a paper with no headings does not swallow the whole PDF.
    stop_names = {"introduction", "methods", "results", "results_discussion",
                  "discussion", "background", "synthesis", "conclusion"}
    limit = len(blocks)
    for section in sections:
        if section["normalized_name"] in stop_names and section["title_block"] is not None:
            limit = section["title_block"]
            break
    leading = [b for b in blocks[:min(limit, 60)]
               if b["type"] == BLOCK_TEXT and b["page_idx"] <= 1 and b["text"].strip()]

    # Route 2: a block that begins with the word "Abstract".
    for block in leading:
        if _ABSTRACT_MARKER.match(block["text"]):
            body = _ABSTRACT_MARKER.sub("", block["text"].strip())
            if _looks_like_prose(body):
                built = build("inline_marker", [block["i"]], body)
                if built:
                    return built
            # A bare "Abstract" heading mis-typed as text: take what follows.
            if len(body) < 40:
                following = [b for b in leading if b["i"] > block["i"]][:3]
                acc: List[int] = []
                text = ""
                for nxt in following:
                    acc.append(nxt["i"])
                    text = (text + " " + nxt["text"]).strip()
                    if _looks_like_prose(text):
                        built = build("marker_next_block", acc, text)
                        if built:
                            return built

    # Route 3: "Abstract" appearing a short way into a merged header block.
    for block in leading:
        match = _ABSTRACT_INLINE.search(block["text"][:400])
        if match:
            body = block["text"][match.end():]
            if _looks_like_prose(body):
                built = build("inline_split", [block["i"]], body)
                if built:
                    return built

    # Route 4: no marker anywhere -- the longest prose block in the leading
    # sequence that is not an affiliation/copyright slab.
    best = None
    for block in leading:
        text = block["text"].strip()
        if not _looks_like_prose(text) or len(text) > ABSTRACT_MAX:
            continue
        head = text[:300]
        if _AFFILIATION_HINT.search(head):
            continue
        if best is None or len(text) > len(best["text"]):
            best = block
    if best is not None:
        return build("leading_block", [best["i"]], best["text"])

    return None


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------

def _count_table_shape(html: str) -> Tuple[int, int]:
    if not html:
        return 0, 0
    rows = len(re.findall(r"<tr[\s>]", html, re.IGNORECASE))
    first_row = re.search(r"<tr[\s>].*?</tr>", html, re.IGNORECASE | re.DOTALL)
    cols = 0
    if first_row:
        cols = len(re.findall(r"<t[dh][\s>]", first_row.group(0), re.IGNORECASE))
    return rows, cols


def _file_facts(pdf_path: Optional[Path]) -> Dict[str, Any]:
    facts: Dict[str, Any] = {}
    try:
        if pdf_path is not None:
            stat = os.stat(pdf_path)
            facts["file_size"] = stat.st_size
            facts["file_mtime"] = int(stat.st_mtime)
    except OSError:
        pass
    return facts


def _base_record(paper_id: str, file_name: str, extractor: str) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "paper_id": paper_id,
        "file_name": file_name,
        "extractor": extractor,
        "extracted_at": int(time.time()),
        "degraded": False,
        "degraded_reason": None,
        "bbox_units": "per_mille_of_page",
        "bbox_origin": "top_left",
        "block_sep": BLOCK_SEP,
        "page_sep": PAGE_SEP,
        "capabilities": {
            "tables": True, "captions": True, "sections": True,
            "bbox": True, "page_offsets": True, "figure_images": False,
        },
        "metadata": {},
        "pages": [],
        "blocks": [],
        "sections": [],
        "abstract": None,
        "tables": [],
        "figures": [],
        "equations": [],
        "caption_blocks": [],
        "assets": {"dir": f"{paper_id}/{FIGURES_SUBDIR}", "n_referenced": 0,
                   "n_present": 0, "bytes": 0, "referenced_by_block_type": {}},
        "stats": {},
    }


def build_record(
    content_list: List[Any],
    pdf_info: List[Dict[str, Any]],
    paper_id: str,
    file_name: str,
    pdf_path: Optional[Path] = None,
    extractor_version: str = "",
    cross_page_map: Optional[Dict] = None,
    image_bbox_map: Optional[Dict] = None,
    assets_base: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build a full paper record from MinerU output.

    Runs **inside the MinerU subprocess** (see ``pdf_processor._subprocess_mineru_extract``)
    so that ``content_list`` and ``pdf_info`` never have to cross the process
    boundary and the pdfium isolation is preserved.
    """
    record = _base_record(paper_id, file_name, "mineru")
    record["extractor_version"] = extractor_version
    record.update(_file_facts(pdf_path))

    page_sizes: Dict[int, Tuple[float, float]] = {}
    for page in pdf_info or []:
        idx = page.get("page_idx")
        size = page.get("page_size") or []
        if idx is not None and len(size) == 2:
            page_sizes[int(idx)] = (float(size[0]), float(size[1]))

    # Which keys MinerU actually emitted, per item type. Kept as a diagnostic so a
    # future MinerU upgrade that renames e.g. `table_body` is visible in the data
    # rather than silently producing empty tables the way R1 did for the whole corpus.
    observed_keys: Dict[str, set] = {}
    for item in content_list or []:
        if isinstance(item, dict):
            observed_keys.setdefault(item.get("type", "?"), set()).update(item.keys())

    blocks = _reorder_by_page(
        _flatten_content_list(content_list, cross_page_map, image_bbox_map))
    pages, _ = _assign_offsets(blocks, page_sizes)
    sections = _build_sections(blocks)
    _tag_blocks_with_sections(blocks, sections)
    abstract = _extract_abstract(blocks, sections)

    tables = []
    figures = []
    equations = []
    caption_blocks = []

    for block in blocks:
        if block["type"] == BLOCK_TABLE:
            rows, cols = _count_table_shape(block["text"])
            tables.append({
                # `figure_id` is the public handle: it is what a chunk payload
                # carries, what `/papers/{id}/figures/{figure_id}` resolves and
                # what the reference resolver points at. `{kind}_{ordinal}`,
                # ordinal being the index in this list.
                "figure_id": f"table_{len(tables)}",
                "kind": "table",
                "index": len(tables),
                "block": block["i"],
                "page_idx": block["page_idx"],
                "bbox": block["bbox"],
                "image_bbox": block.get("image_bbox"),
                "caption_blocks": block.get("caption_blocks", []),
                "footnote_blocks": block.get("footnote_blocks", []),
                "section_id": block.get("section_id"),
                "has_body": bool(block["text"]),
                "body_chars": len(block["text"]),
                "n_rows": rows,
                "n_cols": cols,
                "img_path": block.get("img_path"),
            })
        elif block["type"] == BLOCK_IMAGE:
            figures.append({
                "figure_id": f"figure_{len(figures)}",
                "kind": "figure",
                "index": len(figures),
                "block": block["i"],
                "page_idx": block["page_idx"],
                "bbox": block["bbox"],
                "image_bbox": block.get("image_bbox"),
                "caption_blocks": block.get("caption_blocks", []),
                "footnote_blocks": block.get("footnote_blocks", []),
                "section_id": block.get("section_id"),
                "img_path": block.get("img_path"),
            })
        elif block["type"] == BLOCK_EQUATION:
            equations.append({
                "block": block["i"],
                "page_idx": block["page_idx"],
                "bbox": block["bbox"],
                "has_latex": bool(block["text"]),
                "section_id": block.get("section_id"),
            })
        elif block["type"] == BLOCK_CAPTION:
            caption_blocks.append(block["i"])

    record["pages"] = pages
    record["blocks"] = blocks
    record["sections"] = sections
    record["abstract"] = abstract
    record["tables"] = tables
    record["figures"] = figures
    record["equations"] = equations
    record["caption_blocks"] = caption_blocks
    record["metadata"] = {
        "title": _guess_title(blocks, pdf_info, file_name),
        "num_pages": len(pages),
        "file_name": file_name,
    }
    label_objects(record)
    record["assets"] = asset_summary(record, assets_base)
    record["capabilities"]["figure_images"] = record["assets"]["n_present"] > 0
    record["stats"] = _compute_stats(record)
    record["stats"]["mineru_item_keys"] = {k: sorted(v) for k, v in sorted(observed_keys.items())}
    return record


def _guess_title(blocks, pdf_info, file_name: str) -> str:
    """First level-1 heading on page 0, else the first heading, else the stem."""
    for block in blocks:
        if block["type"] == BLOCK_TITLE and block["page_idx"] == 0:
            text = block["text"].strip()
            if 8 <= len(text) <= 300:
                return text
    for page in pdf_info or []:
        for block in page.get("preproc_blocks", []) or []:
            if block.get("type") == "title" and block.get("text"):
                return str(block["text"]).strip()
        break
    return Path(file_name).stem


def _compute_stats(record: Dict[str, Any]) -> Dict[str, Any]:
    blocks = record["blocks"]
    by_type: Dict[str, int] = {}
    for block in blocks:
        by_type[block["type"]] = by_type.get(block["type"], 0) + 1
    body_chars = sum(len(b["text"]) for b in blocks if b["type"] in BODY_BLOCK_TYPES)
    return {
        "n_pages": len(record["pages"]),
        "n_blocks": len(blocks),
        "blocks_by_type": by_type,
        "n_sections": len(record["sections"]),
        "n_tables": len(record["tables"]),
        "n_tables_with_body": sum(1 for t in record["tables"] if t["has_body"]),
        "n_figures": len(record["figures"]),
        "n_figure_images": sum(1 for f in record["figures"] if f.get("img_path")),
        "n_table_images": sum(1 for t in record["tables"] if t.get("img_path")),
        "n_labeled_objects": sum(1 for e in (record["figures"] + record["tables"])
                                 if e.get("label")),
        "n_equations": len(record["equations"]),
        "n_captions": len(record["caption_blocks"]),
        "has_abstract": record["abstract"] is not None,
        "abstract_source": (record["abstract"] or {}).get("source"),
        "body_chars": body_chars,
        "total_chars": sum(len(b["text"]) for b in blocks),
    }


def build_degraded_record(
    page_texts: List[str],
    paper_id: str,
    file_name: str,
    reason: str = DEGRADED_FALLBACK,
    pdf_path: Optional[Path] = None,
    page_sizes: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """Build a record for the pypdfium2 fallback path (~3.1% of the corpus).

    pypdfium2 returns interleaved two-column text with no layout, so there are no
    captions, no tables and no headings to key sections off.  The record is marked
    ``degraded`` and its ``capabilities`` flags say exactly which claims it cannot
    support, rather than presenting empty lists as "this paper has no tables".
    """
    record = _base_record(paper_id, file_name, "pypdfium2")
    record["degraded"] = True
    record["degraded_reason"] = reason
    record["capabilities"] = {
        "tables": False, "captions": False, "sections": False,
        "bbox": False, "page_offsets": True, "figure_images": False,
    }
    record.update(_file_facts(pdf_path))

    blocks: List[Dict[str, Any]] = []
    for page_idx, text in enumerate(page_texts):
        text = _sanitize(text)
        if not text.strip():
            continue
        blocks.append({
            "i": len(blocks),
            "type": BLOCK_TEXT,
            "text": text,
            "page_idx": page_idx,
            "bbox": None,
        })

    sizes: Dict[int, Tuple[float, float]] = {}
    for idx, size in enumerate(page_sizes or []):
        sizes[idx] = (float(size[0]), float(size[1]))
    if not sizes:
        sizes = {i: (0.0, 0.0) for i in range(len(page_texts))}

    pages, _ = _assign_offsets(blocks, sizes)
    record["pages"] = pages
    record["blocks"] = blocks
    record["sections"] = []
    record["abstract"] = _fallback_abstract(blocks)
    record["metadata"] = {
        "title": Path(file_name).stem,
        "num_pages": len(page_texts),
        "file_name": file_name,
    }
    record["stats"] = _compute_stats(record)
    record["stats"]["degraded"] = True
    return record


def _fallback_abstract(blocks) -> Optional[Dict[str, Any]]:
    """Best effort abstract from unstructured page text, flagged as such."""
    if not blocks:
        return None
    head = blocks[0]
    match = _ABSTRACT_INLINE.search(head["text"][:3000])
    if not match:
        return None
    tail = head["text"][match.end():]
    stop = re.search(r"\n\s*(?:\d+\.?\s*)?(?:introduction|keywords?)\b", tail, re.IGNORECASE)
    if stop:
        tail = tail[:stop.start()]
    text = _clean_abstract(tail)
    if not (ABSTRACT_MIN <= len(text) <= ABSTRACT_MAX):
        return None
    start = head["doc_char_start"] + match.end()
    return {
        "text": text,
        "source": "fallback_regex",
        "block_start": head["i"],
        "block_end": head["i"] + 1,
        "page_idx": head["page_idx"],
        "doc_char_start": start,
        "doc_char_end": start + len(tail),
        "degraded": True,
    }


def build_failed_record(paper_id: str, file_name: str,
                        reason: str = DEGRADED_FAILED,
                        pdf_path: Optional[Path] = None) -> Dict[str, Any]:
    """A record for a paper no extractor could read. Never looks complete."""
    record = _base_record(paper_id, file_name, "none")
    record["degraded"] = True
    record["degraded_reason"] = reason
    record["capabilities"] = {k: False for k in record["capabilities"]}
    record.update(_file_facts(pdf_path))
    record["metadata"] = {"title": Path(file_name).stem, "num_pages": 0, "file_name": file_name}
    record["stats"] = _compute_stats(record)
    record["stats"]["degraded"] = True
    return record


# ---------------------------------------------------------------------------
# Reassembly and verification
# ---------------------------------------------------------------------------

def rebuild_derived(record: Dict[str, Any], base: Optional[Path] = None) -> Dict[str, Any]:
    """Recompute sections, abstract and stats from the stored blocks, in place.

    Blocks, offsets and page geometry are untouched -- only the derived layers are
    rebuilt. This is what makes a change to heading classification or abstract
    detection a milliseconds-per-paper operation instead of a 60-80 h
    re-extraction, and it is the same capability W2 needs for re-chunking.
    """
    blocks = record["blocks"]
    for block in blocks:
        block.pop("section_id", None)
    sections = _build_sections(blocks)
    _tag_blocks_with_sections(blocks, sections)
    record["sections"] = sections
    record["abstract"] = (None if record.get("degraded") and not sections
                          else _extract_abstract(blocks, sections))
    if record["abstract"] is None and record.get("degraded"):
        record["abstract"] = _fallback_abstract(blocks)
    for group in ("tables", "figures"):
        for entry in record.get(group, []):
            entry["section_id"] = blocks[entry["block"]].get("section_id")
    for entry in record.get("equations", []):
        entry["section_id"] = blocks[entry["block"]].get("section_id")
    label_objects(record)
    record["assets"] = asset_summary(record, base)
    record.setdefault("capabilities", {})["figure_images"] = record["assets"]["n_present"] > 0
    keys = record["stats"].get("mineru_item_keys")
    record["stats"] = _compute_stats(record)
    if keys is not None:
        record["stats"]["mineru_item_keys"] = keys
    if record.get("degraded"):
        record["stats"]["degraded"] = True
    return record


def page_text(record: Dict[str, Any], page_idx: int) -> str:
    """Reassemble one page's text -- the string the page offsets index into."""
    parts = [b["text"] for b in record["blocks"]
             if b["page_idx"] == page_idx and b["text"]]
    return BLOCK_SEP.join(parts)


def all_page_texts(record: Dict[str, Any]) -> List[str]:
    by_page: Dict[int, List[str]] = {}
    for block in record["blocks"]:
        if block["text"]:
            by_page.setdefault(block["page_idx"], []).append(block["text"])
    n_pages = max(len(record["pages"]), (max(by_page) + 1) if by_page else 0)
    return [BLOCK_SEP.join(by_page.get(i, [])) for i in range(n_pages)]


def full_text(record: Dict[str, Any]) -> str:
    """Reassemble the document text -- the string the doc offsets index into."""
    return PAGE_SEP.join(all_page_texts(record))


def body_text(record: Dict[str, Any]) -> str:
    """Running prose only: no table HTML, no captions, no page furniture.

    This is what the existing chunker consumes in place of the old ``full_text``.
    Note its offsets are *not* the record's offsets -- use ``full_text`` for those.
    """
    parts = []
    for block in record["blocks"]:
        if block["type"] not in BODY_BLOCK_TYPES or not block["text"]:
            continue
        if block["type"] == BLOCK_TITLE:
            parts.append("\n\n" + block["text"] + "\n")
        else:
            parts.append(block["text"])
    return "\n".join(parts)


def section_text(record: Dict[str, Any], section: Dict[str, Any]) -> str:
    blocks = record["blocks"]
    parts = []
    if section["title_block"] is not None:
        parts.append(blocks[section["title_block"]]["text"])
    for block in blocks[section["block_start"]:section["block_end"]]:
        if block["type"] in BODY_BLOCK_TYPES and block["text"]:
            parts.append(block["text"])
    return "\n\n".join(parts)


def captions(record: Dict[str, Any], kind: Optional[str] = None) -> List[str]:
    """Real captions, from MinerU's caption lists -- never body prose (R2)."""
    out = []
    for i in record.get("caption_blocks", []):
        block = record["blocks"][i]
        if kind and block.get("kind") != kind:
            continue
        out.append(block["text"])
    return out


def table_texts(record: Dict[str, Any], with_caption: bool = True) -> List[str]:
    """Table bodies (HTML) with caption and footnote attached (R1)."""
    blocks = record["blocks"]
    out = []
    for table in record.get("tables", []):
        body = blocks[table["block"]]["text"]
        if not body:
            continue
        parts = []
        if with_caption:
            parts += [blocks[i]["text"] for i in table.get("caption_blocks", [])]
        parts.append(body)
        if with_caption:
            parts += [blocks[i]["text"] for i in table.get("footnote_blocks", [])]
        out.append("\n".join(p for p in parts if p))
    return out


# ---------------------------------------------------------------------------
# Figure and table objects, and their persisted crops
# ---------------------------------------------------------------------------

def assets_dir(paper_id: str, base: Optional[Path] = None) -> Path:
    """``processed_data/{paper_id}/`` -- the paper's own asset directory.

    Sits alongside ``processed_data/{paper_id}.json``, so one ``base`` locates
    both and a sample harness pointing ``base`` at a scratch directory gets its
    crops there too instead of writing into the production library.
    """
    return records_dir(base) / paper_id


def figures_dir(paper_id: str, base: Optional[Path] = None) -> Path:
    return assets_dir(paper_id, base) / FIGURES_SUBDIR


def asset_path(paper_id: str, ref: Optional[str],
               base: Optional[Path] = None) -> Optional[Path]:
    """Resolve a stored ``img_path`` to a file that **exists**, or ``None``.

    Returning ``None`` rather than a Path is deliberate: schema-1 records store a
    temp-directory path that has been deleted, and a crop can also be missing
    because MinerU declined to write one.  Callers that must render an image
    need to branch on that anyway, so make it impossible to skip.
    """
    if not ref:
        return None
    root = assets_dir(paper_id, base).resolve()
    candidate = (root / str(ref)).resolve()
    # Containment check: `ref` comes out of a JSON file, so treat it as input.
    if root != candidate and root not in candidate.parents:
        logger.warning(f"Rejected out-of-tree asset ref for {paper_id}: {ref!r}")
        return None
    return candidate if candidate.is_file() else None


def figure_image(record: Dict[str, Any], entry: Dict[str, Any],
                 base: Optional[Path] = None) -> Optional[Path]:
    """The crop for one ``figures[]`` / ``tables[]`` entry, if it is on disk."""
    return asset_path(record.get("paper_id", ""), entry.get("img_path"), base)


def asset_refs(record: Dict[str, Any]) -> List[str]:
    """Every asset reference the record makes, in block order.

    Includes equation crops: an equation MinerU could not read as LaTeX exists
    *only* as a picture, so dropping those would lose the content outright.
    """
    refs: List[str] = []
    seen = set()
    for block in record.get("blocks") or []:
        ref = block.get("img_path")
        if ref and ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


def asset_summary(record: Dict[str, Any], base: Optional[Path] = None) -> Dict[str, Any]:
    """What the record references vs what is actually on disk."""
    paper_id = record.get("paper_id", "")
    refs = asset_refs(record)
    present = 0
    total_bytes = 0
    for ref in refs:
        path = asset_path(paper_id, ref, base)
        if path is not None:
            present += 1
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass
    by_type: Dict[str, int] = {}
    for block in record.get("blocks") or []:
        if block.get("img_path"):
            by_type[block["type"]] = by_type.get(block["type"], 0) + 1
    return {
        "dir": f"{paper_id}/{FIGURES_SUBDIR}",
        "n_referenced": len(refs),
        "n_present": present,
        "bytes": total_bytes,
        "referenced_by_block_type": by_type,
    }


def prune_assets(record: Dict[str, Any], base: Optional[Path] = None) -> Dict[str, Any]:
    """Delete crops in the paper's figures directory the record does not use.

    Filenames are MinerU content hashes, so a re-extraction of an unchanged
    paper overwrites the same files and nothing is orphaned.  A re-extraction
    that *changes* the crops -- a MinerU upgrade, a re-rasterized page, a
    replaced PDF under the same name -- would otherwise leave the old set behind
    forever.  Run this immediately after ``build_record``, in the same process
    that wrote them.
    """
    paper_id = record.get("paper_id", "")
    directory = figures_dir(paper_id, base)
    keep = {Path(ref).name for ref in asset_refs(record)}
    removed = 0
    freed = 0
    if directory.is_dir():
        for child in directory.iterdir():
            if not child.is_file() or child.name in keep:
                continue
            if child.suffix.lower() not in ASSET_SUFFIXES:
                continue                       # never our file; leave it alone
            try:
                freed += child.stat().st_size
                child.unlink()
                removed += 1
            except OSError as exc:
                logger.warning(f"Could not prune {child}: {exc}")
    return {"removed": removed, "bytes_freed": freed, "kept": len(keep)}


def delete_assets(paper_id: str, base: Optional[Path] = None) -> int:
    """Remove a paper's whole asset directory. Returns files deleted."""
    directory = assets_dir(paper_id, base)
    deleted = 0
    if not directory.is_dir():
        return 0
    for child in sorted(directory.rglob("*"), reverse=True):
        try:
            if child.is_file():
                child.unlink()
                deleted += 1
            elif child.is_dir():
                child.rmdir()
        except OSError as exc:
            logger.warning(f"Could not delete {child}: {exc}")
    try:
        directory.rmdir()
    except OSError:
        pass
    return deleted


def label_objects(record: Dict[str, Any]) -> Dict[str, Any]:
    """Attach a parsed label (``Figure 3``, ``Table II``) to each object.

    Delegates to :mod:`figure_refs`, imported lazily because that module reads
    this one.  Stored on the record so the API, the chunker and the resolver all
    agree on one labelling rather than each re-deriving it.
    """
    from . import figure_refs

    return figure_refs.label_record_objects(record)


def figure_objects(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Figures and tables as one addressable list -- the view the API serves.

    Reading order within the document, each entry carrying its ``figure_id``,
    page, the rectangle to highlight, its caption text and whether a crop is on
    disk.  Schema-1 records work: they simply report ``has_image: False``.
    """
    blocks = record.get("blocks") or []
    out: List[Dict[str, Any]] = []
    for group in ("figures", "tables"):
        for n, entry in enumerate(record.get(group) or []):
            kind = entry.get("kind") or ("table" if group == "tables" else "figure")
            figure_id = entry.get("figure_id") or f"{kind}_{n}"
            caption = " ".join(
                blocks[i]["text"] for i in entry.get("caption_blocks", [])
                if 0 <= i < len(blocks)
            ).strip()
            footnote = " ".join(
                blocks[i]["text"] for i in entry.get("footnote_blocks", [])
                if 0 <= i < len(blocks)
            ).strip()
            out.append({
                "figure_id": figure_id,
                "kind": kind,
                "index": entry.get("index", n),
                "block": entry.get("block"),
                "page_idx": entry.get("page_idx"),
                "page": (entry.get("page_idx") + 1
                         if entry.get("page_idx") is not None else None),
                # The rectangle to highlight: the picture itself where
                # middle_json gave it to us, else MinerU's figure-group box
                # (picture + caption), which still contains the picture.
                "bbox": entry.get("image_bbox") or entry.get("bbox"),
                "group_bbox": entry.get("bbox"),
                "bbox_is_image": bool(entry.get("image_bbox")),
                "label": entry.get("label"),
                "label_kind": entry.get("label_kind"),
                "label_number": entry.get("label_number"),
                "label_source": entry.get("label_source"),
                "caption": caption,
                "footnote": footnote,
                "caption_blocks": list(entry.get("caption_blocks", [])),
                "section_id": entry.get("section_id"),
                "img_path": entry.get("img_path"),
                "has_body": entry.get("has_body"),
                "n_rows": entry.get("n_rows"),
                "n_cols": entry.get("n_cols"),
            })
    out.sort(key=lambda e: (e["page_idx"] if e["page_idx"] is not None else 0,
                            e["block"] if e["block"] is not None else 0))
    return out


def figure_object(record: Dict[str, Any], figure_id: str) -> Optional[Dict[str, Any]]:
    """One ``figures[]``/``tables[]`` entry by ``figure_id``, or ``None``."""
    kind, _, index = str(figure_id).rpartition("_")
    group = {"figure": "figures", "table": "tables"}.get(kind)
    if group is None or not index.isdigit():
        return None
    entries = record.get(group) or []
    n = int(index)
    if 0 <= n < len(entries):
        return entries[n]
    return None


def locate(record: Dict[str, Any], doc_offset: int) -> Optional[Dict[str, Any]]:
    """Map a document character offset to its block -- W3's highlight lookup.

    Returns ``{"block", "type", "page_idx", "bbox", "page_char_start", ...}`` or
    ``None`` if the offset falls in a separator between blocks.
    """
    blocks = record["blocks"]
    starts = record.get("_starts")
    if starts is None:
        starts = [b.get("doc_char_start", 0) for b in blocks]
    i = bisect.bisect_right(starts, doc_offset) - 1
    best = None
    # Zero-length blocks share a start with their neighbour, so walk back over ties.
    while i >= 0:
        block = blocks[i]
        if block.get("doc_char_start") is None:
            break
        if block["doc_char_start"] <= doc_offset < block["doc_char_end"]:
            best = block
            break
        if block["doc_char_start"] < doc_offset:
            break
        i -= 1
    if best is None:
        return None
    return {
        "block": best["i"],
        "type": best["type"],
        "page_idx": best["page_idx"],
        "bbox": best["bbox"],
        "page_char_start": best["page_char_start"],
        "page_char_offset": doc_offset - best["doc_char_start"] + best["page_char_start"],
        "section_id": best.get("section_id"),
    }


def verify_record_offsets(record: Dict[str, Any]) -> Dict[str, Any]:
    """Check the two offset identities the highlight feature depends on.

    Returns a report; ``ok`` is True only if every block round-trips byte-exactly
    through both the page text and the document text, and every stored page md5
    matches the reassembled page.
    """
    page_texts = all_page_texts(record)
    doc = PAGE_SEP.join(page_texts)
    errors: List[str] = []
    checked = 0

    for page in record["pages"]:
        idx = page["page_idx"]
        text = page_texts[idx] if idx < len(page_texts) else ""
        if len(text) != page["char_len"]:
            errors.append(f"page {idx}: char_len {page['char_len']} != {len(text)}")
        digest = hashlib.md5(text.encode("utf-8", "replace")).hexdigest()[:8]
        if digest != page["md5"]:
            errors.append(f"page {idx}: md5 {page['md5']} != {digest}")
        if doc[page["char_start"]:page["char_start"] + len(text)] != text:
            errors.append(f"page {idx}: doc char_start {page['char_start']} misaligned")

    for block in record["blocks"]:
        if "page_char_start" not in block:
            errors.append(f"block {block['i']}: missing offsets")
            continue
        idx = block["page_idx"]
        text = page_texts[idx] if idx < len(page_texts) else ""
        if text[block["page_char_start"]:block["page_char_end"]] != block["text"]:
            errors.append(f"block {block['i']}: page slice mismatch")
        if doc[block["doc_char_start"]:block["doc_char_end"]] != block["text"]:
            errors.append(f"block {block['i']}: doc slice mismatch")
        checked += 1

    return {
        "ok": not errors,
        "blocks_checked": checked,
        "pages_checked": len(record["pages"]),
        "errors": errors[:20],
        "n_errors": len(errors),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def records_dir(base: Optional[Path] = None) -> Path:
    """Resolve ``processed_data/``.

    ``settings.processed_data_dir`` is relative, so it would otherwise land in a
    different place depending on whether the caller is uvicorn (cwd ``backend/``)
    or a script run from the repo root.  Anchor it on the backend package.
    """
    if base is not None:
        return Path(base)
    try:
        from config import settings  # type: ignore
        configured = Path(settings.processed_data_dir)
    except Exception:
        configured = Path("./processed_data")
    if configured.is_absolute():
        return configured
    return (Path(__file__).resolve().parents[1] / configured).resolve()


def record_path(paper_id: str, base: Optional[Path] = None) -> Path:
    return records_dir(base) / f"{paper_id}.json"


def save_record(record: Dict[str, Any], base: Optional[Path] = None) -> Path:
    """Persist atomically so a crash mid-write cannot leave a half record."""
    path = record_path(record["paper_id"], base)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)
    return path


def load_record(paper_id: str, base: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    path = record_path(paper_id, base)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Could not read paper record {path}: {exc}")
        return None
