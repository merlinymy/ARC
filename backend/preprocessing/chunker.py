"""Multi-type chunking from the paper record (W2).

Creates 6 types of chunks from W1's structured paper record:

1. ABSTRACT - the abstract span the record identified
2. SECTION  - one coarse chunk per section, **untruncated** (R3)
3. FINE     - paragraph/sentence-packed chunks over the *complete* section
4. FULL     - mean-pooled paper embedding, built at index time
5. CAPTION  - MinerU's real figure/table captions (never body prose)
6. TABLE    - table body HTML with its caption and footnote

What changed, and why
---------------------
The previous chunker ran ``SectionDetector`` and ``extract_abstract`` over a
flattened string and never read the record: on a paper whose record held two
sections and an abstract it emitted **zero** section, fine and abstract chunks.
W1 made the structure exist; this module is what gets it into the index.

Three invariants hold for every chunk this module produces, and the validation
harness (``scripts/w2_sample_chunk.py``) checks all three:

1. **``text`` is a verbatim, contiguous slice of the record's document text** —
   ``full_text(record)[chunk.char_start:chunk.char_end] == chunk.text``.  Chunks
   are *built by slicing* that string, so a synthetic prefix cannot leak in.
   This is what the Citations API and the click-to-highlight feature stand on.
   The corollary is that a chunk is always a contiguous run of record blocks:
   the chunker breaks a run at a table/figure/caption rather than skipping over
   it, because skipping would make the slice disagree with the text.
2. **Positions are honest.** ``page_start``/``page_end`` are 1-indexed and come
   from the blocks the span actually covers, so a paragraph W1 split across a
   page break (R18) reports both pages instead of collapsing to the first.
3. **Nothing is truncated away.** Fine chunks are built from the complete
   section, not from a truncated copy of it (R3).  Truncation survives in one
   place only: the *coarse* section embedding, which is capped because Voyage
   has a per-request token budget — and that cap now changes only what is
   embedded, never what is stored or retrievable.
"""

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import tiktoken

from . import numeric_facts as nf
from . import paper_record as pr
from .models import Chunk, ChunkType, PaperMetadata
from .section_detector import SectionDetector

logger = logging.getLogger(__name__)

#: Sections never worth retrieving.
SKIP_SECTIONS = frozenset({"references", "acknowledgments", "abbreviations"})

#: Block types that end a body run: they carry their own chunk type, and their
#: text would otherwise be swallowed into a prose chunk's verbatim span.
BREAK_BLOCK_TYPES = frozenset({"table", "image", "caption", "footnote"})

#: ``discarded`` blocks are page furniture (running heads, page numbers, stray
#: column fragments).  A short one is spanned over so that a paragraph W1 split
#: across a page break stays in one chunk; a long one breaks the run, because it
#: is usually a whole mis-classified column and does not belong in a quote.
MAX_SPANNED_DISCARDED_CHARS = 200

#: A section shorter than this is not worth a chunk of its own.
MIN_SECTION_CHARS = 100

_SENTENCE_BOUNDARY = re.compile(
    r'(?<=[.!?])["”’)]?\s+(?=[A-Z(\[“"‘\d])'  # normal sentence end
    r'|(?<=[.!?])["”’)]?\n+'                            # end before newline
)


class PaperChunker:
    """Multi-type chunker driven by the paper record."""

    def __init__(
        self,
        abstract_max_tokens: int = 400,
        section_max_tokens: int = 3000,
        fine_chunk_tokens: int = 500,
        fine_chunk_overlap: int = 100,
        min_chunk_tokens: int = 50,
    ):
        """Initialize chunker.

        Args:
            abstract_max_tokens: Cap on the abstract *embedding* body
            section_max_tokens: Cap on the coarse section *embedding* body (R3:
                this no longer truncates stored text or the fine chunks)
            fine_chunk_tokens: Target tokens for fine chunks
            fine_chunk_overlap: Overlap between fine chunks, in tokens
            min_chunk_tokens: A trailing chunk smaller than this is merged back
                into its predecessor rather than dropped
        """
        self.abstract_max_tokens = abstract_max_tokens
        self.section_max_tokens = section_max_tokens
        self.fine_chunk_tokens = fine_chunk_tokens
        self.fine_chunk_overlap = fine_chunk_overlap
        self.min_chunk_tokens = min_chunk_tokens

        self.section_detector = SectionDetector()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def _truncate_at_sentence(self, text: str, max_tokens: int) -> str:
        """Truncate at a sentence boundary, not mid-sentence."""
        tokens = self.tokenizer.encode(text, disallowed_special=())
        if len(tokens) <= max_tokens:
            return text
        truncated = self.tokenizer.decode(tokens[:max_tokens])
        boundary = max(truncated.rfind(". "), truncated.rfind("? "), truncated.rfind("! "))
        if boundary > len(truncated) * 0.5:
            return truncated[:boundary + 1]
        return truncated

    def _sentence_spans(self, text: str) -> List[Tuple[int, int]]:
        """Sentence spans within ``text``, as ``(start, end)`` offsets."""
        spans: List[Tuple[int, int]] = []
        cursor = 0
        for match in _SENTENCE_BOUNDARY.finditer(text):
            if match.start() > cursor:
                spans.append((cursor, match.start()))
            cursor = match.end()
        if cursor < len(text):
            spans.append((cursor, len(text)))
        return spans or [(0, len(text))]

    # ------------------------------------------------------------------
    # Record → runs of contiguous body blocks
    # ------------------------------------------------------------------

    @staticmethod
    def _body_runs(blocks: Sequence[Dict[str, Any]],
                   block_start: int, block_end: int) -> List[List[Dict[str, Any]]]:
        """Split a block range into maximal contiguous runs of body blocks.

        A run may span short ``discarded`` blocks (page furniture) so that a
        paragraph split across a page break stays whole; it always breaks at a
        table, figure, caption or footnote.
        """
        runs: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        for block in blocks[block_start:block_end]:
            btype = block["type"]
            if btype in pr.BODY_BLOCK_TYPES and block["text"]:
                current.append(block)
            elif btype == "discarded" and len(block["text"]) <= MAX_SPANNED_DISCARDED_CHARS:
                continue                      # spanned: stays inside the slice
            elif btype in BREAK_BLOCK_TYPES or btype == "discarded":
                if current:
                    runs.append(current)
                    current = []
        if current:
            runs.append(current)
        return runs

    def _pack_run(
        self,
        run: Sequence[Dict[str, Any]],
        target_tokens: int,
        overlap_tokens: int,
    ) -> List[Dict[str, Any]]:
        """Pack a body run into spans of ~``target_tokens``.

        Units are whole blocks (paragraphs) where they fit, sentences where a
        single block is larger than the target — so a chunk never starts or ends
        mid-sentence.  Each returned span is
        ``{"char_start", "char_end", "page_start", "page_end", "bbox",
        "block_start", "block_end", "tokens"}`` in *record* coordinates.
        """
        units: List[Dict[str, Any]] = []
        for block in run:
            text = block["text"]
            tokens = self.count_tokens(text)
            if tokens <= target_tokens:
                units.append({
                    "start": block["doc_char_start"],
                    "end": block["doc_char_end"],
                    "tokens": tokens,
                    "page": block["page_idx"],
                    "block": block["i"],
                    "bbox": block["bbox"],
                })
                continue
            for s, e in self._sentence_spans(text):
                units.append({
                    "start": block["doc_char_start"] + s,
                    "end": block["doc_char_start"] + e,
                    "tokens": self.count_tokens(text[s:e]),
                    "page": block["page_idx"],
                    "block": block["i"],
                    "bbox": block["bbox"],
                })

        spans: List[Dict[str, Any]] = []
        group: List[Dict[str, Any]] = []
        group_tokens = 0

        def flush():
            nonlocal group, group_tokens
            if not group:
                return
            spans.append({
                "char_start": group[0]["start"],
                "char_end": group[-1]["end"],
                "page_start": min(u["page"] for u in group),
                "page_end": max(u["page"] for u in group),
                "bbox": group[0]["bbox"],
                "block_start": group[0]["block"],
                "block_end": group[-1]["block"] + 1,
                "tokens": group_tokens,
            })
            group = []
            group_tokens = 0

        for index, unit in enumerate(units):
            if group and group_tokens + unit["tokens"] > target_tokens:
                flush()
                # Re-include trailing units up to the overlap budget, so a claim
                # straddling a chunk boundary is retrievable from both sides.
                back: List[Dict[str, Any]] = []
                budget = 0
                for prior in reversed(units[:index]):
                    if budget + prior["tokens"] > overlap_tokens:
                        break
                    back.insert(0, prior)
                    budget += prior["tokens"]
                group = back
                group_tokens = budget
            group.append(unit)
            group_tokens += unit["tokens"]
        flush()

        # A tiny tail is merged into its predecessor rather than dropped: the old
        # chunker discarded it, which lost the end of the section outright.
        if len(spans) > 1 and spans[-1]["tokens"] < self.min_chunk_tokens:
            tail = spans.pop()
            spans[-1]["char_end"] = max(spans[-1]["char_end"], tail["char_end"])
            spans[-1]["page_end"] = max(spans[-1]["page_end"], tail["page_end"])
            spans[-1]["block_end"] = max(spans[-1]["block_end"], tail["block_end"])
            spans[-1]["tokens"] += tail["tokens"]
        return spans

    @staticmethod
    def _strip_spans(text: str, char_start: int,
                     holes: Sequence[Tuple[int, int]]) -> Optional[str]:
        """``text`` minus the ranges in ``holes`` (document coordinates).

        A run spans short ``discarded`` blocks so that a paragraph split across
        a page break stays in one chunk (R18) — but those blocks are the
        journal's running head, and they sit *inside* the verbatim slice::

            ... energy-filtered TEM (EFTEM). We
            IOP PUBLISHING  doi:10.1088/...  1  S
            find that Cu catalysts disassociate ...

        ``text`` has to keep them: it is a byte-exact span and the offsets are
        load-bearing.  The embedding and the UI do not, so they get this cleaned
        copy.  Returns ``None`` when there is nothing to strip.
        """
        if not holes:
            return None
        keep = []
        cursor = 0
        for h0, h1 in holes:
            a, b = h0 - char_start, h1 - char_start
            if a < 0 or b > len(text) or b <= a:
                continue
            keep.append(text[cursor:a])
            cursor = b
        if not keep:
            return None
        keep.append(text[cursor:])
        cleaned = re.sub(r"\n{3,}", "\n\n", "".join(keep)).strip()
        return cleaned if cleaned and cleaned != text else None

    # ------------------------------------------------------------------
    # Context line
    # ------------------------------------------------------------------

    @staticmethod
    def _page_label(page_start: Optional[int], page_end: Optional[int]) -> str:
        if page_start is None:
            return ""
        if page_end is not None and page_end != page_start:
            return f"p.{page_start}-{page_end}"
        return f"p.{page_start}"

    @staticmethod
    def _section_bits(section_name: Optional[str],
                      section_label: Optional[str],
                      subsection_name: Optional[str]) -> List[str]:
        """``["Results", "3.2 Antibacterial activity"]``.

        The canonical name comes first so it matches a query that says
        "methods"; the paper's own heading follows when it says something the
        canonical name does not.  Without this the literal heading reaches no
        chunk at all — a section's title block sits outside its body range, so
        it is in neither the section chunk's span nor any fine chunk.
        """
        bits: List[str] = []
        if section_name:
            bits.append(section_name.replace("_", " ").title())
        detail = (subsection_name or section_label or "").strip()
        if detail:
            stop = {"and", "the", "of", "a", "an", "in", "for", "to", "on", "with"}
            canon = set(re.findall(r"[a-z]+", (section_name or "").lower()))
            words = set(re.findall(r"[a-z]+", detail.lower())) - stop - canon
            if words or not bits:
                bits.append(detail[:80])
        return bits

    def context_line(
        self,
        title: str,
        section_name: Optional[str],
        subsection_name: Optional[str],
        page_start: Optional[int],
        page_end: Optional[int] = None,
        section_label: Optional[str] = None,
    ) -> str:
        """``[Title — Section > Subsection, p.N]`` — goes in ``embed_text`` only."""
        parts: List[str] = []
        clean_title = (title or "").strip().replace("\n", " ")
        if clean_title:
            parts.append(clean_title[:120])
        tail = " > ".join(self._section_bits(section_name, section_label, subsection_name))
        page = self._page_label(page_start, page_end)
        if tail and page:
            parts.append(f"{tail}, {page}")
        elif tail:
            parts.append(tail)
        elif page:
            parts.append(page)
        return f"[{' — '.join(parts)}]" if parts else ""

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def chunk_record(
        self,
        record: Dict[str, Any],
        metadata: Optional[PaperMetadata] = None,
        llm_contexts: Optional[Dict[str, str]] = None,
    ) -> List[Chunk]:
        """Create all chunk types from a W1 paper record.

        Args:
            record: a paper record (``preprocessing.paper_record``)
            metadata: paper metadata; derived from the record when omitted
            llm_contexts: optional ``{chunk_id: context sentence}`` from the
                contextual-retrieval pass (§3b.2), merged into ``embed_text``
        """
        meta = metadata or self.metadata_from_record(record)
        blocks: List[Dict[str, Any]] = record.get("blocks") or []
        if not blocks:
            logger.warning(f"Record for {meta.paper_id} has no blocks; no chunks")
            return []

        doc = pr.full_text(record)
        chunks: List[Chunk] = []

        chunks.extend(self._abstract_chunks(record, doc, meta))
        chunks.extend(self._section_and_fine_chunks(record, doc, meta))
        # A table chunk's span already contains its caption verbatim, so a
        # standalone caption chunk for it would index the same text twice.
        # Figure captions have no such chunk and always stand alone.
        table_chunks, covered = self._table_chunks(record, doc, meta)
        chunks.extend(self._caption_chunks(record, doc, meta, skip_blocks=covered))
        chunks.extend(table_chunks)

        if llm_contexts:
            for chunk in chunks:
                context = llm_contexts.get(chunk.chunk_id)
                if context:
                    chunk.llm_context = context.strip()

        counts: Dict[str, int] = {}
        for chunk in chunks:
            counts[chunk.chunk_type.value] = counts.get(chunk.chunk_type.value, 0) + 1
        logger.info(
            f"Created {len(chunks)} chunks for {meta.paper_id}: "
            + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        )
        return chunks

    def metadata_from_record(self, record: Dict[str, Any]) -> PaperMetadata:
        """Build ``PaperMetadata`` from what the record knows."""
        rec_meta = record.get("metadata") or {}
        return PaperMetadata(
            paper_id=record.get("paper_id", ""),
            title=rec_meta.get("title") or "",
            authors=rec_meta.get("authors") or [],
            year=rec_meta.get("year"),
            journal=rec_meta.get("journal"),
            doi=rec_meta.get("doi"),
            file_name=rec_meta.get("file_name") or record.get("file_name", ""),
            num_pages=rec_meta.get("num_pages") or len(record.get("pages") or []),
            project_tag=rec_meta.get("project_tag"),
            research_area=rec_meta.get("research_area"),
        )

    # ------------------------------------------------------------------
    # Chunk builders
    # ------------------------------------------------------------------

    def _make(
        self,
        chunk_id: str,
        chunk_type: ChunkType,
        doc: str,
        char_start: int,
        char_end: int,
        meta: PaperMetadata,
        *,
        page_start: Optional[int],
        page_end: Optional[int],
        bbox: Optional[List[int]] = None,
        block_start: Optional[int] = None,
        block_end: Optional[int] = None,
        section_name: Optional[str] = None,
        section_label: Optional[str] = None,
        subsection_name: Optional[str] = None,
        parent_chunk_id: Optional[str] = None,
        figure_id: Optional[str] = None,
        embed_body: Optional[str] = None,
        display_body: Optional[str] = None,
        numeric_source: Optional[str] = None,
    ) -> Chunk:
        """Build one chunk, slicing ``text`` out of the document text.

        ``text`` comes from the slice and only from the slice — that is what
        makes the verbatim guarantee structural rather than aspirational.
        Pages arrive 0-indexed (record ``page_idx``) and are stored 1-indexed.
        """
        text = doc[char_start:char_end]
        p_start = None if page_start is None else page_start + 1
        p_end = None if page_end is None else page_end + 1
        line = self.context_line(meta.title, section_name, subsection_name,
                                 p_start, p_end, section_label=section_label)

        facts: List[Dict[str, Any]] = []
        if numeric_source == "table":
            pass                              # supplied by the caller
        elif numeric_source != "skip":
            facts = nf.facts_from_prose(text)

        return Chunk(
            chunk_id=chunk_id,
            paper_id=meta.paper_id,
            chunk_type=chunk_type,
            text=text,
            context_line=line or None,
            embed_body=embed_body,
            display_body=display_body,
            page_start=p_start,
            page_end=p_end,
            page_numbers=(list(range(p_start, p_end + 1))
                          if p_start is not None and p_end is not None else []),
            char_start=char_start,
            char_end=char_end,
            bbox=bbox,
            block_start=block_start,
            block_end=block_end,
            section_name=section_name,
            subsection_name=subsection_name,
            parent_chunk_id=parent_chunk_id,
            figure_id=figure_id,
            title=meta.title,
            authors=meta.authors,
            year=meta.year,
            doi=meta.doi,
            project_tag=meta.project_tag,
            research_area=meta.research_area,
            file_name=meta.file_name,
            numeric_facts=facts,
            token_count=self.count_tokens(text),
        )

    def _abstract_chunks(self, record, doc, meta) -> List[Chunk]:
        abstract = record.get("abstract")
        if not abstract:
            return []
        start, end = abstract["doc_char_start"], abstract["doc_char_end"]
        if not (0 <= start < end <= len(doc)):
            logger.warning(f"{meta.paper_id}: abstract offsets out of range; skipped")
            return []
        blocks = record["blocks"]
        span_blocks = blocks[abstract["block_start"]:abstract["block_end"]] or [blocks[0]]
        # The record's cleaned abstract text (front matter stripped, "Keywords:"
        # tail removed) is the better thing to embed and to show; the verbatim
        # slice is what gets cited.
        cleaned = abstract.get("text") or ""
        chunk = self._make(
            f"{meta.paper_id}_abstract",
            ChunkType.ABSTRACT,
            doc, start, end, meta,
            page_start=min(b["page_idx"] for b in span_blocks),
            page_end=max(b["page_idx"] for b in span_blocks),
            bbox=span_blocks[0].get("bbox"),
            block_start=abstract["block_start"],
            block_end=abstract["block_end"],
            section_name="abstract",
            embed_body=(self._truncate_at_sentence(cleaned, self.abstract_max_tokens)
                        if cleaned and cleaned != doc[start:end] else None),
            display_body=cleaned if cleaned and cleaned != doc[start:end] else None,
        )
        return [chunk]

    def _sections(self, record) -> List[Dict[str, Any]]:
        """Record sections, or one synthetic whole-document section.

        Degraded (pypdfium2) records carry no sections at all, and a paper whose
        headings MinerU missed has only the ``body`` pseudo-section W1 emits.
        Either way the body must still be chunked, so synthesize a span rather
        than silently indexing nothing — that is the 13.5%-zero-fine-chunks bug.
        """
        raw = record.get("sections") or []
        if raw:
            return [s for s in raw if s["normalized_name"] not in SKIP_SECTIONS]
        blocks = record.get("blocks") or []
        if not blocks:
            return []
        body = [b for b in blocks if b["type"] in pr.BODY_BLOCK_TYPES and b["text"]]
        if not body:
            return []
        return [{
            "id": 0,
            "name": "Document",
            "normalized_name": "body",
            "level": 1,
            "title_block": None,
            "block_start": 0,
            "block_end": len(blocks),
            "parent_id": None,
            "subsection_name": None,
            "doc_char_start": body[0]["doc_char_start"],
            "doc_char_end": body[-1]["doc_char_end"],
            "page_start": body[0]["page_idx"],
            "page_end": body[-1]["page_idx"],
            "n_body_blocks": len(body),
            "char_len": sum(len(b["text"]) for b in body),
            "synthetic": True,
        }]

    def _section_and_fine_chunks(self, record, doc, meta) -> List[Chunk]:
        blocks = record["blocks"]
        chunks: List[Chunk] = []
        fine_idx = 0

        for section in self._sections(record):
            runs = self._body_runs(blocks, section["block_start"], section["block_end"])
            if not runs:
                continue
            body_chars = sum(len(b["text"]) for run in runs for b in run)
            if body_chars < MIN_SECTION_CHARS:
                continue

            section_name = section["normalized_name"]
            section_label = section.get("name")
            subsection = section.get("subsection_name")
            first, last = runs[0][0], runs[-1][-1]

            # Fine spans first, so a section that yields exactly one fine chunk
            # covering its whole span can skip the coarse chunk: the two would be
            # byte-identical points competing for the same reranker slot. Seen on
            # the sample as a `section` and a `fine` hit with the same char range.
            spans = [span for run in runs
                     for span in self._pack_run(run, self.fine_chunk_tokens,
                                                self.fine_chunk_overlap)]
            redundant = (len(spans) == 1
                         and spans[0]["char_start"] == first["doc_char_start"]
                         and spans[0]["char_end"] == last["doc_char_end"])

            # --- coarse section chunk: the complete span, untruncated (R3) ---
            body_text = "\n\n".join(b["text"] for run in runs for b in run)
            section_chunk = None if redundant else self._make(
                f"{meta.paper_id}_section_{section['id']}",
                ChunkType.SECTION,
                doc, first["doc_char_start"], last["doc_char_end"], meta,
                page_start=first["page_idx"],
                page_end=last["page_idx"],
                bbox=first.get("bbox"),
                block_start=section["block_start"],
                block_end=section["block_end"],
                section_name=section_name,
                section_label=section_label,
                subsection_name=subsection,
                # Only the coarse *embedding* is capped — Voyage has a per-request
                # token budget and a whole Results section can be 25k tokens.
                embed_body=self._truncate_at_sentence(body_text, self.section_max_tokens),
                display_body=body_text[:2000] if len(body_text) > 2000 else None,
                # The fine chunks under this section carry the numeric facts;
                # extracting them twice only doubles the payload.
                numeric_source="skip",
            )
            if section_chunk is not None:
                chunks.append(section_chunk)

            # --- fine chunks over the *complete* section (R3) ---
            # Non-body blocks a run spanned (page furniture): kept in `text`,
            # which is verbatim, and stripped from what gets embedded and shown.
            spanned = [(b["doc_char_start"], b["doc_char_end"])
                       for b in blocks[section["block_start"]:section["block_end"]]
                       if b["type"] not in pr.BODY_BLOCK_TYPES and b["text"]]
            for span in spans:
                holes = [(a, b) for a, b in spanned
                         if a >= span["char_start"] and b <= span["char_end"]]
                clean = self._strip_spans(
                    doc[span["char_start"]:span["char_end"]],
                    span["char_start"], holes)
                chunks.append(self._make(
                    f"{meta.paper_id}_fine_{fine_idx}",
                    ChunkType.FINE,
                    doc, span["char_start"], span["char_end"], meta,
                    page_start=span["page_start"],
                    page_end=span["page_end"],
                    bbox=span["bbox"],
                    block_start=span["block_start"],
                    block_end=span["block_end"],
                    section_name=section_name,
                    section_label=section_label,
                    subsection_name=subsection,
                    parent_chunk_id=section_chunk.chunk_id if section_chunk else None,
                    embed_body=clean,
                    display_body=clean,
                ))
                fine_idx += 1
        return chunks

    def _caption_chunks(self, record, doc, meta, skip_blocks=frozenset()) -> List[Chunk]:
        blocks = record["blocks"]
        sections = {s["id"]: s for s in (record.get("sections") or [])}
        chunks: List[Chunk] = []
        for n, block_i in enumerate(record.get("caption_blocks") or []):
            block = blocks[block_i]
            if block_i in skip_blocks or len(block["text"]) < 20:
                continue
            section = sections.get(block.get("section_id"))
            kind = block.get("kind") or "figure"
            chunks.append(self._make(
                f"{meta.paper_id}_caption_{n}",
                ChunkType.CAPTION,
                doc, block["doc_char_start"], block["doc_char_end"], meta,
                page_start=block["page_idx"],
                page_end=block["page_idx"],
                bbox=block.get("bbox"),
                block_start=block_i,
                block_end=block_i + 1,
                # A caption belongs to the section it sits in, so a section-scoped
                # query can reach it.  The old chunker wrote a fake "figures"
                # section name here.
                section_name=section["normalized_name"] if section else None,
                section_label=section.get("name") if section else None,
                figure_id=f"{kind}_{n}",
            ))
        return chunks

    def _table_chunks(self, record, doc, meta) -> Tuple[List[Chunk], set]:
        """Table chunks, plus the caption blocks their spans already cover."""
        blocks = record["blocks"]
        sections = {s["id"]: s for s in (record.get("sections") or [])}
        chunks: List[Chunk] = []
        covered: set = set()
        for n, table in enumerate(record.get("tables") or []):
            body_block = blocks[table["block"]]
            body_html = body_block["text"]
            if not body_html:
                continue

            # Caption blocks are emitted immediately before the body and
            # footnotes immediately after, so caption+body+footnote is one
            # contiguous span and stays verbatim.
            span_blocks = [blocks[i] for i in table.get("caption_blocks", [])]
            span_blocks.append(body_block)
            span_blocks += [blocks[i] for i in table.get("footnote_blocks", [])]
            span_blocks.sort(key=lambda b: b["doc_char_start"])
            char_start = span_blocks[0]["doc_char_start"]
            char_end = span_blocks[-1]["doc_char_end"]

            caption = " ".join(blocks[i]["text"] for i in table.get("caption_blocks", []))
            footnote = " ".join(blocks[i]["text"] for i in table.get("footnote_blocks", []))
            rendered = nf.render_table(body_html)
            readable = "\n".join(p for p in (caption, rendered, footnote) if p)

            section = sections.get(table.get("section_id"))
            chunk = self._make(
                f"{meta.paper_id}_table_{n}",
                ChunkType.TABLE,
                doc, char_start, char_end, meta,
                page_start=min(b["page_idx"] for b in span_blocks),
                page_end=max(b["page_idx"] for b in span_blocks),
                bbox=body_block.get("bbox"),
                block_start=span_blocks[0]["i"],
                block_end=span_blocks[-1]["i"] + 1,
                section_name=section["normalized_name"] if section else None,
                section_label=section.get("name") if section else None,
                figure_id=f"table_{n}",
                # HTML tags are noise to an embedding model and to BM25; the cell
                # values are the signal.  `text` keeps the verbatim HTML.
                embed_body=readable or None,
                display_body=readable or None,
                numeric_source="table",
            )
            chunk.numeric_facts = nf.facts_from_table(body_html, caption)
            covered.update(table.get("caption_blocks", []))
            chunks.append(chunk)
        return chunks, covered

    # ------------------------------------------------------------------
    # Paper-level FULL chunk (built at index time, after embedding)
    # ------------------------------------------------------------------

    def make_full_chunk(self, chunks: Sequence[Chunk]) -> Optional[Chunk]:
        """The paper-level chunk that carries the mean-pooled embedding.

        Its ``text`` used to be the synthetic string ``"[Full paper: <title>]"``,
        which is both uncitable and a useless BM25 document.  It now reuses the
        abstract's verbatim span (or the first section's) so the paper-level
        point carries real text, while ``text_is_verbatim`` stays honest.
        """
        if not chunks:
            return None
        first = chunks[0]
        source = next((c for c in chunks if c.chunk_type == ChunkType.ABSTRACT), None)
        if source is None:
            source = next((c for c in chunks if c.chunk_type == ChunkType.SECTION), None)
        if source is None:
            source = chunks[0]

        text = source.text
        if source.chunk_type == ChunkType.SECTION:
            text = self._truncate_at_sentence(text, self.abstract_max_tokens)
        verbatim = text == source.text

        return Chunk(
            chunk_id=f"{first.paper_id}_full",
            paper_id=first.paper_id,
            chunk_type=ChunkType.FULL,
            text=text,
            context_line=self.context_line(first.title, None, None, None),
            embed_body=f"{first.title}\n\n{text}" if first.title else None,
            display_body=None,
            text_is_verbatim=verbatim,
            char_start=source.char_start if verbatim else None,
            char_end=source.char_end if verbatim else None,
            page_start=source.page_start if verbatim else None,
            page_end=source.page_end if verbatim else None,
            page_numbers=list(source.page_numbers) if verbatim else [],
            section_name=None,
            title=first.title,
            authors=first.authors,
            year=first.year,
            doi=first.doi,
            project_tag=first.project_tag,
            research_area=first.research_area,
            file_name=first.file_name,
            token_count=self.count_tokens(text),
        )

    # ------------------------------------------------------------------
    # Legacy entry point
    # ------------------------------------------------------------------

    def chunk_paper(
        self,
        text: str,
        metadata: PaperMetadata,
        captions: Optional[List[str]] = None,
        tables: Optional[List[str]] = None,
        record: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Chunk a paper.

        Prefers the record (``chunk_record``).  The flat-string path is kept only
        for callers that have no record — it cannot produce offsets, pages or a
        verbatim guarantee, so every chunk it returns is marked
        ``text_is_verbatim=False``.
        """
        if record is not None and (record.get("blocks") or []):
            return self.chunk_record(record, metadata)

        logger.warning(
            f"No paper record for {metadata.paper_id}: falling back to flat-text "
            "chunking. Chunks will carry no offsets, pages or bbox."
        )
        return self._chunk_flat_text(text, metadata, captions, tables)

    def _chunk_flat_text(self, text, metadata, captions, tables) -> List[Chunk]:
        chunks: List[Chunk] = []
        if not text or len(text) < 200:
            return chunks

        def add(chunk_id, chunk_type, body, **kw):
            chunks.append(Chunk(
                chunk_id=chunk_id,
                paper_id=metadata.paper_id,
                chunk_type=chunk_type,
                text=body,
                text_is_verbatim=False,
                title=metadata.title,
                authors=metadata.authors,
                year=metadata.year,
                doi=metadata.doi,
                project_tag=metadata.project_tag,
                research_area=metadata.research_area,
                file_name=metadata.file_name,
                token_count=self.count_tokens(body),
                **kw,
            ))

        abstract = self.section_detector.extract_abstract(text)
        if abstract:
            add(f"{metadata.paper_id}_abstract", ChunkType.ABSTRACT,
                abstract[:self.abstract_max_tokens * 4], section_name="abstract")

        idx = 0
        for section in self.section_detector.detect_sections(text):
            if len(section.text) < MIN_SECTION_CHARS:
                continue
            if section.normalized_name in SKIP_SECTIONS:
                continue
            parent_id = f"{metadata.paper_id}_section_{idx}"
            add(parent_id, ChunkType.SECTION, section.text,
                section_name=section.normalized_name,
                subsection_name=section.subsection_name,
                embed_body=self._truncate_at_sentence(section.text, self.section_max_tokens))
            idx += 1
            for start, end in self._flat_spans(section.text):
                add(f"{metadata.paper_id}_fine_{idx}", ChunkType.FINE,
                    section.text[start:end],
                    section_name=section.normalized_name,
                    subsection_name=section.subsection_name,
                    parent_chunk_id=parent_id)
                idx += 1

        if not chunks:
            for start, end in self._flat_spans(text):
                add(f"{metadata.paper_id}_fine_{idx}", ChunkType.FINE,
                    text[start:end], section_name="body")
                idx += 1

        for n, caption in enumerate(captions or []):
            if len(caption) >= 20:
                add(f"{metadata.paper_id}_caption_{n}", ChunkType.CAPTION, caption,
                    figure_id=f"figure_{n}")
        for n, table in enumerate(tables or []):
            if len(table) >= 50:
                add(f"{metadata.paper_id}_table_{n}", ChunkType.TABLE, table,
                    figure_id=f"table_{n}",
                    embed_body=nf.render_table(table) or None,
                    display_body=nf.render_table(table) or None)
        return chunks

    def _flat_spans(self, text: str) -> List[Tuple[int, int]]:
        """Sentence-packed spans over a flat string (legacy path)."""
        spans: List[Tuple[int, int]] = []
        group: List[Tuple[int, int]] = []
        tokens = 0
        for s, e in self._sentence_spans(text):
            t = self.count_tokens(text[s:e])
            if group and tokens + t > self.fine_chunk_tokens:
                spans.append((group[0][0], group[-1][1]))
                group, tokens = [], 0
            group.append((s, e))
            tokens += t
        if group:
            spans.append((group[0][0], group[-1][1]))
        return spans
