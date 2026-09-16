"""PDF processing utilities using MinerU for high-quality extraction.

Enhanced version with multi-type chunking support:
- Abstract chunks
- Section chunks (with chemistry-aware detection)
- Fine-grained overlapping chunks
- Table chunks (IC50, activity data)
- Caption chunks (figures, tables, schemes)

Uses MinerU (PDF-Extract-Kit) for:
- Layout detection (DocLayout-YOLO)
- Table extraction (StructEqTable)
- Formula recognition (UniMERNet)
- OCR for scanned documents
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import json
import re
from dataclasses import dataclass
import tiktoken
import multiprocessing
import gc

from .models import ChunkType, PaperMetadata, Chunk
from .chunker import PaperChunker
from . import paper_record as pr


def _mineru_version() -> str:
    """Installed MinerU version. The package exposes no ``__version__``."""
    try:
        from importlib.metadata import version
        return version("mineru")
    except Exception:
        return ""


def _cross_page_map(pdf_info) -> Dict[Tuple[int, Tuple[int, ...]], int]:
    """Find paragraphs MinerU merged across a page break.

    ``para_split`` joins a paragraph that continues onto the next page into one
    block, appends the continuation's lines, flags their spans ``cross_page``, and
    keeps the block on the page where the paragraph *starts*.  Measured on a
    40-paper sample, **15.3% of long body paragraphs are such merges** -- so
    without this, a citation landing in a paragraph's tail resolves to the
    previous page and the PDF highlight opens one page early.

    Returns ``{(page_idx, scaled_bbox): chars_belonging_to_the_starting_page}``.
    The key is the same scaled bbox ``make_blocks_to_content_list`` writes into
    ``content_list``, which is what lets the two views be matched up.
    """
    from mineru.backend.pipeline.pipeline_middle_json_mkcontent import merge_para_with_text

    out: Dict[Tuple[int, Tuple[int, ...]], int] = {}
    for page in pdf_info or []:
        page_idx = page.get("page_idx")
        size = page.get("page_size") or []
        if page_idx is None or len(size) != 2 or not size[0] or not size[1]:
            continue
        width, height = size
        for block in page.get("para_blocks") or []:
            lines = block.get("lines") or []
            split = next(
                (i for i, line in enumerate(lines)
                 if any(s.get("cross_page") for s in line.get("spans") or [])),
                None,
            )
            if not split:  # None, or the whole block is the continuation
                continue
            bbox = block.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            try:
                full = merge_para_with_text(block)
                head = merge_para_with_text({**block, "lines": lines[:split]})
            except Exception:
                continue
            # Hyphen handling at the boundary looks at the following line, so the
            # truncated render can differ by a character; take the common prefix.
            n = 0
            for a, b in zip(full, head):
                if a != b:
                    break
                n += 1
            if 0 < n < len(full):
                key = (page_idx, (
                    int(bbox[0] * 1000 / width), int(bbox[1] * 1000 / height),
                    int(bbox[2] * 1000 / width), int(bbox[3] * 1000 / height),
                ))
                out[key] = n
    return out


def _image_body_map(pdf_info) -> Dict[Tuple[int, Tuple[int, ...]], List[int]]:
    """The *picture's own* rectangle for each image/table group.

    ``content_list`` gives one bbox per image or table item, and it is the
    **group** rectangle -- picture plus caption plus footnote.  Highlighting that
    box in a PDF viewer covers the caption too, and it is not what the crop on
    disk shows.  The picture's own box exists only in ``middle_json``, nested in
    ``para_blocks[i]["blocks"]`` as the ``image_body`` / ``table_body`` child.

    Keyed the same way ``_cross_page_map`` is -- on ``(page_idx, scaled group
    bbox)`` -- because that key is exactly what ``make_blocks_to_content_list``
    writes out, which is what lets the two views be joined at all.
    """
    from mineru.utils.enum_class import BlockType

    bodies = {BlockType.IMAGE_BODY, BlockType.TABLE_BODY}
    out: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}
    for page in pdf_info or []:
        page_idx = page.get("page_idx")
        size = page.get("page_size") or []
        if page_idx is None or len(size) != 2 or not size[0] or not size[1]:
            continue
        width, height = size

        def scale(box):
            return [int(box[0] * 1000 / width), int(box[1] * 1000 / height),
                    int(box[2] * 1000 / width), int(box[3] * 1000 / height)]

        for block in page.get("para_blocks") or []:
            if block.get("type") not in (BlockType.IMAGE, BlockType.TABLE):
                continue
            group = block.get("bbox")
            if not group or len(group) != 4:
                continue
            for child in block.get("blocks") or []:
                box = child.get("bbox")
                if child.get("type") in bodies and box and len(box) == 4:
                    out[(int(page_idx), tuple(scale(group)))] = scale(box)
                    break
    return out


def _subprocess_mineru_extract(pdf_path_str: str, lang: str, use_gpu: bool,
                               paper_id: str, result_queue,
                               assets_base_str: Optional[str] = None):
    """Run MinerU extraction in an isolated subprocess.

    The **paper record is built here, inside the subprocess**, not reconstructed by
    the parent afterwards.  MinerU's ``content_list`` and ``middle_json`` never
    cross the process boundary, so the pdfium isolation of
    docs/BUG_REPORT_pdfium_deadlock_2026-05-04.md is preserved and the structure
    that W1 depends on is not lost on the way out.

    The figure and table crops are written here too, for the same reason: the
    image writer is driven from inside ``result_to_middle_json``, so the only
    place it can point at a durable directory is this process.  It used to point
    at a ``TemporaryDirectory`` that was removed on exit, which is why every
    ``img_path`` in the corpus was dangling.

    Must be a top-level function for multiprocessing pickling.
    """
    pdf_path = Path(pdf_path_str)

    from mineru.cli.common import read_fn
    from mineru.backend.pipeline.pipeline_analyze import doc_analyze
    from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json
    from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make
    from mineru.data.data_reader_writer import FileBasedDataWriter
    from mineru.utils.enum_class import MakeMode

    pdf_bytes = read_fn(pdf_path)

    assets_base = Path(assets_base_str) if assets_base_str else None
    crops_dir = pr.figures_dir(paper_id, assets_base)
    crops_dir.mkdir(parents=True, exist_ok=True)

    # The crop writer's root and MinerU's `img_buket_path` are deliberately two
    # halves of one path: the writer gets `.../{paper_id}/figures`, the bucket
    # gets the bare `figures`, so `content_list` carries a record-relative
    # `figures/<sha256>.jpg` that `paper_record.asset_path` can resolve against
    # whatever directory the record itself is read from.
    image_writer = FileBasedDataWriter(str(crops_dir))
    bucket = pr.FIGURES_SUBDIR

    infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled = doc_analyze(
        [pdf_bytes], [lang], parse_method="auto", formula_enable=True, table_enable=True
    )

    middle_json = result_to_middle_json(
        infer_results[0], all_image_lists[0], all_pdf_docs[0],
        image_writer, lang, ocr_enabled[0]
    )

    pdf_info = middle_json.get("pdf_info", [])
    markdown = union_make(pdf_info, MakeMode.MM_MD, bucket)
    content_list = union_make(pdf_info, MakeMode.CONTENT_LIST, bucket)

    record = pr.build_record(
        content_list=content_list,
        pdf_info=pdf_info,
        paper_id=paper_id,
        file_name=pdf_path.name,
        pdf_path=pdf_path,
        extractor_version=_mineru_version(),
        cross_page_map=_cross_page_map(pdf_info),
        image_bbox_map=_image_body_map(pdf_info),
        assets_base=assets_base,
    )

    # Crop filenames are MinerU content hashes, so re-extracting an unchanged
    # paper rewrites the same files. Anything left over belongs to a previous
    # extraction whose crops changed, and would orphan forever otherwise.
    record['stats']['assets_pruned'] = pr.prune_assets(record, assets_base)
    record['assets'] = pr.asset_summary(record, assets_base)
    record['capabilities']['figure_images'] = record['assets']['n_present'] > 0

    result_queue.put({
        'full_text': pr.body_text(record),
        'markdown': markdown,
        'tables': pr.table_texts(record),
        'figures': [
            {
                'figure_id': f.get('figure_id'),
                'img_path': f.get('img_path'),
                'path': str(pr.figure_image(record, f, assets_base) or ''),
                'caption': ' '.join(record['blocks'][c]['text']
                                    for c in f.get('caption_blocks', [])),
                'label': f.get('label'),
                'bbox': f.get('image_bbox') or f['bbox'],
                'page_idx': f['page_idx'],
                'block': f['block'],
            }
            for f in record['figures']
        ],
        'captions': pr.captions(record),
        'metadata': record['metadata'],
        'record': record,
    })


def _discard_partial_crops(record, assets_base) -> Dict[str, Any]:
    """Delete crops MinerU wrote before it timed out or crashed.

    MinerU writes each crop during ``result_to_middle_json``, so a run that dies
    partway leaves real files behind.  The paper then falls through to
    pypdfium2, whose degraded record has no bboxes or captions and therefore
    references none of them -- they are unusable *and* invisible, which is
    exactly how a directory grows without bound.  Pruning against a record that
    references nothing removes all of them.
    """
    try:
        return pr.prune_assets(record, assets_base)
    except OSError as exc:
        logging.getLogger(__name__).warning(f"Could not prune partial crops: {exc}")
        return {"removed": 0, "bytes_freed": 0, "kept": 0, "error": str(exc)}


def _subprocess_fallback_extract(pdf_path_str: str, paper_id: str, result_queue,
                                 assets_base_str: Optional[str] = None):
    """Run pypdfium2 fallback extraction in an isolated subprocess.

    Prevents PDFium native library state corruption in long-running processes.
    Produces a **degraded** record: pypdfium2 gives interleaved two-column text
    with no layout, so there are no captions, no tables and no headings to derive
    sections from.  Per-page text and offsets are still real, so provenance down
    to the page survives; the ``capabilities`` flags say what the record cannot
    support instead of letting empty lists read as "this paper has no tables".

    Must be a top-level function for multiprocessing pickling.
    """
    import pypdfium2 as pdfium

    pdf_path = Path(pdf_path_str)
    assets_base = Path(assets_base_str) if assets_base_str else None
    doc = None
    try:
        doc = pdfium.PdfDocument(pdf_path)
        text_parts = []
        page_sizes = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            textpage = page.get_textpage()
            text_parts.append(textpage.get_text_bounded())
            textpage.close()
            try:
                page_sizes.append((page.get_width(), page.get_height()))
            except Exception:
                page_sizes.append((0.0, 0.0))
        record = pr.build_degraded_record(
            page_texts=text_parts, paper_id=paper_id, file_name=pdf_path.name,
            reason=pr.DEGRADED_FALLBACK, pdf_path=pdf_path, page_sizes=page_sizes,
        )
        full_text = "\n\n".join(text_parts)
        record['stats']['assets_pruned'] = _discard_partial_crops(record, assets_base)
        result_queue.put({
            'full_text': full_text, 'markdown': full_text,
            'tables': [], 'figures': [], 'captions': [],
            'metadata': record['metadata'],
            'record': record,
        })
    except Exception as e:
        logging.getLogger(__name__).error(f"Fallback extraction failed in subprocess: {e}")
        record = pr.build_failed_record(paper_id, pdf_path.name, pdf_path=pdf_path)
        record['stats']['assets_pruned'] = _discard_partial_crops(record, assets_base)
        result_queue.put({
            'full_text': '', 'markdown': '', 'tables': [], 'figures': [], 'captions': [],
            'metadata': record['metadata'], 'record': record,
        })
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass

# Set multiprocessing start method to 'spawn' for macOS safety
# This prevents fork() issues with MinerU's multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PDFChunk:
    """Represents a chunk of content from a PDF."""
    text: str
    page_number: int
    paper_id: str
    paper_title: str
    chunk_index: int
    metadata: Dict
    images: Optional[List[Dict]] = None


@dataclass
class MinerUContent:
    """Parsed content from MinerU extraction.

    ``record`` is W1's paper record (see ``paper_record``).  The four flat fields
    above it are views onto that record, kept so the existing chunker keeps
    working unchanged; new code should read ``record``.
    """
    full_text: str
    markdown: str
    tables: List[str]
    figures: List[Dict]
    captions: List[str]
    metadata: Dict
    record: Optional[Dict[str, Any]] = None


class MinerUExtractor:
    """Extract content from PDFs using MinerU."""

    # Default timeout for PDF extraction (15 minutes - then fallback to simple extraction)
    DEFAULT_TIMEOUT = 900

    def __init__(self, use_gpu: bool = True, lang: str = "en",
                 timeout: int = DEFAULT_TIMEOUT,
                 assets_base: Optional[Path] = None):
        """Initialize MinerU extractor.

        Args:
            use_gpu: Whether to use GPU acceleration
            lang: Language for OCR ('en', 'ch', etc.)
            timeout: Maximum seconds to wait for extraction (default: 300)
            assets_base: Root the figure crops are written under, matching the
                ``base`` the record will be saved with.  ``None`` means the
                configured ``processed_data/``.  A sample harness must pass its
                own scratch directory here or it will write crops into the
                production library.
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.timeout = timeout
        self.assets_base = Path(assets_base) if assets_base else None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of MinerU components."""
        if self._initialized:
            return

        try:
            # MinerU doesn't require prepare_env() when using the library programmatically
            # The actual pipeline functions (doc_analyze, etc.) handle initialization internally
            self._initialized = True
            logger.info("MinerU initialized successfully")
        except Exception as e:
            logger.warning(f"MinerU initialization warning: {e}")
            self._initialized = True  # Continue anyway

    def extract(self, pdf_path: Path, paper_id: Optional[str] = None,
                assets_base: Optional[Path] = None) -> MinerUContent:
        """Extract all content from a PDF using MinerU.

        Runs extraction in a subprocess to prevent PDFium native library
        state corruption in long-running processes.  The paper record is built
        inside that subprocess and returned with the content.

        Args:
            pdf_path: Path to PDF file
            paper_id: Stable paper identifier. Defaults to ``md5(filename)[:12]``,
                which W5's baseline depends on -- do not change the derivation.
            assets_base: Overrides the instance's crop root for this one call.

        Returns:
            MinerUContent with extracted text, tables, figures, and the record.
        """
        import queue

        paper_id = paper_id or pr.paper_id_for_filename(pdf_path.name)
        base = assets_base or self.assets_base
        base_str = str(base) if base else None

        result_queue = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=_subprocess_mineru_extract,
            args=(str(pdf_path), self.lang, self.use_gpu, paper_id,
                  result_queue, base_str),
        )
        proc.start()

        # Read from queue BEFORE joining to avoid deadlock.
        # Queue.put() blocks if the pipe buffer is full, and proc.join()
        # waits for the process to exit — if join() comes first, both sides
        # wait for each other forever.
        try:
            result_dict = result_queue.get(timeout=self.timeout)
            proc.join(10)
            return MinerUContent(**result_dict)
        except queue.Empty:
            logger.warning(f"MinerU extraction timed out after {self.timeout}s for {pdf_path.name}, using fallback")
        except Exception as e:
            logger.warning(f"MinerU extraction failed: {e}, using fallback")
        finally:
            if proc.is_alive():
                proc.terminate()
                proc.join(5)
                if proc.is_alive():
                    proc.kill()

        # Fallback: run pypdfium2 in a separate subprocess
        result_queue = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=_subprocess_fallback_extract,
            args=(str(pdf_path), paper_id, result_queue, base_str),
        )
        proc.start()

        try:
            result_dict = result_queue.get(timeout=60)
            proc.join(10)
            return MinerUContent(**result_dict)
        except queue.Empty:
            logger.error(f"Fallback extraction timed out for {pdf_path.name}")
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
        finally:
            if proc.is_alive():
                proc.terminate()
                proc.join(5)
                if proc.is_alive():
                    proc.kill()

        # Neither extractor produced anything. Emit a record that is explicitly
        # marked failed rather than an empty one that looks complete, and drop
        # any crops the dead MinerU run had already written.
        failed = pr.build_failed_record(paper_id, pdf_path.name, pdf_path=pdf_path)
        failed['stats']['assets_pruned'] = _discard_partial_crops(failed, base)
        return MinerUContent(
            full_text="", markdown="", tables=[], figures=[], captions=[],
            metadata={"title": pdf_path.stem, "file_name": pdf_path.name, "num_pages": 0},
            record=failed,
        )

    # NOTE: the in-process `_extract_with_mineru` and `_fallback_extract` variants
    # were removed in W1. Both loaded pdfium into the long-running backend process,
    # which is exactly the corruption described in
    # docs/BUG_REPORT_pdfium_deadlock_2026-05-04.md. Extraction must go through the
    # `_subprocess_*` functions above, which is also where the paper record is built.

    def _sanitize_text(self, text: str) -> str:
        """Remove invalid Unicode characters that can't be encoded to JSON.

        Handles surrogate characters and other problematic Unicode.
        """
        # Encode to utf-8, replacing errors, then decode back
        return text.encode('utf-8', errors='replace').decode('utf-8')

    def _extract_text_from_content(self, content_list: List) -> str:
        """Extract plain text from MinerU content list."""
        text_parts = []

        for item in content_list:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                if item_type == "text":
                    text_parts.append(item.get("text", ""))
                elif item_type == "title":
                    text_parts.append("\n\n" + item.get("text", "") + "\n")
            elif isinstance(item, str):
                text_parts.append(item)

        return self._sanitize_text("\n".join(text_parts))

    def _extract_tables_from_content(self, content_list: List) -> List[str]:
        """Extract tables from MinerU content list (R1).

        MinerU's pipeline backend emits a table item as
        ``{"type": "table", "table_body": "<table>...", "table_caption": [...],
        "table_footnote": [...]}``.  This previously read ``item["latex"]`` or
        ``item["text"]``; neither key is ever present, so it returned ``[]`` for
        every paper in the corpus.  Verified against mineru 2.7.3
        ``pipeline_middle_json_mkcontent.make_blocks_to_content_list``.
        """
        tables = []

        for item in content_list:
            if not isinstance(item, dict) or item.get("type") != "table":
                continue
            body = item.get("table_body", "") or ""
            if not body:
                continue
            parts = [c for c in (item.get("table_caption") or []) if c]
            parts.append(body)
            parts += [f for f in (item.get("table_footnote") or []) if f]
            tables.append(self._sanitize_text("\n".join(parts)))

        return tables

    def _extract_captions_from_content(self, content_list: List) -> List[str]:
        """Extract real captions from MinerU's caption lists (R2).

        The regex this replaces ran ``finditer`` over the whole markdown, so every
        in-text mention ("as shown in Figure 3, the peak...") became a caption --
        93,144 pseudo-caption chunks, 44% of the index.  MinerU already separates
        captions from body text; use its lists.
        """
        captions = []

        for item in content_list:
            if not isinstance(item, dict):
                continue
            for key in ("image_caption", "table_caption"):
                for caption in item.get(key) or []:
                    caption = (caption or "").strip()
                    if caption:
                        captions.append(self._sanitize_text(caption))

        return captions

    def _extract_figures_from_content(self, content_list: List) -> List[Dict]:
        """Extract figure information from MinerU content list.

        ``img_path`` here is the *record-relative* crop reference
        (``figures/<sha256>.jpg``), not an absolute path: the crop lives under
        ``processed_data/{paper_id}/``, so resolving it needs a paper id.  Use
        ``paper_record.figure_image`` rather than opening this value directly.
        Also note the caption key is ``image_caption`` (a list) -- the old
        ``img_caption`` is not a key MinerU emits, so this returned "" for every
        figure in the corpus.
        """
        figures = []

        for item in content_list:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                if item_type == "image":
                    figures.append({
                        "img_path": item.get("img_path", ""),
                        "caption": " ".join(
                            c for c in (item.get("image_caption") or []) if c),
                        "bbox": item.get("bbox", []),
                        "page_idx": item.get("page_idx", 0),
                    })

        return figures

    def _extract_captions_from_markdown(self, markdown: str) -> List[str]:
        """REMOVED (R2). Kept as a stub so no caller silently reverts to it.

        The body-text caption regex is gone; use ``_extract_captions_from_content``
        or ``paper_record.captions(record)``.
        """
        raise NotImplementedError(
            "Caption extraction by regex over markdown was removed in W1 (R2). "
            "Use paper_record.captions(record) or _extract_captions_from_content()."
        )

    def _extract_metadata(self, pdf_path: Path, middle_json: Dict) -> Dict:
        """Extract metadata from PDF."""
        metadata = {
            "title": pdf_path.stem,
            "num_pages": len(middle_json.get("pdf_info", [])),
            "file_name": pdf_path.name,
        }

        # Try to extract title from first page
        pdf_info = middle_json.get("pdf_info", [])
        if pdf_info:
            first_page = pdf_info[0] if pdf_info else {}
            # Look for title in preproc_blocks
            for block in first_page.get("preproc_blocks", []):
                if block.get("type") == "title":
                    metadata["title"] = block.get("text", pdf_path.stem)
                    break

        return metadata

class PDFProcessor:
    """Process PDF files to extract text, tables, and images.

    Uses token-based chunking with semantic preservation (respects sentence boundaries).
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        """Initialize the PDF processor.

        Args:
            chunk_size: Maximum number of tokens per chunk (default: 512)
            chunk_overlap: Number of tokens to overlap between chunks (default: 128)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extractor = MinerUExtractor()

        # Initialize tokenizer (using OpenAI's cl100k_base encoding)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        logger.info(f"Initialized PDFProcessor with chunk_size={chunk_size} tokens, overlap={chunk_overlap} tokens")

    def extract_text_and_metadata(self, pdf_path: Path) -> Dict:
        """Extract text and metadata from PDF using MinerU."""
        content = self.extractor.extract(pdf_path)

        return {
            'title': content.metadata.get('title', pdf_path.stem),
            'metadata': content.metadata,
            'full_text': content.full_text,
            'markdown': content.markdown,
            'num_pages': content.metadata.get('num_pages', 0)
        }

    def extract_tables(self, pdf_path: Path) -> List[Dict]:
        """Extract tables from PDF using MinerU."""
        content = self.extractor.extract(pdf_path)

        tables = []
        for idx, table in enumerate(content.tables):
            tables.append({
                'page_number': 1,  # MinerU doesn't always provide page info
                'table_index': idx,
                'markdown': table,
                'raw_data': table
            })

        return tables

    def extract_images(self, pdf_path: Path) -> List[Dict]:
        """Extract images from PDF using MinerU."""
        content = self.extractor.extract(pdf_path)

        images = []
        for idx, fig in enumerate(content.figures):
            images.append({
                'page_number': (fig.get('page_idx') or 0) + 1,
                'image_index': idx,
                'figure_id': fig.get('figure_id'),
                # Absolute path to the persisted crop, or "" when MinerU wrote
                # none. `img_path` is the record-relative reference.
                'path': fig.get('path', ''),
                'img_path': fig.get('img_path', ''),
                'caption': fig.get('caption', ''),
            })

        return images

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving semantic meaning."""
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.tokenizer.encode(text))

    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """Split text into token-based chunks while preserving semantic meaning."""
        sentences = self._split_into_sentences(text)

        if not sentences:
            logger.warning("No sentences found in text")
            return []

        chunks = []
        current_sentences = []
        current_token_count = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_token_count = self._count_tokens(sentence)

            if current_token_count + sentence_token_count > self.chunk_size and current_sentences:
                chunk_text = ' '.join(current_sentences)
                chunks.append({
                    'text': chunk_text,
                    'chunk_index': chunk_index,
                    'token_count': current_token_count,
                    'sentence_count': len(current_sentences)
                })

                # Create overlap for next chunk
                overlap_sentences = []
                overlap_token_count = 0

                for s in reversed(current_sentences):
                    s_tokens = self._count_tokens(s)
                    if overlap_token_count + s_tokens > self.chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_token_count += s_tokens

                current_sentences = overlap_sentences
                current_token_count = overlap_token_count
                chunk_index += 1

            current_sentences.append(sentence)
            current_token_count += sentence_token_count

        if current_sentences:
            chunk_text = ' '.join(current_sentences)
            chunks.append({
                'text': chunk_text,
                'chunk_index': chunk_index,
                'token_count': current_token_count,
                'sentence_count': len(current_sentences)
            })

        logger.debug(f"Created {len(chunks)} chunks")
        return chunks

    def process_pdf(self, pdf_path: Path, paper_id: str) -> List[PDFChunk]:
        """Process a single PDF and return chunks."""
        logger.info(f"Processing PDF: {pdf_path.name}")

        extracted = self.extract_text_and_metadata(pdf_path)
        tables = self.extract_tables(pdf_path)
        images = self.extract_images(pdf_path)

        text_chunks = self.chunk_text(extracted['full_text'], extracted['metadata'])

        pdf_chunks = []
        for chunk_data in text_chunks:
            chunk = PDFChunk(
                text=chunk_data['text'],
                page_number=1,
                paper_id=paper_id,
                paper_title=extracted['title'],
                chunk_index=chunk_data['chunk_index'],
                metadata={
                    'num_pages': extracted['num_pages'],
                    'has_tables': len(tables) > 0,
                    'num_tables': len(tables),
                    'num_images': len(images),
                    'file_name': pdf_path.name,
                },
                images=images[:3] if images else None
            )
            pdf_chunks.append(chunk)

        logger.info(f"Created {len(pdf_chunks)} chunks from {pdf_path.name}")
        return pdf_chunks


class EnhancedPDFProcessor:
    """Enhanced PDF processor with multi-type chunking using MinerU.

    Creates 6 types of chunks optimized for different query types:
    - ABSTRACT: Full abstract for overview queries
    - SECTION: Logical sections for framing/methods queries
    - FINE: Overlapping chunks for factual queries
    - TABLE: Extracted tables for data queries
    - CAPTION: Figure/table captions for specific references
    - FULL: Mean-pooled embedding created at index time
    """

    def __init__(
        self,
        abstract_max_tokens: int = 300,
        section_max_tokens: int = 2000,
        fine_chunk_tokens: int = 500,
        fine_chunk_overlap: int = 128,
        extraction_timeout: int = 900,
    ):
        """Initialize enhanced processor.

        Args:
            abstract_max_tokens: Max tokens for abstract
            section_max_tokens: Max tokens per section
            fine_chunk_tokens: Target tokens for fine chunks
            fine_chunk_overlap: Overlap between fine chunks
            extraction_timeout: Max seconds for PDF extraction (default: 900)
        """
        self.chunker = PaperChunker(
            abstract_max_tokens=abstract_max_tokens,
            section_max_tokens=section_max_tokens,
            fine_chunk_tokens=fine_chunk_tokens,
            fine_chunk_overlap=fine_chunk_overlap,
        )
        self.extractor = MinerUExtractor(timeout=extraction_timeout)
        self._legacy_processor = PDFProcessor()

        logger.info("Initialized EnhancedPDFProcessor with MinerU backend")

    def extract_text(self, pdf_path: Path) -> Tuple[str, Dict]:
        """Extract full text and metadata from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (full_text, metadata_dict)
        """
        content = self.extractor.extract(pdf_path)

        # Extract title: PDF metadata > MinerU extraction > filename
        title = self._extract_title_from_pdf_metadata(pdf_path)
        if not title:
            title = content.metadata.get('title', '') or pdf_path.stem
        if not title or title.strip() == '':
            title = pdf_path.stem

        # Extract authors: PDF metadata > text heuristics
        authors = self._extract_authors_from_pdf_metadata(pdf_path)
        if not authors:
            authors = self._extract_authors_from_text(content.full_text)

        metadata = {
            'title': title,
            'authors': authors,
            'year': self._extract_year(pdf_path),
            'num_pages': content.metadata.get('num_pages', 0),
            'file_name': pdf_path.name,
        }

        return content.full_text, metadata

    def _extract_authors_from_text(self, text: str) -> List[str]:
        """Extract authors from text using heuristics.

        Looks for author patterns in the first ~2000 chars (before abstract).
        Common patterns:
        - Names separated by commas or 'and'
        - Names with superscripts (affiliations)
        - Names after title, before Abstract
        """
        if not text:
            return []

        # Focus on first ~2000 chars (title + authors area)
        header_text = text[:2000]

        # Find text before "Abstract" (authors are usually there)
        abstract_match = re.search(r'\bAbstract\b', header_text, re.IGNORECASE)
        if abstract_match:
            header_text = header_text[:abstract_match.start()]

        # Skip the title (usually first line or two)
        lines = header_text.strip().split('\n')
        if len(lines) > 1:
            # Skip first 1-2 lines (likely title)
            author_section = '\n'.join(lines[1:5])
        else:
            author_section = header_text

        # Clean up the text
        author_section = re.sub(r'\d+\s*,?\s*', '', author_section)  # Remove superscript numbers
        author_section = re.sub(r'[*†‡§∥⊥#]', '', author_section)  # Remove symbols
        author_section = re.sub(r'\([^)]*\)', '', author_section)  # Remove parentheticals

        # Pattern for names: Capitalized words that look like names
        # Matches patterns like "John Smith", "Jean-Pierre Martin", "O'Brien"
        name_pattern = r"([A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?))"

        # Find all potential names
        potential_names = re.findall(name_pattern, author_section)

        # Filter and clean names
        authors = []
        seen = set()
        for name in potential_names:
            name = name.strip()
            # Must have at least 2 parts (first + last) and not be common words
            parts = name.split()
            if len(parts) >= 2:
                # Skip common non-name words
                skip_words = {'The', 'This', 'That', 'These', 'From', 'With', 'University',
                             'Department', 'Institute', 'Center', 'Laboratory', 'School'}
                if parts[0] not in skip_words and name.lower() not in seen:
                    seen.add(name.lower())
                    authors.append(name)

            # Limit to reasonable number of authors
            if len(authors) >= 15:
                break

        return authors

    def _extract_authors_from_pdf_metadata(self, pdf_path: Path) -> List[str]:
        """Extract authors from PDF document metadata."""
        import pypdfium2 as pdfium

        doc = None
        try:
            doc = pdfium.PdfDocument(pdf_path)
            metadata = doc.get_metadata_dict()

            author_str = metadata.get('Author', '') or metadata.get('Creator', '')
            if not author_str:
                return []

            # Split by common separators
            authors = []
            for sep in [';', ',', ' and ', '&']:
                if sep in author_str:
                    authors = [a.strip() for a in author_str.split(sep) if a.strip()]
                    break

            if not authors and author_str.strip():
                authors = [author_str.strip()]

            return authors
        except Exception as e:
            logger.debug(f"Could not extract PDF metadata: {e}")
            return []
        finally:
            # Always close the PDF document to prevent file descriptor leaks
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass

    def _extract_title_from_pdf_metadata(self, pdf_path: Path) -> Optional[str]:
        """Extract title from PDF document metadata."""
        import pypdfium2 as pdfium

        doc = None
        try:
            doc = pdfium.PdfDocument(pdf_path)
            metadata = doc.get_metadata_dict()

            title = metadata.get('Title', '')
            if title and len(title) > 5 and title != pdf_path.stem:
                return title.strip()
            return None
        except Exception as e:
            logger.debug(f"Could not extract PDF title metadata: {e}")
            return None
        finally:
            # Always close the PDF document to prevent file descriptor leaks
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass

    def _extract_doi_from_pdf(self, pdf_path: Path) -> Optional[str]:
        """Extract DOI from PDF metadata or first 3 pages text.

        DOI can be in:
        1. PDF metadata (uncommon)
        2. First 3 pages text (common)
        """
        import pypdfium2 as pdfium

        doc = None
        try:
            # Open PDF once and check both metadata and text
            doc = pdfium.PdfDocument(pdf_path)

            # Try PDF metadata first
            try:
                metadata = doc.get_metadata_dict()

                # Check common DOI fields
                for field in ['doi', 'DOI', 'Doi', 'Subject', 'Keywords']:
                    value = metadata.get(field, '')
                    if value and 'doi' in value.lower():
                        # Extract DOI from text
                        doi_match = re.search(r'10\.\d{4,}/[^\s]+', value)
                        if doi_match:
                            doi = doi_match.group(0).strip().rstrip('.,;)')
                            logger.debug(f"Found DOI in metadata: {doi}")
                            return doi
            except Exception as e:
                logger.debug(f"Could not extract DOI from metadata: {e}")

            # Try extracting from first 3 pages text
            try:
                # Check first 3 pages (or fewer if document is shorter)
                pages_to_check = min(3, len(doc))

                for page_num in range(pages_to_check):
                    page = doc[page_num]
                    textpage = page.get_textpage()
                    page_text = textpage.get_text_bounded()

                    # Close the textpage to free resources
                    textpage.close()

                    # Look for DOI patterns (more comprehensive)
                    doi_patterns = [
                        r'doi\.org/([0-9]{2}\.[0-9]{4,}/[^\s\"\'\)]+)',  # doi.org/10.xxxx/...
                        r'DOI:?\s*([0-9]{2}\.[0-9]{4,}/[^\s\"\'\)]+)',  # DOI: 10.xxxx/...
                        r'doi:?\s*([0-9]{2}\.[0-9]{4,}/[^\s\"\'\)]+)',  # doi: 10.xxxx/...
                        r'\bhttps?://dx\.doi\.org/([0-9]{2}\.[0-9]{4,}/[^\s\"\'\)]+)',  # dx.doi.org/...
                        r'\b([0-9]{2}\.[0-9]{4,}/[A-Za-z0-9\.\-_\(\)/]+)',  # standalone 10.xxxx/...
                    ]

                    for pattern in doi_patterns:
                        match = re.search(pattern, page_text, re.IGNORECASE)
                        if match:
                            doi = match.group(1) if len(match.groups()) > 0 else match.group(0)
                            # Clean up DOI
                            doi = doi.strip().rstrip('.,;:\)\"\' ')
                            # Remove trailing punctuation and quotes
                            doi = re.sub(r'[.,;:\)\"\'\s]+$', '', doi)

                            if doi.startswith('10.') and '/' in doi:
                                logger.debug(f"Found DOI on page {page_num + 1}: {doi}")
                                return doi

            except Exception as e:
                logger.debug(f"Could not extract DOI from text: {e}")

            return None

        except Exception as e:
            logger.debug(f"Failed to open PDF for DOI extraction: {e}")
            return None

        finally:
            # CRITICAL: Always close the PDF document to prevent file descriptor leaks
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass  # Ignore errors during cleanup

    def _fetch_metadata_from_doi(self, doi: str) -> Dict[str, Any]:
        """Fetch paper metadata from CrossRef using DOI.

        Returns dict with: title, authors, year, journal
        """
        try:
            import requests

            # CrossRef REST API
            url = f"https://api.crossref.org/works/{doi}"
            headers = {
                'User-Agent': 'ResearchPaperRAG/1.0 (mailto:user@example.com)'  # Polite API usage
            }

            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()['message']

                # Extract title
                title = data.get('title', [None])[0] if data.get('title') else None

                # Extract authors
                authors = []
                for author in data.get('author', []):
                    given = author.get('given', '')
                    family = author.get('family', '')
                    if given and family:
                        authors.append(f"{given} {family}")
                    elif family:
                        authors.append(family)

                # Extract year
                year = None
                if 'published-print' in data:
                    year = data['published-print'].get('date-parts', [[None]])[0][0]
                elif 'published-online' in data:
                    year = data['published-online'].get('date-parts', [[None]])[0][0]

                # Extract journal
                journal = data.get('container-title', [None])[0] if data.get('container-title') else None

                logger.info(f"Fetched metadata from CrossRef for DOI: {doi}")
                return {
                    'title': title,
                    'authors': authors,
                    'year': year,
                    'journal': journal,
                    'doi': doi
                }
            else:
                logger.warning(f"CrossRef API returned {response.status_code} for DOI: {doi}")
                return {}

        except Exception as e:
            logger.warning(f"Failed to fetch metadata from CrossRef: {e}")
            return {}

    def _extract_year(self, pdf_path: Path) -> Optional[int]:
        """Extract publication year from filename."""
        match = re.search(r'(19|20)\d{2}', pdf_path.stem)
        if match:
            return int(match.group(0))
        return None

    def process_pdf(
        self,
        pdf_path: Path,
        paper_id: str,
        project_tag: Optional[str] = None,
        research_area: Optional[str] = None,
    ) -> List[Chunk]:
        """Process PDF into multi-type chunks.

        Args:
            pdf_path: Path to PDF file
            paper_id: Unique identifier for the paper
            project_tag: Optional project tag (e.g., "ERα_inhibitors")
            research_area: Optional research area (e.g., "peptide_imaging")

        Returns:
            List of Chunk objects of various types
        """
        logger.info(f"Processing PDF: {pdf_path.name}")

        # Extract content using MinerU. The paper record is built inside the
        # extraction subprocess and comes back attached to `content`.
        content = self.extractor.extract(pdf_path, paper_id=paper_id)
        full_text = content.full_text
        meta = content.metadata

        # Extract title: PDF metadata > MinerU extraction > filename
        title = self._extract_title_from_pdf_metadata(pdf_path)
        if not title:
            title = meta.get('title', '') or pdf_path.stem
        if not title or title.strip() == '':
            title = pdf_path.stem

        # Extract authors: PDF metadata > text heuristics
        authors = self._extract_authors_from_pdf_metadata(pdf_path)
        if not authors:
            authors = self._extract_authors_from_text(full_text)

        # Extract year
        year = self._extract_year(pdf_path)
        journal = None
        doi = None

        # Try DOI extraction and CrossRef lookup (overrides poor extraction)
        doi_extracted = self._extract_doi_from_pdf(pdf_path)
        if doi_extracted:
            logger.info(f"Found DOI: {doi_extracted}")
            doi = doi_extracted
            crossref_metadata = self._fetch_metadata_from_doi(doi_extracted)

            if crossref_metadata:
                # Override with CrossRef data if available and better quality
                if crossref_metadata.get('title') and len(crossref_metadata['title']) > 10:
                    title = crossref_metadata['title']
                    logger.info(f"Using CrossRef title: {title}")

                if crossref_metadata.get('authors') and len(crossref_metadata['authors']) > 0:
                    authors = crossref_metadata['authors']
                    logger.info(f"Using CrossRef authors: {authors}")

                if crossref_metadata.get('year'):
                    year = crossref_metadata['year']

                if crossref_metadata.get('journal'):
                    journal = crossref_metadata['journal']

        # Create paper metadata
        paper_metadata = PaperMetadata(
            paper_id=paper_id,
            title=title,
            authors=authors,
            year=year,
            journal=journal,
            doi=doi,
            num_pages=meta.get('num_pages', 0),
            file_name=meta.get('file_name', pdf_path.name),
            project_tag=project_tag,
            research_area=research_area,
        )

        # Persist the paper record. This is what makes every later chunking change
        # cost minutes instead of a 60-80 h re-extraction (W1 item 6); W2, W3's
        # long-context path and W5 all read it back from here.
        record = content.record
        if record is not None:
            record["metadata"].update({
                "title": title,
                "authors": authors,
                "year": year,
                "journal": journal,
                "doi": doi,
                "project_tag": project_tag,
                "research_area": research_area,
            })
            try:
                path = pr.save_record(record)
                logger.debug(f"Wrote paper record {path}")
            except OSError as e:
                logger.warning(f"Could not persist paper record for {paper_id}: {e}")

        logger.debug(
            f"Extracted {len(content.tables)} tables, {len(content.captions)} "
            f"captions from {pdf_path.name}"
        )

        # Create multi-type chunks **from the record** (W2). The flat-string
        # arguments are only the fallback for a paper with no record: chunking
        # from them cannot produce character offsets, pages or a verbatim span,
        # which is what W3's citations and PDF highlighting need.
        chunks = self.chunker.chunk_paper(
            text=full_text,
            metadata=paper_metadata,
            captions=content.captions,
            tables=content.tables,
            record=record,
        )

        logger.info(
            f"Created {len(chunks)} chunks from {pdf_path.name}: "
            f"{self._count_by_type(chunks)}"
        )

        return chunks

    def _count_by_type(self, chunks: List[Chunk]) -> str:
        """Count chunks by type for logging."""
        counts = {}
        for chunk in chunks:
            chunk_type = chunk.chunk_type.value
            counts[chunk_type] = counts.get(chunk_type, 0) + 1

        return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))

    def process_pdf_legacy(self, pdf_path: Path, paper_id: str) -> List[PDFChunk]:
        """Process PDF using legacy simple chunking.

        For backwards compatibility.
        """
        return self._legacy_processor.process_pdf(pdf_path, paper_id)


if __name__ == "__main__":
    import sys

    test_pdf = Path("test.pdf")

    if len(sys.argv) > 1:
        test_pdf = Path(sys.argv[1])

    if test_pdf.exists():
        print(f"Testing with: {test_pdf}")

        # Test enhanced processor
        print("\n--- Enhanced Processor (MinerU) ---")
        enhanced_processor = EnhancedPDFProcessor()
        enhanced_chunks = enhanced_processor.process_pdf(test_pdf, "test_paper_001")
        print(f"Enhanced: {len(enhanced_chunks)} chunks")

        # Show breakdown by type
        type_counts = {}
        for chunk in enhanced_chunks:
            t = chunk.chunk_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        print("Chunk types:")
        for chunk_type, count in sorted(type_counts.items()):
            print(f"  {chunk_type}: {count}")

        # Preview first chunk of each type
        seen_types = set()
        for chunk in enhanced_chunks:
            if chunk.chunk_type.value not in seen_types:
                seen_types.add(chunk.chunk_type.value)
                preview = chunk.text[:150].replace('\n', ' ')
                print(f"\n{chunk.chunk_type.value.upper()}: {preview}...")

    else:
        print(f"PDF not found: {test_pdf}")
        print("Usage: python pdf_processor.py [path/to/test.pdf]")
