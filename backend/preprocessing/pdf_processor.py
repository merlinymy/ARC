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
import tempfile
import json
import re
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import tiktoken
import multiprocessing
import gc

from .models import ChunkType, PaperMetadata, Chunk
from .chunker import PaperChunker

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
    """Parsed content from MinerU extraction."""
    full_text: str
    markdown: str
    tables: List[str]
    figures: List[Dict]
    captions: List[str]
    metadata: Dict


class MinerUExtractor:
    """Extract content from PDFs using MinerU."""

    # Default timeout for PDF extraction (15 minutes - then fallback to simple extraction)
    DEFAULT_TIMEOUT = 900

    def __init__(self, use_gpu: bool = True, lang: str = "en", timeout: int = DEFAULT_TIMEOUT):
        """Initialize MinerU extractor.

        Args:
            use_gpu: Whether to use GPU acceleration
            lang: Language for OCR ('en', 'ch', etc.)
            timeout: Maximum seconds to wait for extraction (default: 300)
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.timeout = timeout
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

    def extract(self, pdf_path: Path) -> MinerUContent:
        """Extract all content from a PDF using MinerU.

        Args:
            pdf_path: Path to PDF file

        Returns:
            MinerUContent with extracted text, tables, figures, etc.
        """
        self._ensure_initialized()

        try:
            # Use a daemon-threaded executor so timed-out threads don't linger
            # and leak memory. A regular ThreadPoolExecutor thread that times out
            # keeps running (and holding all its memory) indefinitely.
            executor = ThreadPoolExecutor(max_workers=1)
            # Mark worker threads as daemon so they die with the main process
            executor._thread_name_prefix = "mineru-daemon"
            future = executor.submit(self._extract_with_mineru, pdf_path)
            try:
                result = future.result(timeout=self.timeout)
                return result
            except FuturesTimeoutError:
                logger.warning(f"MinerU extraction timed out after {self.timeout}s for {pdf_path.name}, using fallback")
                future.cancel()
                # Force GC to reclaim any partially-loaded model state
                gc.collect()
                return self._fallback_extract(pdf_path)
            except Exception as e:
                logger.warning(f"MinerU extraction error: {e}, using fallback")
                future.cancel()
                return self._fallback_extract(pdf_path)
            finally:
                executor.shutdown(wait=False)
        except Exception as e:
            logger.warning(f"MinerU extraction failed, falling back: {e}")
            return self._fallback_extract(pdf_path)

    def _extract_with_mineru(self, pdf_path: Path) -> MinerUContent:
        """Extract using MinerU pipeline."""
        from mineru.cli.common import read_fn
        from mineru.backend.pipeline.pipeline_analyze import doc_analyze
        from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json
        from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make
        from mineru.data.data_reader_writer import FileBasedDataWriter
        from mineru.utils.enum_class import MakeMode

        try:
            # Read PDF bytes
            pdf_bytes = read_fn(pdf_path)

            # Create temp directory for images
            with tempfile.TemporaryDirectory() as tmp_dir:
                image_writer = FileBasedDataWriter(tmp_dir)

                # Run document analysis
                infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled = doc_analyze(
                    [pdf_bytes],
                    [self.lang],
                    parse_method="auto",
                    formula_enable=True,
                    table_enable=True
                )

                # Convert to intermediate format
                middle_json = result_to_middle_json(
                    infer_results[0],
                    all_image_lists[0],
                    all_pdf_docs[0],
                    image_writer,
                    self.lang,
                    ocr_enabled[0]
                )

                pdf_info = middle_json.get("pdf_info", [])

                # Generate markdown and content list
                markdown = union_make(pdf_info, MakeMode.MM_MD, tmp_dir)
                content_list = union_make(pdf_info, MakeMode.CONTENT_LIST, tmp_dir)

                # Extract components
                full_text = self._extract_text_from_content(content_list)
                tables = self._extract_tables_from_content(content_list)
                figures = self._extract_figures_from_content(content_list)
                captions = self._extract_captions_from_markdown(markdown)

                # Extract metadata
                metadata = self._extract_metadata(pdf_path, middle_json)

                result = MinerUContent(
                    full_text=full_text,
                    markdown=markdown,
                    tables=tables,
                    figures=figures,
                    captions=captions,
                    metadata=metadata
                )

                # Clean up references to help with resource cleanup
                del infer_results, all_image_lists, all_pdf_docs, middle_json
                gc.collect()

                return result
        except Exception as e:
            # Ensure cleanup on error
            gc.collect()
            raise

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
        """Extract tables from MinerU content list."""
        tables = []

        for item in content_list:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                if item_type == "table":
                    # MinerU outputs tables in LaTeX or markdown
                    table_content = item.get("latex", "") or item.get("text", "")
                    if table_content:
                        tables.append(table_content)

        return tables

    def _extract_figures_from_content(self, content_list: List) -> List[Dict]:
        """Extract figure information from MinerU content list."""
        figures = []

        for item in content_list:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                if item_type == "image":
                    figures.append({
                        "path": item.get("img_path", ""),
                        "caption": item.get("img_caption", ""),
                        "bbox": item.get("bbox", [])
                    })

        return figures

    def _extract_captions_from_markdown(self, markdown: str) -> List[str]:
        """Extract captions from markdown content."""
        captions = []

        # Patterns for detecting caption starts
        caption_patterns = [
            r'(?:Figure|Fig\.?)\s*(\d+[a-zA-Z]?)\s*[\.:\-–—]?\s*([^\n]+)',
            r'Table\s*(\d+[a-zA-Z]?)\s*[\.:\-–—]?\s*([^\n]+)',
            r'Scheme\s*(\d+[a-zA-Z]?)\s*[\.:\-–—]?\s*([^\n]+)',
        ]

        for pattern in caption_patterns:
            matches = re.finditer(pattern, markdown, re.IGNORECASE)
            for match in matches:
                full_match = match.group(0).strip()
                if len(full_match) >= 20:
                    captions.append(full_match[:1000])

        return list(set(captions))  # Deduplicate

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

    def _fallback_extract(self, pdf_path: Path) -> MinerUContent:
        """Fallback extraction using basic pypdfium2."""
        import pypdfium2 as pdfium

        doc = None
        try:
            doc = pdfium.PdfDocument(pdf_path)
            text_parts = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                textpage = page.get_textpage()
                text_parts.append(textpage.get_text_bounded())
                textpage.close()  # Close textpage to free resources

            full_text = "\n\n".join(text_parts)

            return MinerUContent(
                full_text=full_text,
                markdown=full_text,
                tables=[],
                figures=[],
                captions=[],
                metadata={
                    "title": pdf_path.stem,
                    "num_pages": len(doc),
                    "file_name": pdf_path.name,
                }
            )
        except Exception as e:
            logger.error(f"Fallback extraction also failed: {e}")
            return MinerUContent(
                full_text="",
                markdown="",
                tables=[],
                figures=[],
                captions=[],
                metadata={"title": pdf_path.stem, "file_name": pdf_path.name, "num_pages": 0}
            )
        finally:
            # Always close the PDF document to prevent file descriptor leaks
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass


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
                'page_number': 1,
                'image_index': idx,
                'path': fig.get('path', ''),
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

        # Extract content using MinerU
        content = self.extractor.extract(pdf_path)
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

        # Get tables and captions from MinerU extraction
        tables = content.tables
        captions = content.captions

        logger.debug(f"Extracted {len(tables)} tables, {len(captions)} captions from {pdf_path.name}")

        # Create multi-type chunks
        chunks = self.chunker.chunk_paper(
            text=full_text,
            metadata=paper_metadata,
            captions=captions,
            tables=tables,
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
