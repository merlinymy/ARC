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
from typing import Dict, List, Optional, Tuple
import logging
import tempfile
import json
import re
from dataclasses import dataclass
import tiktoken

from .models import ChunkType, PaperMetadata, Chunk
from .chunker import PaperChunker

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

    def __init__(self, use_gpu: bool = True, lang: str = "en"):
        """Initialize MinerU extractor.

        Args:
            use_gpu: Whether to use GPU acceleration
            lang: Language for OCR ('en', 'ch', etc.)
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of MinerU components."""
        if self._initialized:
            return

        try:
            from mineru.cli.common import prepare_env
            prepare_env()
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
            return self._extract_with_mineru(pdf_path)
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

            return MinerUContent(
                full_text=full_text,
                markdown=markdown,
                tables=tables,
                figures=figures,
                captions=captions,
                metadata=metadata
            )

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

        return "\n".join(text_parts)

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
        try:
            import pypdfium2 as pdfium

            doc = pdfium.PdfDocument(pdf_path)
            text_parts = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                textpage = page.get_textpage()
                text_parts.append(textpage.get_text_bounded())

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
    ):
        """Initialize enhanced processor.

        Args:
            abstract_max_tokens: Max tokens for abstract
            section_max_tokens: Max tokens per section
            fine_chunk_tokens: Target tokens for fine chunks
            fine_chunk_overlap: Overlap between fine chunks
        """
        self.chunker = PaperChunker(
            abstract_max_tokens=abstract_max_tokens,
            section_max_tokens=section_max_tokens,
            fine_chunk_tokens=fine_chunk_tokens,
            fine_chunk_overlap=fine_chunk_overlap,
        )
        self.extractor = MinerUExtractor()
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

        title = content.metadata.get('title', '') or pdf_path.stem
        if not title or title.strip() == '':
            title = pdf_path.stem

        metadata = {
            'title': title,
            'authors': self._extract_authors_from_text(content.full_text),
            'year': self._extract_year(pdf_path),
            'num_pages': content.metadata.get('num_pages', 0),
            'file_name': pdf_path.name,
        }

        return content.full_text, metadata

    def _extract_authors_from_text(self, text: str) -> List[str]:
        """Try to extract authors from text (basic heuristic)."""
        # This is a simplified approach - MinerU doesn't always extract author metadata
        return []

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

        title = meta.get('title', '') or pdf_path.stem

        # Create paper metadata
        paper_metadata = PaperMetadata(
            paper_id=paper_id,
            title=title,
            authors=self._extract_authors_from_text(full_text),
            year=self._extract_year(pdf_path),
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
