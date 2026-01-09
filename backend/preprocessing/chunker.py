"""Multi-type chunking logic for research papers.

Creates 6 types of chunks from extracted PDF content:
1. ABSTRACT - Full abstract as single chunk
2. SECTION - Logical sections (Introduction, Methods, etc.)
3. FINE - Semantic paragraph-based chunks with sentence boundaries
4. FULL - Mean-pooled embedding of all chunks (at index time)
5. CAPTION - Figure and table captions
6. TABLE - Extracted table content

Chunking improvements:
- Paragraph-aware: Preserves paragraph boundaries as semantic units
- Sentence boundaries: Never splits mid-sentence
- Contextual headers: Fine chunks include section context
- Recursive chunking: Long sections are intelligently subdivided
"""

import logging
import re
from typing import List, Optional, Tuple
import tiktoken

from .models import Chunk, ChunkType, PaperMetadata
from .section_detector import SectionDetector, Section

logger = logging.getLogger(__name__)


class PaperChunker:
    """Multi-type chunker for research papers with semantic awareness."""

    def __init__(
        self,
        abstract_max_tokens: int = 400,
        section_max_tokens: int = 3000,  # Increased from 2000
        fine_chunk_tokens: int = 500,
        fine_chunk_overlap: int = 100,
        min_chunk_tokens: int = 50,  # Minimum viable chunk size
    ):
        """Initialize chunker.

        Args:
            abstract_max_tokens: Max tokens for abstract chunks
            section_max_tokens: Max tokens for section chunks
            fine_chunk_tokens: Target tokens for fine chunks
            fine_chunk_overlap: Overlap between fine chunks
            min_chunk_tokens: Minimum tokens for a chunk to be valid
        """
        self.abstract_max_tokens = abstract_max_tokens
        self.section_max_tokens = section_max_tokens
        self.fine_chunk_tokens = fine_chunk_tokens
        self.fine_chunk_overlap = fine_chunk_overlap
        self.min_chunk_tokens = min_chunk_tokens

        self.section_detector = SectionDetector()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Sentence-ending patterns
        self.sentence_end_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|'  # Standard sentence end
            r'(?<=[.!?])\s*\n|'          # Sentence end before newline
            r'(?<=\))\.\s+(?=[A-Z])'     # Citation ending: "). Next"
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving boundaries."""
        # Split on sentence boundaries
        parts = self.sentence_end_pattern.split(text)

        sentences = []
        current = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Check if this looks like a complete sentence
            if current:
                current += " " + part
            else:
                current = part

            # If ends with sentence-ending punctuation, it's complete
            if current and current[-1] in '.!?':
                sentences.append(current)
                current = ""

        # Add any remaining text
        if current.strip():
            sentences.append(current.strip())

        return sentences

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs based on double newlines or indentation."""
        # Split on double newlines or significant whitespace patterns
        paragraph_pattern = re.compile(r'\n\s*\n|\n(?=\s{4,})')

        raw_paragraphs = paragraph_pattern.split(text)
        paragraphs = []

        for para in raw_paragraphs:
            para = para.strip()
            if para and len(para) > 20:  # Skip very short fragments
                paragraphs.append(para)

        return paragraphs if paragraphs else [text]

    def _chunk_with_sentence_boundaries(
        self,
        text: str,
        target_tokens: int,
        overlap_tokens: int,
    ) -> List[Tuple[str, int]]:
        """Create chunks that respect sentence boundaries.

        Args:
            text: Text to chunk
            target_tokens: Target chunk size in tokens
            overlap_tokens: Overlap between chunks

        Returns:
            List of (chunk_text, token_count) tuples
        """
        sentences = self._split_into_sentences(text)

        if not sentences:
            return [(text, self.count_tokens(text))]

        chunks = []
        current_sentences = []
        current_tokens = 0
        overlap_sentences = []

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If single sentence exceeds target, add it as its own chunk
            if sentence_tokens > target_tokens:
                # First, save current chunk if exists
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append((chunk_text, current_tokens))
                    # Keep last sentences for overlap
                    overlap_sentences = current_sentences[-2:] if len(current_sentences) > 1 else current_sentences

                # Add the long sentence as its own chunk
                chunks.append((sentence, sentence_tokens))
                current_sentences = []
                current_tokens = 0
                overlap_sentences = []
                continue

            # Check if adding this sentence exceeds target
            if current_tokens + sentence_tokens > target_tokens and current_sentences:
                # Save current chunk
                chunk_text = " ".join(current_sentences)
                chunks.append((chunk_text, current_tokens))

                # Start new chunk with overlap
                overlap_sentences = current_sentences[-2:] if len(current_sentences) > 1 else current_sentences[-1:]
                overlap_text = " ".join(overlap_sentences)
                overlap_token_count = self.count_tokens(overlap_text)

                # Only use overlap if it fits
                if overlap_token_count < target_tokens * 0.3:
                    current_sentences = list(overlap_sentences)
                    current_tokens = overlap_token_count
                else:
                    current_sentences = []
                    current_tokens = 0

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        # Add final chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            if self.count_tokens(chunk_text) >= self.min_chunk_tokens:
                chunks.append((chunk_text, self.count_tokens(chunk_text)))

        return chunks

    def _create_contextual_fine_chunks(
        self,
        text: str,
        section_name: str,
        parent_chunk: Chunk,
        metadata: PaperMetadata,
        start_idx: int,
    ) -> List[Chunk]:
        """Create fine chunks with paragraph awareness and contextual headers.

        Args:
            text: Section text to chunk
            section_name: Name of the section for context
            parent_chunk: Parent section chunk
            metadata: Paper metadata
            start_idx: Starting chunk index

        Returns:
            List of fine-grained Chunk objects
        """
        chunks = []

        # First, try paragraph-based chunking
        paragraphs = self._split_into_paragraphs(text)

        # Format section name for context header
        section_display = section_name.replace("_", " ").title()
        context_header = f"[{section_display}] "
        header_tokens = self.count_tokens(context_header)

        # Effective target size accounting for header
        effective_target = self.fine_chunk_tokens - header_tokens

        idx = start_idx
        current_paragraphs = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # If single paragraph is too large, chunk it with sentence boundaries
            if para_tokens > effective_target:
                # First, save accumulated paragraphs
                if current_paragraphs:
                    chunk_text = context_header + "\n\n".join(current_paragraphs)
                    chunks.append(self._make_fine_chunk(
                        chunk_text, section_name, parent_chunk, metadata, idx
                    ))
                    idx += 1
                    current_paragraphs = []
                    current_tokens = 0

                # Chunk the large paragraph by sentences
                sentence_chunks = self._chunk_with_sentence_boundaries(
                    para, effective_target, self.fine_chunk_overlap
                )

                for chunk_text, token_count in sentence_chunks:
                    full_text = context_header + chunk_text
                    chunks.append(self._make_fine_chunk(
                        full_text, section_name, parent_chunk, metadata, idx
                    ))
                    idx += 1
                continue

            # Check if adding this paragraph exceeds target
            if current_tokens + para_tokens > effective_target and current_paragraphs:
                # Save current accumulated paragraphs
                chunk_text = context_header + "\n\n".join(current_paragraphs)
                chunks.append(self._make_fine_chunk(
                    chunk_text, section_name, parent_chunk, metadata, idx
                ))
                idx += 1

                # Start fresh (no paragraph overlap to preserve semantic units)
                current_paragraphs = []
                current_tokens = 0

            current_paragraphs.append(para)
            current_tokens += para_tokens

        # Add final chunk
        if current_paragraphs:
            chunk_text = context_header + "\n\n".join(current_paragraphs)
            if self.count_tokens(chunk_text) >= self.min_chunk_tokens:
                chunks.append(self._make_fine_chunk(
                    chunk_text, section_name, parent_chunk, metadata, idx
                ))

        return chunks

    def _make_fine_chunk(
        self,
        text: str,
        section_name: str,
        parent_chunk: Chunk,
        metadata: PaperMetadata,
        idx: int,
    ) -> Chunk:
        """Helper to create a fine chunk with all metadata."""
        return Chunk(
            chunk_id=f"{metadata.paper_id}_fine_{idx}",
            paper_id=metadata.paper_id,
            chunk_type=ChunkType.FINE,
            text=text,
            section_name=section_name,
            subsection_name=parent_chunk.subsection_name,  # Inherit from parent
            parent_chunk_id=parent_chunk.chunk_id,
            title=metadata.title,
            authors=metadata.authors,
            year=metadata.year,
            doi=metadata.doi,
            project_tag=metadata.project_tag,
            research_area=metadata.research_area,
            file_name=metadata.file_name,
            token_count=self.count_tokens(text),
        )

    def chunk_paper(
        self,
        text: str,
        metadata: PaperMetadata,
        captions: Optional[List[str]] = None,
        tables: Optional[List[str]] = None,
    ) -> List[Chunk]:
        """Create all chunk types for a paper.

        Args:
            text: Full text of the paper
            metadata: Paper metadata
            captions: Extracted figure/table captions
            tables: Extracted table content

        Returns:
            List of Chunk objects of all types
        """
        chunks = []
        chunk_idx = 0

        # 1. Abstract chunk
        abstract = self.section_detector.extract_abstract(text)
        if abstract:
            chunks.append(Chunk(
                chunk_id=f"{metadata.paper_id}_abstract",
                paper_id=metadata.paper_id,
                chunk_type=ChunkType.ABSTRACT,
                text=abstract[:self.abstract_max_tokens * 4],  # Rough char limit
                section_name="abstract",
                title=metadata.title,
                authors=metadata.authors,
                year=metadata.year,
                doi=metadata.doi,
                project_tag=metadata.project_tag,
                research_area=metadata.research_area,
                file_name=metadata.file_name,
                token_count=self.count_tokens(abstract),
            ))
            chunk_idx += 1

        # 2. Section chunks
        sections = self.section_detector.detect_sections(text)
        section_chunks = []

        # Sections that deserve more token allowance (contain unique insights)
        high_value_sections = {"discussion", "results_discussion", "results", "conclusion"}

        for section in sections:
            # Skip very short sections
            if len(section.text) < 100:
                continue

            # Skip references/acknowledgments for retrieval
            if section.normalized_name in ["references", "acknowledgments", "abbreviations"]:
                continue

            # Use higher token limit for high-value sections
            if section.normalized_name in high_value_sections:
                max_tokens = self.section_max_tokens + 1000  # Extra allowance
            else:
                max_tokens = self.section_max_tokens

            # Truncate at sentence boundary if needed
            section_text = self._truncate_at_sentence(section.text, max_tokens)

            section_chunk = Chunk(
                chunk_id=f"{metadata.paper_id}_section_{chunk_idx}",
                paper_id=metadata.paper_id,
                chunk_type=ChunkType.SECTION,
                text=section_text,
                section_name=section.normalized_name,
                subsection_name=section.subsection_name,
                title=metadata.title,
                authors=metadata.authors,
                year=metadata.year,
                doi=metadata.doi,
                project_tag=metadata.project_tag,
                research_area=metadata.research_area,
                file_name=metadata.file_name,
                token_count=self.count_tokens(section_text),
            )
            chunks.append(section_chunk)
            section_chunks.append(section_chunk)
            chunk_idx += 1

        # 3. Fine chunks (semantic, paragraph-aware)
        for section_chunk in section_chunks:
            fine_chunks = self._create_contextual_fine_chunks(
                text=section_chunk.text,
                section_name=section_chunk.section_name,
                parent_chunk=section_chunk,
                metadata=metadata,
                start_idx=chunk_idx,
            )
            chunks.extend(fine_chunks)
            chunk_idx += len(fine_chunks)

        # 4. Caption chunks (with context)
        if captions:
            for i, caption in enumerate(captions):
                if len(caption) < 20:  # Skip very short captions
                    continue

                # Add figure context
                caption_text = f"[Figure/Table Caption] {caption}"

                chunks.append(Chunk(
                    chunk_id=f"{metadata.paper_id}_caption_{i}",
                    paper_id=metadata.paper_id,
                    chunk_type=ChunkType.CAPTION,
                    text=caption_text,
                    section_name="figures",
                    figure_id=f"figure_{i}",
                    title=metadata.title,
                    authors=metadata.authors,
                    year=metadata.year,
                    doi=metadata.doi,
                    project_tag=metadata.project_tag,
                    research_area=metadata.research_area,
                    file_name=metadata.file_name,
                    token_count=self.count_tokens(caption_text),
                ))
                chunk_idx += 1

        # 5. Table chunks (with context)
        if tables:
            for i, table in enumerate(tables):
                if len(table) < 50:  # Skip very short tables
                    continue

                # Add table context
                table_text = f"[Table Content] {table}"

                chunks.append(Chunk(
                    chunk_id=f"{metadata.paper_id}_table_{i}",
                    paper_id=metadata.paper_id,
                    chunk_type=ChunkType.TABLE,
                    text=table_text,
                    section_name="tables",
                    figure_id=f"table_{i}",
                    title=metadata.title,
                    authors=metadata.authors,
                    year=metadata.year,
                    doi=metadata.doi,
                    project_tag=metadata.project_tag,
                    research_area=metadata.research_area,
                    file_name=metadata.file_name,
                    token_count=self.count_tokens(table_text),
                ))
                chunk_idx += 1

        # 6. Fallback: if no chunks created, do full-text chunking
        if not chunks and len(text) > 200:
            logger.warning(
                f"No structure found for {metadata.paper_id}, using fallback chunking"
            )
            chunks = self._create_fallback_chunks(text, metadata, start_idx=chunk_idx)

        logger.info(
            f"Created {len(chunks)} chunks for {metadata.paper_id}: "
            f"{sum(1 for c in chunks if c.chunk_type == ChunkType.ABSTRACT)} abstract, "
            f"{sum(1 for c in chunks if c.chunk_type == ChunkType.SECTION)} section, "
            f"{sum(1 for c in chunks if c.chunk_type == ChunkType.FINE)} fine, "
            f"{sum(1 for c in chunks if c.chunk_type == ChunkType.CAPTION)} caption, "
            f"{sum(1 for c in chunks if c.chunk_type == ChunkType.TABLE)} table"
        )

        return chunks

    def _create_fallback_chunks(
        self,
        text: str,
        metadata: PaperMetadata,
        start_idx: int = 0,
    ) -> List[Chunk]:
        """Create fine chunks from raw text when no structure is detected.

        Used as a fallback for papers without standard sections/abstract.
        Reuses _create_contextual_fine_chunks to maintain consistent behavior.

        Args:
            text: Full text of the paper
            metadata: Paper metadata
            start_idx: Starting chunk index

        Returns:
            List of Chunk objects
        """
        # Create a dummy parent chunk for the fallback context
        parent_chunk = Chunk(
            chunk_id=f"{metadata.paper_id}_content",
            paper_id=metadata.paper_id,
            chunk_type=ChunkType.SECTION,
            text="",  # Not stored, just for reference
            section_name="content",
            title=metadata.title,
            authors=metadata.authors,
            year=metadata.year,
            doi=metadata.doi,
            project_tag=metadata.project_tag,
            research_area=metadata.research_area,
            file_name=metadata.file_name,
        )

        # Use the existing paragraph-aware chunking
        return self._create_contextual_fine_chunks(
            text=text,
            section_name="content",  # Generic section name
            parent_chunk=parent_chunk,
            metadata=metadata,
            start_idx=start_idx,
        )

    def _truncate_at_sentence(self, text: str, max_tokens: int) -> str:
        """Truncate text at a sentence boundary, not mid-sentence.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed

        Returns:
            Truncated text ending at a sentence boundary
        """
        tokens = self.tokenizer.encode(text)

        if len(tokens) <= max_tokens:
            return text

        # Decode truncated tokens
        truncated = self.tokenizer.decode(tokens[:max_tokens])

        # Find last sentence boundary
        last_period = truncated.rfind('. ')
        last_question = truncated.rfind('? ')
        last_exclaim = truncated.rfind('! ')

        last_boundary = max(last_period, last_question, last_exclaim)

        if last_boundary > len(truncated) * 0.5:  # At least 50% of content
            return truncated[:last_boundary + 1]

        return truncated

    # Legacy method for backwards compatibility
    def _create_fine_chunks(
        self,
        text: str,
        parent_chunk: Chunk,
        metadata: PaperMetadata,
        start_idx: int,
    ) -> List[Chunk]:
        """Create fine chunks - now delegates to contextual version."""
        return self._create_contextual_fine_chunks(
            text=text,
            section_name=parent_chunk.section_name,
            parent_chunk=parent_chunk,
            metadata=metadata,
            start_idx=start_idx,
        )
