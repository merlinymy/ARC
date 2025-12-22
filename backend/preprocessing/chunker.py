"""Multi-type chunking logic for research papers.

Creates 6 types of chunks from extracted PDF content:
1. ABSTRACT - Full abstract as single chunk
2. SECTION - Logical sections (Introduction, Methods, etc.)
3. FINE - 500-token overlapping chunks from sections
4. FULL - Mean-pooled embedding of all chunks (at index time)
5. CAPTION - Figure and table captions
6. TABLE - Extracted table content

Each chunk type serves different query types:
- FACTUAL queries → FINE, TABLE, CAPTION
- FRAMING queries → ABSTRACT, SECTION
- METHODS queries → SECTION (methods), FINE
"""

import logging
from typing import List, Optional
import tiktoken

from .models import Chunk, ChunkType, PaperMetadata
from .section_detector import SectionDetector, Section

logger = logging.getLogger(__name__)


class PaperChunker:
    """Multi-type chunker for research papers."""

    def __init__(
        self,
        abstract_max_tokens: int = 300,
        section_max_tokens: int = 2000,
        fine_chunk_tokens: int = 500,
        fine_chunk_overlap: int = 128,
    ):
        """Initialize chunker.

        Args:
            abstract_max_tokens: Max tokens for abstract chunks
            section_max_tokens: Max tokens for section chunks
            fine_chunk_tokens: Target tokens for fine chunks
            fine_chunk_overlap: Overlap between fine chunks
        """
        self.abstract_max_tokens = abstract_max_tokens
        self.section_max_tokens = section_max_tokens
        self.fine_chunk_tokens = fine_chunk_tokens
        self.fine_chunk_overlap = fine_chunk_overlap

        self.section_detector = SectionDetector()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

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
                project_tag=metadata.project_tag,
                research_area=metadata.research_area,
                token_count=self.count_tokens(abstract),
            ))
            chunk_idx += 1

        # 2. Section chunks
        sections = self.section_detector.detect_sections(text)
        section_chunks = []

        for section in sections:
            # Skip very short sections
            if len(section.text) < 100:
                continue

            # Skip references/acknowledgments for retrieval
            if section.normalized_name in ["references", "acknowledgments", "abbreviations"]:
                continue

            section_chunk = Chunk(
                chunk_id=f"{metadata.paper_id}_section_{chunk_idx}",
                paper_id=metadata.paper_id,
                chunk_type=ChunkType.SECTION,
                text=section.text[:self.section_max_tokens * 4],
                section_name=section.normalized_name,
                title=metadata.title,
                authors=metadata.authors,
                year=metadata.year,
                project_tag=metadata.project_tag,
                research_area=metadata.research_area,
                token_count=self.count_tokens(section.text),
            )
            chunks.append(section_chunk)
            section_chunks.append(section_chunk)
            chunk_idx += 1

        # 3. Fine chunks (from section chunks)
        for section_chunk in section_chunks:
            fine_chunks = self._create_fine_chunks(
                text=section_chunk.text,
                parent_chunk=section_chunk,
                metadata=metadata,
                start_idx=chunk_idx,
            )
            chunks.extend(fine_chunks)
            chunk_idx += len(fine_chunks)

        # 4. Caption chunks
        if captions:
            for i, caption in enumerate(captions):
                if len(caption) < 20:  # Skip very short captions
                    continue

                chunks.append(Chunk(
                    chunk_id=f"{metadata.paper_id}_caption_{i}",
                    paper_id=metadata.paper_id,
                    chunk_type=ChunkType.CAPTION,
                    text=caption,
                    figure_id=f"figure_{i}",
                    title=metadata.title,
                    authors=metadata.authors,
                    year=metadata.year,
                    project_tag=metadata.project_tag,
                    research_area=metadata.research_area,
                    token_count=self.count_tokens(caption),
                ))
                chunk_idx += 1

        # 5. Table chunks
        if tables:
            for i, table in enumerate(tables):
                if len(table) < 50:  # Skip very short tables
                    continue

                chunks.append(Chunk(
                    chunk_id=f"{metadata.paper_id}_table_{i}",
                    paper_id=metadata.paper_id,
                    chunk_type=ChunkType.TABLE,
                    text=table,
                    figure_id=f"table_{i}",
                    title=metadata.title,
                    authors=metadata.authors,
                    year=metadata.year,
                    project_tag=metadata.project_tag,
                    research_area=metadata.research_area,
                    token_count=self.count_tokens(table),
                ))
                chunk_idx += 1

        logger.info(
            f"Created {len(chunks)} chunks for {metadata.paper_id}: "
            f"{sum(1 for c in chunks if c.chunk_type == ChunkType.ABSTRACT)} abstract, "
            f"{sum(1 for c in chunks if c.chunk_type == ChunkType.SECTION)} section, "
            f"{sum(1 for c in chunks if c.chunk_type == ChunkType.FINE)} fine, "
            f"{sum(1 for c in chunks if c.chunk_type == ChunkType.CAPTION)} caption, "
            f"{sum(1 for c in chunks if c.chunk_type == ChunkType.TABLE)} table"
        )

        return chunks

    def _create_fine_chunks(
        self,
        text: str,
        parent_chunk: Chunk,
        metadata: PaperMetadata,
        start_idx: int,
    ) -> List[Chunk]:
        """Create overlapping fine-grained chunks from section text.

        Args:
            text: Section text to chunk
            parent_chunk: Parent section chunk
            metadata: Paper metadata
            start_idx: Starting chunk index

        Returns:
            List of fine-grained Chunk objects
        """
        chunks = []

        # Tokenize
        tokens = self.tokenizer.encode(text)

        if len(tokens) <= self.fine_chunk_tokens:
            # Text is small enough, create single fine chunk
            chunks.append(Chunk(
                chunk_id=f"{metadata.paper_id}_fine_{start_idx}",
                paper_id=metadata.paper_id,
                chunk_type=ChunkType.FINE,
                text=text,
                section_name=parent_chunk.section_name,
                parent_chunk_id=parent_chunk.chunk_id,
                title=metadata.title,
                authors=metadata.authors,
                year=metadata.year,
                project_tag=metadata.project_tag,
                research_area=metadata.research_area,
                token_count=len(tokens),
            ))
            return chunks

        # Create overlapping chunks
        step = self.fine_chunk_tokens - self.fine_chunk_overlap
        idx = start_idx

        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + self.fine_chunk_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunks.append(Chunk(
                chunk_id=f"{metadata.paper_id}_fine_{idx}",
                paper_id=metadata.paper_id,
                chunk_type=ChunkType.FINE,
                text=chunk_text,
                section_name=parent_chunk.section_name,
                parent_chunk_id=parent_chunk.chunk_id,
                title=metadata.title,
                authors=metadata.authors,
                year=metadata.year,
                project_tag=metadata.project_tag,
                research_area=metadata.research_area,
                token_count=len(chunk_tokens),
            ))
            idx += 1

            # Stop if we've covered all tokens
            if i + self.fine_chunk_tokens >= len(tokens):
                break

        return chunks
