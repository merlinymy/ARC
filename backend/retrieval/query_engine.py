"""Full query pipeline with classification, expansion, and retrieval.

Orchestrates the entire retrieval and answer generation workflow:
1. Query classification (determine query type)
2. Query expansion (add domain synonyms)
3. Embedding and vector search
4. Reranking
5. Parent chunk expansion
6. Answer generation with citations
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from anthropic import Anthropic

from .embedder import VoyageEmbedder
from .reranker import CohereReranker
from .qdrant_store import QdrantStore
from .query_classifier import QueryClassifier, QueryType, QueryClassification, RETRIEVAL_STRATEGIES
from .query_expander import QueryExpander

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result from the query pipeline."""
    query: str
    expanded_query: str
    query_type: QueryType
    classification: QueryClassification
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_count: int
    reranked_count: int


# Query-type-specific answer generation prompts
ANSWER_PROMPTS = {
    QueryType.FACTUAL: """You are a research assistant answering factual questions about scientific literature.

Based on the retrieved sources, provide a direct, accurate answer to the question.
Include specific values, definitions, or mechanisms when available.
Cite sources using [Source N] format.

Question: {query}

Retrieved Sources:
{sources}

Provide a clear, factual answer:""",

    QueryType.FRAMING: """You are a research writing assistant helping frame and position research for publication.

Based on how similar concepts are presented in the literature, help the user understand how to frame their research.
Focus on rhetorical strategies, positioning language, and how other authors have approached similar framing challenges.
Cite sources using [Source N] format.

Question: {query}

Retrieved Sources:
{sources}

Provide actionable framing advice based on the literature:""",

    QueryType.METHODS: """You are a research methods expert helping with technical writing.

Based on the retrieved methods sections, provide technical guidance on protocols and procedures.
Include specific details like reagents, conditions, and equipment when available.
Cite sources using [Source N] format.

Question: {query}

Retrieved Sources:
{sources}

Provide detailed technical guidance:""",

    QueryType.SUMMARY: """You are a research assistant summarizing scientific literature.

Based on the retrieved content, provide a structured summary of the key findings.
Include major results, conclusions, and implications.
Cite sources using [Source N] format.

Question: {query}

Retrieved Sources:
{sources}

Provide a comprehensive summary:""",

    QueryType.COMPARATIVE: """You are a research analyst comparing approaches across the literature.

Based on the retrieved sources, compare and contrast the different approaches, methods, or findings.
Highlight similarities, differences, and trade-offs.
Cite sources using [Source N] format.

Question: {query}

Retrieved Sources:
{sources}

Provide a balanced comparative analysis:""",

    QueryType.NOVELTY: """You are a research strategist assessing novelty and contribution.

Based on the retrieved literature, help assess what aspects might be novel or defensible as contributions.
Consider what has been done before and what gaps exist.
Cite sources using [Source N] format.

Question: {query}

Retrieved Sources:
{sources}

Provide analysis of novelty and defensible claims:""",

    QueryType.LIMITATIONS: """You are a research writing assistant helping discuss limitations.

Based on how limitations are discussed in similar literature, help frame constraints and caveats.
Focus on balanced, honest presentation of limitations while maintaining scientific credibility.
Cite sources using [Source N] format.

Question: {query}

Retrieved Sources:
{sources}

Provide guidance on discussing limitations:""",

    QueryType.GENERAL: """You are a research assistant helping with questions about scientific literature.

Based on the retrieved sources, provide a helpful and comprehensive answer to the question.
Draw on relevant information from the sources and cite them using [Source N] format.

Question: {query}

Retrieved Sources:
{sources}

Provide a helpful answer:""",
}


class QueryEngine:
    """Full query pipeline with classification, retrieval, and generation."""

    def __init__(
        self,
        embedder: VoyageEmbedder,
        reranker: CohereReranker,
        store: QdrantStore,
        anthropic_client: Anthropic,
        claude_model: str = "claude-3-5-sonnet-20241022",
        enable_classification: bool = True,
        enable_expansion: bool = True,
    ):
        """Initialize query engine.

        Args:
            embedder: Voyage embedder instance
            reranker: Cohere reranker instance
            store: Qdrant store instance
            anthropic_client: Anthropic client for answer generation
            claude_model: Claude model for generation
            enable_classification: Whether to classify queries
            enable_expansion: Whether to expand queries with synonyms
        """
        self.embedder = embedder
        self.reranker = reranker
        self.store = store
        self.anthropic = anthropic_client
        self.claude_model = claude_model
        self.enable_classification = enable_classification
        self.enable_expansion = enable_expansion

        # Initialize components
        self.classifier = QueryClassifier(anthropic_client) if enable_classification else None
        self.expander = QueryExpander() if enable_expansion else None

        logger.info("Initialized QueryEngine")

    def query(self, query: str, paper_id: Optional[str] = None) -> QueryResult:
        """Execute the full query pipeline.

        Args:
            query: User query
            paper_id: Optional paper ID to limit search to

        Returns:
            QueryResult with answer and sources
        """
        # Step 1: Classify query
        if self.enable_classification and self.classifier:
            classification = self.classifier.classify(query)
            query_type = classification.query_type
        else:
            # Default classification
            classification = QueryClassification(
                query_type=QueryType.FACTUAL,
                confidence=0.5,
                entities=[],
                needs_cross_corpus=False,
                suggested_chunk_types=["fine", "table", "caption"],
                suggested_top_k=50,
                reasoning="Classification disabled",
            )
            query_type = QueryType.FACTUAL

        logger.info(f"Query classified as: {query_type.value}")

        # Step 2: Expand query with synonyms
        expanded_query = query
        if self.enable_expansion and self.expander:
            expanded_query, added_terms = self.expander.expand_query(query)
            if added_terms:
                logger.info(f"Query expanded with: {added_terms}")

        # Step 3: Get retrieval strategy
        strategy = RETRIEVAL_STRATEGIES[query_type]
        chunk_types = strategy["chunk_types"]
        top_k = strategy["top_k"]
        rerank_top_n = strategy["rerank_top_n"]
        max_per_paper = strategy.get("max_per_paper", 3)
        section_filter = strategy.get("section_filter")

        # Step 4: Embed query
        query_embedding = self.embedder.embed_query(expanded_query)

        # Step 5: Search by strategy
        if paper_id:
            # Single paper search
            results = self.store.search(
                query_embedding=query_embedding,
                limit=top_k,
                chunk_types=chunk_types,
                paper_ids=[paper_id],
            )
        else:
            # Multi-paper search
            results = self.store.search_by_strategy(
                query_embedding=query_embedding,
                chunk_types=chunk_types,
                top_k=top_k,
                section_filter=section_filter,
            )

        retrieval_count = len(results)
        logger.info(f"Retrieved {retrieval_count} chunks")

        # Step 6: Rerank
        if results:
            reranked = self.reranker.rerank_with_metadata(
                query=query,  # Use original query for reranking
                documents=results,
                top_n=rerank_top_n,
                max_per_paper=max_per_paper,
            )
        else:
            reranked = []

        reranked_count = len(reranked)
        logger.info(f"Reranked to {reranked_count} chunks")

        # Step 7: Expand fine chunks with parent context
        expanded_sources = self._expand_fine_chunks(reranked)

        # Step 8: Generate answer
        answer = self._generate_answer(
            query=query,
            query_type=query_type,
            sources=expanded_sources,
        )

        return QueryResult(
            query=query,
            expanded_query=expanded_query,
            query_type=query_type,
            classification=classification,
            answer=answer,
            sources=expanded_sources,
            retrieval_count=retrieval_count,
            reranked_count=reranked_count,
        )

    def _expand_fine_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Expand fine chunks by fetching their parent section for context."""
        expanded = []

        for chunk in chunks:
            if chunk.get('chunk_type') == 'fine' and chunk.get('parent_chunk_id'):
                parent = self.store.get_parent_chunk(chunk)
                if parent:
                    chunk['parent_context'] = parent.get('text', '')[:500]  # First 500 chars
            expanded.append(chunk)

        return expanded

    def _generate_answer(
        self,
        query: str,
        query_type: QueryType,
        sources: List[Dict[str, Any]],
    ) -> str:
        """Generate answer using Claude with query-type-specific prompt."""
        if not sources:
            return "I couldn't find relevant information in the literature to answer this question."

        # Format sources
        sources_text = self._format_sources(sources)

        # Get query-type-specific prompt
        prompt_template = ANSWER_PROMPTS.get(query_type, ANSWER_PROMPTS[QueryType.FACTUAL])
        prompt = prompt_template.format(query=query, sources=sources_text)

        try:
            response = self.anthropic.messages.create(
                model=self.claude_model,
                max_tokens=2048,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Error generating answer: {str(e)}"

    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Format sources for the prompt."""
        formatted = []

        for i, source in enumerate(sources, 1):
            title = source.get('title', 'Unknown Title')
            text = source.get('text', '')[:1000]  # Limit text length
            chunk_type = source.get('chunk_type', 'unknown')
            section = source.get('section_name', '')
            paper_id = source.get('paper_id', '')

            source_str = f"[Source {i}] ({chunk_type}"
            if section:
                source_str += f", {section}"
            source_str += f")\nTitle: {title}\nPaper ID: {paper_id}\n{text}\n"

            # Add parent context if available
            if source.get('parent_context'):
                source_str += f"\n[Parent context]: {source['parent_context']}\n"

            formatted.append(source_str)

        return "\n---\n".join(formatted)

    def search_only(
        self,
        query: str,
        top_k: int = 20,
        chunk_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search without answer generation (for evaluation)."""
        # Expand query
        expanded_query = query
        if self.enable_expansion and self.expander:
            expanded_query, _ = self.expander.expand_query(query)

        # Embed and search
        query_embedding = self.embedder.embed_query(expanded_query)
        results = self.store.search(
            query_embedding=query_embedding,
            limit=top_k,
            chunk_types=chunk_types,
        )

        # Rerank
        if results:
            results = self.reranker.rerank(query, results, top_n=top_k)

        return results
