"""Full query pipeline with classification, expansion, and retrieval.

Orchestrates the entire retrieval and answer generation workflow:
1. Query rewriting and spelling correction
2. Query classification (determine query type)
3. Query expansion (add domain synonyms)
4. Embedding (with optional HyDE)
5. Hybrid vector search (dense + sparse)
6. Reranking with entity boosting
7. Parent chunk expansion
8. Answer generation with citations
9. Citation verification
10. Conversation memory for follow-ups

All results are cached for performance with automatic invalidation.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

from anthropic import Anthropic, RateLimitError, APIStatusError

from .embedder import VoyageEmbedder
from .reranker import CohereReranker, RerankResult
from .qdrant_store import QdrantStore
from .query_classifier import QueryClassifier, QueryType, QueryClassification, MultiQueryClassification, RETRIEVAL_STRATEGIES
from .query_expander import QueryExpander

# New imports for advanced features
from .cache import RAGCache
from .hyde import HyDE, HyDEEmbedder
from .query_rewriter import QueryRewriter
from .entity_extractor import EntityExtractor, LLMEntityExtractor
from .citation_verifier import CitationVerifier
from .conversation_memory import ConversationMemory

logger = logging.getLogger(__name__)


# Retry configuration for API calls
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0  # seconds
MAX_RETRY_DELAY = 30.0  # seconds


def retry_with_exponential_backoff(
    func,
    max_retries: int = MAX_RETRIES,
    initial_delay: float = INITIAL_RETRY_DELAY,
    max_delay: float = MAX_RETRY_DELAY,
):
    """Execute a function with exponential backoff retry on rate limit errors.

    Args:
        func: Function to execute (should be a callable)
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds

    Returns:
        Result of the function call

    Raises:
        The last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except RateLimitError as e:
            last_exception = e
            if attempt == max_retries:
                logger.error(f"Rate limit exceeded after {max_retries} retries")
                raise

            # Check for retry-after header
            retry_after = getattr(e, 'retry_after', None)
            if retry_after:
                delay = min(float(retry_after), max_delay)
            else:
                delay = min(delay * 2, max_delay)

            logger.warning(f"Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)

        except APIStatusError as e:
            # Retry on 5xx errors (server errors)
            if e.status_code >= 500:
                last_exception = e
                if attempt == max_retries:
                    raise

                delay = min(delay * 2, max_delay)
                logger.warning(f"Server error {e.status_code}, retrying in {delay:.1f}s")
                time.sleep(delay)
            else:
                raise

    raise last_exception


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
    # New fields for advanced features
    rewritten_query: str = ""
    used_hyde: bool = False
    cache_hit: bool = False
    citation_verified: bool = False
    entities_extracted: List[str] = field(default_factory=list)
    # Pipeline warnings (e.g., rate limits, degraded service)
    warnings: List[str] = field(default_factory=list)


# Query-type-specific system prompts (instructions only, no query/sources)
SYSTEM_PROMPTS = {
    QueryType.FACTUAL: """You are a research assistant answering factual questions about scientific literature.

Based on the retrieved sources, provide a direct, accurate answer to the question.
Include specific values, definitions, or mechanisms when available.
Cite sources using [Source N] format.""",

    QueryType.FRAMING: """You are a research writing strategist helping position research for publication.

Based on the retrieved literature, provide STRATEGIC ADVICE on how to frame and position the research.
Focus on:
- Rhetorical strategies and positioning language
- How to articulate the unique value proposition
- Key differentiating factors to emphasize
- Language patterns from successful papers

IMPORTANT: End your response with a "## Recommended Positioning Framework" section that provides a concise, actionable strategy the user can directly apply to their writing.

Cite sources using [Source N] format.""",

    QueryType.METHODS: """You are a research methods expert helping with technical writing.

Based on the retrieved methods sections, provide technical guidance on protocols and procedures.
Include specific details like reagents, conditions, and equipment when available.
Cite sources using [Source N] format.""",

    QueryType.SUMMARY: """You are a research assistant summarizing scientific literature.

Based on the retrieved content, provide a structured summary of the key findings.
Include major results, conclusions, and implications.
Cite sources using [Source N] format.""",

    QueryType.COMPARATIVE: """You are a research analyst comparing approaches across the literature.

Based on the retrieved sources, compare and contrast the different approaches, methods, or findings.
Highlight similarities, differences, and trade-offs.
Cite sources using [Source N] format.""",

    QueryType.NOVELTY: """You are a research strategist assessing novelty and contribution.

Based on the retrieved literature, help assess what aspects might be novel or defensible as contributions.
Consider what has been done before and what gaps exist.
Cite sources using [Source N] format.""",

    QueryType.LIMITATIONS: """You are a research writing assistant helping discuss limitations.

Based on how limitations are discussed in similar literature, help frame constraints and caveats.
Focus on balanced, honest presentation of limitations while maintaining scientific credibility.
Cite sources using [Source N] format.""",

    QueryType.GENERAL: """You are a research assistant helping with questions about scientific literature.

Based on the retrieved sources, provide a helpful and comprehensive answer to the question.
Draw on relevant information from the sources and cite them using [Source N] format.""",
}


class QueryEngine:
    """Full query pipeline with classification, retrieval, and generation."""

    def __init__(
        self,
        embedder: VoyageEmbedder,
        reranker: CohereReranker,
        store: QdrantStore,
        anthropic_client: Anthropic,
        claude_model: str = "claude-opus-4-5-20251101",
        claude_model_fast: str = "claude-3-haiku-20240307",
        claude_model_classifier: str = "claude-sonnet-4-20250514",
        enable_classification: bool = True,
        enable_expansion: bool = True,
        # New options for advanced features
        enable_caching: bool = True,
        enable_hyde: bool = False,
        enable_query_rewriting: bool = True,
        enable_entity_extraction: bool = True,
        enable_citation_verification: bool = False,
        enable_conversation_memory: bool = True,
        enable_hybrid_search: bool = False,
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
            enable_caching: Whether to cache embeddings and results
            enable_hyde: Whether to use HyDE for query embedding
            enable_query_rewriting: Whether to rewrite/correct queries
            enable_entity_extraction: Whether to extract entities for boosting
            enable_citation_verification: Whether to verify LLM citations
            enable_conversation_memory: Whether to track conversation context
            enable_hybrid_search: Whether to use hybrid (dense+sparse) search
        """
        self.embedder = embedder
        self.reranker = reranker
        self.store = store
        self.anthropic = anthropic_client
        self.claude_model = claude_model
        self.enable_classification = enable_classification
        self.enable_expansion = enable_expansion
        self.enable_hybrid_search = enable_hybrid_search

        # Initialize core components
        self.classifier = QueryClassifier(
            anthropic_client,
            model=claude_model_classifier,
        ) if enable_classification else None
        self.expander = QueryExpander() if enable_expansion else None

        # Initialize advanced components
        self.cache = RAGCache() if enable_caching else None
        if enable_hyde:
            hyde = HyDE(anthropic_client=anthropic_client, model=claude_model_fast)
            self.hyde_embedder = HyDEEmbedder(
                hyde=hyde,
                embedder=embedder,
                cache=self.cache,
            )
        else:
            self.hyde_embedder = None
        self.query_rewriter = QueryRewriter(
            anthropic_client=anthropic_client,
            model=claude_model_fast,
            enable_llm_rewrite=False,  # Use rule-based only by default
        ) if enable_query_rewriting else None
        self.entity_extractor = LLMEntityExtractor(
            anthropic_client=anthropic_client,
            model=claude_model_fast,
        ) if enable_entity_extraction else None
        self.citation_verifier = CitationVerifier(
            anthropic_client=anthropic_client,
            model=claude_model_fast,
        ) if enable_citation_verification else None
        self.conversation_memory = ConversationMemory() if enable_conversation_memory else None

        logger.info(
            f"Initialized QueryEngine (cache={enable_caching}, hyde={enable_hyde}, "
            f"rewrite={enable_query_rewriting}, entities={enable_entity_extraction}, "
            f"citations={enable_citation_verification}, memory={enable_conversation_memory})"
        )

    def query(
        self,
        query: str,
        paper_ids: Optional[List[str]] = None,
        max_chunks_per_paper: Optional[int] = None,
        progress_callback: Optional[callable] = None,
        query_type_override: Optional[str] = None,
        enable_hyde_override: Optional[bool] = None,
        enable_expansion_override: Optional[bool] = None,
        enable_citation_check_override: Optional[bool] = None,
    ) -> QueryResult:
        """Execute the full query pipeline.

        Args:
            query: User query
            paper_ids: Optional list of paper IDs to limit search to
            max_chunks_per_paper: Optional user-specified max chunks per paper (None = auto)
            progress_callback: Optional callback(step_name, step_data) for real-time progress
            query_type_override: Optional query type override (skips classification if provided)
            enable_hyde_override: Optional override for HyDE (None = use system default)
            enable_expansion_override: Optional override for query expansion (None = use system default)
            enable_citation_check_override: Optional override for citation verification (None = use system default)

        Returns:
            QueryResult with answer and sources
        """
        def emit(step: str, data: dict = None):
            """Emit progress if callback is provided."""
            if progress_callback:
                progress_callback(step, data or {})

        cache_hit = False
        used_hyde = False
        rewritten_query = query
        entities_extracted = []

        # Determine effective settings (overrides take precedence over system defaults)
        use_hyde = enable_hyde_override if enable_hyde_override is not None else (self.hyde_embedder is not None)
        use_expansion = enable_expansion_override if enable_expansion_override is not None else self.enable_expansion
        use_citation_check = enable_citation_check_override if enable_citation_check_override is not None else (self.citation_verifier is not None)

        # Step -1: Check for cache invalidation (if new papers were indexed)
        if self.cache:
            recently_indexed = self.store.get_recently_upserted_papers()
            if recently_indexed:
                self.cache.invalidate_if_needed(recently_indexed)
                logger.info(f"Invalidated cache due to {len(recently_indexed)} newly indexed papers")

        # Step 0: Resolve references from conversation history
        if self.conversation_memory:
            resolved_query = self.conversation_memory.resolve_references(query)
            if resolved_query != query:
                logger.debug(f"Resolved references: '{query}' -> '{resolved_query}'")
                query = resolved_query

        # Step 1: Query rewriting and spelling correction
        emit("rewriting", {"status": "starting"})
        if self.query_rewriter:
            rewrite_result = self.query_rewriter.rewrite(query)
            rewritten_query = rewrite_result.rewritten
            if rewrite_result.corrections:
                logger.info(f"Corrected spelling: {rewrite_result.corrections}")
            if rewritten_query != query:
                logger.debug(f"Query rewritten: '{query}' -> '{rewritten_query}'")
                emit("rewriting", {"original": query, "rewritten": rewritten_query, "changed": True})
            else:
                # Ran but no changes - treat as skipped
                emit("rewriting", {"query": query, "changed": False, "skipped": True})
        else:
            emit("rewriting", {"query": query, "changed": False, "skipped": True})

        # Step 2: Entity extraction for later boosting
        emit("entities", {"status": "starting"})
        if self.entity_extractor:
            entities = self.entity_extractor.extract(rewritten_query)
            entities_extracted = entities.all_entities()
            if entities_extracted:
                logger.debug(f"Extracted entities: {entities_extracted[:5]}")
                emit("entities", {"found": entities_extracted})
            else:
                # Ran but found nothing - treat as skipped
                emit("entities", {"found": [], "skipped": True})
        else:
            emit("entities", {"found": [], "skipped": True})

        # Step 3: Classify query (or use override if provided)
        emit("classification", {"status": "starting"})
        if query_type_override:
            # User specified query type - skip classification
            try:
                query_type = QueryType(query_type_override)
            except ValueError:
                logger.warning(f"Invalid query_type_override '{query_type_override}', falling back to auto-detection")
                query_type = self._detect_targeted_query_type(rewritten_query)
            strategy = RETRIEVAL_STRATEGIES[query_type]
            classification = QueryClassification(
                query_type=query_type,
                confidence=1.0,
                entities=entities_extracted[:5],
                needs_cross_corpus=True,
                suggested_chunk_types=strategy["chunk_types"],
                suggested_top_k=strategy["top_k"],
                reasoning=f"User-specified query type: {query_type.value}",
            )
            logger.info(f"Using user-specified query type: {query_type.value}")
        elif self.enable_classification and self.classifier:
            # Full LLM classification
            classification = self.classifier.classify(rewritten_query)
            query_type = classification.query_type
        else:
            # Hybrid approach: targeted retrieval for methods/limitations, universal for rest
            query_type = self._detect_targeted_query_type(rewritten_query)
            strategy = RETRIEVAL_STRATEGIES[query_type]
            classification = QueryClassification(
                query_type=query_type,
                confidence=1.0,
                entities=entities_extracted[:5],
                needs_cross_corpus=True,
                suggested_chunk_types=strategy["chunk_types"],
                suggested_top_k=strategy["top_k"],
                reasoning=f"Hybrid retrieval - {query_type.value}",
            )

        logger.info(f"Query classified as: {query_type.value}")
        emit("classification", {
            "type": query_type.value,
            "confidence": classification.confidence,
            "reasoning": classification.reasoning,
            "chunk_types": classification.suggested_chunk_types,
        })

        # Step 4: Expand query with synonyms (respects override)
        emit("expansion", {"status": "starting"})
        expanded_query = rewritten_query
        added_terms = []
        if use_expansion and self.expander:
            expanded_query, added_terms = self.expander.expand_query(rewritten_query)
            if added_terms:
                logger.info(f"Query expanded with: {added_terms}")
                emit("expansion", {"added_terms": added_terms, "expanded": expanded_query})
            else:
                # Ran but no terms added - treat as skipped
                emit("expansion", {"added_terms": [], "expanded": expanded_query, "skipped": True})
        else:
            emit("expansion", {"added_terms": [], "expanded": expanded_query, "skipped": True})

        # Step 5: Get retrieval strategy
        strategy = RETRIEVAL_STRATEGIES[query_type]
        chunk_types = strategy["chunk_types"]
        top_k = strategy["top_k"]
        rerank_top_n = strategy["rerank_top_n"]
        max_per_paper = strategy.get("max_per_paper", 3)
        section_filter = strategy.get("section_filter")

        # User-specified max_chunks_per_paper takes priority
        if max_chunks_per_paper is not None:
            max_per_paper = max_chunks_per_paper
            # Also adjust rerank_top_n to ensure we have enough candidates
            rerank_top_n = max(rerank_top_n, max_per_paper + 5)
            logger.debug(f"User-specified max_chunks_per_paper={max_per_paper}")
        elif paper_ids:
            # Auto mode: Override max_per_paper when specific papers are selected
            # Users selecting specific papers want deep analysis, not diversity
            num_papers = len(paper_ids)
            if num_papers == 1:
                # Single paper focus - allow many chunks for comprehensive analysis
                max_per_paper = 25
                rerank_top_n = min(rerank_top_n * 2, 30)
            elif num_papers <= 3:
                # Few papers - allow more chunks per paper
                max_per_paper = 15
                rerank_top_n = min(rerank_top_n + 10, 30)
            else:
                # Multiple papers but still filtered - moderate increase
                max_per_paper = max(max_per_paper, 8)
            logger.debug(f"Auto mode, paper filter active ({num_papers} papers): max_per_paper={max_per_paper}, rerank_top_n={rerank_top_n}")

        # Step 6: Check cache for search results
        results = None
        if self.cache:
            results = self.cache.get_search_results(
                expanded_query, chunk_types, section_filter
            )
            if results:
                cache_hit = True
                logger.info("Cache hit for search results")

        if not results:
            # Step 7: Embed query (with optional HyDE, respects override)
            emit("hyde", {"status": "starting"})
            if use_hyde and self.hyde_embedder:
                query_embedding, _ = self.hyde_embedder.embed_query_with_hyde(
                    query=expanded_query,
                    query_type=query_type.value if query_type else None,
                )
                used_hyde = True
                emit("hyde", {"used": True})
            else:
                emit("hyde", {"used": False, "skipped": True})
                # Check embedding cache
                if self.cache:
                    query_embedding = self.cache.get_embedding(expanded_query)
                    if query_embedding:
                        logger.debug("Cache hit for embedding")
                    else:
                        query_embedding = self.embedder.embed_query(expanded_query)
                        self.cache.set_embedding(expanded_query, query_embedding)
                else:
                    query_embedding = self.embedder.embed_query(expanded_query)

            # Step 8: Search (hybrid or dense-only)
            if self.enable_hybrid_search and hasattr(self.store, 'hybrid_search'):
                # Hybrid search (dense + sparse)
                results = self.store.hybrid_search(
                    query=expanded_query,
                    query_embedding=query_embedding,
                    limit=top_k,
                    chunk_types=chunk_types,
                    section_names=section_filter,
                    paper_ids=paper_ids,
                )
            else:
                # Dense-only search
                results = self.store.search_by_strategy(
                    query_embedding=query_embedding,
                    chunk_types=chunk_types,
                    top_k=top_k,
                    section_filter=section_filter,
                    paper_ids=paper_ids,
                )

            # Cache search results
            if self.cache and results:
                self.cache.set_search_results(
                    expanded_query, results, chunk_types, section_filter
                )

        retrieval_count = len(results) if results else 0
        logger.info(f"Retrieved {retrieval_count} chunks")
        emit("retrieval", {"count": retrieval_count, "cache_hit": cache_hit})

        # Step 9: Boost results by entity overlap
        if self.entity_extractor and entities_extracted and results:
            for result in results:
                text = result.get('text', '')
                entity_score, _ = self.entity_extractor.score_chunk_relevance(rewritten_query, text)
                result['entity_boost'] = entity_score
                # Slightly boost score for entity matches
                result['score'] = result.get('score', 0) * (1 + 0.1 * entity_score)

        # Step 10: Rerank
        emit("reranking", {"status": "starting", "input_count": len(results) if results else 0})
        warnings: List[str] = []
        if results:
            rerank_result = self.reranker.rerank_with_metadata(
                query=query,  # Use original query for reranking
                documents=results,
                top_n=rerank_top_n,
                max_per_paper=max_per_paper,
            )
            reranked = rerank_result.documents
            if not rerank_result.success and rerank_result.error:
                warnings.append(rerank_result.error)
        else:
            reranked = []

        reranked_count = len(reranked)
        logger.info(f"Reranked to {reranked_count} chunks")
        emit("reranking", {
            "output_count": reranked_count,
            "success": not warnings,
            "error": warnings[0] if warnings else None
        })

        # Step 11: Expand fine chunks with parent context
        expanded_sources = self._expand_fine_chunks(reranked)

        # Step 12: Generate answer
        emit("generation", {"status": "starting", "source_count": len(expanded_sources)})
        answer = self._generate_answer(
            query=query,
            query_type=query_type,
            sources=expanded_sources,
        )
        emit("generation", {"status": "complete"})

        # Step 13: Verify citations (if enabled, respects override)
        emit("verification", {"status": "starting"})
        citation_verified = False
        if use_citation_check and self.citation_verifier and expanded_sources:
            verification = self.citation_verifier.verify_answer(answer, expanded_sources)
            citation_verified = verification.is_trustworthy
            if not citation_verified:
                logger.warning(f"Citation verification warnings: {verification.warnings}")
            emit("verification", {"verified": citation_verified, "warnings": verification.warnings if not citation_verified else []})
        else:
            emit("verification", {"skipped": True})

        # Step 14: Update conversation memory
        if self.conversation_memory:
            self.conversation_memory.add_user_message(query)
            self.conversation_memory.add_assistant_message(answer, sources=expanded_sources)

        return QueryResult(
            query=query,
            expanded_query=expanded_query,
            query_type=query_type,
            classification=classification,
            answer=answer,
            sources=expanded_sources,
            retrieval_count=retrieval_count,
            reranked_count=reranked_count,
            rewritten_query=rewritten_query,
            used_hyde=used_hyde,
            cache_hit=cache_hit,
            citation_verified=citation_verified,
            entities_extracted=entities_extracted,
            warnings=warnings,
        )

    def _detect_targeted_query_type(self, query: str) -> QueryType:
        """Lightweight detection for queries that benefit from targeted retrieval.

        Only detects METHODS and LIMITATIONS queries which benefit from section filtering.
        Everything else uses universal retrieval (GENERAL).
        """
        query_lower = query.lower()

        # METHODS: queries about technical procedures benefit from section filtering
        methods_signals = [
            "protocol", "procedure", "synthesized", "synthesis", "purified", "purification",
            "buffer", "concentration", "incubation", "temperature",
            "how was", "how were", "how is", "how are",
            "what method", "what protocol", "what buffer", "what concentration",
            "cell line", "assay", "measured", "performed",
        ]
        if any(signal in query_lower for signal in methods_signals):
            return QueryType.METHODS

        # LIMITATIONS: queries about constraints benefit from discussion/conclusion sections
        limitations_signals = [
            "limitation", "caveat", "constraint", "weakness",
            "drawback", "shortcoming", "without", "lacking",
        ]
        if any(signal in query_lower for signal in limitations_signals):
            return QueryType.LIMITATIONS

        # NOVELTY: queries about implications, significance, findings benefit from discussion
        novelty_signals = [
            "implication", "significance", "finding", "conclude", "conclusion",
            "novel", "unique", "important", "significance", "contribution",
            "what did they find", "what were the results", "what does this mean",
            "why is this important", "key insight", "main finding", "discovered",
            "demonstrate", "shown", "proved", "established",
        ]
        if any(signal in query_lower for signal in novelty_signals):
            return QueryType.NOVELTY

        # Everything else: universal retrieval
        return QueryType.GENERAL

    def _expand_fine_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Expand fine chunks by fetching their parent section for context.

        Uses batch retrieval for efficiency instead of individual lookups.
        """
        if not chunks:
            return chunks

        # Collect all parent chunk IDs needed
        parent_id_to_indices: Dict[str, List[int]] = {}
        for i, chunk in enumerate(chunks):
            if chunk.get('chunk_type') == 'fine' and chunk.get('parent_chunk_id'):
                parent_id = chunk['parent_chunk_id']
                if parent_id not in parent_id_to_indices:
                    parent_id_to_indices[parent_id] = []
                parent_id_to_indices[parent_id].append(i)

        # Batch retrieve all parent chunks at once
        if parent_id_to_indices:
            parent_ids = list(parent_id_to_indices.keys())
            parent_chunks = self.store.get_chunks_by_ids(parent_ids)

            # Create lookup by chunk_id
            parent_by_id = {p.get('_chunk_id'): p for p in parent_chunks if p}

            # Attach parent context to fine chunks
            for parent_id, indices in parent_id_to_indices.items():
                parent = parent_by_id.get(parent_id)
                if parent:
                    parent_text = parent.get('text', '')[:500]  # First 500 chars
                    for idx in indices:
                        chunks[idx]['parent_context'] = parent_text

        return chunks

    def _generate_answer(
        self,
        query: str,
        query_type: QueryType,
        sources: List[Dict[str, Any]],
    ) -> str:
        """Generate answer using Claude with query-type-specific prompt.

        Includes conversation history for multi-turn context and retry logic
        for rate limits and transient errors.
        """
        if not sources:
            return "I couldn't find relevant information in the literature to answer this question."

        # Format sources
        sources_text = self._format_sources(sources)

        # Get query-type-specific system prompt
        system_prompt = SYSTEM_PROMPTS.get(query_type, SYSTEM_PROMPTS[QueryType.FACTUAL])

        # Build messages array with conversation history
        messages = []

        # Add conversation history (previous turns only - current query not yet in memory)
        if self.conversation_memory:
            history = self.conversation_memory.get_chat_history(max_tokens=2000)
            messages.extend(history)

        # Add current query with retrieved sources
        current_message = f"""Question: {query}

Retrieved Sources:
{sources_text}

Please provide your answer based on the sources above."""
        messages.append({"role": "user", "content": current_message})

        def make_api_call():
            return self.anthropic.messages.create(
                model=self.claude_model,
                max_tokens=2048,
                temperature=0.3,
                system=system_prompt,
                messages=messages,
            )

        try:
            response = retry_with_exponential_backoff(make_api_call)
            return response.content[0].text

        except (RateLimitError, APIStatusError) as e:
            logger.error(f"Answer generation failed after retries: {e}")
            return "Error generating answer (API unavailable): Please try again later."

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
            rerank_result = self.reranker.rerank(query, results, top_n=top_k)
            results = rerank_result.documents

        return results

    # Helper methods for advanced features

    def clear_conversation(self) -> None:
        """Clear conversation memory for a new session."""
        if self.conversation_memory:
            self.conversation_memory.clear()
            logger.info("Cleared conversation memory")

    def get_conversation_context(self) -> Optional[str]:
        """Get formatted conversation context for debugging."""
        if self.conversation_memory:
            return self.conversation_memory.format_context_for_prompt()
        return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.cache:
            return self.cache.stats()
        return {"enabled": False}

    def clear_cache(self) -> None:
        """Clear all caches."""
        if self.cache:
            self.cache.clear_all()
            logger.info("Cleared all caches")

    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        if self.conversation_memory:
            return self.conversation_memory.get_stats()
        return {"enabled": False}
