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
from collections import OrderedDict
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
from .citation_verifier import CitationVerifier, VerificationResult, StreamingCitationVerifier
from .conversation_memory import ConversationMemory
from .analytics import get_analytics_tracker, StepTimings, CitationResult

logger = logging.getLogger(__name__)


# Retry configuration for API calls
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0  # seconds
MAX_RETRY_DELAY = 30.0  # seconds

# Valid values for output_config.effort
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
DEFAULT_EFFORT = "high"

# Adaptive thinking is on by default on Opus 5; "summarized" keeps the reasoning
# visible so streaming clients see progress instead of a dead pause.
ADAPTIVE_THINKING = {"type": "adaptive", "display": "summarized"}

# Prompt caching. Breakpoints only create an entry when the prefix that precedes
# them is at least the model's minimum cacheable length; below it the API returns
# cache_creation_input_tokens == 0 with no error.  Measured with
# messages.count_tokens against the prompts in this module (Opus 5 floor = 512):
#
#   system prompt              concise        detailed
#   base only                   57- 108  tok   158- 295 tok   <- never cacheable
#   base + general knowledge   383- 434  tok   484- 621 tok   <- 6/8 detailed types
#   base + pdf upload          477- 528  tok   578- 715 tok
#   base + both                803- 854  tok   926-1041 tok   <- always cacheable
#
# So the breakpoint has to sit on the *last* system block, not after the stable
# base: no base prompt in this file reaches 512 tokens on its own.
CACHE_CONTROL_EPHEMERAL: Dict[str, str] = {"type": "ephemeral"}

# The API rejects a request carrying more than four cache_control markers.
MAX_CACHE_BREAKPOINTS = 4

# Character budget for the replayed conversation history (see
# ConversationMemory.get_chat_history). Shared with the cache-stability check so
# both use the same window.
HISTORY_TOKEN_BUDGET = 2000


def _output_config(effort: str) -> Dict[str, Any]:
    """Build the `output_config` argument, validating the requested effort level."""
    return {"effort": effort if effort in EFFORT_LEVELS else DEFAULT_EFFORT}


def _mark_cacheable(block: Dict[str, Any]) -> Dict[str, Any]:
    """Attach a prompt-cache breakpoint to a content block, in place."""
    block["cache_control"] = dict(CACHE_CONTROL_EPHEMERAL)
    return block


def _enforce_breakpoint_budget(
    system_blocks: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
) -> int:
    """Drop the earliest cache breakpoints until at most four remain.

    Three independent places add markers to an answer request - the system
    prompt, the replayed conversation history, and the uploaded PDF documents -
    so the limit is enforced here rather than assumed at each site. Earliest
    markers go first: a breakpoint placed later covers a longer prefix, so it is
    worth more per read.

    Returns:
        The number of breakpoints left on the request.
    """
    marked: List[Dict[str, Any]] = [b for b in system_blocks if "cache_control" in b]
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            marked.extend(
                b for b in content if isinstance(b, dict) and "cache_control" in b
            )

    excess = len(marked) - MAX_CACHE_BREAKPOINTS
    if excess > 0:
        for block in marked[:excess]:
            block.pop("cache_control", None)
        logger.warning(
            "Dropped %d cache breakpoint(s) to stay within the limit of %d",
            excess,
            MAX_CACHE_BREAKPOINTS,
        )
    return min(len(marked), MAX_CACHE_BREAKPOINTS)


def _log_cache_usage(usage: Any, label: str) -> Dict[str, int]:
    """Log the prompt-cache counters for one request and return them.

    Caching fails silently - a volatile prefix produces correct answers at full
    price with no error - so these counters are the only evidence it still works.
    """
    stats = {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }
    logger.info(
        "[CACHE] %s: uncached=%d written=%d read=%d",
        label,
        stats["input_tokens"],
        stats["cache_creation_input_tokens"],
        stats["cache_read_input_tokens"],
    )
    return stats


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
class CitationCheckResult:
    """A single citation verification result for API response."""
    citation_id: int
    claim: str
    confidence: float
    is_valid: bool
    explanation: str


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
    # Citation verification details for inline display
    citation_checks: List[CitationCheckResult] = field(default_factory=list)
    # Web search results (separate from RAG answer)
    web_search_answer: str = ""
    web_search_sources: List[Dict[str, str]] = field(default_factory=list)


# Query-type-specific system prompts with concise and detailed variants
# Each query type has two modes: "concise" (brief, focused) and "detailed" (comprehensive, in-depth)

SYSTEM_PROMPTS_CONCISE = {
    QueryType.FACTUAL: """You are a research assistant answering factual questions about scientific literature.

Based on the retrieved sources, provide a direct, accurate answer to the question.
Include specific values, definitions, or mechanisms when available.
Keep your response focused and concise.
Cite sources using [Source N] format.""",

    QueryType.FRAMING: """You are a research writing strategist helping position research for publication.

Based on the retrieved literature, provide STRATEGIC ADVICE on how to frame and position the research.
Focus on key differentiating factors and positioning language.

End with a brief "## Recommended Positioning Framework" section.
Cite sources using [Source N] format.""",

    QueryType.METHODS: """You are a research methods expert helping with technical writing.

Based on the retrieved methods sections, provide technical guidance on protocols and procedures.
Include key details like reagents, conditions, and equipment.
Keep your response focused on the essential steps.
Cite sources using [Source N] format.""",

    QueryType.SUMMARY: """You are a research assistant summarizing scientific literature.

Based on the retrieved content, provide a concise summary of the key findings.
Focus on major results and conclusions.
Cite sources using [Source N] format.""",

    QueryType.COMPARATIVE: """You are a research analyst comparing approaches across the literature.

Based on the retrieved sources, briefly compare and contrast the different approaches.
Highlight key similarities and differences.
Cite sources using [Source N] format.""",

    QueryType.NOVELTY: """You are a research strategist assessing novelty and contribution.

Based on the retrieved literature, briefly assess what aspects might be novel.
Identify key gaps and potential contributions.
Cite sources using [Source N] format.""",

    QueryType.LIMITATIONS: """You are a research writing assistant helping discuss limitations.

Based on how limitations are discussed in similar literature, briefly frame constraints and caveats.
Focus on the most important limitations.
Cite sources using [Source N] format.""",

    QueryType.GENERAL: """You are a research assistant helping with questions about scientific literature.

Based on the retrieved sources, provide a focused answer to the question.
Cite sources using [Source N] format.""",
}

SYSTEM_PROMPTS_DETAILED = {
    QueryType.FACTUAL: """You are a research assistant answering factual questions about scientific literature.

Based on the retrieved sources, provide a comprehensive, accurate answer to the question.
Include:
- Specific values, definitions, and mechanisms with full context
- Background information that helps understand the answer
- Multiple perspectives or values if sources differ
- Relevant caveats or conditions that affect the answer

Explain the significance and implications where relevant.
Cite sources using [Source N] format throughout your response.""",

    QueryType.FRAMING: """You are a research writing strategist helping position research for publication.

Based on the retrieved literature, provide COMPREHENSIVE STRATEGIC ADVICE on how to frame and position the research.

Cover these aspects in depth:
- Rhetorical strategies and positioning language used by successful papers
- How to articulate the unique value proposition with specific examples
- Key differentiating factors to emphasize and why they matter
- Language patterns and phrases from successful papers you can adapt
- Common pitfalls to avoid in framing
- How different journals/audiences might respond to different framings

IMPORTANT: End your response with a detailed "## Recommended Positioning Framework" section that provides:
1. Suggested narrative arc
2. Key claims to emphasize
3. Specific language recommendations
4. Positioning relative to existing literature

Cite sources using [Source N] format throughout your response.""",

    QueryType.METHODS: """You are a research methods expert helping with technical writing.

Based on the retrieved methods sections, provide comprehensive technical guidance on protocols and procedures.

Include detailed information on:
- Step-by-step protocols with all relevant parameters
- Specific reagents, concentrations, and preparation details
- Equipment specifications and settings
- Timing, temperatures, and critical conditions
- Controls and validation steps
- Common variations across different papers
- Tips for reproducibility and troubleshooting
- Quality control checkpoints

Explain the rationale behind key methodological choices where evident from the sources.
Cite sources using [Source N] format throughout your response.""",

    QueryType.SUMMARY: """You are a research assistant summarizing scientific literature.

Based on the retrieved content, provide a comprehensive, structured summary covering:

1. **Background & Context**: The research landscape and why this work matters
2. **Key Findings**: Major results with specific data points and statistics
3. **Methodology Highlights**: How key findings were obtained
4. **Implications**: What these findings mean for the field
5. **Connections**: How different findings relate to each other
6. **Open Questions**: What remains to be addressed

Organize the information logically and explain the significance of findings.
Cite sources using [Source N] format throughout your response.""",

    QueryType.COMPARATIVE: """You are a research analyst comparing approaches across the literature.

Based on the retrieved sources, provide a thorough comparison covering:

1. **Overview of Approaches**: Brief description of each approach/method being compared
2. **Key Similarities**: What the approaches share in common
3. **Important Differences**: Where they diverge and why
4. **Trade-offs**: Advantages and disadvantages of each approach
5. **Context-Dependent Recommendations**: When each approach might be preferred
6. **Performance Metrics**: Quantitative comparisons where available
7. **Practical Considerations**: Implementation complexity, resource requirements, etc.

Use tables or structured formats where helpful for clarity.
Cite sources using [Source N] format throughout your response.""",

    QueryType.NOVELTY: """You are a research strategist assessing novelty and contribution.

Based on the retrieved literature, provide a comprehensive assessment covering:

1. **Prior Art Summary**: What has been done before in this area
2. **Gap Analysis**: What hasn't been addressed or remains unresolved
3. **Potential Novel Contributions**: Aspects that could be claimed as new
4. **Strength of Novelty Claims**: How defensible each potential contribution is
5. **Differentiation Strategy**: How to position work relative to existing literature
6. **Risk Assessment**: Potential challenges to novelty claims
7. **Supporting Evidence**: What evidence from literature supports your assessment

Be specific about what has and hasn't been done, with citations.
Cite sources using [Source N] format throughout your response.""",

    QueryType.LIMITATIONS: """You are a research writing assistant helping discuss limitations.

Based on how limitations are discussed in similar literature, provide comprehensive guidance on:

1. **Common Limitations**: What limitations are typically acknowledged in this area
2. **How to Frame Each Limitation**: Language and approaches that maintain credibility
3. **Mitigation Strategies**: How papers address or contextualize their limitations
4. **Balancing Act**: How to be honest without undermining your work
5. **Field-Specific Conventions**: What's expected in this research area
6. **Reviewer Anticipation**: Limitations reviewers are likely to raise
7. **Future Work Connections**: How to turn limitations into future directions

Include specific examples of effective limitation discussions from the sources.
Cite sources using [Source N] format throughout your response.""",

    QueryType.GENERAL: """You are a research assistant helping with questions about scientific literature.

Based on the retrieved sources, provide a comprehensive and well-organized answer to the question.

Structure your response to:
- Address all aspects of the question thoroughly
- Provide relevant background context
- Include specific details, data points, and examples
- Explain connections between different pieces of information
- Note any important caveats or nuances
- Suggest related topics or follow-up questions if relevant

Draw on all relevant information from the sources.
Cite sources using [Source N] format throughout your response.""",
}

# General knowledge addendum for when enable_general_knowledge is True
GENERAL_KNOWLEDGE_ADDENDUM = """

---
IMPORTANT: General Knowledge Mode is ENABLED.

In addition to the retrieved sources, you may draw on your general scientific knowledge to provide a more complete answer. However, you MUST follow this structure:

**CRITICAL: ALWAYS start your response by answering based on the retrieved sources with [Source N] citations. NEVER skip the RAG citations.**

1. FIRST: Answer the question using ONLY the retrieved sources. Cite every claim with [Source N] format.
2. THEN: After fully addressing the question with source citations, you may add:

## Additional Context (General Knowledge)

In this section, clearly indicate that this information comes from your general training knowledge, not the uploaded papers. Use phrases like:
- "Based on general scientific knowledge..."
- "From broader literature (not in uploaded papers)..."
- "General background that may be relevant..."

This separation helps users distinguish between information from their specific papers vs. general knowledge."""

# PDF upload addendum for when full PDF documents are sent to Claude
PDF_UPLOAD_ADDENDUM = """

---
IMPORTANT: Full PDF Document Mode is ENABLED.

You have access to BOTH:
1. **Full PDF documents** of the selected papers (sent as complete documents)
2. **Retrieved source chunks** from RAG (pre-identified relevant sections)

How to use these resources together:

**CRITICAL: You MUST cite the retrieved sources using [Source N] format throughout your response.**

- The retrieved source chunks (RAG) are pre-identified relevant sections that directly address the query
- The full PDFs provide broader context, additional details, and sections not captured in the RAG chunks
- **Always start by thoroughly addressing the question using the retrieved source chunks with [Source N] citations**
- You may then draw on the full PDFs to provide additional context, figures, methods details, or related information not in the RAG chunks
- When referencing information from the PDFs that is NOT in the retrieved sources, clearly indicate this (e.g., "From the broader paper context..." or "As shown in the full document...")
- Cross-reference between RAG chunks and full PDFs to provide comprehensive, well-cited answers

This dual approach allows you to provide highly relevant cited information (from RAG) while having access to the complete paper context (from PDFs)."""

# Web search system prompt - used for the separate web search call
WEB_SEARCH_SYSTEM_PROMPT = """You are a helpful research assistant. Search the web for publicly available information related to the user's question. Focus on recent publications, news, educational resources, and general background information.

Provide factual information with source URLs. This is for educational and research purposes.

You may use markdown formatting (headers, bold, lists) to organize your response clearly."""

# Legacy alias for backward compatibility
SYSTEM_PROMPTS = SYSTEM_PROMPTS_CONCISE


def get_effective_prompt(
    query_type: QueryType,
    response_mode: str,
    custom_prompts: Optional[Dict[str, Any]] = None
) -> str:
    """Get the effective system prompt, preferring custom over default.

    Args:
        query_type: The type of query
        response_mode: 'concise' or 'detailed'
        custom_prompts: Optional dict of custom prompts from user preferences

    Returns:
        The effective system prompt to use
    """
    # Get default prompt
    if response_mode == "detailed":
        default = SYSTEM_PROMPTS_DETAILED.get(query_type, SYSTEM_PROMPTS_DETAILED[QueryType.FACTUAL])
    else:
        default = SYSTEM_PROMPTS_CONCISE.get(query_type, SYSTEM_PROMPTS_CONCISE[QueryType.FACTUAL])

    # Check for custom override
    if custom_prompts and response_mode in custom_prompts:
        mode_prompts = custom_prompts[response_mode]
        if query_type.value in mode_prompts:
            return mode_prompts[query_type.value]

    return default


def get_effective_addendum(
    addendum_type: str,
    custom_prompts: Optional[Dict[str, Any]] = None
) -> str:
    """Get the effective addendum prompt, preferring custom over default.

    Args:
        addendum_type: 'general_knowledge', 'web_search', or 'pdf_upload'
        custom_prompts: Optional dict of custom prompts from user preferences

    Returns:
        The effective addendum to use
    """
    # Default addendums
    defaults = {
        "general_knowledge": GENERAL_KNOWLEDGE_ADDENDUM,
        "web_search": WEB_SEARCH_SYSTEM_PROMPT,
        "pdf_upload": PDF_UPLOAD_ADDENDUM,
    }

    default = defaults.get(addendum_type, "")

    # Check for custom override
    if custom_prompts and "addendums" in custom_prompts:
        addendums = custom_prompts["addendums"]
        if addendum_type in addendums:
            return addendums[addendum_type]

    return default


class QueryEngine:
    """Full query pipeline with classification, retrieval, and generation."""

    # Bound on the number of per-conversation memories held by the process-wide engine
    MAX_CONVERSATION_MEMORIES = 32

    # Characters of the parent section sent alongside a matched fine chunk
    PARENT_CONTEXT_WINDOW = 2000

    def __init__(
        self,
        embedder: VoyageEmbedder,
        reranker: CohereReranker,
        store: QdrantStore,
        anthropic_client: Anthropic,
        claude_model: str = "claude-opus-5",
        claude_model_fast: str = "claude-haiku-4-5",
        claude_model_classifier: str = "claude-sonnet-5",
        claude_model_web_search: str = "claude-sonnet-5",
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
        pdf_service: Optional["PDFService"] = None,
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
        self.claude_model_classifier = claude_model_classifier
        self.claude_model_web_search = claude_model_web_search
        self.enable_classification = enable_classification
        self.enable_expansion = enable_expansion
        self.enable_hybrid_search = enable_hybrid_search
        self.pdf_service = pdf_service  # Optional PDF service for sending full PDFs to Claude

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
        self.enable_conversation_memory = enable_conversation_memory
        self._conversation_memories: "OrderedDict[str, ConversationMemory]" = OrderedDict()
        # Prompt-cache counters from the most recent answer call. Diagnostic only
        # (last writer wins under concurrency); the authoritative record is the
        # "[CACHE]" log line written by _log_cache_usage.
        self.last_answer_usage: Optional[Dict[str, int]] = None

        logger.info(
            f"Initialized QueryEngine (cache={enable_caching}, hyde={enable_hyde}, "
            f"rewrite={enable_query_rewriting}, entities={enable_entity_extraction}, "
            f"citations={enable_citation_verification}, memory={enable_conversation_memory})"
        )

    def _get_conversation_memory(
        self,
        conversation_id: Optional[str] = None,
    ) -> Optional[ConversationMemory]:
        """Get the ConversationMemory for a conversation, creating it on first use.

        The engine is a process-wide singleton, so memory is keyed by conversation_id
        to keep separate chat threads from contaminating each other. The keyed store is
        a bounded LRU; requests without a conversation_id share one anonymous slot.
        """
        if not self.enable_conversation_memory:
            return None

        key = conversation_id or "__anonymous__"
        if key in self._conversation_memories:
            self._conversation_memories.move_to_end(key)
        else:
            self._conversation_memories[key] = ConversationMemory()
            while len(self._conversation_memories) > self.MAX_CONVERSATION_MEMORIES:
                evicted, _ = self._conversation_memories.popitem(last=False)
                logger.debug(f"Evicted conversation memory for {evicted}")

        return self._conversation_memories[key]

    def query(
        self,
        query: str,
        paper_ids: Optional[List[str]] = None,
        max_chunks_per_paper: Optional[int] = None,
        top_k: Optional[int] = None,
        effort: Optional[str] = None,
        conversation_id: Optional[str] = None,
        progress_callback: Optional[callable] = None,
        query_type_override: Optional[str] = None,
        enable_hyde_override: Optional[bool] = None,
        enable_expansion_override: Optional[bool] = None,
        enable_citation_check_override: Optional[bool] = None,
        response_mode: str = "detailed",
        enable_general_knowledge: bool = True,
        enable_web_search: bool = False,
        enable_pdf_upload: bool = False,
        custom_prompts: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """Execute the full query pipeline.

        Args:
            query: User query
            paper_ids: Optional list of paper IDs to limit search to
            max_chunks_per_paper: Optional user-specified max chunks per paper (None = auto)
            top_k: Optional user-specified number of results to retrieve (None = use strategy default)
            effort: Optional answer depth - low|medium|high|xhigh|max (None = high)
            conversation_id: Optional conversation ID that scopes conversation memory
            progress_callback: Optional callback(step_name, step_data) for real-time progress
            query_type_override: Optional query type override (skips classification if provided)
            enable_hyde_override: Optional override for HyDE (None = use system default)
            custom_prompts: Optional dict of custom system prompts from user preferences
            enable_expansion_override: Optional override for query expansion (None = use system default)
            enable_citation_check_override: Optional override for citation verification (None = use system default)
            response_mode: "concise" for brief answers, "detailed" for comprehensive responses
            enable_general_knowledge: Whether to allow LLM to supplement with general knowledge
            enable_web_search: Whether to allow Claude to search the web for additional context

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
        entities_by_category = {}  # For analytics

        # Timing tracking for analytics
        timing_query_processing_start = time.perf_counter()
        timing_embedding_ms = 0.0
        timing_retrieval_ms = 0.0
        timing_reranking_ms = 0.0
        timing_generation_ms = 0.0
        verification_result = None

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
        conversation_memory = self._get_conversation_memory(conversation_id)
        if conversation_memory:
            resolved_query = conversation_memory.resolve_references(query)
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
            entities_by_category = entities.to_dict()  # For analytics
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

        # End query processing timing (steps 1-4)
        timing_query_processing_ms = (time.perf_counter() - timing_query_processing_start) * 1000

        # Step 5: Get retrieval strategy
        strategy = RETRIEVAL_STRATEGIES[query_type]
        chunk_types = strategy["chunk_types"]
        strategy_top_k = strategy["top_k"]
        rerank_top_n = strategy["rerank_top_n"]
        max_per_paper = strategy.get("max_per_paper", 3)
        section_filter = strategy.get("section_filter")

        # User-specified top_k controls the FINAL output count (rerank_top_n), not retrieval
        # Retrieval should cast a wider net to ensure good reranking candidates
        if top_k is not None:
            rerank_top_n = top_k
            # Ensure retrieval gets enough candidates for reranking (at least 2x the final count)
            effective_top_k = max(strategy_top_k, top_k * 2)
            logger.debug(f"User-specified top_k={top_k} -> rerank_top_n={rerank_top_n}, retrieval={effective_top_k}")
        else:
            effective_top_k = strategy_top_k

        # User-specified max_chunks_per_paper takes priority
        if max_chunks_per_paper is not None:
            max_per_paper = max_chunks_per_paper
            # Only adjust rerank_top_n if user didn't explicitly set top_k
            if top_k is None:
                rerank_top_n = max(rerank_top_n, max_per_paper + 5)
            logger.debug(f"User-specified max_chunks_per_paper={max_per_paper}")
        elif paper_ids:
            # Auto mode: Override max_per_paper when specific papers are selected
            # Users selecting specific papers want deep analysis, not diversity
            num_papers = len(paper_ids)
            if num_papers == 1:
                # Single paper focus - allow many chunks for comprehensive analysis
                max_per_paper = 25
                # Only adjust rerank_top_n if user didn't explicitly set top_k
                if top_k is None:
                    rerank_top_n = min(rerank_top_n * 2, 30)
            elif num_papers <= 3:
                # Few papers - allow more chunks per paper
                max_per_paper = 15
                if top_k is None:
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
            timing_embedding_start = time.perf_counter()
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
            timing_embedding_ms = (time.perf_counter() - timing_embedding_start) * 1000

            # Step 8: Search (hybrid or dense-only)
            timing_retrieval_start = time.perf_counter()
            if self.enable_hybrid_search and hasattr(self.store, 'hybrid_search'):
                # Hybrid search (dense + sparse)
                results = self.store.hybrid_search(
                    query=expanded_query,
                    query_embedding=query_embedding,
                    limit=effective_top_k,
                    chunk_types=chunk_types,
                    section_names=section_filter,
                    paper_ids=paper_ids,
                )
            else:
                # Dense-only search
                results = self.store.search_by_strategy(
                    query_embedding=query_embedding,
                    chunk_types=chunk_types,
                    top_k=effective_top_k,
                    section_filter=section_filter,
                    paper_ids=paper_ids,
                )
            timing_retrieval_ms = (time.perf_counter() - timing_retrieval_start) * 1000

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
        timing_reranking_start = time.perf_counter()
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
        timing_reranking_ms = (time.perf_counter() - timing_reranking_start) * 1000

        reranked_count = len(reranked)
        logger.info(f"Reranked to {reranked_count} chunks")
        emit("reranking", {
            "output_count": reranked_count,
            "success": not warnings,
            "error": warnings[0] if warnings else None
        })

        # Step 11: Expand fine chunks with parent context
        expanded_sources = self._expand_fine_chunks(reranked)

        # Step 12: Generate answer (with streaming if callback provided)
        emit("generation", {"status": "starting", "source_count": len(expanded_sources)})
        timing_generation_start = time.perf_counter()

        # Prepare streaming citation verification if enabled
        streaming_citation_checks: List[CitationCheckResult] = []
        streaming_verifier = None
        use_streaming_verification = (
            progress_callback and
            use_citation_check and
            self.citation_verifier and
            expanded_sources
        )

        if use_streaming_verification:
            # Create callback to emit citation verification results in real-time
            def on_citation_verified(check):
                check_result = CitationCheckResult(
                    citation_id=check.citation_id,
                    claim=check.claim,
                    confidence=check.confidence,
                    is_valid=check.is_valid,
                    explanation=check.explanation,
                )
                streaming_citation_checks.append(check_result)
                emit("citation_verified", {
                    "citation_id": check.citation_id,
                    "claim": check.claim,
                    "confidence": check.confidence,
                    "is_valid": check.is_valid,
                    "explanation": check.explanation,
                })

            # Format sources for verification (need 'text' and 'title' keys)
            verification_sources = [
                {
                    'text': s.get('text', ''),
                    'title': s.get('title', f'Source {i+1}'),
                }
                for i, s in enumerate(expanded_sources)
            ]

            streaming_verifier = StreamingCitationVerifier(
                verifier=self.citation_verifier,
                sources=verification_sources,
                on_citation_verified=on_citation_verified,
            )

        # Create stream callback that emits answer chunks AND processes for citations
        def answer_stream_callback(chunk: str):
            emit("answer_chunk", {"chunk": chunk})
            # Process chunk for real-time citation verification
            if streaming_verifier:
                streaming_verifier.process_chunk(chunk)

        # Use user-specified effort or fall back to the default depth
        effective_effort = effort if effort in EFFORT_LEVELS else DEFAULT_EFFORT

        # STEP 1: Generate RAG answer first
        logger.info(f"[QUERY_ENGINE] Starting RAG answer generation with effort={effective_effort}")
        answer = self._generate_answer(
            query=query,
            query_type=query_type,
            sources=expanded_sources,
            stream_callback=answer_stream_callback if progress_callback else None,
            response_mode=response_mode,
            enable_general_knowledge=enable_general_knowledge,
            enable_web_search=enable_web_search,
            progress_emitter=emit if progress_callback else None,
            effort=effective_effort,
            custom_prompts=custom_prompts,
            paper_ids=paper_ids,
            enable_pdf_upload=enable_pdf_upload,
            conversation_memory=conversation_memory,
        )
        timing_generation_ms = (time.perf_counter() - timing_generation_start) * 1000

        # Verify what we got back from _generate_answer
        import hashlib
        answer_hash = hashlib.md5(answer.encode()).hexdigest()
        logger.info(f"[QUERY_ENGINE] RAG answer generation completed in {timing_generation_ms:.2f}ms")
        logger.info(f"[QUERY_ENGINE] Returned answer length: {len(answer)} chars")
        logger.info(f"[QUERY_ENGINE] Returned answer MD5: {answer_hash}")
        logger.info(f"[QUERY_ENGINE] Returned answer preview: {answer[:200]}...")

        emit("generation", {"status": "complete"})
        emit("answer_complete", {"answer": answer})

        # STEP 2: Start web search immediately after RAG completes (same thread, streams properly)
        web_search_answer = ""
        web_search_sources = []

        if enable_web_search and enable_general_knowledge:
            logger.info("Starting web search after RAG completion")
            emit("web_search", {"status": "starting"})

            # Create progress callback for web search
            def web_search_progress(msg):
                emit("web_search_progress", {"message": msg})

            # Create stream callback for web search text chunks
            def web_search_stream(chunk):
                logger.info(f"[STREAMING] Emitting web_search_chunk: {len(chunk)} chars")
                emit("web_search_chunk", {"chunk": chunk})

            try:
                web_search_answer, web_search_sources = self._perform_web_search(
                    query=query,
                    stream_callback=web_search_stream if progress_callback else None,
                    progress_callback=web_search_progress,
                    custom_prompts=custom_prompts,
                )
                emit("web_search", {"status": "complete"})
                logger.info(f"Web search completed: {len(web_search_answer)} chars, {len(web_search_sources)} sources")
            except Exception as e:
                logger.error(f"Web search failed: {e}", exc_info=True)
                web_search_answer = f"*Web search failed: {str(e)}*"
                web_search_sources = []
                emit("web_search", {"status": "error", "error": str(e)})

        # Flush streaming verifier to catch any remaining citations
        if streaming_verifier:
            streaming_verifier.flush()

        # Final pass: verify any citations that weren't caught during streaming
        if use_streaming_verification and answer:
            # Extract all citation IDs from the final answer
            all_citation_ids = set()
            citation_matches = self.citation_verifier.extract_citations(answer)
            for source_id in citation_matches.keys():
                all_citation_ids.add(source_id)

            # Find which citations weren't verified during streaming
            verified_ids = {c.citation_id for c in streaming_citation_checks}
            missing_ids = all_citation_ids - verified_ids

            if missing_ids:
                logger.debug(f"Final pass: verifying {len(missing_ids)} missed citations: {missing_ids}")
                verification_sources = [
                    {
                        'text': s.get('text', ''),
                        'title': s.get('title', f'Source {i+1}'),
                    }
                    for i, s in enumerate(expanded_sources)
                ]
                for source_id in missing_ids:
                    # Get all claims for this citation from the full answer
                    claims = citation_matches.get(source_id, [])
                    for claim in claims:
                        check = self.citation_verifier.verify_single_citation(
                            claim=claim,
                            source_id=source_id,
                            sources=verification_sources,
                        )
                        if check:
                            check_result = CitationCheckResult(
                                citation_id=check.citation_id,
                                claim=check.claim,
                                confidence=check.confidence,
                                is_valid=check.is_valid,
                                explanation=check.explanation,
                            )
                            streaming_citation_checks.append(check_result)
                            emit("citation_verified", {
                                "citation_id": check.citation_id,
                                "claim": check.claim,
                                "confidence": check.confidence,
                                "is_valid": check.is_valid,
                                "explanation": check.explanation,
                            })

        # Step 13: Verify citations (if enabled, respects override)
        emit("verification", {"status": "starting"})
        citation_verified = False
        citation_checks: List[CitationCheckResult] = []

        if use_streaming_verification:
            # Use results from streaming verification
            citation_checks = streaming_citation_checks
            if citation_checks:
                overall_confidence = sum(c.confidence for c in citation_checks) / len(citation_checks)
                citation_verified = overall_confidence >= 0.7
                # Create a VerificationResult for analytics
                verification_result = VerificationResult(
                    total_citations=len(citation_checks),
                    valid_citations=sum(1 for c in citation_checks if c.is_valid),
                    invalid_citations=sum(1 for c in citation_checks if not c.is_valid),
                    checks=[],  # Original checks not needed for analytics
                    overall_confidence=overall_confidence,
                    warnings=[],
                )
            emit("verification", {"verified": citation_verified, "warnings": []})
        elif use_citation_check and self.citation_verifier and expanded_sources:
            # Non-streaming fallback: verify all citations after answer is complete
            verification = self.citation_verifier.verify_answer(answer, expanded_sources)
            citation_verified = verification.is_trustworthy
            verification_result = verification  # Store for analytics
            # Convert checks to API-friendly format and emit each one
            for check in verification.checks:
                check_result = CitationCheckResult(
                    citation_id=check.citation_id,
                    claim=check.claim,
                    confidence=check.confidence,
                    is_valid=check.is_valid,
                    explanation=check.explanation,
                )
                citation_checks.append(check_result)
                # Emit individual citation verification result
                emit("citation_verified", {
                    "citation_id": check.citation_id,
                    "claim": check.claim,
                    "confidence": check.confidence,
                    "is_valid": check.is_valid,
                    "explanation": check.explanation,
                })
            if not citation_verified:
                logger.warning(f"Citation verification warnings: {verification.warnings}")
            emit("verification", {"verified": citation_verified, "warnings": verification.warnings if not citation_verified else []})
        else:
            emit("verification", {"skipped": True})

        # Step 14: Update conversation memory
        if conversation_memory:
            conversation_memory.add_user_message(query)
            conversation_memory.add_assistant_message(answer, sources=expanded_sources)

        # Step 15: Record analytics
        try:
            analytics = get_analytics_tracker()
            step_timings = StepTimings(
                query_processing_ms=timing_query_processing_ms,
                embedding_ms=timing_embedding_ms,
                retrieval_ms=timing_retrieval_ms,
                reranking_ms=timing_reranking_ms,
                generation_ms=timing_generation_ms,
            )
            citation_analytics = None
            if verification_result and verification_result.checks:
                # Count citations by confidence threshold (mutually exclusive)
                # Verified: confidence >= 0.7 (high confidence)
                # Partial: 0.3 <= confidence < 0.7 (moderate confidence)
                # Failed: confidence < 0.3 (low confidence)
                verified_count = sum(1 for c in verification_result.checks if c.confidence >= 0.7)
                partial_count = sum(1 for c in verification_result.checks if 0.3 <= c.confidence < 0.7)
                failed_count = sum(1 for c in verification_result.checks if c.confidence < 0.3)

                citation_analytics = CitationResult(
                    overall_score=verification_result.overall_confidence,
                    total_citations=len(verification_result.checks),
                    valid_citations=verified_count,
                    partial_citations=partial_count,
                    invalid_citations=failed_count,
                )
            analytics.record_query(
                query_type=query_type.value,
                step_timings=step_timings,
                citation_result=citation_analytics,
                entities=entities_by_category if entities_by_category else None,
            )
        except Exception as e:
            logger.warning(f"Failed to record analytics: {e}")

        # Log the QueryResult being returned
        import hashlib
        result_answer_hash = hashlib.md5(answer.encode()).hexdigest()
        logger.info(f"[QUERY_ENGINE] Creating QueryResult object")
        logger.info(f"[QUERY_ENGINE] QueryResult.answer length: {len(answer)} chars, MD5: {result_answer_hash}")
        if web_search_answer:
            ws_hash = hashlib.md5(web_search_answer.encode()).hexdigest()
            logger.info(f"[QUERY_ENGINE] QueryResult.web_search_answer length: {len(web_search_answer)} chars, MD5: {ws_hash}")

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
            citation_checks=citation_checks,
            web_search_answer=web_search_answer,
            web_search_sources=web_search_sources,
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

    def _parent_window(self, parent_text: str, child_text: str) -> str:
        """Take a window of the parent section centred on where the child appears in it.

        Falls back to the head of the section when the child text cannot be located.
        """
        if not parent_text:
            return ""
        if len(parent_text) <= self.PARENT_CONTEXT_WINDOW:
            return parent_text

        half = self.PARENT_CONTEXT_WINDOW // 2
        # The child's stored text may carry a context header, so match on an interior
        # slice of it rather than the whole string.
        probe = child_text[:400].strip()
        offset = parent_text.find(probe) if probe else -1
        if offset < 0 and len(probe) > 80:
            offset = parent_text.find(probe[40:200])
        if offset < 0:
            return parent_text[:self.PARENT_CONTEXT_WINDOW]

        centre = offset + len(probe) // 2
        start = max(0, centre - half)
        end = min(len(parent_text), start + self.PARENT_CONTEXT_WINDOW)
        start = max(0, end - self.PARENT_CONTEXT_WINDOW)
        window = parent_text[start:end]
        if start > 0:
            window = f"...{window}"
        if end < len(parent_text):
            window = f"{window}..."
        return window

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
                    parent_text = parent.get('text', '')
                    for idx in indices:
                        chunks[idx]['parent_context'] = self._parent_window(
                            parent_text, chunks[idx].get('text', '')
                        )

        return chunks

    def _generate_answer(
        self,
        query: str,
        query_type: QueryType,
        sources: List[Dict[str, Any]],
        stream_callback: Optional[callable] = None,
        response_mode: str = "detailed",
        enable_general_knowledge: bool = True,
        enable_web_search: bool = False,
        progress_emitter: Optional[callable] = None,
        effort: str = DEFAULT_EFFORT,
        custom_prompts: Optional[Dict[str, Any]] = None,
        paper_ids: Optional[List[str]] = None,
        enable_pdf_upload: bool = False,
        conversation_memory: Optional[ConversationMemory] = None,
    ) -> str:
        """Generate answer using Claude with query-type-specific prompt.

        Includes conversation history for multi-turn context and retry logic
        for rate limits and transient errors.

        Args:
            query: The user query
            query_type: Classification of the query type
            sources: Retrieved source documents
            stream_callback: Optional callback(chunk: str) for streaming response chunks
            response_mode: "concise" for brief answers, "detailed" for comprehensive responses
            enable_general_knowledge: Whether to allow LLM to supplement with general knowledge
            enable_web_search: Whether to allow Claude to search the web
            progress_emitter: Optional callback(step, data) for progress events
            effort: Answer depth - low|medium|high|xhigh|max (default high)
            custom_prompts: Optional dict of custom system prompts from user preferences
            conversation_memory: Conversation memory scoped to this conversation

        Returns:
            Complete answer text
        """
        # Log the received parameters
        logger.info(f"_generate_answer called - response_mode: {response_mode}, enable_general_knowledge: {enable_general_knowledge}, enable_web_search: {enable_web_search}, query_type: {query_type}")

        if not sources:
            if enable_general_knowledge:
                # Allow general knowledge response even without sources
                logger.info("No sources but general knowledge enabled - proceeding with general knowledge response")
                pass
            else:
                return "I couldn't find relevant information in the literature to answer this question."

        # Format sources
        sources_text = self._format_sources(sources) if sources else ""

        # Get query-type-specific system prompt (using custom prompts if available)
        base_system_prompt = get_effective_prompt(query_type, response_mode, custom_prompts)
        is_custom = custom_prompts and response_mode in custom_prompts and query_type.value in custom_prompts.get(response_mode, {})

        # Set max tokens based on response mode. Adaptive thinking shares this budget
        # with the answer text, so both modes need headroom above the answer length.
        if response_mode == "detailed":
            max_tokens = 64000  # More tokens for detailed responses
            logger.info(f"Using DETAILED prompt (custom={is_custom}) with max_tokens={max_tokens}")
        else:
            max_tokens = 32000
            logger.info(f"Using CONCISE prompt (custom={is_custom}) with max_tokens={max_tokens}")

        # Web search requires general knowledge to be enabled
        if enable_web_search and not enable_general_knowledge:
            logger.warning("Web search requires general knowledge - enabling general knowledge")
            enable_general_knowledge = True

        # Build `system` as content blocks rather than one concatenated string:
        # cache_control attaches to blocks, and separate blocks keep the stable
        # base prompt byte-identical no matter which addendums follow it.
        system_blocks: List[Dict[str, Any]] = [
            {"type": "text", "text": base_system_prompt}
        ]

        # Add general knowledge addendum if enabled (using custom addendum if available)
        if enable_general_knowledge:
            system_blocks.append({
                "type": "text",
                "text": get_effective_addendum("general_knowledge", custom_prompts),
            })
            logger.info("Added general knowledge addendum to system prompt")

        # Add PDF upload addendum if enabled (using custom addendum if available)
        if enable_pdf_upload:
            system_blocks.append({
                "type": "text",
                "text": get_effective_addendum("pdf_upload", custom_prompts),
            })
            logger.info("Added PDF upload addendum to system prompt")

        # The breakpoint goes on the last block, which is the only placement that
        # can clear Opus 5's 512-token floor (see CACHE_CONTROL_EPHEMERAL above).
        _mark_cacheable(system_blocks[-1])
        # With both addendums present, also break after the general-knowledge block
        # so the (base + general knowledge) prefix gets its own entry and toggling
        # PDF upload reads it instead of rewriting the whole system prompt. That
        # prefix is 484-621 tokens in detailed mode, so it caches there; in concise
        # mode it is 383-434 and the API skips it silently.
        if len(system_blocks) == 3:
            _mark_cacheable(system_blocks[1])

        # Build messages array with conversation history
        messages = []

        # Add conversation history (previous turns only - current query not yet in memory)
        if conversation_memory:
            history = conversation_memory.get_chat_history(max_tokens=HISTORY_TOKEN_BUDGET)
            messages.extend(history)

            # Cache the replayed history so later turns read it instead of
            # re-billing it. Only worth a breakpoint when this history will still
            # be a prefix of the next turn's history - get_chat_history() rebuilds
            # the window from the newest message backwards, so once the window is
            # full the next request starts at a different message and an entry
            # written here could never be read, leaving only the write premium.
            if history and conversation_memory.history_survives_next_turn(
                history, max_tokens=HISTORY_TOKEN_BUDGET
            ):
                last = messages[-1]
                if isinstance(last.get("content"), str):
                    last["content"] = [{"type": "text", "text": last["content"]}]
                _mark_cacheable(last["content"][-1])
                logger.info(
                    "Cache breakpoint on conversation history (%d messages)", len(history)
                )

        # Add current query with retrieved sources
        # If PDF upload is enabled and we have the service, use it to create a message with PDFs
        if enable_pdf_upload and self.pdf_service and paper_ids:
            logger.info(f"Creating user message with PDF documents for {len(paper_ids)} papers")
            user_message = self.pdf_service.create_user_message_with_pdfs(
                query=query,
                paper_ids=paper_ids,
                sources_text=sources_text if sources else None
            )
            messages.append(user_message)
        else:
            # Original text-only message
            if sources:
                current_message = f"""Question: {query}

Retrieved Sources from Uploaded Papers:
{sources_text}

IMPORTANT: You MUST cite these sources using [Source N] format in your response. Start by answering the question using these sources with proper citations."""
            else:
                current_message = f"""Question: {query}

No sources were retrieved from the uploaded papers. Please answer based on your general scientific knowledge, clearly indicating that this response comes from general knowledge rather than the uploaded papers."""

            messages.append({"role": "user", "content": current_message})

        breakpoints = _enforce_breakpoint_budget(system_blocks, messages)

        # Debug logging
        logger.info(
            f"System prompt length: {sum(len(b['text']) for b in system_blocks)} chars "
            f"in {len(system_blocks)} block(s); {breakpoints} cache breakpoint(s)"
        )
        logger.info(f"Number of sources provided: {len(sources) if sources else 0}")
        logger.info(f"Web search enabled: {enable_web_search}, General knowledge enabled: {enable_general_knowledge}")

        try:
            # STEP 1: Generate RAG-based answer (no web search tool)
            stream_kwargs = {
                "model": self.claude_model,
                "max_tokens": max_tokens,
                "system": system_blocks,
                "messages": messages,
                "thinking": ADAPTIVE_THINKING,
                "output_config": _output_config(effort),
            }

            if stream_callback:
                # Use streaming API for RAG answer
                full_response = []

                logger.info("[STREAMING] Starting RAG answer generation with streaming")
                with self.anthropic.messages.stream(**stream_kwargs) as stream:
                    for event in stream:
                        if event.type == "message_start":
                            # Cache counters arrive with message_start, before any
                            # content; this is the only proof caching still works.
                            self.last_answer_usage = _log_cache_usage(
                                event.message.usage, "answer (streaming)"
                            )
                            continue
                        if event.type != "content_block_delta":
                            continue
                        delta = event.delta
                        if delta.type == "text_delta":
                            full_response.append(delta.text)
                            stream_callback(delta.text)
                        elif delta.type == "thinking_delta" and progress_emitter:
                            # Kept off the answer stream - the frontend concatenates
                            # answer_chunk into the rendered answer.
                            progress_emitter("thinking_chunk", {"chunk": delta.thinking})

                rag_response = "".join(full_response)

                # Log content verification
                logger.info(f"[STREAMING] RAG answer streaming completed - Total length: {len(rag_response)} chars")
                logger.info(f"[STREAMING] RAG answer preview: {rag_response[:200]}...")
                logger.info(f"[STREAMING] RAG answer hash: {hash(rag_response)}")

                # Web search is now handled separately in the query() method
                return rag_response
            else:
                # Non-streaming fallback. Still goes over the streaming transport so
                # the SDK does not reject these max_tokens values as timeout-prone.
                def make_api_call():
                    with self.anthropic.messages.stream(**stream_kwargs) as stream:
                        return stream.get_final_message()

                response = retry_with_exponential_backoff(make_api_call)
                self.last_answer_usage = _log_cache_usage(response.usage, "answer")
                # Handle response - extract text from content blocks
                text_parts = []
                for block in response.content:
                    if getattr(block, 'type', None) == 'text':
                        text_parts.append(block.text)
                rag_response = "".join(text_parts) if text_parts else ""

                # Web search is now handled separately in the query() method
                return rag_response

        except (RateLimitError, APIStatusError) as e:
            logger.error(f"Answer generation failed after retries: {e}")
            return "Error generating answer (API unavailable): Please try again later."

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Error generating answer: {str(e)}"

    def _perform_web_search(
        self,
        query: str,
        stream_callback: Optional[callable] = None,
        progress_callback: Optional[callable] = None,
        custom_prompts: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, List[Dict[str, str]]]:
        """Perform a separate web search using Anthropic's server-side web_search tool.

        Args:
            query: The original user query
            stream_callback: Optional callback for streaming answer text chunks
            progress_callback: Optional callback for progress updates (search status)
            custom_prompts: Optional dict of custom system prompts from user preferences

        Returns:
            Tuple of (answer_text, sources_list) where sources_list contains dicts with 'url' and 'title' keys
        """
        try:
            # Simplify the query to avoid refusals
            search_query = query[:500] if len(query) > 500 else query

            # Get web search system prompt (using custom if available)
            web_search_prompt = get_effective_addendum("web_search", custom_prompts)

            # Use sonnet for web search (good balance of speed and quality)
            web_search_model = self.claude_model_web_search

            logger.info(f"Starting web search with streaming for query: {search_query[:100]}...")

            # Prepare the web search tool definition
            tools = [{
                "type": "web_search_20260209",
                "name": "web_search",
                "max_uses": 5
            }]

            # Messages for the web search
            web_search_messages = [{
                "role": "user",
                "content": f"Search the web and provide information about: {search_query}\n\nProvide a comprehensive answer with proper citations."
            }]

            # Track collected data
            collected_text_chunks = []
            collected_urls = []

            # Use the streaming API with text_stream helper
            with self.anthropic.messages.stream(
                model=web_search_model,
                max_tokens=64000,
                system=web_search_prompt,
                messages=web_search_messages,
                tools=tools,
            ) as stream:
                # Collect all streamed text first
                for text in stream.text_stream:
                    logger.info(f"[TEXT STREAM] Received {len(text)} chars")
                    collected_text_chunks.append(text)
                    # Stream everything in real-time
                    if stream_callback:
                        stream_callback(text)

                # After stream completes, get the final message
                final_message = stream.get_final_message()

            logger.info(f"Web search response received, content blocks: {len(final_message.content)}, stop_reason: {final_message.stop_reason}")

            # Handle refusal case
            if final_message.stop_reason == 'refusal':
                logger.warning(f"Web search was refused by Claude for query: {search_query[:100]}")
                refusal_msg = "*Web search declined for this query. The AI determined it couldn't helpfully search for this specific topic.*"
                return (refusal_msg, [])

            # Extract URLs and identify which text blocks are answer vs progress
            # Structure: [Text (thinking)] -> [ToolUse] -> [ToolResult] -> [Text (answer with citations)]
            tool_result_index = -1
            answer_text_blocks = []

            # Match on the wire `type` discriminator, not the Python class name:
            # the SDK returns subclasses whose __name__ varies (get_final_message()
            # yields ParsedTextBlock, not TextBlock), while `type` is API contract.
            for i, block in enumerate(final_message.content):
                block_type = getattr(block, 'type', None)
                logger.debug(f"Processing block {i}: {block_type} ({type(block).__name__})")

                if block_type == 'server_tool_use':
                    if hasattr(block, 'input') and isinstance(block.input, dict):
                        search_query_text = block.input.get('query', '')
                        logger.info(f"Web search executed query: {search_query_text}")

                elif block_type == 'web_search_tool_result':
                    tool_result_index = i
                    # Extract URLs from search results
                    if hasattr(block, 'content'):
                        for result in block.content:
                            if hasattr(result, 'url') and hasattr(result, 'title'):
                                collected_urls.append({
                                    'url': result.url,
                                    'title': result.title
                                })
                                logger.debug(f"Found search result: {result.title}")

                elif block_type == 'text':
                    text = block.text if hasattr(block, 'text') else ''
                    has_citations = hasattr(block, 'citations') and block.citations

                    # Text blocks AFTER tool results = answer content (keep)
                    # Text blocks with citations = answer content (keep)
                    # Text blocks BEFORE tool results without citations = thinking (discard)
                    is_answer = (tool_result_index >= 0 and i > tool_result_index) or has_citations

                    if is_answer:
                        answer_text_blocks.append(text)
                        logger.debug(f"Answer text block ({len(text)} chars): {text[:100]}")
                    else:
                        logger.debug(f"Progress text block (discarded): {text[:100]}")

                    # Extract citations from text blocks
                    if has_citations:
                        for citation in block.citations:
                            if hasattr(citation, 'url') and hasattr(citation, 'title'):
                                collected_urls.append({
                                    'url': citation.url,
                                    'title': citation.title
                                })
                                logger.debug(f"Found citation: {citation.title}")

            # Build final result from answer text blocks only (not all streamed text)
            final_text = "".join(answer_text_blocks).strip()
            logger.info(f"[WEB_SEARCH] Building final result: {len(final_text)} chars of text, {len(collected_urls)} URLs collected")

            if final_text:
                # Deduplicate URLs
                seen_urls = set()
                unique_urls = []
                for url_info in collected_urls:
                    if url_info['url'] not in seen_urls:
                        seen_urls.add(url_info['url'])
                        unique_urls.append(url_info)

                # Log what we're returning
                import hashlib
                ws_hash = hashlib.md5(final_text.encode()).hexdigest()
                logger.info(f"[WEB_SEARCH] SUCCESS: Returning {len(final_text)} chars with {len(unique_urls)} unique URLs")
                logger.info(f"[WEB_SEARCH] Content MD5: {ws_hash}")
                logger.info(f"[WEB_SEARCH] Content preview: {final_text[:200]}...")
                return (final_text, unique_urls[:10])  # Limit to 10 sources
            else:
                logger.warning(f"Web search returned no answer text (but found {len(collected_urls)} URLs)")
                no_results_msg = "*No additional web results found for this query.*"
                return (no_results_msg, [])

        except Exception as e:
            logger.error(f"Web search failed: {e}", exc_info=True)
            error_msg = f"*Web search unavailable: {str(e)}*"
            return (error_msg, [])

    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Format sources for the prompt."""
        formatted = []

        for i, source in enumerate(sources, 1):
            title = source.get('title', 'Unknown Title')
            text = source.get('text', '')
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

    def clear_conversation(self, conversation_id: Optional[str] = None) -> None:
        """Clear conversation memory, for one conversation or for all of them."""
        if conversation_id:
            self._conversation_memories.pop(conversation_id, None)
            logger.info(f"Cleared conversation memory for {conversation_id}")
        else:
            self._conversation_memories.clear()
            logger.info("Cleared all conversation memory")

    def get_conversation_context(self, conversation_id: Optional[str] = None) -> Optional[str]:
        """Get formatted conversation context for debugging."""
        memory = self._conversation_memories.get(conversation_id or "__anonymous__")
        if memory:
            return memory.format_context_for_prompt()
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
        """Get conversation statistics aggregated across tracked conversations."""
        if not self.enable_conversation_memory:
            return {"enabled": False}

        stats = {
            "conversations_tracked": len(self._conversation_memories),
            "total_messages": 0,
            "user_messages": 0,
            "papers_discussed": 0,
            "entities_tracked": 0,
        }
        for memory in self._conversation_memories.values():
            memory_stats = memory.get_stats()
            for key in ("total_messages", "user_messages", "papers_discussed", "entities_tracked"):
                stats[key] += memory_stats[key]
        return stats
