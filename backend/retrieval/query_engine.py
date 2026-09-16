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
from enum import Enum

from anthropic import Anthropic, RateLimitError, APIStatusError

from .embedder import VoyageEmbedder
from .reranker import CohereReranker, RerankResult
from .qdrant_store import QdrantStore
from .query_classifier import (
    QueryClassifier,
    QueryType,
    QueryClassification,
    MultiQueryClassification,
    BROAD_RETRIEVAL,
    RESPONSE_SHAPE,
    RETRIEVAL_STRATEGIES,
)
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
# cache_creation_input_tokens == 0 with no error.
#
# Re-measured 2026-09-15 with messages.count_tokens after the 16 query-type
# prompts were replaced by one base prompt plus a stance modifier (Opus 5 floor
# = 512; figures below include 7 tokens of request overhead):
#
#   stance-independent prefix              concise   detailed
#   base only (both addendums off)             867        889
#   base + general knowledge  <- default      1019       1041
#   base + pdf upload                         1032       1054
#   base + both addendums                     1184       1206
#
#   stance block alone                     106 (ask) .. 179 (critique)
#
# The old table said no base prompt reached 512 tokens on its own, so the only
# useful breakpoint was the last system block. That is no longer true: the base
# prompt now carries the invariants and clears the floor by itself in every
# configuration. So there are two system breakpoints - one on the last block
# (whole system prompt) and one on the block before the stance, so that flipping
# stance mid-conversation reads the ~870-1200 token prefix instead of rewriting
# it. Ordering the stance last is what makes that prefix stance-independent.
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
    # Response stance the answer was generated under - an AnswerMode value.
    # Literal rather than DEFAULT_ANSWER_MODE.value because AnswerMode is
    # defined below this dataclass.
    mode: str = "ask"
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


# ---------------------------------------------------------------------------
# Answer prompts: one base prompt plus a user-selected stance
# ---------------------------------------------------------------------------
#
# This replaces 8 query types x 2 response modes = 16 hand-written prompts, each
# of which mandated a numbered report scaffold ("1. Background & Context,
# 2. Key Findings, 3. Methodology Highlights..."). Measured consequence: a
# 5,161-character average assistant message against a 344-character average
# question, opening "## Answer from the Uploaded Sources". See
# docs/IMPROVEMENT_PLAN_2026-09.md section 4b (R12, requests 7 and 8).
#
# Four axes, now orthogonal:
#   BASE_SYSTEM_PROMPT  invariants - cite, ground, admit gaps, keep every number
#   ANSWER_STANCES      what kind of response the user asked for (AnswerMode)
#   LENGTH_DIAL         how much room the answer may take (response_mode)
#   output_config.effort  how much thinking the model spends (not a prompt)
#
# `QueryType` still exists, but after W5 it no longer routes retrieval either:
# there is one broad path (BROAD_RETRIEVAL) and QueryType only shapes how many
# results come back (RESPONSE_SHAPE). It does not select a prompt: retrieval is
# a retrieval concern, stance is a generation concern, and conflating them is
# what produced the book-report voice.


class AnswerMode(str, Enum):
    """The stance the user picked for this turn - what kind of reply they want.

    Orthogonal to QueryType (which chunks to retrieve) and to effort (how hard
    the model thinks).
    """

    ASK = "ask"                # answer the question
    BRAINSTORM = "brainstorm"  # generate possibilities and directions
    DEVELOP = "develop"        # take a stated idea and build it out
    REFINE = "refine"          # tighten an existing idea, sharpen the framing
    CRITIQUE = "critique"      # argue against it, find the weaknesses
    DRAFT = "draft"            # start turning it into writing


DEFAULT_ANSWER_MODE = AnswerMode.ASK
ANSWER_MODES = tuple(m.value for m in AnswerMode)


def coerce_answer_mode(value: Optional[str]) -> AnswerMode:
    """Map an arbitrary string onto an AnswerMode, defaulting to Ask."""
    if not value:
        return DEFAULT_ANSWER_MODE
    try:
        return AnswerMode(value.strip().lower())
    except ValueError:
        logger.warning("Unknown answer mode %r, falling back to %s", value, DEFAULT_ANSWER_MODE.value)
        return DEFAULT_ANSWER_MODE


# The invariants. Everything that must hold no matter which stance is selected.
# Rules 2 and 3 are the resolution of conflict C3 in the plan: request 7 wants
# conversational brevity, request 3 wants every number kept with its provenance.
# Brevity removes scaffolding, never data.
BASE_SYSTEM_PROMPT = """You are a research assistant working over a scientist's own library of papers. Talk to them the way a colleague at the next bench would - directly, in prose, assuming they know their field. Not like a student handing in a report.

Ground rules, in priority order:

1. Lead with the answer. The first sentence answers the question. No preamble, no restating the question back, no "Based on the retrieved sources", no "## Answer from the Uploaded Sources" heading. Just the answer.

2. Keep every number. Concentrations, ranges, equivalents, molar ratios, temperatures, times, yields, pH, solvent ratios, molecular weights, IC50s, detection limits, error bars - if a source gives a value that bears on the question, it goes in your answer with its unit and its [Source N]. Report values exactly as the source gives them: do not round them, average them across sources, or replace them with a vaguer statement like "low micromolar". When you are unsure whether a value bears on the question, include it - a number costs one clause, and leaving it out is the one failure this tool cannot afford. This precision is the most valuable thing you do.

3. Brevity means cutting scaffolding, not data. Cut: section headings, numbered outlines, restatement, hedging, throat-clearing, "it is worth noting", recaps of what you just said, and offers to elaborate further. Never cut: numbers, units, conditions, the caveats that actually change the answer, or citations. An answer that got shorter by dropping a concentration range has failed, not succeeded.

4. Cite as you go. Put [Source N] on the specific claim it supports, inline, not parked at the end of a paragraph.

5. When the library does not cover it, say so in a sentence or two and stop. "Nothing in your library addresses X directly - [Source 3] is the closest and only covers Y." Do not write five paragraphs around an absence. Naming what would answer it is useful; padding is not.

6. Length follows the question, not a setting. A one-line question gets a one-line answer. A question that genuinely spans five papers gets the room the evidence needs and not a word more.

7. Structure only when the content is structured. Prose by default. A list when you are genuinely listing parallel items - three solvents, four conditions. A table when you are genuinely comparing the same measured quantity across papers. Never impose a section scaffold on a two-sentence answer.

8. Sources that disagree are information. Give both values with both citations and name the condition that differs, rather than smoothing it into a range."""


# response_mode is a length dial and nothing else - it no longer selects a
# prompt. It cannot override rules 1-8; it only widens or narrows the room.
LENGTH_DIAL: Dict[str, str] = {
    "concise": """Room: tight. Answer, evidence, stop. If the whole answer is one sentence carrying two numbers and two citations, that is the right answer.""",
    "detailed": """Room: you may explain mechanism, conditions, and where sources disagree, when that genuinely helps. More room is not permission to add headings, outlines, or restatement - rules 1, 3 and 7 still hold.""",
}
DEFAULT_RESPONSE_MODE = "detailed"


# The stance modifiers. Each is composed onto BASE_SYSTEM_PROMPT; none of them
# repeats an invariant, and none of them may relax one.
ANSWER_STANCES: Dict[AnswerMode, str] = {
    AnswerMode.ASK: """Stance - Ask. Answer the question. That is the whole job. If the answer is "1 M NaOH, adjusted to pH 8.0 [Source 2]", say that and stop; do not build a report around a one-line fact. Follow-up context is welcome only when it changes what the scientist would actually do.""",

    AnswerMode.BRAINSTORM: """Stance - Brainstorm. They want possibilities, not a summary of the literature. Generate concrete directions grounded in what the library actually shows: what has been tried and at what conditions [Source N], what has not, what an adjacent system does that could transfer here, what the obvious next experiment is. A handful of specific, testable ideas beats an exhaustive taxonomy. Mark clearly which ideas the library supports and which are your extrapolation beyond it.""",

    AnswerMode.DEVELOP: """Stance - Develop. They have an idea and want it built out. Take it seriously and push it forward: what it would take to make it work, which pieces the library already establishes and at which values [Source N], what has to be decided, where the unknowns sit. Add substance, not affirmation - do not spend lines telling them the idea is promising. Where a step has no support in the library, say so plainly instead of papering over it.""",

    AnswerMode.REFINE: """Stance - Refine. They have an idea and want it sharpened, not expanded. Tighten the framing: what exactly is being claimed, what the minimal defensible version is, which words are doing work and which are filler. Offer a sharper restatement of the idea itself, in their voice. Cut scope the library does not support. Your output should be shorter and harder-edged than the input, not longer.""",

    AnswerMode.CRITIQUE: """Stance - Critique. Argue against it. Take the reviewer's seat and find what breaks: unsupported steps, missing controls, confounds, alternative explanations, prior work that already did this or already failed at it [Source N], numbers that will not carry the claim being hung on them. Be specific, cite, and quantify the objection wherever the library lets you. Do not balance this with praise - they asked for the weaknesses, and a critique that opens by saying what is great is a wasted turn. If something genuinely holds up, one line is enough.""",

    AnswerMode.DRAFT: """Stance - Draft. Start turning this into writing: an outline, a section, a paragraph, a set of specific aims - whichever they asked for. Produce the actual prose, not advice about how to write it. Carry [Source N] citations inline so they can be resolved later, and carry every number into the draft. Match the register of the target: aims are terse and declarative, a discussion paragraph argues, a methods paragraph is procedural and keeps every concentration, equivalent and temperature.""",
}


# General knowledge addendum for when enable_general_knowledge is True.
# Rewritten alongside the base prompt: the old version mandated a
# "## Additional Context (General Knowledge)" section, which is where the
# "## Direct Answer from the Retrieved Sources" style preamble came from.
GENERAL_KNOWLEDGE_ADDENDUM = """You may also draw on your own scientific knowledge beyond the retrieved sources.

Answer from the library first, cited. Where your own knowledge adds something the library does not have, add it inline and mark it in the sentence itself - "not in your library, but standard practice is...", "no paper here reports it; generally...". Marking it in the sentence is enough: do not open a separate section or heading for it, and do not restate the library answer inside it.

If the library already covers the question, add nothing."""

# PDF upload addendum for when full PDF documents are sent to Claude
PDF_UPLOAD_ADDENDUM = """You also have the full PDFs of the selected papers alongside the retrieved chunks.

The chunks are the passages already identified as relevant - cite those as [Source N]. The PDFs are there for whatever the chunks cut off: a figure, a table, a methods detail, the rest of a truncated sentence. When you use something from the full PDF that is not in the chunks, name where it came from ("from the Methods of <paper>"). Everything else is unchanged: lead with the answer, keep every number, no scaffolding."""

# Web search system prompt - used for the separate web search call
WEB_SEARCH_SYSTEM_PROMPT = """You are a helpful research assistant. Search the web for publicly available information related to the user's question. Focus on recent publications, news, educational resources, and general background information.

Provide factual information with source URLs. This is for educational and research purposes.

You may use markdown formatting (headers, bold, lists) to organize your response clearly."""


# The three editable prompt groups exposed by /user/prompts, and the prompt
# types inside each. Custom overrides are stored as
# {"base": {...}, "stance": {...}, "addendums": {...}}.
PROMPT_GROUP_BASE = "base"
PROMPT_GROUP_STANCE = "stance"
PROMPT_GROUP_ADDENDUMS = "addendums"

PROMPT_TYPES: Dict[str, tuple] = {
    PROMPT_GROUP_BASE: ("base", "concise", "detailed"),
    PROMPT_GROUP_STANCE: ANSWER_MODES,
    PROMPT_GROUP_ADDENDUMS: ("general_knowledge", "web_search", "pdf_upload"),
}

DEFAULT_ADDENDUMS: Dict[str, str] = {
    "general_knowledge": GENERAL_KNOWLEDGE_ADDENDUM,
    "web_search": WEB_SEARCH_SYSTEM_PROMPT,
    "pdf_upload": PDF_UPLOAD_ADDENDUM,
}


def get_default_prompt_groups() -> Dict[str, Dict[str, str]]:
    """Every default prompt, grouped the way /user/prompts exposes them."""
    return {
        PROMPT_GROUP_BASE: {
            "base": BASE_SYSTEM_PROMPT,
            "concise": LENGTH_DIAL["concise"],
            "detailed": LENGTH_DIAL["detailed"],
        },
        PROMPT_GROUP_STANCE: {m.value: ANSWER_STANCES[m] for m in AnswerMode},
        PROMPT_GROUP_ADDENDUMS: dict(DEFAULT_ADDENDUMS),
    }


def _custom_override(
    custom_prompts: Optional[Dict[str, Any]],
    group: str,
    prompt_type: str,
) -> Optional[str]:
    """Look up one user override, tolerating a malformed stored structure."""
    if not custom_prompts:
        return None
    group_prompts = custom_prompts.get(group)
    if not isinstance(group_prompts, dict):
        return None
    value = group_prompts.get(prompt_type)
    return value if isinstance(value, str) and value.strip() else None


def build_base_block(
    response_mode: str = DEFAULT_RESPONSE_MODE,
    custom_prompts: Optional[Dict[str, Any]] = None,
) -> str:
    """The stance-independent half of the system prompt: invariants + length dial.

    Kept in its own content block so that switching stance does not invalidate
    it in the prompt cache.
    """
    base = _custom_override(custom_prompts, PROMPT_GROUP_BASE, "base") or BASE_SYSTEM_PROMPT

    mode_key = response_mode if response_mode in LENGTH_DIAL else DEFAULT_RESPONSE_MODE
    length = _custom_override(custom_prompts, PROMPT_GROUP_BASE, mode_key) or LENGTH_DIAL[mode_key]

    return f"{base}\n\n{length}"


def build_stance_block(
    mode: AnswerMode = DEFAULT_ANSWER_MODE,
    custom_prompts: Optional[Dict[str, Any]] = None,
) -> str:
    """The stance modifier for this turn."""
    return (
        _custom_override(custom_prompts, PROMPT_GROUP_STANCE, mode.value)
        or ANSWER_STANCES.get(mode, ANSWER_STANCES[DEFAULT_ANSWER_MODE])
    )


def get_effective_addendum(
    addendum_type: str,
    custom_prompts: Optional[Dict[str, Any]] = None,
) -> str:
    """Get the effective addendum prompt, preferring a user override.

    Args:
        addendum_type: 'general_knowledge', 'web_search', or 'pdf_upload'
        custom_prompts: Optional dict of custom prompts from user preferences

    Returns:
        The effective addendum to use
    """
    return (
        _custom_override(custom_prompts, PROMPT_GROUP_ADDENDUMS, addendum_type)
        or DEFAULT_ADDENDUMS.get(addendum_type, "")
    )


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
        mode: Optional[str] = None,
        response_mode: str = DEFAULT_RESPONSE_MODE,
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
            mode: Response stance - ask|brainstorm|develop|refine|critique|draft (None = ask).
                Orthogonal to retrieval: it selects the stance modifier on the
                answer prompt and nothing else.
            response_mode: Length dial - "concise" or "detailed"
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
            classification = QueryClassification(
                query_type=query_type,
                confidence=1.0,
                entities=entities_extracted[:5],
                needs_cross_corpus=True,
                suggested_chunk_types=BROAD_RETRIEVAL["chunk_types"],
                suggested_top_k=BROAD_RETRIEVAL["top_k"],
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
            classification = QueryClassification(
                query_type=query_type,
                confidence=1.0,
                entities=entities_extracted[:5],
                needs_cross_corpus=True,
                suggested_chunk_types=BROAD_RETRIEVAL["chunk_types"],
                suggested_top_k=BROAD_RETRIEVAL["top_k"],
                reasoning=f"Broad retrieval - {query_type.value} (shaping only)",
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

        # Step 5: Retrieval parameters.
        #
        # One broad path (§4b, measured in W5): no chunk_type or section
        # exclusions, because those are applied *before* scoring and make a
        # correct answer invisible rather than low-ranked. `query_type` survives
        # only as post-retrieval shaping — how many results to return and how
        # many per paper — which cannot hide anything.
        chunk_types = BROAD_RETRIEVAL["chunk_types"]
        section_filter = BROAD_RETRIEVAL["section_filter"]
        strategy_top_k = BROAD_RETRIEVAL["top_k"]
        shape = RESPONSE_SHAPE.get(query_type, BROAD_RETRIEVAL)
        rerank_top_n = shape.get("rerank_top_n", BROAD_RETRIEVAL["rerank_top_n"])
        max_per_paper = shape.get("max_per_paper", BROAD_RETRIEVAL["max_per_paper"])

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
                expanded_query, chunk_types, section_filter,
                paper_ids=paper_ids, top_k=effective_top_k,
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
                    expanded_query, results, chunk_types, section_filter,
                    paper_ids=paper_ids, top_k=effective_top_k,
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
        answer_mode = coerce_answer_mode(mode)

        # STEP 1: Generate RAG answer first
        logger.info(
            f"[QUERY_ENGINE] Starting RAG answer generation with "
            f"mode={answer_mode.value} effort={effective_effort}"
        )
        answer = self._generate_answer(
            query=query,
            sources=expanded_sources,
            stream_callback=answer_stream_callback if progress_callback else None,
            mode=answer_mode,
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
            mode=answer_mode.value,
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
        sources: List[Dict[str, Any]],
        stream_callback: Optional[callable] = None,
        mode: AnswerMode = DEFAULT_ANSWER_MODE,
        response_mode: str = DEFAULT_RESPONSE_MODE,
        enable_general_knowledge: bool = True,
        enable_web_search: bool = False,
        progress_emitter: Optional[callable] = None,
        effort: str = DEFAULT_EFFORT,
        custom_prompts: Optional[Dict[str, Any]] = None,
        paper_ids: Optional[List[str]] = None,
        enable_pdf_upload: bool = False,
        conversation_memory: Optional[ConversationMemory] = None,
    ) -> str:
        """Generate the answer with Claude: base prompt + the selected stance.

        Includes conversation history for multi-turn context and retry logic
        for rate limits and transient errors.

        Args:
            query: The user query
            sources: Retrieved source documents
            stream_callback: Optional callback(chunk: str) for streaming response chunks
            mode: Response stance (AnswerMode) - selects the stance modifier
            response_mode: Length dial - "concise" or "detailed"
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
        logger.info(
            f"_generate_answer called - mode: {mode.value}, response_mode: {response_mode}, "
            f"enable_general_knowledge: {enable_general_knowledge}, enable_web_search: {enable_web_search}"
        )

        if not sources:
            if enable_general_knowledge:
                # Allow general knowledge response even without sources
                logger.info("No sources but general knowledge enabled - proceeding with general knowledge response")
                pass
            else:
                return "I couldn't find relevant information in the literature to answer this question."

        # Format sources
        sources_text = self._format_sources(sources) if sources else ""

        # Set max tokens. This is a ceiling, not a target - the prompt decides
        # length. Adaptive thinking shares this budget with the answer text, so
        # both modes keep headroom well above any answer we expect.
        max_tokens = 64000 if response_mode == "detailed" else 32000

        # Web search requires general knowledge to be enabled
        if enable_web_search and not enable_general_knowledge:
            logger.warning("Web search requires general knowledge - enabling general knowledge")
            enable_general_knowledge = True

        # Build `system` as content blocks rather than one concatenated string:
        # cache_control attaches to blocks, and separate blocks keep the stable
        # prefix byte-identical no matter what follows it.
        #
        # Order matters for the cache. The stance goes LAST, after the addendums,
        # because it is the block the user flips between turns ("ask" then
        # "critique"); everything before it then stays a valid cached prefix
        # across a stance switch. The addendums are set-and-forget settings, so
        # they sit in the stable region.
        system_blocks: List[Dict[str, Any]] = [
            {"type": "text", "text": build_base_block(response_mode, custom_prompts)}
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

        # The stance modifier - always present, always last.
        system_blocks.append(
            {"type": "text", "text": build_stance_block(mode, custom_prompts)}
        )

        # Breakpoint on the last block, which covers the whole system prompt.
        _mark_cacheable(system_blocks[-1])
        # And one on the block just before the stance, so the stance-independent
        # prefix gets its own entry and switching stance reads it instead of
        # rewriting everything. Re-measured with messages.count_tokens (table
        # above CACHE_CONTROL_EPHEMERAL): that prefix is 1,041 tokens in the
        # default configuration and 867 at its smallest, both well clear of
        # Opus 5's 512-token floor.
        if len(system_blocks) > 1:
            _mark_cacheable(system_blocks[-2])

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

Retrieved sources from the library:
{sources_text}

Cite these as [Source N]. Answer the question first."""
            else:
                current_message = f"""Question: {query}

Nothing was retrieved from the library for this question. Say so in a sentence, then answer from your own scientific knowledge if you can, marked as such."""

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
