"""Query classification for retrieval strategy selection.

Classifies user queries to determine the optimal retrieval strategy.
Different query types benefit from different chunk types and retrieval parameters.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from anthropic import Anthropic

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Types of queries with different retrieval needs."""
    FACTUAL = "factual"           # Specific facts, definitions, values
    FRAMING = "framing"           # How to position/describe research
    METHODS = "methods"           # Technical protocols, procedures
    SUMMARY = "summary"           # Summarize a paper or findings
    COMPARATIVE = "comparative"   # Compare methods/findings across papers
    NOVELTY = "novelty"           # Assess what's new or defensible
    LIMITATIONS = "limitations"   # Discuss constraints or caveats
    GENERAL = "general"           # Fallback for uncategorized queries


@dataclass
class QueryClassification:
    """Result of query classification."""
    query_type: QueryType
    confidence: float
    entities: list
    needs_cross_corpus: bool
    suggested_chunk_types: list
    suggested_top_k: int
    reasoning: str


@dataclass
class MultiQueryClassification:
    """Result of multi-label query classification (top 3 categories)."""
    query_types: List[Tuple[QueryType, float]]  # List of (type, confidence) pairs
    primary_type: QueryType  # Highest confidence type
    entities: list
    needs_cross_corpus: bool
    merged_chunk_types: list  # Union of chunk types from all categories
    reasoning: str


# Retrieval strategies per query type
RETRIEVAL_STRATEGIES: Dict[QueryType, Dict[str, Any]] = {
    QueryType.FACTUAL: {
        "chunk_types": ["fine", "table", "caption"],
        "top_k": 50,
        "rerank_top_n": 15,
        "max_per_paper": 3,
    },
    QueryType.FRAMING: {
        "chunk_types": ["abstract", "section"],  # Need broader context
        "top_k": 30,
        "rerank_top_n": 20,
        "max_per_paper": 2,  # Want diversity across papers
    },
    QueryType.METHODS: {
        "chunk_types": ["section", "fine"],  # Focus on methods sections
        "section_filter": ["methods", "experimental", "synthesis"],
        "top_k": 50,
        "rerank_top_n": 15,
        "max_per_paper": 5,  # Methods details often span chunks
    },
    QueryType.SUMMARY: {
        "chunk_types": ["abstract", "section", "full"],
        "top_k": 20,
        "rerank_top_n": 10,
        "max_per_paper": 10,  # Focus on single paper
    },
    QueryType.COMPARATIVE: {
        "chunk_types": ["abstract", "section"],
        "top_k": 100,  # Need many papers
        "rerank_top_n": 25,
        "max_per_paper": 2,  # Want breadth
    },
    QueryType.NOVELTY: {
        "chunk_types": ["abstract", "section"],
        "section_filter": ["introduction", "discussion", "conclusion", "results_discussion", "results"],
        "top_k": 50,
        "rerank_top_n": 20,
        "max_per_paper": 3,
    },
    QueryType.LIMITATIONS: {
        "chunk_types": ["section", "abstract"],
        "section_filter": ["discussion", "conclusion", "results", "results_discussion", "limitations", "future work", "caveats"],
        "top_k": 50,
        "rerank_top_n": 15,
        "max_per_paper": 4,
    },
    QueryType.GENERAL: {
        # Balanced strategy for uncategorized queries - search all chunk types
        "chunk_types": ["abstract", "section", "fine", "table", "caption"],
        "top_k": 50,
        "rerank_top_n": 20,
        "max_per_paper": 3,
    },
}

# Confidence threshold - below this, fall back to GENERAL
LOW_CONFIDENCE_THRESHOLD = 0.6


CLASSIFICATION_PROMPT = '''Classify this research question into exactly one category.

Categories:
- FACTUAL: Asking for specific facts, definitions, numeric values, or mechanisms. The answer is objective and verifiable. Examples: "What is the IC50?", "What mutations cause resistance?", "What is the molecular weight?"
- FRAMING: Asking how to POSITION, JUSTIFY, or ARGUE for research significance in a publication context. Focus is on rhetorical strategy, not technical details. Examples: "How do I frame X as novel?", "How do I justify choosing method Y?", "How should I argue that X addresses an unmet need?"
- METHODS: Asking about technical protocols, procedures, experimental parameters, or HOW something is DONE/SYNTHESIZED/MEASURED. Even if phrased as "how should I describe", if it asks about WHAT was done technically, it's METHODS. Examples: "What buffer was used?", "How are stapled peptides synthesized?", "How should I describe the SRS microscopy setup?"
- SUMMARY: Asking to summarize a paper or its findings. Examples: "Summarize this paper", "What are the key findings?"
- COMPARATIVE: Asking to compare or contrast multiple things. Examples: "How does X compare to Y?", "What is the difference between X and Y?"
- NOVELTY: Asking about what's NEW, what's been done BEFORE, what GAPS exist, or what claims are DEFENSIBLE. Assessing prior work to identify opportunities. Examples: "Has X been done before?", "What aspects are underexplored?", "Are there existing inhibitors for X?", "What is the strongest novelty claim?"
- LIMITATIONS: Asking about constraints, caveats, weaknesses, or how to address missing data/controls. Examples: "What are the limitations of X?", "How do I address the limitation that...?"
- GENERAL: Query doesn't clearly fit the above categories.

CRITICAL DISAMBIGUATION RULES:
1. METHODS vs FRAMING: If the query asks about technical details (buffers, concentrations, synthesis steps, equipment settings, protocols) even with "how should I describe" phrasing → METHODS. If it asks about rhetorical positioning, significance, or justification without technical specifics → FRAMING.
2. NOVELTY vs FACTUAL: If the query assesses prior work, asks "has X been done", "are there existing", "what's underexplored", or seeks to identify gaps → NOVELTY. If it asks for a specific fact without assessing novelty → FACTUAL.
3. NOVELTY vs COMPARATIVE: If the query asks what makes something DIFFERENT from prior work to assess novelty → NOVELTY. If it asks to neutrally compare two approaches → COMPARATIVE.

FEW-SHOT EXAMPLES:

Query: "How should I describe the SRS microscopy setup and imaging parameters?"
→ METHODS (asks about technical equipment/parameters, not rhetorical positioning)

Query: "How are stapled peptides typically synthesized and characterized?"
→ METHODS (asks about synthesis protocol)

Query: "How should I describe Affinity FPLC vs Ion-Exchange FPLC in a paper?"
→ METHODS (asks about technical procedures for chromatography)

Query: "How do I frame Raman imaging as an enabling platform?"
→ FRAMING (asks about rhetorical positioning, no technical details)

Query: "How do I justify that stapled peptides are better than small molecules?"
→ FRAMING (asks for justification/argument strategy)

Query: "Has SRS imaging been applied to biofilm penetration before?"
→ NOVELTY (assessing prior work to identify gaps)

Query: "Are there existing stapled peptide inhibitors targeting ERα mutants?"
→ NOVELTY (assessing what exists to identify opportunities)

Query: "What is the mechanism of action of LL-37?"
→ FACTUAL (asking for specific mechanistic fact)

Query: "What makes this approach distinct from previous work?"
→ NOVELTY (assessing differentiation from prior art)

Query: "How does SRS compare to fluorescence microscopy?"
→ COMPARATIVE (neutral comparison of two techniques)

Also identify:
1. Key domain entities (gene names, proteins, techniques, chemicals)
2. Whether this needs single-paper focus (false) or cross-corpus synthesis (true)

Question: {query}

Respond in this exact format:
QUERY_TYPE: <one of the categories above>
CONFIDENCE: <0.0-1.0, use lower values if the query doesn't fit well>
ENTITIES: <comma-separated list or "none">
CROSS_CORPUS: <true or false>
REASONING: <brief explanation>'''


MULTI_CLASSIFICATION_PROMPT = '''Classify this research question into the TOP 3 most relevant categories, ranked by relevance.

Categories:
- FACTUAL: Specific facts, definitions, values, mechanisms
- FRAMING: How to position/justify research for publication
- METHODS: Technical protocols, procedures, experimental details
- SUMMARY: Summarize findings
- COMPARATIVE: Compare methods/approaches across papers
- NOVELTY: Assess prior work, gaps, what's new/defensible
- LIMITATIONS: Constraints, caveats, weaknesses

Question: {query}

Return the top 3 categories that could help answer this query, even if some are secondary.
Many queries have multiple aspects - a methods question might also need framing context.

Respond in this exact format:
TOP_1: <category> (<confidence 0.0-1.0>)
TOP_2: <category> (<confidence 0.0-1.0>)
TOP_3: <category> (<confidence 0.0-1.0>)
ENTITIES: <domain entities, or "none">
CROSS_CORPUS: <true or false>
REASONING: <brief explanation of why these 3 categories>'''


class QueryClassifier:
    """Classify queries to determine retrieval strategy."""

    def __init__(
        self,
        anthropic_client: Anthropic,
        model: str = "claude-sonnet-4-5-20250929"
    ):
        """Initialize classifier.

        Args:
            anthropic_client: Anthropic API client
            model: Claude model to use for classification
        """
        self.client = anthropic_client
        self.model = model

    def classify(self, query: str) -> QueryClassification:
        """Classify a query.

        Args:
            query: User query to classify

        Returns:
            QueryClassification with type and retrieval strategy
        """
        prompt = CLASSIFICATION_PROMPT.format(query=query)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            return self._parse_response(response.content[0].text, query)

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            # Default to factual on error
            return self._default_classification(query)

    def _parse_response(self, response: str, query: str) -> QueryClassification:
        """Parse Claude's classification response."""
        lines = response.strip().split('\n')

        query_type = QueryType.FACTUAL
        confidence = 0.5
        entities = []
        cross_corpus = False
        reasoning = ""

        for line in lines:
            line = line.strip()
            if line.startswith("QUERY_TYPE:"):
                type_str = line.split(":", 1)[1].strip().upper()
                try:
                    query_type = QueryType(type_str.lower())
                except ValueError:
                    query_type = QueryType.FACTUAL

            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    confidence = 0.5

            elif line.startswith("ENTITIES:"):
                entities_str = line.split(":", 1)[1].strip()
                if entities_str.lower() != "none":
                    entities = [e.strip() for e in entities_str.split(",")]

            elif line.startswith("CROSS_CORPUS:"):
                cross_corpus = line.split(":", 1)[1].strip().lower() == "true"

            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        # Fall back to GENERAL if confidence is low
        if confidence < LOW_CONFIDENCE_THRESHOLD and query_type != QueryType.GENERAL:
            logger.info(f"Low confidence ({confidence:.2f}) for {query_type.value}, falling back to GENERAL")
            query_type = QueryType.GENERAL
            reasoning = f"Low confidence classification, using general strategy. Original: {reasoning}"

        strategy = RETRIEVAL_STRATEGIES[query_type]

        return QueryClassification(
            query_type=query_type,
            confidence=confidence,
            entities=entities,
            needs_cross_corpus=cross_corpus,
            suggested_chunk_types=strategy["chunk_types"],
            suggested_top_k=strategy["top_k"],
            reasoning=reasoning,
        )

    def _default_classification(self, query: str) -> QueryClassification:
        """Return default classification when parsing fails."""
        strategy = RETRIEVAL_STRATEGIES[QueryType.GENERAL]
        return QueryClassification(
            query_type=QueryType.GENERAL,
            confidence=0.5,
            entities=[],
            needs_cross_corpus=False,
            suggested_chunk_types=strategy["chunk_types"],
            suggested_top_k=strategy["top_k"],
            reasoning="Default classification due to parsing error",
        )

    def get_retrieval_strategy(self, query_type: QueryType) -> Dict[str, Any]:
        """Get retrieval strategy for a query type."""
        return RETRIEVAL_STRATEGIES.get(query_type, RETRIEVAL_STRATEGIES[QueryType.GENERAL])

    def classify_multi(self, query: str) -> MultiQueryClassification:
        """Classify a query into top 3 categories for broader retrieval.

        Args:
            query: User query to classify

        Returns:
            MultiQueryClassification with top 3 types and merged chunk types
        """
        prompt = MULTI_CLASSIFICATION_PROMPT.format(query=query)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            return self._parse_multi_response(response.content[0].text, query)

        except Exception as e:
            logger.error(f"Multi-classification failed: {e}")
            return self._default_multi_classification(query)

    def _parse_multi_response(self, response: str, query: str) -> MultiQueryClassification:
        """Parse Claude's multi-classification response."""
        lines = response.strip().split('\n')

        query_types: List[Tuple[QueryType, float]] = []
        entities = []
        cross_corpus = False
        reasoning = ""

        for line in lines:
            line = line.strip()

            # Parse TOP_1, TOP_2, TOP_3
            for i in range(1, 4):
                if line.startswith(f"TOP_{i}:"):
                    try:
                        content = line.split(":", 1)[1].strip()
                        # Parse "METHODS (0.8)" format
                        if "(" in content and ")" in content:
                            type_str = content.split("(")[0].strip().upper()
                            conf_str = content.split("(")[1].split(")")[0].strip()
                            query_type = QueryType(type_str.lower())
                            confidence = float(conf_str)
                            query_types.append((query_type, confidence))
                    except (ValueError, IndexError):
                        pass

            if line.startswith("ENTITIES:"):
                entities_str = line.split(":", 1)[1].strip()
                if entities_str.lower() != "none":
                    entities = [e.strip() for e in entities_str.split(",")]

            elif line.startswith("CROSS_CORPUS:"):
                cross_corpus = line.split(":", 1)[1].strip().lower() == "true"

            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        # Fallback if parsing failed
        if not query_types:
            query_types = [(QueryType.GENERAL, 0.5)]

        # Merge chunk types from all categories
        merged_chunk_types = set()
        for qt, _ in query_types:
            strategy = RETRIEVAL_STRATEGIES.get(qt, RETRIEVAL_STRATEGIES[QueryType.GENERAL])
            merged_chunk_types.update(strategy["chunk_types"])

        return MultiQueryClassification(
            query_types=query_types,
            primary_type=query_types[0][0],
            entities=entities,
            needs_cross_corpus=cross_corpus,
            merged_chunk_types=list(merged_chunk_types),
            reasoning=reasoning,
        )

    def _default_multi_classification(self, query: str) -> MultiQueryClassification:
        """Return default multi-classification when parsing fails."""
        return MultiQueryClassification(
            query_types=[(QueryType.GENERAL, 0.5)],
            primary_type=QueryType.GENERAL,
            entities=[],
            needs_cross_corpus=False,
            merged_chunk_types=RETRIEVAL_STRATEGIES[QueryType.GENERAL]["chunk_types"],
            reasoning="Default classification due to parsing error",
        )


# Technical terms that signal METHODS even with "how should I describe" phrasing
METHODS_TECHNICAL_TERMS = {
    # Equipment and setup
    "setup", "microscopy", "microscope", "imaging parameters", "laser", "wavelength",
    "objective", "power", "detector", "spectrometer",
    # Protocols and procedures
    "protocol", "procedure", "synthesized", "synthesis", "purified", "purification",
    "expressed", "expression", "incubation", "incubated", "buffer", "concentration",
    "temperature", "ph", "chromatography", "fplc", "hplc", "column",
    # Assays
    "assay", "mic", "ic50", "ec50", "cc50", "measured", "quantified",
    # Cell/molecular biology
    "cell line", "cell lines", "transfected", "cultured", "stained",
    # Specific techniques
    "metathesis", "click chemistry", "labeling", "tagged", "conjugated",
}

# Terms that signal NOVELTY (assessing prior work)
NOVELTY_SIGNAL_TERMS = {
    "novel", "new", "unique", "first", "defensible", "underexplored", "gap",
    "been done before", "been applied", "been used", "been reported",
    "existing", "prior work", "previous work", "prior art", "distinct from",
    "different from previous", "what makes", "strongest claim",
    "are there", "has anyone", "has this", "have others",
}


# Simple heuristic classifier (no API call, for testing)
def classify_query_heuristic(query: str) -> QueryType:
    """Classify query using keyword heuristics.

    Use for testing or when API calls should be avoided.
    """
    query_lower = query.lower()

    # Check for technical terms first - these override framing patterns
    has_technical_terms = any(term in query_lower for term in METHODS_TECHNICAL_TERMS)

    # Framing patterns - but only if NO technical terms present
    framing_patterns = [
        "how do i frame", "how do i position",
        "how do i justify", "how can i present",
        "how do i argue", "how should i argue",
        "significance of", "importance of",
    ]
    # "how should i describe" and "how should i write" are ambiguous - check for technical terms
    ambiguous_framing = ["how should i describe", "how should i write", "how do i explain"]

    if any(p in query_lower for p in framing_patterns) and not has_technical_terms:
        return QueryType.FRAMING

    # If using ambiguous phrasing but has technical terms → METHODS
    if any(p in query_lower for p in ambiguous_framing):
        if has_technical_terms:
            return QueryType.METHODS
        else:
            return QueryType.FRAMING

    # Methods patterns - expanded list
    methods_patterns = [
        "what buffer", "what concentration", "how was", "protocol",
        "what temperature", "incubation", "procedure", "what method",
        "how are", "how is", "typically synthesized", "typically purified",
        "commonly used", "what protocols", "what controls", "what cell line",
    ]
    if any(p in query_lower for p in methods_patterns):
        return QueryType.METHODS

    # Explicit technical queries
    if has_technical_terms and ("how" in query_lower or "what" in query_lower):
        return QueryType.METHODS

    # Summary patterns
    if any(p in query_lower for p in ["summarize", "summary", "key findings", "main points"]):
        return QueryType.SUMMARY

    # Comparative patterns
    if any(p in query_lower for p in ["compare", "difference between", "versus", " vs "]):
        return QueryType.COMPARATIVE

    # Novelty patterns - expanded to catch prior work assessment
    if any(term in query_lower for term in NOVELTY_SIGNAL_TERMS):
        return QueryType.NOVELTY

    # Limitations patterns
    if any(p in query_lower for p in ["limitation", "caveat", "constraint", "address the limitation"]):
        return QueryType.LIMITATIONS

    # Default to general for uncategorized queries
    return QueryType.GENERAL
