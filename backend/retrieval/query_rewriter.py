"""Advanced query rewriting for improved retrieval.

Features:
- LLM-based query clarification
- Query decomposition for complex questions
- Synonym expansion with context awareness
- Spelling correction for scientific terms
"""

import logging
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RewrittenQuery:
    """Result of query rewriting."""
    original: str
    rewritten: str
    sub_queries: List[str]  # For decomposed queries
    expansions: List[str]  # Synonym expansions added
    corrections: List[Tuple[str, str]]  # (original, corrected) pairs


# Common scientific term misspellings
SPELLING_CORRECTIONS = {
    # Chemistry
    "chromotography": "chromatography",
    "chromtography": "chromatography",
    "spectometry": "spectrometry",
    "flourescence": "fluorescence",
    "flourescent": "fluorescent",
    "catalisis": "catalysis",
    "catalist": "catalyst",
    "moleculer": "molecular",
    "sythesis": "synthesis",
    "syntheis": "synthesis",
    "crystalization": "crystallization",
    "crystalisation": "crystallisation",
    "soluable": "soluble",
    "titration": "titration",
    "equilibrum": "equilibrium",
    "stoichiomtery": "stoichiometry",

    # Biology
    "protien": "protein",
    "protiens": "proteins",
    "enzyem": "enzyme",
    "enzym": "enzyme",
    "mitocondria": "mitochondria",
    "mitocondrial": "mitochondrial",
    "nuclueus": "nucleus",
    "nuclues": "nucleus",
    "recptor": "receptor",
    "recpetor": "receptor",
    "antibodiy": "antibody",
    "antobody": "antibody",
    "apoptsis": "apoptosis",
    "apotosis": "apoptosis",
    "phosphorlyation": "phosphorylation",
    "phosphrylation": "phosphorylation",

    # Methods
    "electrophorsis": "electrophoresis",
    "electrophresis": "electrophoresis",
    "centerfugation": "centrifugation",
    "centrifigation": "centrifugation",
    "lyophilzation": "lyophilization",
    "incubaton": "incubation",
    "transfection": "transfection",

    # General
    "signifcant": "significant",
    "significent": "significant",
    "concentraion": "concentration",
    "concentraton": "concentration",
    "efficacy": "efficacy",
    "efficency": "efficiency",
    "dependant": "dependent",
    "independant": "independent",
}

# Scientific acronyms and their expansions
ACRONYM_EXPANSIONS = {
    "PCR": ["polymerase chain reaction", "PCR"],
    "HPLC": ["high performance liquid chromatography", "HPLC"],
    "GC-MS": ["gas chromatography mass spectrometry", "GC-MS"],
    "LC-MS": ["liquid chromatography mass spectrometry", "LC-MS"],
    "NMR": ["nuclear magnetic resonance", "NMR"],
    "IC50": ["half maximal inhibitory concentration", "IC50"],
    "EC50": ["half maximal effective concentration", "EC50"],
    "MIC": ["minimum inhibitory concentration", "MIC"],
    "SAR": ["structure activity relationship", "SAR"],
    "QSAR": ["quantitative structure activity relationship", "QSAR"],
    "ELISA": ["enzyme linked immunosorbent assay", "ELISA"],
    "SDS-PAGE": ["sodium dodecyl sulfate polyacrylamide gel electrophoresis", "SDS-PAGE"],
    "Western blot": ["immunoblot", "Western blot", "western blotting"],
    "qPCR": ["quantitative PCR", "real-time PCR", "qPCR"],
    "siRNA": ["small interfering RNA", "siRNA"],
    "shRNA": ["short hairpin RNA", "shRNA"],
    "CRISPR": ["CRISPR-Cas9", "CRISPR", "gene editing"],
    "RNAi": ["RNA interference", "RNAi", "gene silencing"],
    "SPR": ["surface plasmon resonance", "SPR"],
    "ITC": ["isothermal titration calorimetry", "ITC"],
    "CD": ["circular dichroism", "CD spectroscopy"],
    "DLS": ["dynamic light scattering", "DLS"],
    "TEM": ["transmission electron microscopy", "TEM"],
    "SEM": ["scanning electron microscopy", "SEM"],
    "AFM": ["atomic force microscopy", "AFM"],
    "XRD": ["X-ray diffraction", "XRD"],
    "FTIR": ["Fourier transform infrared spectroscopy", "FTIR"],
}

# Query patterns for decomposition
COMPLEX_QUERY_PATTERNS = [
    # Comparison queries
    (r"compare\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+)", "comparison"),
    (r"difference\s+between\s+(.+?)\s+and\s+(.+)", "comparison"),
    (r"(.+?)\s+(?:vs\.?|versus)\s+(.+)", "comparison"),

    # Multi-part queries
    (r"(.+?)\s+and\s+(?:also|additionally)\s+(.+)", "multi_part"),
    (r"(.+?)\.?\s+(?:Also|Additionally|Furthermore),?\s+(.+)", "multi_part"),

    # Cause-effect queries
    (r"how\s+does\s+(.+?)\s+affect\s+(.+)", "cause_effect"),
    (r"what\s+is\s+the\s+effect\s+of\s+(.+?)\s+on\s+(.+)", "cause_effect"),
    (r"relationship\s+between\s+(.+?)\s+and\s+(.+)", "cause_effect"),
]


class QueryRewriter:
    """Rewrite and enhance queries for better retrieval."""

    def __init__(
        self,
        anthropic_client=None,
        model: str = "claude-3-haiku-20240307",
        enable_llm_rewrite: bool = True,
    ):
        """Initialize query rewriter.

        Args:
            anthropic_client: Anthropic client for LLM rewriting
            model: Claude model to use
            enable_llm_rewrite: Whether to use LLM for rewriting
        """
        self.client = anthropic_client
        self.model = model
        self.enable_llm_rewrite = enable_llm_rewrite and anthropic_client is not None

    def rewrite(
        self,
        query: str,
        context: Optional[str] = None,
        use_llm: bool = True,
    ) -> RewrittenQuery:
        """Rewrite query for improved retrieval.

        Args:
            query: Original user query
            context: Optional conversation context
            use_llm: Whether to use LLM for this query

        Returns:
            RewrittenQuery with rewritten query and metadata
        """
        # Step 1: Spelling correction
        corrected, corrections = self._correct_spelling(query)

        # Step 2: Acronym expansion
        expanded, expansions = self._expand_acronyms(corrected)

        # Step 3: Check for complex queries that need decomposition
        sub_queries = self._decompose_if_complex(expanded)

        # Step 4: LLM rewriting for ambiguous queries (if enabled)
        if use_llm and self.enable_llm_rewrite and self._is_ambiguous(expanded):
            rewritten = self._llm_rewrite(expanded, context)
        else:
            rewritten = expanded

        return RewrittenQuery(
            original=query,
            rewritten=rewritten,
            sub_queries=sub_queries,
            expansions=expansions,
            corrections=corrections,
        )

    def _correct_spelling(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """Correct common scientific misspellings.

        Args:
            text: Input text

        Returns:
            Tuple of (corrected_text, list of (original, correction) pairs)
        """
        corrections = []
        result = text

        for wrong, correct in SPELLING_CORRECTIONS.items():
            pattern = re.compile(r'\b' + re.escape(wrong) + r'\b', re.IGNORECASE)
            if pattern.search(result):
                corrections.append((wrong, correct))
                result = pattern.sub(correct, result)

        return result, corrections

    def _expand_acronyms(self, text: str) -> Tuple[str, List[str]]:
        """Add acronym expansions to improve retrieval.

        Args:
            text: Input text

        Returns:
            Tuple of (expanded_text, list of expansions added)
        """
        expansions = []
        expansion_terms = []

        for acronym, alternatives in ACRONYM_EXPANSIONS.items():
            pattern = re.compile(r'\b' + re.escape(acronym) + r'\b', re.IGNORECASE)
            if pattern.search(text):
                # Add the expansion (not the acronym itself which is already there)
                for alt in alternatives:
                    if alt.lower() != acronym.lower() and alt.lower() not in text.lower():
                        expansion_terms.append(alt)
                        expansions.append(alt)
                        break  # Just add one expansion per acronym

        if expansion_terms:
            text = f"{text} ({', '.join(expansion_terms[:3])})"

        return text, expansions

    def _decompose_if_complex(self, query: str) -> List[str]:
        """Decompose complex queries into sub-queries.

        Args:
            query: Input query

        Returns:
            List of sub-queries (empty if not decomposable)
        """
        for pattern, query_type in COMPLEX_QUERY_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                groups = match.groups()
                if query_type == "comparison":
                    return [
                        f"What is {groups[0].strip()}?",
                        f"What is {groups[1].strip()}?",
                        query,  # Keep original for comparison aspect
                    ]
                elif query_type == "cause_effect":
                    return [
                        f"What is {groups[0].strip()}?",
                        f"How does {groups[0].strip()} affect {groups[1].strip()}?",
                    ]
                elif query_type == "multi_part":
                    return [
                        groups[0].strip() + "?",
                        groups[1].strip() + "?",
                    ]

        return []

    def _is_ambiguous(self, query: str) -> bool:
        """Check if query is ambiguous and needs LLM rewriting.

        Args:
            query: Input query

        Returns:
            True if query is ambiguous
        """
        # Short queries are often ambiguous
        if len(query.split()) <= 3:
            return True

        # Queries with pronouns might need context
        pronouns = ["it", "this", "that", "they", "these", "those"]
        if any(f" {p} " in f" {query.lower()} " for p in pronouns):
            return True

        # Very generic queries
        generic_starts = [
            "what is", "how does", "why is", "tell me about",
            "explain", "describe",
        ]
        query_lower = query.lower()
        for start in generic_starts:
            if query_lower.startswith(start) and len(query.split()) <= 5:
                return True

        return False

    def _llm_rewrite(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> str:
        """Use LLM to rewrite ambiguous query.

        Args:
            query: Input query
            context: Optional conversation context

        Returns:
            Rewritten query
        """
        if not self.client:
            return query

        context_text = f"\nConversation context: {context}" if context else ""

        prompt = f"""Rewrite this search query to be more specific and better for searching a scientific paper database.
Keep the same meaning but make it clearer and more searchable.
Only output the rewritten query, nothing else.

Original query: {query}{context_text}

Rewritten query:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )

            rewritten = response.content[0].text.strip()

            # Sanity check: rewritten shouldn't be too different
            if len(rewritten) > len(query) * 3:
                return query

            logger.debug(f"LLM rewrote: '{query}' -> '{rewritten}'")
            return rewritten

        except Exception as e:
            logger.warning(f"LLM rewrite failed: {e}")
            return query


def rewrite_query(
    query: str,
    context: Optional[str] = None,
    anthropic_client=None,
) -> str:
    """Convenience function to rewrite a query.

    Args:
        query: Original query
        context: Optional conversation context
        anthropic_client: Optional Anthropic client

    Returns:
        Rewritten query string
    """
    rewriter = QueryRewriter(anthropic_client=anthropic_client)
    result = rewriter.rewrite(query, context)
    return result.rewritten


def correct_scientific_spelling(text: str) -> str:
    """Convenience function to correct scientific spelling.

    Args:
        text: Input text

    Returns:
        Text with corrected spelling
    """
    rewriter = QueryRewriter(enable_llm_rewrite=False)
    corrected, _ = rewriter._correct_spelling(text)
    return corrected
