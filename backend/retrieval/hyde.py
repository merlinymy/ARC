"""HyDE - Hypothetical Document Embeddings.

Instead of embedding the user query directly, HyDE:
1. Generates a hypothetical answer to the query using an LLM
2. Embeds the hypothetical answer
3. Searches for real documents similar to the hypothetical answer

This often produces better retrieval because the hypothetical answer
is closer to actual document content than a short query.

Reference: https://arxiv.org/abs/2212.10496
"""

import logging
from typing import List, Optional, Tuple
from anthropic import Anthropic

logger = logging.getLogger(__name__)


HYDE_PROMPT = '''You are a research paper excerpt generator. Given a question about scientific research,
generate a detailed, factual paragraph that would likely appear in a research paper answering this question.

The paragraph should:
- Be written in formal academic style
- Include specific technical details, methods, or findings
- Be 100-200 words long
- Sound like it came from a real Methods, Results, or Discussion section

Question: {query}

Generate a hypothetical research paper excerpt that would answer this question:'''


HYDE_PROMPT_BY_TYPE = {
    "methods": '''Generate a detailed Methods section excerpt that would describe the experimental procedure for:
Question: {query}

Write as if from a real research paper, including specific reagents, conditions, and steps:''',

    "results": '''Generate a Results section excerpt presenting findings related to:
Question: {query}

Write as if from a real research paper, including specific data and observations:''',

    "discussion": '''Generate a Discussion section excerpt analyzing:
Question: {query}

Write as if from a real research paper, including interpretation and implications:''',

    "factual": '''Generate a technical paragraph that would answer:
Question: {query}

Include specific facts, values, or mechanisms:''',
}


# Diversity axis for generate_multiple(). Sampling temperature used to supply the
# variation between the three hypothetical documents, but the API no longer accepts
# it, so the variation now comes from the prompt instead: three different section
# framings of the same question, reusing HYDE_PROMPT_BY_TYPE above rather than
# duplicating it. "methods" pulls protocol and reagent language, "results" pulls
# data and numeric findings, "discussion" pulls background and mechanism - so the
# three embeddings land in genuinely different neighbourhoods of the corpus.
HYDE_DIVERSITY_FRAMINGS = ("methods", "results", "discussion")


class HyDE:
    """Hypothetical Document Embeddings for improved retrieval."""

    def __init__(
        self,
        anthropic_client: Anthropic,
        model: str = "claude-haiku-4-5",  # Use fast model for HyDE
        enabled: bool = True,
    ):
        """Initialize HyDE.

        Args:
            anthropic_client: Anthropic client for generation
            model: Model to use for hypothetical generation (prefer fast model)
            enabled: Whether HyDE is enabled
        """
        self.anthropic = anthropic_client
        self.model = model
        self.enabled = enabled

        logger.info(f"Initialized HyDE with model {model}, enabled={enabled}")

    def generate_hypothetical(
        self,
        query: str,
        query_type: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Generate a hypothetical document that would answer the query.

        Args:
            query: User's question
            query_type: Optional query type for specialized prompts

        Returns:
            Tuple of (hypothetical_document, combined_text_for_embedding)
        """
        if not self.enabled:
            return "", query

        try:
            # Select appropriate prompt
            if query_type and query_type.lower() in HYDE_PROMPT_BY_TYPE:
                prompt = HYDE_PROMPT_BY_TYPE[query_type.lower()].format(query=query)
            else:
                prompt = HYDE_PROMPT.format(query=query)

            response = self.anthropic.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )

            hypothetical = response.content[0].text.strip()

            # Combine query + hypothetical for embedding
            # This preserves the original intent while adding document-like context
            combined = f"Query: {query}\n\nRelevant excerpt: {hypothetical}"

            logger.debug(f"HyDE generated {len(hypothetical)} chars for: {query[:50]}...")
            return hypothetical, combined

        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}, falling back to query")
            return "", query

    def generate_multiple(
        self,
        query: str,
        n: int = 3,
        query_type: Optional[str] = None,
    ) -> List[str]:
        """Generate multiple hypothetical documents for diversity.

        Args:
            query: User's question
            n: Number of hypotheticals to generate
            query_type: Optional query type

        Returns:
            List of hypothetical documents
        """
        if not self.enabled:
            return [query]

        hypotheticals = []

        # One call per framing; n caps how many framings are used.
        for framing in HYDE_DIVERSITY_FRAMINGS[:n]:
            try:
                prompt = HYDE_PROMPT_BY_TYPE[framing].format(query=query)

                response = self.anthropic.messages.create(
                    model=self.model,
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt}]
                )

                hypotheticals.append(response.content[0].text.strip())

            except Exception as e:
                logger.warning(f"HyDE multi-generation ({framing}) failed: {e}")

        return hypotheticals if hypotheticals else [query]


class HyDEEmbedder:
    """Combines HyDE with an embedder for end-to-end hypothetical embedding."""

    def __init__(
        self,
        hyde: HyDE,
        embedder,  # VoyageEmbedder
        cache=None,  # Optional RAGCache
    ):
        """Initialize HyDE embedder.

        Args:
            hyde: HyDE instance
            embedder: Embedding model (VoyageEmbedder)
            cache: Optional cache for HyDE results
        """
        self.hyde = hyde
        self.embedder = embedder
        self.cache = cache

    def embed_query_with_hyde(
        self,
        query: str,
        query_type: Optional[str] = None,
        use_cache: bool = True,
    ) -> Tuple[List[float], str]:
        """Embed query using HyDE.

        Args:
            query: User's question
            query_type: Optional query type for specialized prompts
            use_cache: Whether to use cached results

        Returns:
            Tuple of (embedding, text_that_was_embedded)
        """
        # Check cache for HyDE result
        if use_cache and self.cache:
            cached_hyde = self.cache.get_hyde_answer(query)
            if cached_hyde:
                combined = f"Query: {query}\n\nRelevant excerpt: {cached_hyde}"
                embedding = self.embedder.embed_query(combined)
                return embedding, combined

        # Generate hypothetical
        hypothetical, combined = self.hyde.generate_hypothetical(query, query_type)

        # Cache the result
        if use_cache and self.cache and hypothetical:
            self.cache.set_hyde_answer(query, hypothetical)

        # Embed the combined text
        embedding = self.embedder.embed_query(combined)

        return embedding, combined

    def embed_query_multi_hyde(
        self,
        query: str,
        n: int = 3,
    ) -> List[List[float]]:
        """Generate multiple HyDE embeddings for diversity.

        Args:
            query: User's question
            n: Number of hypotheticals

        Returns:
            List of embeddings
        """
        hypotheticals = self.hyde.generate_multiple(query, n)
        embeddings = []

        for hyp in hypotheticals:
            combined = f"Query: {query}\n\nRelevant excerpt: {hyp}"
            embedding = self.embedder.embed_query(combined)
            embeddings.append(embedding)

        return embeddings
