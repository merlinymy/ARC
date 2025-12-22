"""Cohere reranker wrapper for result reranking.

Reranks retrieval results using Cohere's rerank-v3.5 model
to improve relevance ranking beyond embedding similarity.
"""

import logging
from typing import List, Dict, Any
import cohere

logger = logging.getLogger(__name__)


class CohereReranker:
    """Cohere reranker client."""

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-v3.5"
    ):
        """Initialize reranker.

        Args:
            api_key: Cohere API key
            model: Reranker model name
        """
        self.client = cohere.Client(api_key)
        self.model = model

        logger.info(f"Initialized CohereReranker with model {model}")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = 15,
        text_field: str = "text"
    ) -> List[Dict[str, Any]]:
        """Rerank documents by relevance to query.

        Args:
            query: The search query
            documents: List of document dicts with text content
            top_n: Number of top results to return
            text_field: Field name containing the text to rerank on

        Returns:
            Top N documents sorted by rerank score, with 'rerank_score' added
        """
        if not documents:
            return []

        # Extract texts for reranking
        texts = [doc[text_field] for doc in documents]

        try:
            # Call Cohere rerank
            response = self.client.rerank(
                query=query,
                documents=texts,
                model=self.model,
                top_n=min(top_n, len(documents))
            )

            # Build result with rerank scores
            reranked = []
            for result in response.results:
                doc = documents[result.index].copy()
                doc['rerank_score'] = result.relevance_score
                reranked.append(doc)

            logger.debug(f"Reranked {len(documents)} docs to top {len(reranked)}")
            return reranked

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original documents without reranking on error
            return documents[:top_n]

    def rerank_with_metadata(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = 15,
        text_field: str = "text",
        max_per_paper: int = 3
    ) -> List[Dict[str, Any]]:
        """Rerank and deduplicate by paper_id.

        Args:
            query: The search query
            documents: List of document dicts
            top_n: Number of top results to return
            text_field: Field containing text
            max_per_paper: Maximum chunks per paper in results

        Returns:
            Reranked and deduplicated results
        """
        # First rerank all documents
        reranked = self.rerank(query, documents, top_n=len(documents), text_field=text_field)

        # Deduplicate by paper_id
        paper_counts: Dict[str, int] = {}
        deduplicated = []

        for doc in reranked:
            paper_id = doc.get('paper_id', 'unknown')

            if paper_id not in paper_counts:
                paper_counts[paper_id] = 0

            if paper_counts[paper_id] < max_per_paper:
                deduplicated.append(doc)
                paper_counts[paper_id] += 1

            if len(deduplicated) >= top_n:
                break

        return deduplicated
