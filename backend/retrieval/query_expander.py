"""Query expansion using domain synonyms.

Expands user queries with domain-specific synonyms to improve
retrieval of documents that use different terminology for the
same concepts.
"""

import logging
from typing import List, Tuple, Set

from .domain_synonyms import DOMAIN_SYNONYMS, get_synonyms, find_entities_in_text

logger = logging.getLogger(__name__)


class QueryExpander:
    """Expand queries with domain synonyms."""

    def __init__(self, max_expansion_terms: int = 5):
        """Initialize query expander.

        Args:
            max_expansion_terms: Maximum number of synonym terms to add
        """
        self.max_expansion_terms = max_expansion_terms
        self.synonyms = DOMAIN_SYNONYMS

    def expand_query(self, query: str) -> Tuple[str, List[str]]:
        """Expand query with domain synonyms.

        Args:
            query: Original user query

        Returns:
            Tuple of (expanded_query, list_of_added_terms)
        """
        query_lower = query.lower()
        added_terms = []

        # Find entities in query and collect synonyms
        for canonical, aliases in self.synonyms.items():
            all_terms = [canonical.lower()] + [a.lower() for a in aliases]

            # Check if any form of this entity is in the query
            found_in_query = False
            for term in all_terms:
                if term in query_lower:
                    found_in_query = True
                    break

            if found_in_query:
                # Add synonyms that aren't already in query
                for alias in aliases[:3]:  # Limit per entity
                    if alias.lower() not in query_lower:
                        added_terms.append(alias)

        # Limit total expansion terms
        added_terms = added_terms[:self.max_expansion_terms]

        if added_terms:
            expanded = f"{query} (related terms: {', '.join(added_terms)})"
            logger.debug(f"Expanded query: {query} -> added {added_terms}")
            return expanded, added_terms

        return query, []

    def extract_entities(self, query: str) -> Set[str]:
        """Extract known domain entities from query.

        Args:
            query: User query

        Returns:
            Set of canonical entity names found
        """
        return find_entities_in_text(query)


# Convenience function
def expand_query(query: str, max_terms: int = 5) -> str:
    """Expand query with domain synonyms.

    Args:
        query: Original query
        max_terms: Maximum expansion terms

    Returns:
        Expanded query string
    """
    expander = QueryExpander(max_expansion_terms=max_terms)
    expanded, _ = expander.expand_query(query)
    return expanded
