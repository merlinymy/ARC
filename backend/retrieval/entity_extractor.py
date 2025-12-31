"""Entity extraction for research queries.

Extracts scientific entities from queries for:
- Metadata filtering (find papers mentioning specific compounds)
- Query enhancement (add related terms)
- Result validation (verify retrieved chunks contain relevant entities)

Entity types:
- Chemicals/Compounds (drug names, chemical formulas)
- Proteins/Genes (protein names, gene symbols)
- Methods/Techniques (assay names, equipment)
- Organisms (species, cell lines)
- Metrics (IC50, EC50, Ki, etc.)
"""

import re
import logging
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntities:
    """Container for extracted entities."""
    chemicals: List[str] = field(default_factory=list)
    proteins: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    organisms: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    other: List[str] = field(default_factory=list)

    def all_entities(self) -> List[str]:
        """Get all entities as a flat list (deduplicated)."""
        all_ents = (
            self.chemicals + self.proteins + self.methods +
            self.organisms + self.metrics + self.other
        )
        # Deduplicate while preserving order
        seen = set()
        result = []
        for e in all_ents:
            e_lower = e.lower()
            if e_lower not in seen:
                seen.add(e_lower)
                result.append(e)
        return result

    def is_empty(self) -> bool:
        """Check if no entities were extracted."""
        return len(self.all_entities()) == 0

    def to_dict(self) -> Dict[str, List[str]]:
        """Convert to dictionary."""
        return {
            "chemicals": self.chemicals,
            "proteins": self.proteins,
            "methods": self.methods,
            "organisms": self.organisms,
            "metrics": self.metrics,
            "other": self.other,
        }


class EntityExtractor:
    """Rule-based entity extraction for scientific text."""

    def __init__(self):
        """Initialize with pattern libraries."""
        # Chemical patterns
        self.chemical_patterns = [
            r'\b[A-Z]{2,4}-\d{3,6}\b',  # Drug codes: AB-12345
            r'\b\d+-[A-Z][a-z]+-\d+\b',  # Chemical names with numbers
            r'\b[A-Z][a-z]+(?:in|ol|ine|ate|ide|ase|one)\b',  # Common drug suffixes
            r'\b(?:sodium|potassium|calcium|chloride|sulfate|phosphate)\b',
        ]

        # Protein/Gene patterns - more specific to avoid false positives
        self.protein_patterns = [
            r'\b[A-Z]{2,5}\d{1,2}\b',  # Gene symbols with numbers: BRCA1, TP53, HER2
            r'\b(?:ERα|ERβ|ER-alpha|ER-beta)\b',  # Estrogen receptors
            r'\b(?:kinase|receptor|enzyme|antibody|peptide)\b',
            r'\b[A-Z][a-z]+(?:ase|globin)\b',  # Enzyme/protein suffixes: lipase, hemoglobin
            r'\bLL-?37\b|\bLL37\b',  # Specific peptides
        ]

        # Method/Technique patterns
        self.method_patterns = [
            r'\b(?:HPLC|LC-MS|MS|NMR|PCR|ELISA|Western\s*blot)\b',
            r'\b(?:SRS|SERS|Raman|FDTD|DFT|spectroscopy)\b',
            r'\b(?:assay|microscopy|chromatography|electrophoresis)\b',
            r'\b(?:flow\s*cytometry|mass\s*spec(?:trometry)?)\b',
            r'\b(?:synthesis|purification|extraction|incubation)\b',
        ]

        # Organism patterns
        self.organism_patterns = [
            r'\b(?:E\.\s*coli|S\.\s*aureus|HeLa|HEK\s*293|CHO)\b',
            r'\b(?:mouse|mice|rat|human|yeast|bacteria)\b',
            r'\b[A-Z]\.\s*[a-z]+\b',  # Species abbreviations
        ]

        # Metric patterns
        self.metric_patterns = [
            r'\bIC50\b|\bEC50\b|\bKi\b|\bKd\b|\bKm\b',
            r'\b(?:nM|μM|mM|pM)\b',
            r'\b\d+(?:\.\d+)?\s*(?:nM|μM|mM|ng/mL|μg/mL)\b',
            r'\bpH\s*\d+(?:\.\d+)?\b',
        ]

        # Compile patterns
        self._compiled = {
            'chemicals': [re.compile(p, re.IGNORECASE) for p in self.chemical_patterns],
            'proteins': [re.compile(p, re.IGNORECASE) for p in self.protein_patterns],
            'methods': [re.compile(p, re.IGNORECASE) for p in self.method_patterns],
            'organisms': [re.compile(p, re.IGNORECASE) for p in self.organism_patterns],
            'metrics': [re.compile(p, re.IGNORECASE) for p in self.metric_patterns],
        }

        # Common words to exclude (false positives)
        self.exclude_words = {
            'the', 'and', 'for', 'with', 'from', 'that', 'this', 'are', 'was',
            'were', 'been', 'have', 'has', 'what', 'when', 'where', 'which',
            'how', 'why', 'who', 'can', 'could', 'would', 'should', 'may',
            'protein', 'method', 'result', 'study', 'paper', 'research',
        }

    def extract(self, text: str) -> ExtractedEntities:
        """Extract all entity types from text.

        Args:
            text: Input text (query or document)

        Returns:
            ExtractedEntities with categorized entities
        """
        entities = ExtractedEntities()

        # Extract each entity type
        entities.chemicals = self._extract_type(text, 'chemicals')
        entities.proteins = self._extract_type(text, 'proteins')
        entities.methods = self._extract_type(text, 'methods')
        entities.organisms = self._extract_type(text, 'organisms')
        entities.metrics = self._extract_type(text, 'metrics')

        return entities

    def _extract_type(self, text: str, entity_type: str) -> List[str]:
        """Extract entities of a specific type."""
        matches = set()

        for pattern in self._compiled.get(entity_type, []):
            for match in pattern.finditer(text):
                entity = match.group().strip()
                # Filter out common words and very short matches
                if (entity.lower() not in self.exclude_words and
                    len(entity) > 2):
                    matches.add(entity)

        return sorted(list(matches))

    def extract_for_filtering(self, query: str) -> Dict[str, List[str]]:
        """Extract entities suitable for metadata filtering.

        Returns entities that could match paper metadata fields.
        """
        entities = self.extract(query)

        # Return non-empty categories only
        result = {}
        for key, values in entities.to_dict().items():
            if values:
                result[key] = values

        return result

    def score_chunk_relevance(
        self,
        query: str,
        chunk_text: str,
    ) -> Tuple[float, List[str]]:
        """Score chunk relevance based on entity overlap.

        Args:
            query: User query
            chunk_text: Retrieved chunk text

        Returns:
            Tuple of (relevance_score, matched_entities)
        """
        query_entities = self.extract(query)
        chunk_entities = self.extract(chunk_text)

        query_all = set(e.lower() for e in query_entities.all_entities())
        chunk_all = set(e.lower() for e in chunk_entities.all_entities())

        if not query_all:
            return 1.0, []  # No entities to match, neutral score

        matched = query_all.intersection(chunk_all)
        score = len(matched) / len(query_all) if query_all else 0

        return score, sorted(list(matched))


class LLMEntityExtractor:
    """LLM-based entity extraction for complex cases."""

    EXTRACTION_PROMPT = '''Extract scientific entities from this research question.

Categories:
- CHEMICALS: Drug names, compound codes, chemical names
- PROTEINS: Protein names, gene symbols, enzymes
- METHODS: Techniques, assays, equipment
- ORGANISMS: Species, cell lines, model organisms
- METRICS: IC50, EC50, concentrations, measurements

Question: {query}

Return as JSON:
{{"chemicals": [...], "proteins": [...], "methods": [...], "organisms": [...], "metrics": [...]}}

Only include entities actually mentioned. Return empty lists for missing categories.'''

    def __init__(self, anthropic_client, model: str = "claude-3-haiku-20240307"):
        """Initialize with Anthropic client."""
        self.anthropic = anthropic_client
        self.model = model
        self.rule_based = EntityExtractor()  # Fallback

    def extract(self, query: str) -> ExtractedEntities:
        """Extract entities using LLM with rule-based fallback."""
        try:
            response = self.anthropic.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": self.EXTRACTION_PROMPT.format(query=query)
                }]
            )

            # Parse JSON response
            import json
            text = response.content[0].text
            # Find JSON in response
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return ExtractedEntities(
                    chemicals=data.get('chemicals', []),
                    proteins=data.get('proteins', []),
                    methods=data.get('methods', []),
                    organisms=data.get('organisms', []),
                    metrics=data.get('metrics', []),
                )

        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}, using rule-based")

        # Fallback to rule-based
        return self.rule_based.extract(query)

    def score_chunk_relevance(
        self,
        query: str,
        chunk_text: str,
    ) -> Tuple[float, List[str]]:
        """Score chunk relevance based on entity overlap.

        Delegates to rule-based extractor for scoring (fast, no API call).
        """
        return self.rule_based.score_chunk_relevance(query, chunk_text)
