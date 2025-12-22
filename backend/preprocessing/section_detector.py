"""Detect and classify sections in research papers.

Enhanced patterns for chemistry and biology papers including:
- Combined Results and Discussion sections
- Chemistry-specific sections (Synthesis, Characterization)
- Various numbering schemes
"""

import re
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Section:
    """A detected section in a paper."""
    name: str
    normalized_name: str  # e.g., "methods", "results", "discussion"
    start_idx: int
    end_idx: int
    text: str
    level: int  # 1 for main sections, 2 for subsections


# Common section headers in scientific papers (case-insensitive patterns)
# Order matters - more specific patterns first
SECTION_PATTERNS = [
    # Combined sections (common in chemistry)
    (r"^(?:\d+\.?\s*)?results?\s*(?:and|&)\s*discussion", "results_discussion", 1),
    (r"^(?:\d+\.?\s*)?materials?\s*(?:and|&)\s*methods?", "methods", 1),

    # Standard sections
    (r"^(?:1\.?\s*)?(?:introduction|background)", "introduction", 1),
    (r"^(?:2\.?\s*)?(?:methods?|experimental\s*(?:section|procedures?)?)", "methods", 1),
    (r"^(?:3\.?\s*)?(?:results?)", "results", 1),
    (r"^(?:4\.?\s*)?(?:discussion)", "discussion", 1),
    (r"^(?:5\.?\s*)?(?:conclusion|conclusions|summary|concluding\s*remarks)", "conclusion", 1),
    (r"^abstract", "abstract", 1),

    # Chemistry-specific sections
    (r"^(?:\d+\.?\s*)?(?:synthesis|synthetic\s*procedures?|general\s*synthesis)", "synthesis", 1),
    (r"^(?:\d+\.?\s*)?(?:characterization|compound\s*characterization)", "characterization", 1),
    (r"^(?:\d+\.?\s*)?(?:biological\s*evaluation|bioactivity|biological\s*activity)", "bioactivity", 1),
    (r"^(?:\d+\.?\s*)?(?:molecular\s*docking|docking\s*studies|computational)", "computational", 1),
    (r"^(?:\d+\.?\s*)?(?:structure[- ]activity|SAR|structure-activity\s*relationship)", "sar", 1),

    # Other common sections
    (r"^(?:references?|bibliography|literature\s*cited)", "references", 1),
    (r"^(?:acknowledg|funding|support|author\s*contributions)", "acknowledgments", 1),
    (r"^(?:supplementa|supporting\s*information|appendix|SI\s)", "supplementary", 1),
    (r"^(?:abbreviations?|glossary)", "abbreviations", 1),

    # Numbered subsections (level 2)
    (r"^\d+\.\d+\.?\s+", "subsection", 2),
]


class SectionDetector:
    """Detect sections in extracted PDF text."""

    def __init__(self):
        self.patterns = [(re.compile(p, re.IGNORECASE | re.MULTILINE), name, level)
                         for p, name, level in SECTION_PATTERNS]

    def detect_sections(self, text: str) -> List[Section]:
        """Detect all sections in the text.

        Args:
            text: Full text of the paper

        Returns:
            List of Section objects in order of appearance
        """
        # Find all potential section headers
        candidates = []
        lines = text.split('\n')
        current_pos = 0

        for line_idx, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                current_pos += len(line) + 1
                continue

            # Skip lines that are too long (unlikely to be headers)
            if len(line_stripped) > 100:
                current_pos += len(line) + 1
                continue

            # Check if line matches any section pattern
            for pattern, normalized_name, level in self.patterns:
                if pattern.match(line_stripped):
                    candidates.append({
                        'name': line_stripped,
                        'normalized_name': normalized_name,
                        'start_idx': current_pos,
                        'level': level,
                        'line_idx': line_idx
                    })
                    break

            current_pos += len(line) + 1

        # Build sections with end indices
        sections = []
        for i, candidate in enumerate(candidates):
            end_idx = candidates[i + 1]['start_idx'] if i + 1 < len(candidates) else len(text)
            section_text = text[candidate['start_idx']:end_idx].strip()

            sections.append(Section(
                name=candidate['name'],
                normalized_name=candidate['normalized_name'],
                start_idx=candidate['start_idx'],
                end_idx=end_idx,
                text=section_text,
                level=candidate['level']
            ))

        return sections

    def extract_abstract(self, text: str) -> Optional[str]:
        """Extract abstract from paper text.

        Handles common formats:
        - "Abstract" followed by text
        - "Abstract:" followed by text
        - Text before "Introduction" if paper starts with abstract
        """
        # Try to find explicit abstract section
        abstract_pattern = re.compile(
            r'abstract[:\s]*\n*(.*?)(?=\n\s*(?:introduction|keywords?|1\.|1\s|background)|\Z)',
            re.IGNORECASE | re.DOTALL
        )
        match = abstract_pattern.search(text[:8000])  # Abstract should be near start

        if match:
            abstract = match.group(1).strip()
            # Clean up: remove excessive whitespace
            abstract = re.sub(r'\s+', ' ', abstract)
            # Remove common artifacts
            abstract = re.sub(r'^[:\s]+', '', abstract)
            return abstract if len(abstract) > 50 else None

        return None

    def get_section_by_type(
        self,
        sections: List[Section],
        section_types: List[str]
    ) -> List[Section]:
        """Filter sections by normalized type.

        Args:
            sections: List of detected sections
            section_types: Types to include (e.g., ["methods", "results"])

        Returns:
            Filtered list of sections
        """
        return [s for s in sections if s.normalized_name in section_types]
