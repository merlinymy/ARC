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
    parent_section: Optional[str] = None  # Parent section name for subsections
    subsection_name: Optional[str] = None  # Original subsection header text


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
        current_parent_section = None  # Track current level-1 section

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
                    # Track parent section for subsections
                    if level == 1:
                        current_parent_section = normalized_name
                        candidates.append({
                            'name': line_stripped,
                            'normalized_name': normalized_name,
                            'start_idx': current_pos,
                            'level': level,
                            'line_idx': line_idx,
                            'parent_section': None,
                            'subsection_name': None,
                        })
                    else:
                        # Level 2 subsection - inherit parent and extract header text
                        # Remove numbering prefix to get clean subsection name
                        subsection_header = re.sub(r'^\d+\.[\d.]*\s*', '', line_stripped).strip()
                        candidates.append({
                            'name': line_stripped,
                            'normalized_name': current_parent_section or 'body',  # Use parent instead of "subsection"
                            'start_idx': current_pos,
                            'level': level,
                            'line_idx': line_idx,
                            'parent_section': current_parent_section,
                            'subsection_name': subsection_header if subsection_header else None,
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
                level=candidate['level'],
                parent_section=candidate.get('parent_section'),
                subsection_name=candidate.get('subsection_name'),
            ))

        return sections

    def extract_abstract(self, text: str) -> Optional[str]:
        """Extract abstract from paper text.

        Handles common formats:
        - "Abstract" followed by text
        - "Abstract:" followed by text
        - Text before "Introduction" or numbered sections
        - Various section delimiters and formatting
        """
        # Try multiple patterns in order of specificity

        # Pattern 1: Abstract followed by common section headers
        # More flexible - handles numbered sections, various headers
        patterns = [
            # Most common: Abstract followed by Introduction, numbered section, or keywords
            re.compile(
                r'abstract[:\s\-—]*\s*(.*?)(?=\n\s*(?:\d+\.?\s+)?(?:introduction|keywords?|background|methods?|materials?\s+and\s+methods?|results?)[\s\n]|\n\s*\d+[\.\)]\s+[A-Z]|\Z)',
                re.IGNORECASE | re.DOTALL
            ),
            # Fallback: Just grab text after "Abstract" until double newline or section marker
            re.compile(
                r'abstract[:\s\-—]*\s*(.*?)(?=\n\n|\n\s*[A-Z][a-z]+:|\Z)',
                re.IGNORECASE | re.DOTALL
            ),
        ]

        # Search in first 10000 chars (increased from 8000)
        search_text = text[:10000]

        for pattern in patterns:
            match = pattern.search(search_text)
            if match:
                abstract = match.group(1).strip()

                # Clean up: remove excessive whitespace
                abstract = re.sub(r'\s+', ' ', abstract)

                # Remove common artifacts
                abstract = re.sub(r'^[:\s\-—]+', '', abstract)

                # Remove trailing artifacts like "Keywords:", "Key words:", etc.
                abstract = re.sub(r'\s*(?:key\s*words?|keywords?)[\s:]*.*$', '', abstract, flags=re.IGNORECASE)

                # Valid abstract should be at least 50 chars and not too long (max 3000 chars)
                if 50 <= len(abstract) <= 3000:
                    return abstract

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
