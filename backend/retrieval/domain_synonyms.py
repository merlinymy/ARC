"""Domain-specific synonym dictionary for query expansion.

This module contains synonyms specific to the research domain:
- Estrogen receptor research
- Raman/SRS imaging
- Antimicrobial peptides
- Protein purification methods

Expand this dictionary based on failure analysis during evaluation.
"""

from typing import Dict, List, Set


# Domain synonym dictionary
# Format: canonical_term -> [list of synonyms/aliases]
DOMAIN_SYNONYMS: Dict[str, List[str]] = {
    # === Estrogen Receptor Research ===
    "ERα": [
        "estrogen receptor alpha",
        "ER-alpha",
        "ESR1",
        "estrogen receptor",
        "ER alpha",
    ],
    "ERα LBD": [
        "estrogen receptor ligand binding domain",
        "ER LBD",
        "ligand binding domain",
    ],
    "Y537S": [
        "Y537S mutation",
        "tyrosine 537 serine",
        "Y537S mutant",
    ],
    "D538G": [
        "D538G mutation",
        "aspartate 538 glycine",
        "D538G mutant",
    ],
    "hormone-resistant breast cancer": [
        "endocrine-resistant",
        "ER+ breast cancer",
        "estrogen receptor positive",
        "hormone receptor positive",
    ],

    # === Imaging Techniques ===
    "SRS": [
        "stimulated Raman scattering",
        "coherent Raman",
        "SRS imaging",
        "SRS microscopy",
    ],
    "Raman": [
        "Raman spectroscopy",
        "Raman imaging",
        "vibrational imaging",
        "Raman microscopy",
    ],
    "Raman tag": [
        "vibrational tag",
        "Raman probe",
        "vibrational probe",
    ],

    # === Peptides ===
    "LL-37": [
        "LL37",
        "cathelicidin",
        "CAP18",
        "CAMP",
        "human cathelicidin",
    ],
    "AMP": [
        "antimicrobial peptide",
        "antimicrobial peptides",
        "host defense peptide",
        "host defense peptides",
        "HDPs",
    ],
    "stapled peptide": [
        "stapled peptides",
        "hydrocarbon-stapled",
        "macrocyclic peptide",
        "macrocyclic peptides",
        "peptide macrocycle",
    ],
    "cell-penetrating peptide": [
        "CPP",
        "cell penetrating peptide",
        "membrane-penetrating peptide",
    ],

    # === Chemical Tags and Labels ===
    "alkyne tag": [
        "alkyne",
        "diyne",
        "diyne tag",
        "alkyne label",
        "clickable tag",
        "bioorthogonal tag",
    ],
    "FITC": [
        "fluorescein",
        "fluorescein isothiocyanate",
        "fluorescent label",
        "fluorescent tag",
    ],
    "fluorescent label": [
        "fluorophore",
        "fluorescent dye",
        "fluorescent probe",
    ],

    # === Assays and Measurements ===
    "IC50": [
        "IC-50",
        "IC₅₀",
        "half-maximal inhibitory concentration",
        "inhibitory concentration",
    ],
    "CC50": [
        "CC-50",
        "CC₅₀",
        "half-maximal cytotoxic concentration",
        "cytotoxic concentration",
    ],
    "EC50": [
        "EC-50",
        "EC₅₀",
        "half-maximal effective concentration",
        "effective concentration",
    ],
    "MIC": [
        "minimum inhibitory concentration",
        "minimal inhibitory concentration",
    ],

    # === Purification Methods ===
    "FPLC": [
        "fast protein liquid chromatography",
        "protein chromatography",
        "liquid chromatography",
    ],
    "affinity chromatography": [
        "affinity FPLC",
        "affinity purification",
        "His-tag purification",
        "Ni-NTA",
        "nickel affinity",
        "immobilized metal affinity",
        "IMAC",
    ],
    "ion exchange": [
        "ion-exchange FPLC",
        "ion exchange chromatography",
        "IEX",
        "anion exchange",
        "cation exchange",
        "Q column",
        "SP column",
    ],
    "size exclusion": [
        "SEC",
        "gel filtration",
        "size exclusion chromatography",
        "molecular sieve",
    ],

    # === Biological Contexts ===
    "biofilm": [
        "biofilms",
        "bacterial biofilm",
        "biofilm matrix",
        "sessile bacteria",
    ],
    "planktonic": [
        "planktonic cells",
        "free-floating bacteria",
        "planktonic bacteria",
        "planktonic culture",
    ],
    "membrane permeability": [
        "membrane disruption",
        "membrane integrity",
        "membrane permeabilization",
        "pore formation",
    ],

    # === Structural Biology ===
    "coactivator": [
        "coactivator binding",
        "coactivator recruitment",
        "SRC",
        "steroid receptor coactivator",
    ],
    "allosteric": [
        "allosteric site",
        "allosteric binding",
        "allosteric modulation",
    ],
}

# Reverse lookup: alias -> canonical term
ALIAS_TO_CANONICAL: Dict[str, str] = {}
for canonical, aliases in DOMAIN_SYNONYMS.items():
    for alias in aliases:
        ALIAS_TO_CANONICAL[alias.lower()] = canonical


def get_synonyms(term: str) -> List[str]:
    """Get all synonyms for a term.

    Args:
        term: The term to look up (case-insensitive)

    Returns:
        List of synonyms including the canonical form.
        Returns empty list if term not found.
    """
    term_lower = term.lower()

    # Check if it's a canonical term
    for canonical, aliases in DOMAIN_SYNONYMS.items():
        if canonical.lower() == term_lower:
            return [canonical] + aliases

    # Check if it's an alias
    if term_lower in ALIAS_TO_CANONICAL:
        canonical = ALIAS_TO_CANONICAL[term_lower]
        return [canonical] + DOMAIN_SYNONYMS[canonical]

    return []


def find_entities_in_text(text: str) -> Set[str]:
    """Find known domain entities in text.

    Args:
        text: Text to search

    Returns:
        Set of canonical entity names found
    """
    text_lower = text.lower()
    found = set()

    for canonical, aliases in DOMAIN_SYNONYMS.items():
        all_terms = [canonical] + aliases
        for term in all_terms:
            if term.lower() in text_lower:
                found.add(canonical)
                break

    return found
