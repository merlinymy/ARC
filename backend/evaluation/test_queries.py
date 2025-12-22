"""Test query definitions based on actual usage patterns.

Contains 50 test queries distributed across query types:
- Framing/positioning: 12 queries
- Methods writing: 8 queries
- Factual: 8 queries
- Controls/validation: 6 queries
- Novelty: 6 queries
- Limitations: 5 queries
- Comparative: 5 queries
"""

import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum


class QueryType(str, Enum):
    """Query types matching the classifier."""
    FACTUAL = "factual"
    FRAMING = "framing"
    METHODS = "methods"
    SUMMARY = "summary"
    COMPARATIVE = "comparative"
    NOVELTY = "novelty"
    LIMITATIONS = "limitations"


@dataclass
class TestQuery:
    """A test query for evaluation."""
    query_id: str
    query: str
    query_type: QueryType
    expected_topics: List[str]  # Topics that should appear in results
    expected_entities: List[str]  # Domain entities that should be found
    expected_chunk_types: List[str]  # Chunk types likely to be relevant
    requires_human_eval: bool = False  # True for subjective queries like framing
    notes: str = ""


# Test queries based on actual usage patterns from the design document
TEST_QUERIES: List[TestQuery] = [
    # === FRAMING/POSITIONING (12 queries) ===
    TestQuery(
        query_id="framing_001",
        query="How do I frame Raman imaging as an enabling platform rather than a single-use technique?",
        query_type=QueryType.FRAMING,
        expected_topics=["platform", "versatile", "applications", "label-free"],
        expected_entities=["Raman", "SRS"],
        expected_chunk_types=["abstract", "section"],
        requires_human_eval=True,
    ),
    TestQuery(
        query_id="framing_002",
        query="How should I position Raman imaging in biofilms as fundamentally different from planktonic cell imaging?",
        query_type=QueryType.FRAMING,
        expected_topics=["biofilm", "planktonic", "spatial", "heterogeneity"],
        expected_entities=["Raman", "biofilm", "planktonic"],
        expected_chunk_types=["abstract", "section"],
        requires_human_eval=True,
    ),
    TestQuery(
        query_id="framing_003",
        query="How do I frame Y537S and D538G targeting as clinically motivated but mechanistically rigorous?",
        query_type=QueryType.FRAMING,
        expected_topics=["clinical", "resistance", "mechanism", "therapeutic"],
        expected_entities=["Y537S", "D538G", "ERα"],
        expected_chunk_types=["abstract", "section"],
        requires_human_eval=True,
    ),
    TestQuery(
        query_id="framing_004",
        query="How can PET and Raman imaging be presented as complementary rather than redundant?",
        query_type=QueryType.FRAMING,
        expected_topics=["complementary", "multimodal", "resolution", "specificity"],
        expected_entities=["Raman"],
        expected_chunk_types=["abstract", "section"],
        requires_human_eval=True,
    ),
    TestQuery(
        query_id="framing_005",
        query="How do I justify that stapled peptides are a better approach than small molecules for ERα?",
        query_type=QueryType.FRAMING,
        expected_topics=["selectivity", "protein-protein", "interface", "helix"],
        expected_entities=["stapled peptide", "ERα"],
        expected_chunk_types=["abstract", "section"],
        requires_human_eval=True,
    ),
    TestQuery(
        query_id="framing_006",
        query="How should I describe the significance of LL-37 in antimicrobial research?",
        query_type=QueryType.FRAMING,
        expected_topics=["host defense", "innate immunity", "antimicrobial", "therapeutic"],
        expected_entities=["LL-37", "AMP"],
        expected_chunk_types=["abstract", "section"],
        requires_human_eval=True,
    ),
    TestQuery(
        query_id="framing_007",
        query="How do I position vibrational tags as superior to fluorescent labels for live cell imaging?",
        query_type=QueryType.FRAMING,
        expected_topics=["non-perturbative", "small", "bioorthogonal", "native"],
        expected_entities=["alkyne tag", "fluorescent label", "Raman tag"],
        expected_chunk_types=["abstract", "section"],
        requires_human_eval=True,
    ),
    TestQuery(
        query_id="framing_008",
        query="How do I frame this work as advancing the field beyond existing Raman imaging studies?",
        query_type=QueryType.FRAMING,
        expected_topics=["novel", "advance", "previous", "limitation"],
        expected_entities=["Raman"],
        expected_chunk_types=["abstract", "section"],
        requires_human_eval=True,
    ),
    TestQuery(
        query_id="framing_009",
        query="How should I describe the clinical relevance of hormone-resistant breast cancer research?",
        query_type=QueryType.FRAMING,
        expected_topics=["resistance", "clinical", "treatment", "patient"],
        expected_entities=["hormone-resistant breast cancer", "ERα"],
        expected_chunk_types=["abstract", "section"],
        requires_human_eval=True,
    ),
    TestQuery(
        query_id="framing_010",
        query="How do I justify the choice of protein purification strategy for ERα LBD?",
        query_type=QueryType.FRAMING,
        expected_topics=["purity", "yield", "activity", "stability"],
        expected_entities=["ERα LBD", "affinity chromatography"],
        expected_chunk_types=["section"],
        requires_human_eval=True,
    ),
    TestQuery(
        query_id="framing_011",
        query="How do I argue that biofilm-targeting antimicrobials address an unmet clinical need?",
        query_type=QueryType.FRAMING,
        expected_topics=["resistance", "chronic", "infection", "unmet"],
        expected_entities=["biofilm", "AMP"],
        expected_chunk_types=["abstract", "section"],
        requires_human_eval=True,
    ),
    TestQuery(
        query_id="framing_012",
        query="How should I describe SRS imaging advantages over conventional Raman?",
        query_type=QueryType.FRAMING,
        expected_topics=["speed", "sensitivity", "label-free", "nonlinear"],
        expected_entities=["SRS", "Raman"],
        expected_chunk_types=["abstract", "section"],
        requires_human_eval=True,
    ),

    # === METHODS WRITING (8 queries) ===
    TestQuery(
        query_id="methods_001",
        query="How should I describe Affinity FPLC vs Ion-Exchange FPLC in a paper rather than vendor language?",
        query_type=QueryType.METHODS,
        expected_topics=["chromatography", "purification", "column", "buffer"],
        expected_entities=["affinity chromatography", "ion exchange", "FPLC"],
        expected_chunk_types=["section", "fine"],
    ),
    TestQuery(
        query_id="methods_002",
        query="How do I write a technical but concise justification for purifying ERα LBD WT, D538G, and Y537S?",
        query_type=QueryType.METHODS,
        expected_topics=["expression", "purification", "mutant", "wild-type"],
        expected_entities=["ERα LBD", "D538G", "Y537S"],
        expected_chunk_types=["section", "fine"],
    ),
    TestQuery(
        query_id="methods_003",
        query="What buffer conditions are typically used for ERα LBD purification?",
        query_type=QueryType.METHODS,
        expected_topics=["buffer", "pH", "salt", "glycerol"],
        expected_entities=["ERα LBD"],
        expected_chunk_types=["section", "fine"],
    ),
    TestQuery(
        query_id="methods_004",
        query="How should I describe the SRS microscopy setup and imaging parameters?",
        query_type=QueryType.METHODS,
        expected_topics=["laser", "wavelength", "power", "objective"],
        expected_entities=["SRS"],
        expected_chunk_types=["section", "fine"],
    ),
    TestQuery(
        query_id="methods_005",
        query="What protocols are used for antimicrobial activity testing against biofilms?",
        query_type=QueryType.METHODS,
        expected_topics=["MIC", "biofilm", "assay", "colony"],
        expected_entities=["biofilm", "MIC"],
        expected_chunk_types=["section", "fine"],
    ),
    TestQuery(
        query_id="methods_006",
        query="How are stapled peptides typically synthesized and characterized?",
        query_type=QueryType.METHODS,
        expected_topics=["synthesis", "olefin", "metathesis", "HPLC"],
        expected_entities=["stapled peptide"],
        expected_chunk_types=["section", "fine"],
    ),
    TestQuery(
        query_id="methods_007",
        query="What cell lines are commonly used for ERα inhibitor testing?",
        query_type=QueryType.METHODS,
        expected_topics=["MCF-7", "cell line", "breast cancer", "assay"],
        expected_entities=["ERα"],
        expected_chunk_types=["section", "fine"],
    ),
    TestQuery(
        query_id="methods_008",
        query="How should I describe the synthesis of alkyne-tagged compounds for Raman imaging?",
        query_type=QueryType.METHODS,
        expected_topics=["synthesis", "alkyne", "click", "labeling"],
        expected_entities=["alkyne tag"],
        expected_chunk_types=["section", "fine"],
    ),

    # === FACTUAL (8 queries) ===
    TestQuery(
        query_id="factual_001",
        query="What is the difference between CC50 and IC50?",
        query_type=QueryType.FACTUAL,
        expected_topics=["cytotoxic", "inhibitory", "concentration", "viability"],
        expected_entities=["CC50", "IC50"],
        expected_chunk_types=["fine", "table"],
    ),
    TestQuery(
        query_id="factual_002",
        query="What is the mechanism of action of LL-37?",
        query_type=QueryType.FACTUAL,
        expected_topics=["membrane", "disruption", "antimicrobial", "mechanism"],
        expected_entities=["LL-37"],
        expected_chunk_types=["fine", "section"],
    ),
    TestQuery(
        query_id="factual_003",
        query="What mutations in ERα cause hormone resistance?",
        query_type=QueryType.FACTUAL,
        expected_topics=["mutation", "resistance", "ligand", "binding"],
        expected_entities=["ERα", "Y537S", "D538G"],
        expected_chunk_types=["fine", "table"],
    ),
    TestQuery(
        query_id="factual_004",
        query="What are the characteristic Raman peaks for alkyne tags?",
        query_type=QueryType.FACTUAL,
        expected_topics=["wavenumber", "cm-1", "peak", "vibration"],
        expected_entities=["alkyne tag", "Raman"],
        expected_chunk_types=["fine", "table", "caption"],
    ),
    TestQuery(
        query_id="factual_005",
        query="What is the typical MIC of LL-37 against E. coli?",
        query_type=QueryType.FACTUAL,
        expected_topics=["MIC", "concentration", "bacteria", "activity"],
        expected_entities=["LL-37", "MIC"],
        expected_chunk_types=["fine", "table"],
    ),
    TestQuery(
        query_id="factual_006",
        query="What coactivators interact with ERα LBD?",
        query_type=QueryType.FACTUAL,
        expected_topics=["coactivator", "binding", "SRC", "recruitment"],
        expected_entities=["ERα LBD", "coactivator"],
        expected_chunk_types=["fine", "section"],
    ),
    TestQuery(
        query_id="factual_007",
        query="What is the resolution of SRS microscopy?",
        query_type=QueryType.FACTUAL,
        expected_topics=["resolution", "spatial", "diffraction", "micron"],
        expected_entities=["SRS"],
        expected_chunk_types=["fine", "section"],
    ),
    TestQuery(
        query_id="factual_008",
        query="What is the molecular weight of LL-37 peptide?",
        query_type=QueryType.FACTUAL,
        expected_topics=["molecular weight", "amino acid", "sequence", "Da"],
        expected_entities=["LL-37"],
        expected_chunk_types=["fine", "table"],
    ),

    # === CONTROLS/VALIDATION (6 queries) ===
    TestQuery(
        query_id="controls_001",
        query="How do I justify that diyne-girder or alkyne tags do not disrupt biological activity?",
        query_type=QueryType.FRAMING,  # This is framing about controls
        expected_topics=["control", "activity", "native", "perturbation"],
        expected_entities=["alkyne tag"],
        expected_chunk_types=["section"],
        requires_human_eval=True,
    ),
    TestQuery(
        query_id="controls_002",
        query="How do I describe controls so reviewers see Raman imaging as quantitative enough for biology journals?",
        query_type=QueryType.FRAMING,
        expected_topics=["quantitative", "control", "calibration", "validation"],
        expected_entities=["Raman"],
        expected_chunk_types=["section"],
        requires_human_eval=True,
    ),
    TestQuery(
        query_id="controls_003",
        query="How do I argue that fluorescent tags alter permeability while vibrational tags preserve native behavior?",
        query_type=QueryType.FRAMING,
        expected_topics=["permeability", "native", "perturbation", "small"],
        expected_entities=["fluorescent label", "Raman tag"],
        expected_chunk_types=["section"],
        requires_human_eval=True,
    ),
    TestQuery(
        query_id="controls_004",
        query="What controls are needed to validate antimicrobial peptide activity assays?",
        query_type=QueryType.METHODS,
        expected_topics=["control", "positive", "negative", "validation"],
        expected_entities=["AMP", "MIC"],
        expected_chunk_types=["section", "fine"],
    ),
    TestQuery(
        query_id="controls_005",
        query="How do I demonstrate that ERα LBD mutations don't affect protein folding?",
        query_type=QueryType.METHODS,
        expected_topics=["folding", "circular dichroism", "stability", "structure"],
        expected_entities=["ERα LBD"],
        expected_chunk_types=["section", "fine"],
    ),
    TestQuery(
        query_id="controls_006",
        query="What specificity controls are used in Raman imaging experiments?",
        query_type=QueryType.METHODS,
        expected_topics=["specificity", "control", "background", "signal"],
        expected_entities=["Raman"],
        expected_chunk_types=["section", "fine"],
    ),

    # === NOVELTY (6 queries) ===
    TestQuery(
        query_id="novelty_001",
        query="What is the strongest defensible novelty claim for Raman imaging of antimicrobial peptides?",
        query_type=QueryType.NOVELTY,
        expected_topics=["novel", "first", "unique", "contribution"],
        expected_entities=["Raman", "AMP"],
        expected_chunk_types=["abstract", "section"],
        requires_human_eval=True,
    ),
    TestQuery(
        query_id="novelty_002",
        query="Has SRS imaging been applied to biofilm penetration before?",
        query_type=QueryType.NOVELTY,
        expected_topics=["previous", "prior", "existing", "novel"],
        expected_entities=["SRS", "biofilm"],
        expected_chunk_types=["abstract", "section"],
    ),
    TestQuery(
        query_id="novelty_003",
        query="What aspects of ERα mutant targeting are underexplored?",
        query_type=QueryType.NOVELTY,
        expected_topics=["gap", "underexplored", "opportunity", "novel"],
        expected_entities=["ERα"],
        expected_chunk_types=["abstract", "section"],
    ),
    TestQuery(
        query_id="novelty_004",
        query="What makes this approach to peptide imaging distinct from previous work?",
        query_type=QueryType.NOVELTY,
        expected_topics=["distinct", "different", "advance", "improvement"],
        expected_entities=["Raman"],
        expected_chunk_types=["abstract", "section"],
        requires_human_eval=True,
    ),
    TestQuery(
        query_id="novelty_005",
        query="Are there existing stapled peptide inhibitors targeting ERα mutants?",
        query_type=QueryType.NOVELTY,
        expected_topics=["existing", "prior", "inhibitor", "targeting"],
        expected_entities=["stapled peptide", "ERα"],
        expected_chunk_types=["abstract", "section"],
    ),
    TestQuery(
        query_id="novelty_006",
        query="What is the first reported use of alkyne tags for antimicrobial peptide imaging?",
        query_type=QueryType.NOVELTY,
        expected_topics=["first", "reported", "pioneer", "initial"],
        expected_entities=["alkyne tag", "AMP"],
        expected_chunk_types=["abstract", "section"],
    ),

    # === LIMITATIONS (5 queries) ===
    TestQuery(
        query_id="limitations_001",
        query="How do I explain peptide uptake and localization without fluorescent colocalization markers?",
        query_type=QueryType.LIMITATIONS,
        expected_topics=["limitation", "colocalization", "indirect", "inference"],
        expected_entities=["fluorescent label"],
        expected_chunk_types=["section"],
        requires_human_eval=True,
    ),
    TestQuery(
        query_id="limitations_002",
        query="What are the limitations of SRS imaging for live cell studies?",
        query_type=QueryType.LIMITATIONS,
        expected_topics=["limitation", "phototoxicity", "depth", "sensitivity"],
        expected_entities=["SRS"],
        expected_chunk_types=["section"],
    ),
    TestQuery(
        query_id="limitations_003",
        query="How do I address the limitation of in vitro binding assays for ERα?",
        query_type=QueryType.LIMITATIONS,
        expected_topics=["limitation", "in vitro", "cellular", "context"],
        expected_entities=["ERα"],
        expected_chunk_types=["section"],
        requires_human_eval=True,
    ),
    TestQuery(
        query_id="limitations_004",
        query="What caveats should I mention about MIC assays for biofilm-active peptides?",
        query_type=QueryType.LIMITATIONS,
        expected_topics=["caveat", "limitation", "biofilm", "planktonic"],
        expected_entities=["MIC", "biofilm"],
        expected_chunk_types=["section"],
    ),
    TestQuery(
        query_id="limitations_005",
        query="How do I discuss the limitations of using model membranes?",
        query_type=QueryType.LIMITATIONS,
        expected_topics=["limitation", "model", "membrane", "physiological"],
        expected_entities=["membrane permeability"],
        expected_chunk_types=["section"],
        requires_human_eval=True,
    ),

    # === COMPARATIVE (5 queries) ===
    TestQuery(
        query_id="comparative_001",
        query="How does SRS compare to fluorescence microscopy for peptide imaging?",
        query_type=QueryType.COMPARATIVE,
        expected_topics=["compare", "fluorescence", "sensitivity", "resolution"],
        expected_entities=["SRS", "fluorescent label"],
        expected_chunk_types=["abstract", "section"],
    ),
    TestQuery(
        query_id="comparative_002",
        query="What is the difference between affinity and ion exchange chromatography for protein purification?",
        query_type=QueryType.COMPARATIVE,
        expected_topics=["difference", "selectivity", "purity", "binding"],
        expected_entities=["affinity chromatography", "ion exchange"],
        expected_chunk_types=["section"],
    ),
    TestQuery(
        query_id="comparative_003",
        query="How do stapled peptides compare to small molecule inhibitors for ERα?",
        query_type=QueryType.COMPARATIVE,
        expected_topics=["compare", "selectivity", "potency", "stability"],
        expected_entities=["stapled peptide", "ERα"],
        expected_chunk_types=["abstract", "section"],
    ),
    TestQuery(
        query_id="comparative_004",
        query="Compare the antimicrobial mechanisms of LL-37 versus other AMPs",
        query_type=QueryType.COMPARATIVE,
        expected_topics=["mechanism", "compare", "membrane", "activity"],
        expected_entities=["LL-37", "AMP"],
        expected_chunk_types=["abstract", "section"],
    ),
    TestQuery(
        query_id="comparative_005",
        query="What are the differences between Y537S and D538G ERα mutants?",
        query_type=QueryType.COMPARATIVE,
        expected_topics=["difference", "mutation", "activity", "structure"],
        expected_entities=["Y537S", "D538G", "ERα"],
        expected_chunk_types=["section", "table"],
    ),
]


def load_test_queries(path: Optional[Path] = None) -> List[TestQuery]:
    """Load test queries from JSON file or return built-in queries.

    Args:
        path: Optional path to JSON file with queries

    Returns:
        List of TestQuery objects
    """
    if path and path.exists():
        with open(path) as f:
            data = json.load(f)
        return [
            TestQuery(
                query_id=q["query_id"],
                query=q["query"],
                query_type=QueryType(q["query_type"]),
                expected_topics=q.get("expected_topics", []),
                expected_entities=q.get("expected_entities", []),
                expected_chunk_types=q.get("expected_chunk_types", []),
                requires_human_eval=q.get("requires_human_eval", False),
                notes=q.get("notes", ""),
            )
            for q in data["queries"]
        ]

    return TEST_QUERIES


def save_test_queries(queries: List[TestQuery], path: Path):
    """Save test queries to JSON file.

    Args:
        queries: List of TestQuery objects
        path: Output path
    """
    data = {
        "queries": [
            {
                "query_id": q.query_id,
                "query": q.query,
                "query_type": q.query_type.value,
                "expected_topics": q.expected_topics,
                "expected_entities": q.expected_entities,
                "expected_chunk_types": q.expected_chunk_types,
                "requires_human_eval": q.requires_human_eval,
                "notes": q.notes,
            }
            for q in queries
        ]
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_queries_by_type(query_type: QueryType) -> List[TestQuery]:
    """Get all test queries of a specific type."""
    return [q for q in TEST_QUERIES if q.query_type == query_type]


def get_human_eval_queries() -> List[TestQuery]:
    """Get queries that require human evaluation."""
    return [q for q in TEST_QUERIES if q.requires_human_eval]
