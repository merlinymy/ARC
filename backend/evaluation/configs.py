"""Named retrieval configurations for the W5 harness.

Each configuration is a *description* of how to drive the existing retrieval
components. Nothing here changes retrieval behaviour; the runner composes the
real `QdrantStore`, `VoyageEmbedder`, `CohereReranker`, `QueryClassifier`,
`QueryExpander`, `QueryRewriter` and `HyDE` objects exactly as production does.

The arms are laid out so that single-variable comparisons exist:

    hybrid              vs  classifier_filters   -> isolates the hard chunk_type /
                                                    section_filter exclusions, with
                                                    search mode, depth, N and
                                                    max_per_paper all held fixed
    hybrid              vs  broad_hybrid         -> isolates retrieval depth
    dense_only / sparse_only / hybrid            -> isolates search mode
    keyword_routed      vs  broad_hybrid_hyde    -> the live product question
    classifier_routed   vs  broad_hybrid_hyde    -> the SS4b product question

`keyword_routed` is what ARC actually runs today: backend/.env sets
ENABLE_QUERY_CLASSIFICATION=false, so `QueryEngine.query()` falls through to
`_detect_targeted_query_type()` - a keyword matcher - rather than the Sonnet
classifier. `classifier_routed` is the LLM-classifier path that SS4b argues
about. Both are measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Depth used by every non-routed arm. 100 is the production `top_k` for the
# FACTUAL/METHODS/NOVELTY/LIMITATIONS strategies' neighbourhood (50) doubled, so
# the unfiltered arms are not handicapped by depth when compared against a
# filtered arm that searches a much smaller subset of the collection.
STANDARD_TOP_K = 100
BROAD_TOP_K = 200
STANDARD_RERANK_N = 20
STANDARD_MAX_PER_PAPER = 3


@dataclass(frozen=True)
class RetrievalConfig:
    """One named retrieval arm."""

    name: str
    description: str

    # "dense" | "sparse" | "hybrid"
    search_mode: str = "hybrid"

    # "none"       - no routing, no chunk_type/section filters
    # "classifier" - LLM classifier (claude-sonnet-5) picks a QueryType
    # "keyword"    - QueryEngine._detect_targeted_query_type keyword matcher
    route: str = "none"

    # When routed: apply the RETRIEVAL_STRATEGIES chunk_types / section_filter
    # exclusions. When False, the route is still computed (for reporting and for
    # HyDE prompt selection) but no filter is applied.
    apply_filters: bool = False

    # When routed and True, also take top_k / rerank_top_n / max_per_paper from
    # the strategy instead of the fixed values below.
    use_strategy_sizes: bool = False

    top_k: int = STANDARD_TOP_K
    rerank_top_n: int = STANDARD_RERANK_N
    max_per_paper: int = STANDARD_MAX_PER_PAPER

    use_hyde: bool = False
    use_expansion: bool = True
    use_rewrite: bool = True

    # Arms contributing documents to the judging pool.
    pooled: bool = True


CONFIGS: Dict[str, RetrievalConfig] = {}


def _add(cfg: RetrievalConfig) -> None:
    CONFIGS[cfg.name] = cfg


_add(RetrievalConfig(
    name="dense_only",
    description="Voyage dense vectors only, no filters, top_k=100 -> rerank 20.",
    search_mode="dense",
))

_add(RetrievalConfig(
    name="sparse_only",
    description="BM25 sparse vectors only, no filters, top_k=100 -> rerank 20.",
    search_mode="sparse",
))

_add(RetrievalConfig(
    name="hybrid",
    description="Dense + BM25 fused with Qdrant RRF, no filters, top_k=100 -> rerank 20.",
    search_mode="hybrid",
))

_add(RetrievalConfig(
    name="broad_hybrid",
    description=(
        "SS4b proposal: one broad path. Hybrid over all chunk types, no "
        "chunk_types/section_filter exclusions, wide top_k=200, rerank to 20."
    ),
    search_mode="hybrid",
    top_k=BROAD_TOP_K,
))

_add(RetrievalConfig(
    name="classifier_filters",
    description=(
        "Single-variable control for `hybrid`: identical search mode, depth, N "
        "and max_per_paper, but with the classifier's chunk_types/section_filter "
        "exclusions applied."
    ),
    search_mode="hybrid",
    route="classifier",
    apply_filters=True,
))

_add(RetrievalConfig(
    name="classifier_routed",
    description=(
        "The SS4b target: LLM classifier (claude-sonnet-5) -> RETRIEVAL_STRATEGIES "
        "entry, with that entry's chunk_types, section_filter, top_k, rerank_top_n "
        "and max_per_paper, hybrid search, HyDE on."
    ),
    search_mode="hybrid",
    route="classifier",
    apply_filters=True,
    use_strategy_sizes=True,
    use_hyde=True,
))

_add(RetrievalConfig(
    name="keyword_routed",
    description=(
        "Current live production. ENABLE_QUERY_CLASSIFICATION=false, so routing "
        "is the _detect_targeted_query_type keyword matcher; strategy sizes and "
        "filters applied, hybrid search, HyDE on."
    ),
    search_mode="hybrid",
    route="keyword",
    apply_filters=True,
    use_strategy_sizes=True,
    use_hyde=True,
))

_add(RetrievalConfig(
    name="broad_hybrid_hyde",
    description=(
        "broad_hybrid with HyDE on, so the routed arms (which run HyDE in "
        "production) can be compared without HyDE confounding the result."
    ),
    search_mode="hybrid",
    top_k=BROAD_TOP_K,
    use_hyde=True,
))


DEFAULT_ARMS: List[str] = list(CONFIGS.keys())


def get(name: str) -> RetrievalConfig:
    try:
        return CONFIGS[name]
    except KeyError:
        raise SystemExit(f"unknown config '{name}'. Known: {', '.join(CONFIGS)}")
