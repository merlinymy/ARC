"""Loader and literal-term matching for the W5 golden query set.

The JSON artifact (`golden_queries_v{N}.json`) is produced by
`build_query_set.py` straight from `backend/data/app.db`; this module only reads
it and provides the term-normalisation used by the literal-presence metric.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_QUERY_SET = EVAL_DIR / "golden_queries_v1.json"

KINDS = ("factual", "methods", "library_filtered", "synthesis")

# Unicode sub/superscript digits used in chemical formulae, e.g. Pd(PPh3)2Cl2
# written as Pd(PPh<sub>3</sub>)<sub>2</sub>Cl<sub>2</sub>.
_DIGIT_FOLD = {
    ord(c): d
    for c, d in zip("₀₁₂₃₄₅₆₇₈₉", "0123456789")
}
_DIGIT_FOLD.update({
    ord(c): d
    for c, d in zip("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
})
_DASHES = {ord(c): "-" for c in "‐‑‒–—―−⁃"}
_QUOTES = {ord(c): "'" for c in "‘’ʼ´"}


def normalize(text: str) -> str:
    """Fold a string so literal chemical terms match across typographic variants."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_DIGIT_FOLD).translate(_DASHES).translate(_QUOTES)
    text = text.lower()
    return re.sub(r"\s+", " ", text)


@dataclass
class GoldenQuery:
    query_id: str
    text: str
    kind: str
    terms: List[List[str]]
    source_message_id: int
    source_conversation_id: str
    occurrences: int
    conversations: int

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    def terms_present(self, text: str) -> List[int]:
        """Indices of the term-groups whose any-variant appears in `text`."""
        hay = normalize(text)
        return [
            i
            for i, group in enumerate(self.terms)
            if any(normalize(v) in hay for v in group)
        ]


@dataclass
class QuerySet:
    version: str
    built_at: str
    source: Dict[str, Any]
    composition: Dict[str, Any]
    dropped: Dict[str, str]
    queries: List[GoldenQuery]

    def __iter__(self) -> Iterable[GoldenQuery]:
        return iter(self.queries)

    def __len__(self) -> int:
        return len(self.queries)

    def by_id(self, query_id: str) -> GoldenQuery:
        for q in self.queries:
            if q.query_id == query_id:
                return q
        raise KeyError(query_id)

    def of_kind(self, kind: str) -> List[GoldenQuery]:
        return [q for q in self.queries if q.kind == kind]


def load_query_set(path: Optional[Path] = None) -> QuerySet:
    path = path or DEFAULT_QUERY_SET
    data = json.loads(Path(path).read_text())
    return QuerySet(
        version=data["version"],
        built_at=data["built_at"],
        source=data["source"],
        composition=data["composition"],
        dropped=data.get("dropped", {}),
        queries=[GoldenQuery(**q) for q in data["queries"]],
    )
