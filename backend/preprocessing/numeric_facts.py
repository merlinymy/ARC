"""Numeric facts — value + unit + entity triples (W2 item 5).

The user's most-valued capability is "keep the numbers advantage, and tell me
where the number came from".  Today a number is only ever *incidentally*
retrievable: it is a few characters inside a chunk of prose, and nothing in the
index knows it is a number, what it measures, or what it measures it of.

This module lifts those numbers out into a queryable payload field.  Two
sources, in order of density:

1. **Tables** (``record["tables"]`` → MinerU ``table_body`` HTML).  A PrAMP
   activity table is a grid of ``entity × property = value``, which is exactly
   the triple shape, so the table path reconstructs the grid (including
   ``rowspan`` / ``colspan``) and reads properties and units out of the header
   rows.  This is where the densest facts live.
2. **Prose**, best-effort: ``"an IC50 of 12 nM"``, ``"85% yield"``, ``"stirred
   at 80 °C"``, ``"2.0 equiv"``.

A fact is ``{"property", "value", "value_text", "unit", "entity", "source",
"context"}``.  ``property`` is canonical (``ic50``, ``mic``, ``kd``, ``yield``,
``temperature``, ``concentration``, ``equivalents``, ...) so it can be filtered
on; ``value_text`` keeps the original string, including ranges and ``<``/``>``
qualifiers, because ``"MIC < 0.5"`` is not the same claim as ``"MIC = 0.5"``.

Deliberately conservative: a missed fact costs recall on one query, but a wrong
entity/property pairing puts a false number in front of the user with a citation
attached.  Where the entity cannot be established it is left ``None`` rather
than guessed.
"""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Hard cap per chunk, so a 40-row table cannot dominate a Qdrant payload.
MAX_FACTS_PER_CHUNK = 60


# ---------------------------------------------------------------------------
# Units and properties
# ---------------------------------------------------------------------------

# Canonical unit spellings.  MinerU emits both µ (micro sign) and μ (Greek mu),
# and papers use "uM" as ASCII, so all three fold together.
_MICRO = "µ"  # micro sign
_UNIT_CANON = {
    "um": f"{_MICRO}M", "μm": f"{_MICRO}M", f"{_MICRO}m": f"{_MICRO}M",
    "nm_conc": "nM", "mm": "mM", "m": "M", "pm": "pM",
    "umol/l": f"{_MICRO}mol/L", "μmol/l": f"{_MICRO}mol/L",
    f"{_MICRO}mol/l": f"{_MICRO}mol/L",
    "nmol/l": "nmol/L", "mmol/l": "mmol/L", "mol/l": "mol/L",
    "ug/ml": f"{_MICRO}g/mL", "μg/ml": f"{_MICRO}g/mL",
    f"{_MICRO}g/ml": f"{_MICRO}g/mL",
    "mg/ml": "mg/mL", "ng/ml": "ng/mL", "g/l": "g/L", "mg/l": "mg/L",
    "mg/kg": "mg/kg", "%": "%", "°c": "°C", "k": "K",
    "h": "h", "hr": "h", "hrs": "h", "min": "min", "s": "s", "d": "d",
    "equiv": "equiv", "equiv.": "equiv", "eq": "equiv", "eq.": "equiv",
    "mol%": "mol%", "nm": "nm", "µm_len": f"{_MICRO}m",
    "da": "Da", "kda": "kDa", "kcal/mol": "kcal/mol", "kj/mol": "kJ/mol",
    "mpa": "MPa", "bar": "bar", "psi": "psi", "rpm": "rpm",
    "cm-1": "cm⁻¹", "cm⁻¹": "cm⁻¹",
}

#: Units that denote a concentration, used to type a bare value+unit match.
_CONCENTRATION_UNITS = {
    f"{_MICRO}M", "nM", "mM", "M", "pM", f"{_MICRO}mol/L", "nmol/L",
    "mmol/L", "mol/L", f"{_MICRO}g/mL", "mg/mL", "ng/mL", "g/L", "mg/L",
}

# Unit alternatives for the regexes, longest first so "µmol/L" wins over "µm".
_UNIT_ALTERNATIVES = sorted(
    {
        "%", "°C", "K", f"{_MICRO}M", "μM", "uM", "nM", "mM", "pM",
        f"{_MICRO}mol/L", "μmol/L", "umol/L", "nmol/L", "mmol/L", "mol/L",
        f"{_MICRO}g/mL", "μg/mL", "ug/mL", "mg/mL", "ng/mL", "mg/kg",
        "g/L", "mg/L", "mol%", "equiv.", "equiv", "eq.", "nm", "kDa", "Da",
        "kcal/mol", "kJ/mol", "MPa", "bar", "psi", "rpm", "h", "min",
    },
    key=len,
    reverse=True,
)
_UNIT_RE = "|".join(re.escape(u) for u in _UNIT_ALTERNATIVES)

#: Property name → canonical key.  Order matters: longest/most specific first.
_PROPERTY_PATTERNS: List[Tuple[str, str]] = [
    (r"IC\s?-?\s?50", "ic50"),
    (r"EC\s?-?\s?50", "ec50"),
    (r"LD\s?-?\s?50", "ld50"),
    (r"CC\s?-?\s?50", "cc50"),
    (r"GI\s?-?\s?50", "gi50"),
    (r"ED\s?-?\s?50", "ed50"),
    (r"MBC", "mbc"),
    (r"MIC(?:\s?-?\s?50|\s?-?\s?90)?", "mic"),
    (r"K\s?[_\s]?\{?\s?[dD]\s?\}?", "kd"),
    (r"K\s?[_\s]?\{?\s?[iI]\s?\}?", "ki"),
    (r"K\s?[_\s]?\{?\s?[mM]\s?\}?", "km"),
    (r"K\s?[_\s]?\{?\s?cat\s?\}?", "kcat"),
    (r"EE|enantiomeric\s+excess", "ee"),
    (r"(?:isolated\s+|chemical\s+)?yield", "yield"),
    (r"conversion", "conversion"),
    (r"selectivity", "selectivity"),
    (r"loading", "loading"),
    (r"temperature", "temperature"),
    (r"equivalents?", "equivalents"),
    (r"(?:reaction\s+)?time", "time"),
    (r"pH", "ph"),
    (r"concentrations?", "concentration"),
    (r"limit\s+of\s+detection|LOD", "lod"),
    (r"molecular\s+weight|MW", "molecular_weight"),
    (r"wavelength|λ(?:max)?", "wavelength"),
    (r"TOF", "tof"),
    (r"TON", "ton"),
]
_PROPERTY_RE = re.compile(
    "|".join(f"(?P<p{i}>{pat})" for i, (pat, _) in enumerate(_PROPERTY_PATTERNS)),
    re.IGNORECASE,
)
_PROPERTY_BY_GROUP = {f"p{i}": key for i, (_, key) in enumerate(_PROPERTY_PATTERNS)}

#: Units that imply a property when no property word is nearby.
_UNIT_IMPLIED_PROPERTY = {
    "°C": "temperature", "K": None, "equiv": "equivalents",
    "h": "time", "min": "time", "mol%": "loading", "nm": "wavelength",
    "Da": "molecular_weight", "kDa": "molecular_weight",
}

# A number: optional qualifier, digits, optional decimal / exponent, optional
# range or +- error.  "0.382 [0.337-0.432]", "<0.5", "2.0 ± 0.1", "1.2 x 10^3".
_NUMBER = (
    r"(?P<qual>[<>≤≥~≈]?\s?)"
    r"(?P<num>\d{1,6}(?:[.,]\d+)?(?:\s?[x×]\s?10\s?[\^⁻]?\s?-?\d+)?)"
    r"(?P<tail>\s?(?:[±+]/?-?\s?\d+(?:\.\d+)?)?)"
)
_LATEX_STRIP = re.compile(r"\$|\\mathrm|\\mathit|\\text|\\left|\\right|[{}]|\\,|\\;|\\ ")
_BRACKET_UNIT = re.compile(r"[\[\(]\s*([^\]\)]{1,20}?)\s*[\]\)]")
_NUM_ONLY = re.compile(r"^\s*" + _NUMBER + r"\s*$")

#: A *value* cell: a number, optionally with an error term, a confidence
#: interval in brackets, a footnote marker, or a unit — but no trailing words.
#: "0.382 [0.337-0.432]" and ">128" are values; "25% MHBII" is a header.
_VALUE_CELL = re.compile(
    r"^\s*[<>≤≥~≈]?\s*\d{1,6}(?:[.,]\d+)?"
    r"(?:\s?[x×]\s?10\s?[\^⁻]?\s?-?\d+)?"
    r"(?:\s*[±+]/?-?\s*\d+(?:[.,]\d+)?)?"
    r"(?:\s*[\[\(][^\]\)]{0,24}[\]\)])?"
    r"(?:\s*(?:" + _UNIT_RE + r"))?"
    r"\s*[*†‡§]{0,3}\s*$"
)


def _canon_unit(raw: Optional[str], *, numeric_context: bool = True) -> Optional[str]:
    """Fold a unit spelling onto its canonical form."""
    if not raw:
        return None
    text = _LATEX_STRIP.sub("", raw).strip().rstrip(".,;:")
    if not text:
        return None
    key = text.lower().replace("µ", _MICRO)
    # "nM" is nanomolar; "nm" is nanometres.  Case is the only signal.
    if text in ("nM", "nm"):
        return "nM" if text == "nM" else "nm"
    if key in ("um", "μm", f"{_MICRO}m"):
        return f"{_MICRO}M" if numeric_context else f"{_MICRO}m"
    return _UNIT_CANON.get(key, text)


def _canon_property(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    match = _PROPERTY_RE.search(_LATEX_STRIP.sub("", raw))
    if not match:
        return None
    for name, value in match.groupdict().items():
        if value:
            return _PROPERTY_BY_GROUP[name]
    return None


def _parse_value(num: str) -> Optional[float]:
    cleaned = num.replace(",", ".").replace(" ", "")
    exp_match = re.match(r"^([\d.]+)[x×]10\^?(-?\d+)$", cleaned)
    try:
        if exp_match:
            return float(exp_match.group(1)) * (10 ** int(exp_match.group(2)))
        return float(cleaned)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Table grid
# ---------------------------------------------------------------------------

class _TableGrid(HTMLParser):
    """Expand a MinerU ``<table>`` into a rectangular grid of strings.

    ``rowspan`` / ``colspan`` are carried, which is what makes a two-row header
    like ``MIC [µg/mL]`` over ``BW | Ros`` readable: the value cell's column
    inherits both header levels.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: List[List[Optional[str]]] = []
        self._pending: Dict[Tuple[int, int], str] = {}   # (row, col) -> text
        self._row = -1
        self._col = 0
        self._cell: Optional[List[str]] = None
        self._cell_span: Tuple[int, int] = (1, 1)
        self._header_rows: List[int] = []
        self._row_has_header = False

    # -- parser hooks ------------------------------------------------------
    def handle_starttag(self, tag, attrs):
        attr = {k.lower(): (v or "") for k, v in attrs}
        if tag == "tr":
            self._row += 1
            self._col = 0
            self._row_has_header = False
        elif tag in ("td", "th"):
            if self._row < 0:            # a <td> outside any <tr>
                self._row = 0
            self._cell = []
            try:
                rowspan = max(1, min(int(attr.get("rowspan", 1) or 1), 20))
                colspan = max(1, min(int(attr.get("colspan", 1) or 1), 20))
            except ValueError:
                rowspan = colspan = 1
            self._cell_span = (rowspan, colspan)
            if tag == "th":
                self._row_has_header = True
        elif tag == "br":
            if self._cell is not None:
                self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            text = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            rowspan, colspan = self._cell_span
            while (self._row, self._col) in self._pending:
                self._col += 1
            for dr in range(rowspan):
                for dc in range(colspan):
                    self._pending[(self._row + dr, self._col + dc)] = text
            self._col += colspan
            self._cell = None
        elif tag == "tr" and self._row_has_header:
            self._header_rows.append(self._row)

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    # -- result ------------------------------------------------------------
    def grid(self) -> List[List[str]]:
        if not self._pending:
            return []
        n_rows = max(r for r, _ in self._pending) + 1
        n_cols = max(c for _, c in self._pending) + 1
        return [[self._pending.get((r, c), "") for c in range(n_cols)]
                for r in range(n_rows)]

    def header_rows(self) -> List[int]:
        return sorted(set(self._header_rows))


def parse_table_grid(html: str) -> List[List[str]]:
    """Parse MinerU table HTML into a rectangular grid (``[]`` on failure)."""
    if not html or "<t" not in html.lower():
        return []
    parser = _TableGrid()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:                     # a malformed table is not fatal
        logger.debug(f"table parse failed: {exc}")
        return []
    return parser.grid()


def render_table(html: str, max_rows: int = 40) -> str:
    """Render table HTML as pipe-delimited rows — readable, and good for BM25.

    HTML tags are noise to an embedding model and to BM25; the cell values are
    the signal.  Used for both ``display_text`` and ``embed_text`` of a table
    chunk (never for ``text``, which stays the verbatim source span).
    """
    grid = parse_table_grid(html)
    if not grid:
        return ""
    lines = []
    for row in grid[:max_rows]:
        cells = [_LATEX_STRIP.sub("", c).strip() for c in row]
        if not any(cells):
            continue
        lines.append(" | ".join(cells))
    if len(grid) > max_rows:
        lines.append(f"... ({len(grid) - max_rows} more rows)")
    return "\n".join(lines)


def _header_depth(grid: List[List[str]]) -> int:
    """How many leading rows are header rows.

    A header row is one whose cells are mostly *not* value cells.  Two-level
    headers are common in activity tables (property over strain), and getting
    the depth wrong either loses the unit or turns a data row into a header —
    which silently drops the first compound in the table.

    The test is ``_VALUE_CELL``, not "parses as a number", because the cells
    that matter here are ``0.382 [0.337-0.432]`` (a value) and ``25% MHBII``
    (a header), and a leading-number test gets both wrong.
    """
    depth = 0
    for row in grid[:3]:
        values = [_LATEX_STRIP.sub("", c) for c in row[1:] if c.strip()]
        if not values:
            values = [_LATEX_STRIP.sub("", c) for c in row if c.strip()]
        if not values:
            break
        numeric = sum(1 for c in values if _VALUE_CELL.match(c))
        if numeric > len(values) / 2:
            break
        depth += 1
    return max(depth, 1) if grid else 0


def facts_from_table(
    html: str,
    caption: str = "",
    max_facts: int = MAX_FACTS_PER_CHUNK,
) -> List[Dict[str, Any]]:
    """Read ``entity × property = value unit`` triples out of a table."""
    grid = parse_table_grid(html)
    if len(grid) < 2:
        return []

    depth = _header_depth(grid)
    if depth >= len(grid):
        return []
    n_cols = len(grid[0])

    # Column property + unit, taken from the deepest header cell that names a
    # known property; the unit may sit on a different header level than the name.
    col_property: List[Optional[str]] = [None] * n_cols
    col_unit: List[Optional[str]] = [None] * n_cols
    col_label: List[str] = [""] * n_cols
    caption_property = _canon_property(caption)
    caption_unit = None
    for candidate in _BRACKET_UNIT.findall(caption or ""):
        caption_unit = caption_unit or _canon_unit(candidate)

    for col in range(n_cols):
        labels = [grid[r][col] for r in range(depth) if col < len(grid[r])]
        col_label[col] = " ".join(lbl for lbl in labels if lbl).strip()
        for label in labels:
            col_property[col] = col_property[col] or _canon_property(label)
            for candidate in _BRACKET_UNIT.findall(label):
                col_unit[col] = col_unit[col] or _canon_unit(candidate)
            if col_unit[col] is None:
                bare = _LATEX_STRIP.sub("", label).strip()
                if bare in ("%", "°C"):
                    col_unit[col] = bare

    # The entity column: the first column with non-numeric data cells.
    entity_col = 0
    for col in range(n_cols):
        values = [grid[r][col] for r in range(depth, len(grid)) if grid[r][col].strip()]
        if values and not any(_NUM_ONLY.match(_LATEX_STRIP.sub("", v)) for v in values):
            entity_col = col
            break

    facts: List[Dict[str, Any]] = []
    for row_idx in range(depth, len(grid)):
        row = grid[row_idx]
        entity = _LATEX_STRIP.sub("", row[entity_col]).strip() if entity_col < len(row) else ""
        for col in range(n_cols):
            if col == entity_col or col >= len(row):
                continue
            cell = _LATEX_STRIP.sub("", row[col]).strip()
            if not cell:
                continue
            match = re.match(r"^\s*" + _NUMBER, cell)
            if not match:
                continue
            value = _parse_value(match.group("num"))
            if value is None:
                continue
            unit = col_unit[col] or caption_unit
            inline_unit = re.match(r"^\s*" + _NUMBER + r"\s*(?P<unit>" + _UNIT_RE + r")\b", cell)
            if inline_unit:
                unit = _canon_unit(inline_unit.group("unit")) or unit
            prop = col_property[col] or caption_property
            if prop is None and unit in _CONCENTRATION_UNITS:
                prop = "concentration"
            if prop is None:
                continue                      # unnamed column: not a claim we can make
            facts.append({
                "property": prop,
                "value": value,
                "value_text": cell[:60],
                "unit": unit,
                "entity": entity[:80] or None,
                "source": "table",
                "context": (col_label[col] or "")[:80],
            })
            if len(facts) >= max_facts:
                return facts
    return facts


# ---------------------------------------------------------------------------
# Prose
# ---------------------------------------------------------------------------

# "IC50 of 12 nM", "MIC = 2 µg/mL", "Kd values of 0.38 µmol/L"
_PROP_FIRST = re.compile(
    r"(?P<prop>" + "|".join(pat for pat, _ in _PROPERTY_PATTERNS) + r")"
    r"(?P<mid>[^.;:\n]{0,24}?)"
    r"(?:of|=|:|was|were|is|are|at|≈|~)?\s*"
    + _NUMBER +
    r"\s*(?P<unit>" + _UNIT_RE + r")\b",
    re.IGNORECASE,
)
# "85% yield", "12 nM IC50", "80 °C"
_VALUE_FIRST = re.compile(
    _NUMBER + r"\s*(?P<unit>" + _UNIT_RE + r")\b"
    r"(?P<after>\s{0,2}(?:of\s+)?(?P<prop>" + "|".join(pat for pat, _ in _PROPERTY_PATTERNS) + r"))?",
    re.IGNORECASE,
)
#: A compound-or-peptide-looking name: Api137, Pd(PPh3)2Cl2, TMEDA, 4-OHT.
_ENTITY_NEAR = re.compile(
    r"\b(?:[A-Z][A-Za-z]{1,14}\s?-?\d{1,4}[A-Za-z]?"      # Api137, SRC-2, Onc112
    r"|[A-Z][a-z]?\([A-Za-z0-9()]{2,20}\)\d?[A-Za-z0-9]{0,6}"  # Pd(PPh3)2Cl2
    r"|[A-Z]{2,8}"                                          # TMEDA, DMSO
    r"|compound\s+\d+[a-z]?)\b"
)


def _entity_before(text: str, start: int, window: int = 90) -> Optional[str]:
    """Nearest compound-shaped name in the preceding window, if any."""
    left = text[max(0, start - window):start]
    matches = list(_ENTITY_NEAR.finditer(left))
    if not matches:
        return None
    name = matches[-1].group(0).strip()
    if name.lower() in ("the", "and", "with", "for", "was", "were", "this", "that",
                        "figure", "table", "fig", "eq", "si", "nmr", "hplc", "ms",
                        "using", "from", "after", "about", "approximately"):
        return None
    return name[:80]


def facts_from_prose(text: str, max_facts: int = MAX_FACTS_PER_CHUNK) -> List[Dict[str, Any]]:
    """Best-effort value+unit(+property)(+entity) triples from running prose."""
    if not text:
        return []
    facts: List[Dict[str, Any]] = []
    seen = set()

    def add(prop, qual, num, unit, entity, span):
        value = _parse_value(num)
        if value is None or prop is None:
            return
        key = (prop, round(value, 6), unit, (entity or "").lower())
        if key in seen:
            return
        seen.add(key)
        raw = text[max(0, span[0]):min(len(text), span[1])]
        facts.append({
            "property": prop,
            "value": value,
            "value_text": (qual.strip() + raw.strip())[:60] if qual.strip() else raw.strip()[:60],
            "unit": unit,
            "entity": entity,
            "source": "prose",
            "context": re.sub(r"\s+", " ", text[max(0, span[0] - 60):span[1] + 20])[:160],
        })

    for match in _PROP_FIRST.finditer(text):
        # A long filler between the property word and the number usually means
        # they are unrelated ("MIC values, measured in triplicate, 12 nM" is ok;
        # a whole clause is not).
        if len(match.group("mid") or "") > 24:
            continue
        prop = _canon_property(match.group("prop"))
        unit = _canon_unit(match.group("unit"))
        add(prop, match.group("qual") or "", match.group("num"), unit,
            _entity_before(text, match.start()), match.span())
        if len(facts) >= max_facts:
            return facts

    for match in _VALUE_FIRST.finditer(text):
        unit = _canon_unit(match.group("unit"))
        prop = _canon_property(match.group("prop")) if match.group("prop") else None
        if prop is None:
            prop = _UNIT_IMPLIED_PROPERTY.get(unit) if unit else None
        if prop is None and unit in _CONCENTRATION_UNITS:
            prop = "concentration"
        if prop is None:
            continue
        add(prop, match.group("qual") or "", match.group("num"), unit,
            _entity_before(text, match.start()), match.span())
        if len(facts) >= max_facts:
            break
    return facts
