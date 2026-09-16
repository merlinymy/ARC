"""Contextual retrieval — one LLM context line per chunk (§3b.2, W2 item 4).

A fine chunk pulled out of a Results section often reads as "The values were
markedly lower in the mutant strain (Table 2)": correct, and unretrievable,
because nothing in it names the compound, the assay or the paper.  Prepending a
short model-written line that says what the chunk *is about* fixes that, and
2026 benchmarks put the gain specifically on **numeric** queries, which is this
library's strongest use case.

Cost is the whole design constraint.  The naive implementation re-sends the
paper for every chunk.  This module makes **one call per paper** that emits a
line for every chunk in it at once, so the paper text is paid for once instead
of ~45 times.

Where the line goes
-------------------
Into ``Chunk.llm_context``, which reaches the index only through
``embed_text`` — never through ``text``.  That is the C2 resolution: the
Citations API sees the byte-exact source span, while Voyage and BM25 see the
span plus its context.  Putting the line in ``text`` would make every quote
begin with a sentence that is not in the paper.

Running it
----------
- ``contextualize_paper``   one paper, synchronously (validation, uploads)
- ``batch_request``         one Batch API request per paper (the corpus run, 50% off)
- ``parse_response``        model output → ``{chunk_id: context}``
- ``estimate_cost``         projected spend from measured token counts

Not run corpus-wide by W2 — it is designed here and validated on a handful of
papers; the corpus pass belongs to the single W1+W2 reindex.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .models import Chunk, ChunkType

logger = logging.getLogger(__name__)

#: Haiku 4.5 rates, USD per million tokens (input / output).
HAIKU_INPUT_PER_MTOK = 1.00
HAIKU_OUTPUT_PER_MTOK = 5.00
#: The Batch API is 50% off.
BATCH_DISCOUNT = 0.5

DEFAULT_MODEL = "claude-haiku-4-5"

#: How much of the paper to send.  A context line needs the paper's subject,
#: system and structure, not its every sentence; the cap also bounds the cost of
#: a 60-page review.
MAX_PAPER_CHARS = 48_000
#: Each chunk is identified to the model by a short head+tail stub.
STUB_HEAD_CHARS = 180
STUB_TAIL_CHARS = 60
#: Chunk types that need a context line.  `abstract`, `section` and `full`
#: already carry their own context, and a caption already opens with "Table 2.
#: Antibacterial activities of ..." — paying for those is waste.
CONTEXTUALIZED_TYPES = frozenset({ChunkType.FINE, ChunkType.TABLE})
#: Guardrail: a context line is a sentence, not a summary.
MAX_CONTEXT_CHARS = 300

SYSTEM_PROMPT = (
    "You situate excerpts within the scientific paper they came from, so that "
    "the excerpt can be found by search on its own. You are given the paper and "
    "a numbered list of excerpts from it. For each excerpt, write ONE short "
    "sentence (<= 20 words) that states what the excerpt is about in the context "
    "of the paper: name the compounds, materials, assays, techniques or systems "
    "it concerns and, where the excerpt reports numbers, say what those numbers "
    "measure and of what. Use the paper's own terminology, expand an acronym the "
    "first time it appears, and never add a fact the paper does not state. Do not "
    "summarise the excerpt's conclusion and do not begin with 'This excerpt' or "
    "'This chunk'."
)

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "contexts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "i": {"type": "integer"},
                    "context": {"type": "string"},
                },
                "required": ["i", "context"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["contexts"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _stub(text: str) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    if len(flat) <= STUB_HEAD_CHARS + STUB_TAIL_CHARS + 5:
        return flat
    return f"{flat[:STUB_HEAD_CHARS]} […] {flat[-STUB_TAIL_CHARS:]}"


def targets(chunks: Sequence[Chunk]) -> List[Chunk]:
    """The chunks worth a context line, in index order."""
    return [c for c in chunks if c.chunk_type in CONTEXTUALIZED_TYPES]


def build_messages(
    paper_text: str,
    chunks: Sequence[Chunk],
    title: str = "",
) -> Tuple[List[Dict[str, Any]], List[Chunk]]:
    """Build the single per-paper request body.  Returns ``(messages, targets)``."""
    wanted = targets(chunks)
    if not wanted:
        return [], []

    body = paper_text[:MAX_PAPER_CHARS]
    truncated = len(paper_text) > MAX_PAPER_CHARS

    lines = []
    for n, chunk in enumerate(wanted):
        where = chunk.section_name or "body"
        page = f"p.{chunk.page_start}" if chunk.page_start else "?"
        lines.append(f"[{n}] ({chunk.chunk_type.value}, {where}, {page}) {_stub(chunk.text)}")

    prompt = (
        f"<paper title=\"{title[:200]}\">\n{body}\n"
        f"{'[paper truncated]' if truncated else ''}</paper>\n\n"
        f"<excerpts>\n" + "\n".join(lines) + "\n</excerpts>\n\n"
        f"Return one context sentence for each of the {len(wanted)} excerpts, "
        f"keyed by its number."
    )
    return [{"role": "user", "content": prompt}], wanted


def max_tokens_for(n_chunks: int) -> int:
    """Output budget per paper.

    A 20-word line is ~30 tokens and its JSON wrapper another ~12, so 60 per
    line leaves headroom.  Under-sizing this does not truncate one line — it
    truncates the JSON array and loses the whole paper, which is how the first
    run of this failed.
    """
    return max(1024, min(16384, 400 + 60 * n_chunks))


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

def parse_response(text: str, wanted: Sequence[Chunk]) -> Dict[str, str]:
    """Map the model's JSON onto ``{chunk_id: context}``.

    Tolerant on purpose: a malformed line costs one chunk its context, and must
    not cost the paper its chunks.
    """
    if not text:
        return {}
    payload: Any = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = None

    items: List[Any]
    if isinstance(payload, dict):
        items = payload.get("contexts") or []
    else:
        # Truncated output (a `max_tokens` stop) leaves a valid prefix of the
        # array. Salvage the complete objects rather than losing the paper.
        items = []
        for raw in re.finditer(r"\{[^{}]*\}", text):
            try:
                items.append(json.loads(raw.group(0)))
            except json.JSONDecodeError:
                continue
        if items:
            logger.warning(f"contextualizer: response was not valid JSON; "
                           f"salvaged {len(items)} objects")
        else:
            logger.warning("contextualizer: could not parse response as JSON")
            return {}

    out: Dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("i"))
        except (TypeError, ValueError):
            continue
        context = (item.get("context") or "").strip()
        if not context or not (0 <= index < len(wanted)):
            continue
        out[wanted[index].chunk_id] = context[:MAX_CONTEXT_CHARS]
    missing = len(wanted) - len(out)
    if missing:
        logger.info(f"contextualizer: {missing}/{len(wanted)} chunks got no context line")
    return out


# ---------------------------------------------------------------------------
# Synchronous single-paper call
# ---------------------------------------------------------------------------

def contextualize_paper(
    paper_text: str,
    chunks: Sequence[Chunk],
    client: Any,
    title: str = "",
    model: str = DEFAULT_MODEL,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """One Haiku call for one paper.  Returns ``({chunk_id: context}, usage)``.

    ``usage`` carries the measured token counts and dollar cost, so the corpus
    projection is arithmetic on measurements rather than an estimate.
    """
    messages, wanted = build_messages(paper_text, chunks, title)
    if not wanted:
        return {}, {}

    kwargs: Dict[str, Any] = dict(
        model=model,
        max_tokens=max_tokens_for(len(wanted)),
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    try:
        response = client.messages.create(
            **kwargs,
            output_config={"format": {"type": "json_schema", "schema": _RESPONSE_SCHEMA}},
        )
    except Exception as exc:
        # Structured output is the nicety, not the feature.
        logger.info(f"contextualizer: structured output unavailable ({exc}); retrying plain")
        kwargs["system"] = SYSTEM_PROMPT + (
            ' Reply with JSON only: {"contexts":[{"i":0,"context":"..."}]}'
        )
        response = client.messages.create(**kwargs)

    if getattr(response, "stop_reason", None) == "max_tokens":
        logger.warning(
            f"contextualizer: hit max_tokens on {len(wanted)} chunks — raise "
            f"max_tokens_for(); salvaging what parsed"
        )
    text = "".join(block.text for block in response.content if block.type == "text")
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "n_chunks": len(wanted),
        "cost_usd": (
            response.usage.input_tokens / 1e6 * HAIKU_INPUT_PER_MTOK
            + response.usage.output_tokens / 1e6 * HAIKU_OUTPUT_PER_MTOK
        ),
    }
    return parse_response(text, wanted), usage


# ---------------------------------------------------------------------------
# Batch API (the corpus run)
# ---------------------------------------------------------------------------

def batch_request(
    paper_id: str,
    paper_text: str,
    chunks: Sequence[Chunk],
    title: str = "",
    model: str = DEFAULT_MODEL,
) -> Optional[Dict[str, Any]]:
    """One Batch API request for one paper, or ``None`` if it needs no lines.

    Submit with ``client.messages.batches.create(requests=[...])`` and key the
    results by ``custom_id`` — batch results come back in any order.
    """
    messages, wanted = build_messages(paper_text, chunks, title)
    if not wanted:
        return None
    return {
        "custom_id": paper_id,
        "params": {
            "model": model,
            "max_tokens": max_tokens_for(len(wanted)),
            "system": SYSTEM_PROMPT,
            "messages": messages,
            "output_config": {"format": {"type": "json_schema", "schema": _RESPONSE_SCHEMA}},
        },
    }


def estimate_cost(
    n_papers: int,
    mean_input_tokens: float,
    mean_output_tokens: float,
    batch: bool = False,
) -> Dict[str, float]:
    """Projected corpus spend from measured per-paper token means."""
    per_paper = (mean_input_tokens / 1e6 * HAIKU_INPUT_PER_MTOK
                 + mean_output_tokens / 1e6 * HAIKU_OUTPUT_PER_MTOK)
    if batch:
        per_paper *= BATCH_DISCOUNT
    return {
        "per_paper_usd": per_paper,
        "total_usd": per_paper * n_papers,
        "n_papers": n_papers,
        "mean_input_tokens": mean_input_tokens,
        "mean_output_tokens": mean_output_tokens,
        "batch": batch,
    }
