"""LLM relevance judging for the W5 qrels, using Claude Haiku 4.5.

TREC-style graded judgments on pooled (query, chunk) pairs:

    2 - directly answers the question / contains the specific asked-for content
    1 - partially relevant: real supporting material, does not answer on its own
    0 - irrelevant

Judgments are cached on disk keyed by sha256(query_text || chunk_id), so a
re-run of the harness costs nothing and every configuration sees exactly the
same labels for the same pair. The judge never learns which retrieval arm
surfaced a chunk - the pool is a de-duplicated union and each pair is judged
once - so no arm can be favoured by the labelling step.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

JUDGE_MODEL = "claude-haiku-4-5"
# Claude Haiku 4.5 list price, USD per million tokens.
PRICE_IN = 1.00
PRICE_OUT = 5.00

# Chunk text sent to the judge. Deliberately above the longest chunk in the
# corpus (~15.2k chars), i.e. no truncation in practice.
#
# An earlier 3000-char cap was removed because it was *differentially* unfair:
# 16.5% of retrieved chunks are longer than that, and almost all of them are
# `section` chunks - precisely the chunk type the FACTUAL retrieval strategy
# excludes. Truncating them would have under-graded the evidence for the arms
# that retrieve sections and flattered the arms that filter them out, which is
# the exact comparison this harness exists to adjudicate.
MAX_CHUNK_CHARS = 20000

SYSTEM = """You are a relevance assessor building a test collection for a chemistry and \
chemical-biology research-paper search engine. You judge whether a retrieved passage is \
relevant to a researcher's question. You are strict, consistent, and you never reward a \
passage for merely sharing vocabulary with the question."""

PROMPT = """Judge how relevant this passage is to the researcher's question.

<question>
{query}
</question>

<passage>
Paper: {title}
Section: {section} (passage type: {chunk_type})
---
{text}
</passage>

Grading scale:
- 2 = DIRECTLY RELEVANT. The passage contains information that directly answers the \
question, or is exactly the kind of material the request asked to be found (the specific \
value, protocol, reagent, definition, finding, or the named paper itself).
- 1 = PARTIALLY RELEVANT. The passage is genuinely on-topic and a good answer could \
usefully cite it for background or supporting context, but it does not answer the \
question on its own.
- 0 = NOT RELEVANT. Off-topic, or related only by shared words / the same general field.

Notes:
- The question may be open-ended (framing, brainstorming, "summarise X", "propose Y"). \
Grade such questions on whether the passage supplies substantive material a good answer \
would actually draw on - not on whether it is a complete answer.
- A passage that merely mentions a term from the question, in an unrelated context, is 0.
- Figure and table captions can be 2 if the caption itself carries the answer.

Answer with JSON only: {{"grade": 0|1|2, "reason": "<10 words or fewer>"}}"""


@dataclass
class Judgment:
    grade: int
    reason: str
    model: str = JUDGE_MODEL


def pair_key(query_text: str, chunk_id: str) -> str:
    return hashlib.sha256(f"{query_text}\x00{chunk_id}".encode("utf-8")).hexdigest()


class RelevanceJudge:
    def __init__(
        self,
        client,
        cache_path: Path,
        max_spend_usd: float = 5.0,
        workers: int = 8,
    ):
        self.client = client
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, Dict[str, Any]] = (
            json.loads(cache_path.read_text()) if cache_path.exists() else {}
        )
        self.max_spend_usd = max_spend_usd
        self.workers = workers

        self._lock = threading.Lock()
        self.calls = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self.failures = 0
        self.stopped_on_budget = False

    # -- accounting -------------------------------------------------------

    @property
    def spend_usd(self) -> float:
        return (self.in_tokens / 1e6) * PRICE_IN + (self.out_tokens / 1e6) * PRICE_OUT

    def stats(self) -> Dict[str, Any]:
        return {
            "model": JUDGE_MODEL,
            "api_calls": self.calls,
            "input_tokens": self.in_tokens,
            "output_tokens": self.out_tokens,
            "spend_usd": round(self.spend_usd, 4),
            "cached_judgments": len(self.cache),
            "failures": self.failures,
            "stopped_on_budget": self.stopped_on_budget,
        }

    def save(self) -> None:
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache, indent=1, ensure_ascii=False))
        tmp.replace(self.cache_path)

    # -- judging ----------------------------------------------------------

    def _call(self, query_text: str, chunk: Dict[str, Any]) -> Judgment:
        text = (chunk.get("text") or "")[:MAX_CHUNK_CHARS]
        prompt = PROMPT.format(
            query=query_text,
            title=chunk.get("title") or "(untitled)",
            section=chunk.get("section_name") or "(unknown)",
            chunk_type=chunk.get("chunk_type") or "(unknown)",
            text=text,
        )
        resp = self.client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=90,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        with self._lock:
            self.calls += 1
            self.in_tokens += getattr(resp.usage, "input_tokens", 0) or 0
            self.out_tokens += getattr(resp.usage, "output_tokens", 0) or 0

        raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return _parse(raw)

    def judge_pairs(
        self,
        pairs: List[Tuple[str, str, Dict[str, Any]]],
        progress_every: int = 100,
    ) -> Dict[str, Judgment]:
        """Judge (query_id, query_text, chunk) triples; returns {pair_key: Judgment}.

        Already-cached pairs are returned without an API call.
        """
        todo = []
        out: Dict[str, Judgment] = {}
        for query_id, query_text, chunk in pairs:
            key = pair_key(query_text, chunk["chunk_id"])
            hit = self.cache.get(key)
            if hit is not None:
                out[key] = Judgment(**hit)
            else:
                todo.append((key, query_id, query_text, chunk))

        if not todo:
            return out

        logger.info(
            "judging %d new pairs (%d already cached)", len(todo), len(pairs) - len(todo)
        )

        done = 0
        with cf.ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {}
            for item in todo:
                futures[pool.submit(self._safe_call, item)] = item
            for fut in cf.as_completed(futures):
                key, query_id, query_text, chunk = futures[fut]
                judgment = fut.result()
                if judgment is None:
                    continue
                out[key] = judgment
                with self._lock:
                    self.cache[key] = {
                        "grade": judgment.grade,
                        "reason": judgment.reason,
                        "model": judgment.model,
                    }
                done += 1
                if done % progress_every == 0:
                    self.save()
                    logger.info(
                        "judged %d/%d  spend=$%.3f", done, len(todo), self.spend_usd
                    )
                if self.spend_usd > self.max_spend_usd:
                    self.stopped_on_budget = True

        self.save()
        return out

    def _safe_call(self, item) -> Optional[Judgment]:
        key, query_id, query_text, chunk = item
        if self.stopped_on_budget:
            return None
        try:
            return self._call(query_text, chunk)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.failures += 1
            logger.warning("judge failed for %s/%s: %s", query_id, chunk["chunk_id"], exc)
            return None


_JSON = re.compile(r"\{.*\}", re.S)


def _parse(raw: str) -> Judgment:
    m = _JSON.search(raw or "")
    if not m:
        return Judgment(grade=0, reason="unparseable judge output")
    try:
        data = json.loads(m.group(0))
        grade = int(data.get("grade", 0))
        if grade not in (0, 1, 2):
            grade = 0
        return Judgment(grade=grade, reason=str(data.get("reason", ""))[:120])
    except (json.JSONDecodeError, TypeError, ValueError):
        return Judgment(grade=0, reason="unparseable judge output")


def estimate_cost(pairs: List[Tuple[str, str, Dict[str, Any]]]) -> Dict[str, float]:
    """Rough pre-flight cost estimate (4 chars/token, ~40 output tokens)."""
    fixed = len(SYSTEM) + len(PROMPT)
    chars = sum(
        fixed + len(q) + len((c.get("text") or "")[:MAX_CHUNK_CHARS]) + 120
        for _, q, c in pairs
    )
    in_tok = chars / 4
    out_tok = len(pairs) * 40
    return {
        "pairs": len(pairs),
        "est_input_tokens": round(in_tok),
        "est_output_tokens": round(out_tok),
        "est_usd": round((in_tok / 1e6) * PRICE_IN + (out_tok / 1e6) * PRICE_OUT, 3),
    }
