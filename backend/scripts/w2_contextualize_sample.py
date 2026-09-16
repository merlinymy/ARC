#!/usr/bin/env python
"""W2: validate contextual retrieval (§3b.2) on a handful of papers and price it.

One Haiku call per paper emits a context line for every chunk in that paper.
This script runs it on a few sample records, shows the lines, and turns the
*measured* token counts into a corpus projection — the plan's ≈$96 / ≈$48 was an
estimate, and this replaces it with arithmetic on real usage.

It deliberately does **not** run corpus-wide. The corpus pass belongs to the
single W1+W2 reindex, via the Batch API (`contextualizer.batch_request`).

Usage:
    python scripts/w2_contextualize_sample.py --papers 3
    python scripts/w2_contextualize_sample.py --dry-run     # tokens only, no spend
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from config import settings                              # noqa: E402
from preprocessing import contextualizer as ctx          # noqa: E402
from preprocessing import paper_record as pr             # noqa: E402
from preprocessing.chunker import PaperChunker           # noqa: E402

CORPUS_PAPERS = 4797
#: Numeric-dense first: the Api137 MIC/K_d/K_i paper, a synthesis paper, a review.
PREFERRED = ["73fc45a3d6ad", "e6d4c4c705f8", "f1b1fe44c739"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="data/w1_sample_v2/records")
    ap.add_argument("--papers", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true",
                    help="count tokens with count_tokens, make no billed call")
    ap.add_argument("--model", default=ctx.DEFAULT_MODEL)
    args = ap.parse_args()

    directory = BACKEND / args.records
    paths = {p.stem: p for p in sorted(directory.glob("*.json"))}
    order = [paths[k] for k in PREFERRED if k in paths]
    order += [p for k, p in paths.items() if k not in PREFERRED]
    order = order[:args.papers]
    if not order:
        print(f"no records under {args.records}")
        return 1

    import anthropic
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    chunker = PaperChunker()

    print("=" * 78)
    print(f"W2 CONTEXTUAL RETRIEVAL — {len(order)} papers, model {args.model}"
          f"{' (dry run)' if args.dry_run else ''}")
    print("=" * 78)

    rows = []
    for path in order:
        record = json.loads(path.read_text())
        chunks = chunker.chunk_record(record)
        wanted = ctx.targets(chunks)
        paper_text = pr.body_text(record)
        messages, _ = ctx.build_messages(paper_text, chunks,
                                         record["metadata"].get("title", ""))

        if args.dry_run:
            counted = client.messages.count_tokens(
                model=args.model, system=ctx.SYSTEM_PROMPT, messages=messages)
            row = {"paper_id": record["paper_id"], "n_chunks": len(wanted),
                   "input_tokens": counted.input_tokens,
                   "output_tokens": ctx.max_tokens_for(len(wanted)) * 0.6,
                   "cost_usd": None, "contexts": {}}
        else:
            contexts, usage = ctx.contextualize_paper(
                paper_text, chunks, client,
                title=record["metadata"].get("title", ""), model=args.model)
            row = {"paper_id": record["paper_id"], "n_chunks": len(wanted),
                   **usage, "contexts": contexts}

        rows.append(row)
        print(f"\n--- {record['paper_id']}  {record['file_name'][:52]}")
        print(f"    {len(chunks)} chunks, {len(wanted)} need a line | "
              f"in {row['input_tokens']:,} tok, out {row['output_tokens']:,.0f} tok"
              + (f", ${row['cost_usd']:.4f}" if row.get("cost_usd") else ""))
        if row["contexts"]:
            print(f"    {len(row['contexts'])}/{len(wanted)} lines returned")
            for chunk in wanted[:4]:
                line = row["contexts"].get(chunk.chunk_id)
                if not line:
                    continue
                print(f"      {chunk.chunk_id} ({chunk.chunk_type.value})")
                print(f"        chunk : {chunk.text[:100].strip()!r}")
                print(f"        line  : {line}")

    mean_in = statistics.mean(r["input_tokens"] for r in rows)
    mean_out = statistics.mean(r["output_tokens"] for r in rows)
    print("\n--- PROJECTED CORPUS COST (4,797 papers) ---")
    for batch in (False, True):
        est = ctx.estimate_cost(CORPUS_PAPERS, mean_in, mean_out, batch=batch)
        print(f"  {'Batch API (50% off)' if batch else 'Standard API':22s} "
              f"${est['per_paper_usd']:.4f}/paper -> ${est['total_usd']:.2f} total")
    print(f"  measured means: {mean_in:,.0f} input / {mean_out:,.0f} output tokens "
          f"per paper over {len(rows)} papers")
    print(f"  plan's estimate was ~$96 standard / ~$48 batch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
