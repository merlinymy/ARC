#!/usr/bin/env python
"""W2: index the sample into a THROWAWAY Qdrant collection and query it.

Proves the last link in the chain: that the new payload survives a round trip
through Qdrant, that pages/offsets/bbox come back on the other side, and that
`table` chunks — a type with **zero** members corpus-wide today — are actually
retrievable by a real user query.

Safety
------
* The collection name is generated per run (``w2_sample_<pid>_<ts>``) and is
  **asserted** not to be the live one before anything is written.
* The collection is deleted in a ``finally`` block.
* The BM25 IDF cache is loaded read-only; nothing calls ``save_idf_cache``.

Usage:
    python scripts/w2_index_sample.py
    python scripts/w2_index_sample.py --keep      # leave the collection for poking
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from config import settings                              # noqa: E402
from preprocessing import paper_record as pr             # noqa: E402
from preprocessing.chunker import PaperChunker           # noqa: E402
from preprocessing.models import (                       # noqa: E402
    ChunkType,
    display_text_from_payload,
    embed_text_from_payload,
)
from retrieval.bm25 import BM25Vectorizer                # noqa: E402
from retrieval.embedder import VoyageEmbedder            # noqa: E402
from retrieval.qdrant_store import QdrantStore           # noqa: E402

LIVE_COLLECTION = settings.qdrant_collection_name

#: A few of W5's golden queries, chosen to exercise what W2 changed.
GOLDEN = [
    ("q16", "what is the mic of apidaecin analogs including api 137"),
    ("q07", "tell me how to efficiently and with the highest yield "
            "on tetramethyl guanidinylate the N terminus of my peptide"),
    ("q06", "can you tell me where apidaecin is fitc tagged"),
    ("q13", "What are the cellular concentrations that some drugs and peptide "
            "accumulate and what is the limit of detection using SRS"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="data/w1_sample/records")
    ap.add_argument("--extra-records", default="data/w1_sample_v2/records")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="papers (0 = all)")
    args = ap.parse_args()

    collection = f"w2_sample_{os.getpid()}_{int(time.time())}"
    assert collection != LIVE_COLLECTION, "refusing to touch the live collection"
    assert not collection.startswith("research"), "refusing to touch the live collection"

    records: Dict[str, Path] = {}
    for directory in (args.records, args.extra_records):
        path = BACKEND / directory
        if path.exists():
            for file in sorted(path.glob("*.json")):
                records[file.stem] = file
    paths = list(records.values())[:args.limit or None]

    print("=" * 78)
    print(f"W2 SAMPLE INDEX -> throwaway collection '{collection}'")
    print(f"live collection '{LIVE_COLLECTION}' is not touched")
    print("=" * 78)

    chunker = PaperChunker()
    chunks = []
    docs: Dict[str, str] = {}
    for path in paths:
        record = json.loads(path.read_text())
        docs[record["paper_id"]] = pr.full_text(record)
        paper_chunks = chunker.chunk_record(record)
        full = chunker.make_full_chunk(paper_chunks)
        if full is not None:
            paper_chunks.append(full)
        chunks.extend(paper_chunks)
    print(f"\n{len(chunks)} chunks from {len(paths)} papers")

    bm25 = BM25Vectorizer()
    if bm25.load_idf_cache():
        print(f"BM25 IDF cache loaded read-only ({len(bm25._idf_cache)} terms, "
              f"doc_count={bm25._doc_count})")

    embedder = VoyageEmbedder(api_key=settings.voyage_api_key,
                              model=settings.embedding_model)
    store = QdrantStore(collection_name=collection, enable_hybrid=True,
                        bm25_vectorizer=bm25,
                        embedding_dimension=settings.embedding_dimension)

    try:
        store.ensure_collection()
        texts = [c.embed_text for c in chunks]
        print(f"embedding {len(texts)} chunks with {settings.embedding_model}...")
        embeddings = embedder.embed_documents(texts)
        store.upsert_chunks([c.chunk_id for c in chunks], embeddings,
                            [c.to_payload() for c in chunks])
        info = store.client.get_collection(collection)
        print(f"indexed: {info.points_count} points")

        # ---- payload round trip ----------------------------------------
        print("\n--- PAYLOAD ROUND TRIP ---")
        sample_id = next(c.chunk_id for c in chunks
                         if c.chunk_type == ChunkType.TABLE and c.numeric_facts)
        stored = store.get_chunk_by_id(sample_id)
        for key in ("chunk_type", "page_start", "page_end", "page_numbers",
                    "char_start", "char_end", "bbox", "section_name",
                    "text_is_verbatim", "has_numeric_facts"):
            print(f"  {key:18s} {stored.get(key)!r}"[:150])
        print(f"  numeric_properties {stored.get('numeric_properties')}")
        print(f"  numeric_entities   {str(stored.get('numeric_entities'))[:110]}")
        print(f"  numeric_facts[0]   {stored.get('numeric_facts', [{}])[0]}")
        print(f"  embed_text recomposed from payload: "
              f"{embed_text_from_payload(stored)[:90]!r}")

        # The offsets must still resolve against the record after the round trip:
        # this is the exact lookup W3's PDF highlight will perform.
        doc = docs[stored["paper_id"]]
        verbatim = doc[stored["char_start"]:stored["char_end"]] == stored["text"]
        print(f"  full_text[char_start:char_end] == stored text: "
              f"{'PASS' if verbatim else 'FAIL'}")

        missing_pages = store.client.count(
            collection_name=collection,
            count_filter=_null_filter("page_start"),
        ).count
        print(f"  points with a null page_start: {missing_pages} "
              f"(full chunks have no span: expected {sum(1 for c in chunks if c.page_start is None)})")

        # ---- golden queries --------------------------------------------
        print("\n--- W5 GOLDEN QUERIES (hybrid, broad path: no chunk_type filter) ---")
        for qid, query in GOLDEN:
            embedding = embedder.embed_query(query)
            results = store.hybrid_search(query=query, query_embedding=embedding,
                                          limit=10)
            print(f"\n  [{qid}] {query[:74]}")
            by_type: Dict[str, int] = {}
            for r in results:
                by_type[r.get("chunk_type", "?")] = by_type.get(r.get("chunk_type", "?"), 0) + 1
            print(f"       types returned: {by_type}")
            for r in results[:3]:
                pages = (f"p.{r.get('page_start')}"
                         + (f"-{r.get('page_end')}" if r.get('page_end') != r.get('page_start') else ""))
                print(f"       {r.get('chunk_type'):8s} {pages:8s} "
                      f"chars {r.get('char_start')}-{r.get('char_end')} "
                      f"bbox={r.get('bbox')} {r.get('section_name')}")
                print(f"           {display_text_from_payload(r)[:110].strip()!r}")
                if r.get("numeric_facts"):
                    facts = r["numeric_facts"][:3]
                    print(f"           facts: "
                          + "; ".join(f"{f['entity']} {f['property']}={f['value']}{f['unit'] or ''}"
                                      for f in facts))

        # ---- is the table type reachable at all? -----------------------
        print("\n--- TABLE CHUNKS ARE REACHABLE ---")
        embedding = embedder.embed_query("MIC values of Api137 against E. coli")
        table_hits = store.search(query_embedding=embedding, limit=5,
                                  chunk_types=["table"])
        print(f"  chunk_types=['table'] returns {len(table_hits)} hits "
              f"(corpus today: 0 table chunks exist)")
        for r in table_hits[:2]:
            print(f"    {r.get('chunk_id')} p.{r.get('page_start')} "
                  f"{len(r.get('numeric_facts') or [])} facts "
                  f"{display_text_from_payload(r)[:80].strip()!r}")
        return 0

    finally:
        if args.keep:
            print(f"\n-- keeping '{collection}' as asked; delete it yourself")
        else:
            store.client.delete_collection(collection)
            print(f"\n-- deleted throwaway collection '{collection}'")
            remaining = [c.name for c in store.client.get_collections().collections]
            print(f"-- collections now: {remaining}")


def _null_filter(field: str):
    from qdrant_client.models import Filter, IsNullCondition, PayloadField
    return Filter(must=[IsNullCondition(is_null=PayloadField(key=field))])


if __name__ == "__main__":
    raise SystemExit(main())
