#!/usr/bin/env python3
"""Interactive RAG testing script.

Usage:
    python test_rag.py                    # Interactive mode
    python test_rag.py "your question"    # Single query mode
    python test_rag.py --show-context     # Show chunks sent to LLM
    python test_rag.py --verbose          # Show full pipeline steps

Features enabled (all advanced RAG features):
    - Query classification (LLM-based)
    - Query expansion (domain synonyms)
    - Query rewriting (spelling correction)
    - Entity extraction & boosting
    - HyDE (hypothetical document embeddings)
    - Hybrid search (dense + BM25)
    - Caching (embeddings, search results)
    - Conversation memory (multi-turn)
    - Cohere reranking
    - Citation verification
"""

import sys

from dependencies import get_dependencies
from retrieval.query_engine import QueryEngine


def main():
    # Check for flags
    show_context = "--show-context" in sys.argv
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    if show_context:
        sys.argv.remove("--show-context")
    if "--verbose" in sys.argv:
        sys.argv.remove("--verbose")
    if "-v" in sys.argv:
        sys.argv.remove("-v")

    print("Initializing RAG components (all features enabled)...")

    # Use shared dependencies - includes all advanced features:
    # - Caching, query rewriting, entity extraction
    # - Conversation memory, hybrid search (BM25)
    deps = get_dependencies()
    engine = deps.query_engine

    print("Ready! All features enabled:")
    print("  - Query classification & expansion")
    print("  - Query rewriting (spelling correction)")
    print("  - Entity extraction & boosting")
    print("  - HyDE (hypothetical document embeddings)")
    print("  - Hybrid search (dense + BM25)")
    print("  - Caching & conversation memory")
    print("  - Cohere reranking")
    print("  - Citation verification")
    print("Flags: --show-context (chunks) | --verbose (pipeline)")
    print("-" * 60)

    # Single query mode
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_query(engine, query, show_context, verbose)
        return

    # Interactive mode
    print("Enter your questions (Ctrl+C to exit):\n")
    while True:
        try:
            query = input("Query: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                break
            run_query(engine, query, show_context, verbose)
            print()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def create_progress_callback():
    """Create a callback that prints pipeline steps in real-time."""
    step_icons = {
        "rewriting": ("1", "QUERY REWRITING"),
        "entities": ("2", "ENTITY EXTRACTION"),
        "classification": ("3", "QUERY CLASSIFICATION"),
        "expansion": ("4", "QUERY EXPANSION"),
        "hyde": ("5", "HYDE"),
        "retrieval": ("6", "RETRIEVAL"),
        "reranking": ("7", "RERANKING"),
        "generation": ("8", "ANSWER GENERATION"),
        "verification": ("9", "CITATION VERIFICATION"),
    }

    def callback(step: str, data: dict):
        if step not in step_icons:
            return

        num, name = step_icons[step]

        # Handle "starting" status
        if data.get("status") == "starting":
            print(f"  [{num}] {name}...", end="", flush=True)
            return

        # Handle completion/results
        if step == "rewriting":
            if data.get("changed"):
                print(f" ✓ Rewritten")
                print(f"      → {data.get('rewritten', '')[:60]}...")
            else:
                print(f" ○ No change")

        elif step == "entities":
            found = data.get("found", [])
            if found:
                print(f" ✓ Found {len(found)}: {', '.join(found[:5])}")
            else:
                print(f" ○ None found")

        elif step == "classification":
            if "type" in data:
                conf = data.get("confidence", 0)
                print(f" → {data['type'].upper()} ({conf:.0%})")
                chunks = data.get("chunk_types", [])
                if chunks:
                    print(f"      Chunks: {', '.join(chunks)}")

        elif step == "expansion":
            terms = data.get("added_terms", [])
            if terms:
                print(f" ✓ Added: {', '.join(terms[:5])}")
            else:
                print(f" ○ No expansion")

        elif step == "hyde":
            if data.get("used"):
                print(f" ✓ Hypothetical doc embedded")
            else:
                print(f" ○ Skipped")

        elif step == "retrieval":
            count = data.get("count", 0)
            cache = "cached" if data.get("cache_hit") else "fresh"
            print(f" → {count} chunks ({cache})")

        elif step == "reranking":
            if "output_count" in data:
                print(f" → {data['output_count']} chunks")

        elif step == "generation":
            if data.get("status") == "complete":
                print(f" ✓ Done")

        elif step == "verification":
            if data.get("skipped"):
                print(f" ○ Skipped")
            elif data.get("verified"):
                print(f" ✓ Verified")
            else:
                print(f" ⚠ Warnings found")

    return callback


def print_pipeline_steps(result, original_query: str):
    """Print detailed pipeline steps (summary mode - for non-verbose)."""
    print(f"\nPipeline: {result.query_type.value} | {result.retrieval_count}→{result.reranked_count} chunks")


def run_query(engine: QueryEngine, query: str, show_context: bool = False, verbose: bool = False):
    """Run a single query and display results with proper source attribution."""
    print(f"\nProcessing: {query}")
    print("-" * 40)

    if verbose:
        print("\n📊 PIPELINE (real-time):")
        callback = create_progress_callback()
        result = engine.query(query, progress_callback=callback)
        print()  # newline after pipeline
    else:
        result = engine.query(query)
        print(f"Query Type: {result.query_type.value}")
        print(f"Retrieved: {result.retrieval_count} -> Reranked: {result.reranked_count}")

    # Show context sent to LLM if requested
    if show_context:
        print("\n" + "=" * 60)
        print("CONTEXT SENT TO LLM (chunks)")
        print("=" * 60)
        for i, source in enumerate(result.sources, 1):
            title = source.get('title', 'Unknown')
            section = source.get('section_name', 'N/A')
            chunk_type = source.get('chunk_type', 'unknown')
            text = source.get('text', '')[:300]  # First 300 chars
            print(f"\n[{i}] {title}")
            print(f"    Section: {section} | Type: {chunk_type}")
            print(f"    Text: {text}...")
        print("\n" + "=" * 60)

    # Print LLM answer
    print("\n--- ANSWER ---")
    print(result.answer)

    # Append sources manually (don't rely on LLM citations)
    print("\n--- REFERENCES ---")

    # Group chunks by paper, preserving chunk numbers
    papers = {}
    for chunk_idx, source in enumerate(result.sources, 1):
        title = source.get('title', 'Unknown')
        if title not in papers:
            papers[title] = {
                'title': title,
                'file_name': source.get('file_name', ''),
                'authors': source.get('authors', []),
                'year': source.get('year'),
                'chunks': [],
            }
        section = source.get('section_name') or 'N/A'
        chunk_type = source.get('chunk_type', 'unknown')
        papers[title]['chunks'].append({
            'idx': chunk_idx,
            'section': section,
            'type': chunk_type,
        })

    # Print summary
    total_chunks = len(result.sources)
    total_papers = len(papers)
    print(f"\n{total_papers} papers, {total_chunks} chunks")

    # Print each paper with its chunks
    for paper_idx, paper in enumerate(papers.values(), 1):
        title = paper['title'][:55]
        file_name = paper['file_name'] or 'N/A'
        authors = ", ".join(paper['authors'][:2]) if paper['authors'] else "Unknown"
        year = paper['year'] or "N/A"

        # Format chunks list: [1] abstract, [2] methods, ...
        chunk_strs = [f"[{c['idx']}] {c['section']}" for c in paper['chunks']]
        chunks_line = ", ".join(chunk_strs)

        print(f"\n[Paper {paper_idx}] {title}")
        print(f"           Authors: {authors} ({year})")
        print(f"           File: {file_name}")
        print(f"           Chunks: {chunks_line}")


if __name__ == "__main__":
    main()
