# Research Paper Agent - Complete Pipeline Documentation

This document provides a detailed walkthrough of the entire RAG (Retrieval-Augmented Generation) pipeline, from PDF ingestion to answer generation.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Indexing Pipeline](#indexing-pipeline)
   - [PDF Processing (MinerU)](#1-pdf-processing-mineru)
   - [Multi-Type Chunking](#2-multi-type-chunking)
   - [Embedding Generation](#3-embedding-generation)
   - [Vector Storage (Qdrant)](#4-vector-storage-qdrant)
   - [BM25 Index Update](#5-bm25-index-update)
3. [Query Pipeline](#query-pipeline)
   - [Cache Invalidation Check](#step--1-cache-invalidation-check)
   - [Conversation Reference Resolution](#step-0-conversation-reference-resolution)
   - [Query Rewriting & Spelling Correction](#step-1-query-rewriting--spelling-correction)
   - [Entity Extraction](#step-2-entity-extraction)
   - [Query Classification](#step-3-query-classification)
   - [Query Expansion](#step-4-query-expansion)
   - [Retrieval Strategy Selection](#step-5-retrieval-strategy-selection)
   - [Cache Lookup](#step-6-cache-lookup)
   - [Query Embedding (with optional HyDE)](#step-7-query-embedding-with-optional-hyde)
   - [Hybrid Search (Dense + Sparse)](#step-8-hybrid-search-dense--sparse)
   - [Entity Boosting](#step-9-entity-boosting)
   - [Reranking](#step-10-reranking)
   - [Parent Chunk Expansion](#step-11-parent-chunk-expansion)
   - [Answer Generation](#step-12-answer-generation)
   - [Citation Verification](#step-13-citation-verification)
   - [Conversation Memory Update](#step-14-conversation-memory-update)
4. [Component Reference](#component-reference)
5. [Configuration](#configuration)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INDEXING PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   PDF Files                                                              │
│       │                                                                  │
│       ▼                                                                  │
│   ┌──────────────────┐                                                   │
│   │ MinerU Extractor │  ← DocLayout-YOLO, StructEqTable, OCR            │
│   └────────┬─────────┘                                                   │
│            │                                                             │
│            ▼                                                             │
│   ┌──────────────────┐                                                   │
│   │  PaperChunker    │  → 6 chunk types: abstract, section, fine,       │
│   │                  │    caption, table, full                          │
│   └────────┬─────────┘                                                   │
│            │                                                             │
│            ▼                                                             │
│   ┌──────────────────┐                                                   │
│   │ VoyageEmbedder   │  ← voyage-3-large (1024 dims)                    │
│   └────────┬─────────┘                                                   │
│            │                                                             │
│            ▼                                                             │
│   ┌──────────────────┐    ┌─────────────────┐                           │
│   │  QdrantStore     │ +  │ BM25Vectorizer  │  ← Hybrid vectors         │
│   └──────────────────┘    └─────────────────┘                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          QUERY PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   User Query                                                             │
│       │                                                                  │
│       ▼                                                                  │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ Step 0-1: Conversation Resolution → Query Rewriting/Correction  │   │
│   └────────────────────────────┬────────────────────────────────────┘   │
│                                │                                         │
│                                ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ Step 2-4: Entity Extraction → Classification → Expansion        │   │
│   └────────────────────────────┬────────────────────────────────────┘   │
│                                │                                         │
│                                ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ Step 5-8: Strategy Selection → Cache Check → Embed → Search     │   │
│   └────────────────────────────┬────────────────────────────────────┘   │
│                                │                                         │
│                                ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ Step 9-11: Entity Boost → Rerank (Cohere) → Parent Expansion    │   │
│   └────────────────────────────┬────────────────────────────────────┘   │
│                                │                                         │
│                                ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ Step 12-14: Generate Answer (Claude) → Verify → Update Memory   │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Indexing Pipeline

Entry point: `backend/index_papers.py`

### 1. PDF Processing (MinerU)

**File:** `backend/preprocessing/pdf_processor.py`

MinerU (PDF-Extract-Kit) provides high-quality PDF extraction using:
- **DocLayout-YOLO**: Layout detection for figures, tables, text blocks
- **StructEqTable**: Table structure recognition
- **UniMERNet**: Mathematical formula recognition
- **OCR**: For scanned documents

```python
class MinerUExtractor:
    def extract(self, pdf_path: Path) -> MinerUContent:
        # 1. Read PDF bytes
        pdf_bytes = read_fn(pdf_path)

        # 2. Run document analysis (GPU-accelerated)
        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled = doc_analyze(
            [pdf_bytes], [self.lang], parse_method="auto",
            formula_enable=True, table_enable=True
        )

        # 3. Convert to intermediate JSON format
        middle_json = result_to_middle_json(...)

        # 4. Generate markdown and content list
        markdown = union_make(pdf_info, MakeMode.MM_MD, tmp_dir)
        content_list = union_make(pdf_info, MakeMode.CONTENT_LIST, tmp_dir)

        # 5. Extract components
        return MinerUContent(
            full_text=self._extract_text_from_content(content_list),
            markdown=markdown,
            tables=self._extract_tables_from_content(content_list),
            figures=self._extract_figures_from_content(content_list),
            captions=self._extract_captions_from_markdown(markdown),
            metadata=self._extract_metadata(pdf_path, middle_json)
        )
```

**Output:** `MinerUContent` containing:
- `full_text`: Extracted plain text
- `markdown`: Structured markdown representation
- `tables`: List of extracted tables (LaTeX/markdown format)
- `figures`: List of figure metadata
- `captions`: Figure/table captions
- `metadata`: Title, page count, filename

---

### 2. Multi-Type Chunking

**File:** `backend/preprocessing/chunker.py`

The `PaperChunker` creates 6 distinct chunk types optimized for different query types:

| Chunk Type | Description | Token Limit | Use Case |
|------------|-------------|-------------|----------|
| `ABSTRACT` | Full abstract as single chunk | 400 | Overview queries |
| `SECTION` | Logical sections (Methods, Results, etc.) | 3000 | Context-rich queries |
| `FINE` | Semantic paragraph-based chunks | 500 | Precise factual queries |
| `CAPTION` | Figure/table captions | - | Specific figure references |
| `TABLE` | Extracted table content | - | Data queries |
| `FULL` | Mean-pooled embedding (created at index time) | - | Paper-level similarity |

**Chunking Process:**

```python
def chunk_paper(self, text: str, metadata: PaperMetadata,
                captions: List[str], tables: List[str]) -> List[Chunk]:
    chunks = []

    # 1. ABSTRACT chunk - extract using section detector
    abstract = self.section_detector.extract_abstract(text)
    if abstract:
        chunks.append(Chunk(chunk_type=ChunkType.ABSTRACT, text=abstract, ...))

    # 2. SECTION chunks - detect logical sections
    sections = self.section_detector.detect_sections(text)
    for section in sections:
        # Skip references, acknowledgments
        if section.normalized_name in ["references", "acknowledgments"]:
            continue
        section_chunk = Chunk(chunk_type=ChunkType.SECTION,
                              text=section.text,
                              section_name=section.normalized_name, ...)
        chunks.append(section_chunk)

    # 3. FINE chunks - semantic paragraph-aware splitting
    for section_chunk in section_chunks:
        fine_chunks = self._create_contextual_fine_chunks(
            text=section_chunk.text,
            section_name=section_chunk.section_name,
            parent_chunk=section_chunk,  # For later parent expansion
            ...
        )
        chunks.extend(fine_chunks)

    # 4. CAPTION chunks
    for caption in captions:
        chunks.append(Chunk(chunk_type=ChunkType.CAPTION,
                            text=f"[Figure/Table Caption] {caption}", ...))

    # 5. TABLE chunks
    for table in tables:
        chunks.append(Chunk(chunk_type=ChunkType.TABLE,
                            text=f"[Table Content] {table}", ...))

    return chunks
```

**Fine Chunk Features:**
- Paragraph-aware: Preserves paragraph boundaries as semantic units
- Sentence boundaries: Never splits mid-sentence
- Contextual headers: Includes section context (e.g., `[Methods] ...`)
- Parent linkage: Links to parent section chunk for expansion

---

### 3. Embedding Generation

**File:** `backend/retrieval/embedder.py`

Uses Voyage AI's `voyage-3-large` model (1024 dimensions):

```python
class VoyageEmbedder:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with rate limiting and retry."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = self._embed_with_retry(batch, input_type="document")
            all_embeddings.extend(embeddings)
        return all_embeddings

    def compute_mean_pooled_embedding(self, texts: List[str]) -> List[float]:
        """Create full-paper embedding via mean pooling.

        Used for FULL chunk type - preserves information from all sections
        (unlike truncation which loses end content).
        """
        embeddings = self.embed_documents(texts)
        embeddings_array = np.array(embeddings)
        mean_embedding = np.mean(embeddings_array, axis=0)

        # L2 normalize
        norm = np.linalg.norm(mean_embedding)
        return (mean_embedding / norm).tolist()
```

**Rate Limiting:**
- 2000 RPM (requests per minute)
- 3M TPM (tokens per minute)
- Exponential backoff on rate limit errors

---

### 4. Vector Storage (Qdrant)

**File:** `backend/retrieval/qdrant_store.py`

Single collection with all chunk types, differentiated by metadata:

```python
class QdrantStore:
    def ensure_collection(self) -> bool:
        """Create collection with hybrid vector support."""
        # Dense vectors (Voyage embeddings)
        vectors_config = VectorParams(
            size=self.embedding_dimension,  # 1024
            distance=Distance.COSINE
        )

        # Sparse vectors (BM25 for hybrid search)
        sparse_config = {
            "bm25": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        }

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config,
        )

        # Create payload indices for efficient filtering
        self._ensure_payload_indices()  # chunk_type, paper_id, section_name

    def upsert_chunks(self, chunk_ids, embeddings, payloads) -> int:
        """Upsert chunks with dense + sparse vectors."""
        for chunk_id, embedding, payload in zip(chunk_ids, embeddings, payloads):
            # Generate deterministic UUID from chunk_id
            point_id = self._chunk_id_to_point_id(chunk_id)

            # Build hybrid vector (dense + sparse)
            sparse_vec = self.bm25_vectorizer.vectorize(payload['text'], is_query=False)
            vector_data = {
                "": embedding,  # Default dense vector
                "bm25": QdrantSparseVector(
                    indices=sparse_vec.indices,
                    values=sparse_vec.values,
                )
            }

            points.append(PointStruct(
                id=point_id,
                vector=vector_data,
                payload={**payload, '_chunk_id': chunk_id}
            ))

        self.client.upsert(collection_name=self.collection_name, points=points)
```

**Payload Schema:**
```python
{
    "_chunk_id": "paper123_section_0",
    "chunk_type": "section",
    "paper_id": "paper123",
    "section_name": "methods",
    "text": "...",
    "title": "Paper Title",
    "authors": ["Author 1", "Author 2"],
    "year": 2024,
    "parent_chunk_id": null,  # For fine chunks, links to parent section
    "token_count": 1500,
    "file_name": "paper.pdf"
}
```

---

### 5. BM25 Index Update

**File:** `backend/retrieval/bm25.py`

After indexing, the BM25 IDF cache is updated:

```python
class BM25Vectorizer:
    def update_idf_incremental(self, new_documents: List[str]) -> None:
        """Incrementally update IDF with new documents."""
        for doc in new_documents:
            tokens = set(self.tokenize(doc))
            for token in tokens:
                self._doc_freq[token] += 1

        self._doc_count += len(new_documents)
        self._recalculate_idf()

    def save_idf_cache(self, path: Path = DEFAULT_IDF_CACHE_PATH) -> None:
        """Persist IDF cache for reuse across sessions."""
        data = {
            "idf_cache": self._idf_cache,
            "doc_count": self._doc_count,
            "doc_freq": self._doc_freq,
        }
        with open(path, 'w') as f:
            json.dump(data, f)
```

---

## Query Pipeline

Entry point: `backend/retrieval/query_engine.py`

### Step -1: Cache Invalidation Check

Checks if new papers have been indexed since last query:

```python
if self.cache:
    recently_indexed = self.store.get_recently_upserted_papers()
    if recently_indexed:
        self.cache.invalidate_if_needed(recently_indexed)
        logger.info(f"Invalidated cache due to {len(recently_indexed)} newly indexed papers")
```

---

### Step 0: Conversation Reference Resolution

**File:** `backend/retrieval/conversation_memory.py`

Resolves pronouns and references from conversation history:

```python
class ConversationMemory:
    def resolve_references(self, query: str) -> str:
        """Resolve pronouns like 'it', 'the paper', 'this method'."""
        # Example: "What is its IC50?" → "What is the IC50 of LL-37?"

        # Pronoun patterns to resolve
        PRONOUN_PATTERNS = {
            r'\bit\b': 'the compound',
            r'\bthe paper\b': self._get_last_paper_title(),
            r'\bthis method\b': self._get_last_method(),
        }

        resolved = query
        for pattern, replacement in PRONOUN_PATTERNS.items():
            if replacement:
                resolved = re.sub(pattern, replacement, resolved, flags=re.IGNORECASE)

        return resolved
```

---

### Step 1: Query Rewriting & Spelling Correction

**File:** `backend/retrieval/query_rewriter.py`

Corrects scientific terminology misspellings:

```python
SPELLING_CORRECTIONS = {
    "chromotography": "chromatography",
    "flourescence": "fluorescence",
    "protien": "protein",
    "sythesis": "synthesis",
    # ... 50+ scientific term corrections
}

class QueryRewriter:
    def rewrite(self, query: str) -> RewrittenQuery:
        rewritten = query
        corrections = []

        # Apply spelling corrections
        for wrong, correct in SPELLING_CORRECTIONS.items():
            if wrong in rewritten.lower():
                rewritten = re.sub(wrong, correct, rewritten, flags=re.IGNORECASE)
                corrections.append((wrong, correct))

        return RewrittenQuery(
            original=query,
            rewritten=rewritten,
            corrections=corrections,
            ...
        )
```

---

### Step 2: Entity Extraction

**File:** `backend/retrieval/entity_extractor.py`

Extracts scientific entities for later entity boosting:

```python
class EntityExtractor:
    def extract(self, text: str) -> ExtractedEntities:
        """Extract entities by category."""
        return ExtractedEntities(
            chemicals=self._extract_type(text, 'chemicals'),   # Drug names, compound codes
            proteins=self._extract_type(text, 'proteins'),     # Gene symbols, enzymes
            methods=self._extract_type(text, 'methods'),       # HPLC, PCR, SRS
            organisms=self._extract_type(text, 'organisms'),   # E. coli, HeLa
            metrics=self._extract_type(text, 'metrics'),       # IC50, nM, μM
        )

    # Example patterns:
    chemical_patterns = [
        r'\b[A-Z]{2,4}-\d{3,6}\b',  # Drug codes: AB-12345
        r'\b[A-Z][a-z]+(?:in|ol|ine|ate|ide|ase|one)\b',  # Drug suffixes
    ]

    protein_patterns = [
        r'\b[A-Z]{2,5}\d{0,2}\b',  # Gene symbols: BRCA1, TP53
        r'\bLL-?37\b',  # Specific peptides
    ]
```

---

### Step 3: Query Classification

**File:** `backend/retrieval/query_classifier.py`

Classifies queries into 8 types, each with optimized retrieval strategy:

| Query Type | Description | Chunk Types | Top-K |
|------------|-------------|-------------|-------|
| `FACTUAL` | Specific facts, values | fine, table, caption | 50 |
| `FRAMING` | How to position research | abstract, section | 30 |
| `METHODS` | Technical protocols | section, fine (Methods filter) | 50 |
| `SUMMARY` | Summarize findings | abstract, section, full | 20 |
| `COMPARATIVE` | Compare approaches | abstract, section | 100 |
| `NOVELTY` | Prior work, gaps | abstract, section | 50 |
| `LIMITATIONS` | Constraints, caveats | section, abstract | 50 |
| `GENERAL` | Uncategorized | all types | 50 |

```python
class QueryClassifier:
    def classify(self, query: str) -> QueryClassification:
        """Use Claude to classify query type."""
        prompt = CLASSIFICATION_PROMPT.format(query=query)

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        return self._parse_response(response.content[0].text, query)
```

**Hybrid Fallback:** When LLM classification is disabled, uses keyword heuristics:

```python
def _detect_targeted_query_type(self, query: str) -> QueryType:
    """Lightweight keyword-based detection."""
    query_lower = query.lower()

    # METHODS signals
    if any(s in query_lower for s in ["protocol", "synthesized", "buffer", "concentration"]):
        return QueryType.METHODS

    # NOVELTY signals
    if any(s in query_lower for s in ["novel", "been done before", "existing", "gap"]):
        return QueryType.NOVELTY

    return QueryType.GENERAL
```

---

### Step 4: Query Expansion

**File:** `backend/retrieval/query_expander.py`

Expands queries with domain-specific synonyms:

```python
DOMAIN_SYNONYMS = {
    "LL-37": ["LL37", "cathelicidin", "CAMP", "hCAP18"],
    "stapled peptide": ["hydrocarbon-stapled", "i,i+4 staple", "i,i+7 staple"],
    "SRS microscopy": ["stimulated Raman", "SRS imaging", "coherent Raman"],
    # ... extensive synonym dictionary
}

class QueryExpander:
    def expand_query(self, query: str) -> Tuple[str, List[str]]:
        """Add synonyms for domain terms found in query."""
        added_terms = []

        for canonical, aliases in DOMAIN_SYNONYMS.items():
            if canonical.lower() in query.lower():
                for alias in aliases[:3]:
                    if alias.lower() not in query.lower():
                        added_terms.append(alias)

        if added_terms:
            return f"{query} (related terms: {', '.join(added_terms)})", added_terms
        return query, []
```

---

### Step 5: Retrieval Strategy Selection

Based on query classification, select appropriate parameters:

```python
RETRIEVAL_STRATEGIES = {
    QueryType.FACTUAL: {
        "chunk_types": ["fine", "table", "caption"],
        "top_k": 50,
        "rerank_top_n": 15,
        "max_per_paper": 3,
    },
    QueryType.METHODS: {
        "chunk_types": ["section", "fine"],
        "section_filter": ["methods", "experimental", "synthesis"],
        "top_k": 50,
        "rerank_top_n": 15,
        "max_per_paper": 5,
    },
    # ... etc
}

strategy = RETRIEVAL_STRATEGIES[query_type]
chunk_types = strategy["chunk_types"]
top_k = strategy["top_k"]
section_filter = strategy.get("section_filter")
```

---

### Step 6: Cache Lookup

**File:** `backend/retrieval/cache.py`

Check if results are already cached:

```python
class RAGCache:
    def get_search_results(self, query: str, chunk_types: List[str],
                           section_filter: Optional[List[str]]) -> Optional[List[Dict]]:
        """Get cached search results."""
        key = self.search_cache._make_key(query, chunk_types, section_filter)
        return self.search_cache.get(key)

    def get_embedding(self, query: str) -> Optional[List[float]]:
        """Get cached query embedding."""
        key = self.embedding_cache._make_key(query)
        return self.embedding_cache.get(key)
```

**Cache TTLs:**
- Embeddings: 24 hours
- Search results: 10 minutes (short for freshness)
- HyDE results: 30 minutes

---

### Step 7: Query Embedding (with optional HyDE)

**File:** `backend/retrieval/hyde.py`

**Standard Embedding:**
```python
query_embedding = self.embedder.embed_query(expanded_query)
```

**HyDE (Hypothetical Document Embeddings):**

Instead of embedding the query directly, generate a hypothetical answer first:

```python
class HyDE:
    def generate_hypothetical(self, query: str, query_type: str) -> Tuple[str, str]:
        """Generate hypothetical answer that would appear in a research paper."""
        prompt = HYDE_PROMPT_BY_TYPE.get(query_type, HYDE_PROMPT).format(query=query)

        response = self.anthropic.messages.create(
            model="claude-3-haiku-20240307",  # Fast model
            max_tokens=500,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )

        hypothetical = response.content[0].text.strip()
        combined = f"Query: {query}\n\nRelevant excerpt: {hypothetical}"

        return hypothetical, combined

class HyDEEmbedder:
    def embed_with_hyde(self, query: str, query_type: str) -> List[float]:
        """Embed using HyDE."""
        hypothetical, combined = self.hyde.generate_hypothetical(query, query_type)
        return self.embedder.embed_query(combined)
```

**Why HyDE?** The hypothetical answer is closer to actual document content than a short query, often producing better retrieval.

---

### Step 8: Hybrid Search (Dense + Sparse)

**Dense-Only Search:**
```python
results = self.store.search(
    query_embedding=query_embedding,
    limit=top_k,
    chunk_types=chunk_types,
    section_names=section_filter,
)
```

**Hybrid Search (Dense + BM25):**

Uses Qdrant's native Prefetch + RRF (Reciprocal Rank Fusion):

```python
def hybrid_search(self, query: str, query_embedding: List[float],
                  limit: int, chunk_types: List[str], ...) -> List[Dict]:
    # Generate sparse BM25 vector
    sparse_vec = self.bm25_vectorizer.vectorize(query, is_query=True)

    # Hybrid search with prefetch and fusion
    results = self.client.query_points(
        collection_name=self.collection_name,
        prefetch=[
            # Dense vector prefetch
            Prefetch(
                query=query_embedding,
                using="",  # Default dense vector
                limit=limit * 2,
                filter=query_filter,
            ),
            # Sparse vector prefetch
            Prefetch(
                query=QdrantSparseVector(
                    indices=sparse_vec.indices,
                    values=sparse_vec.values,
                ),
                using="bm25",
                limit=limit * 2,
                filter=query_filter,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),  # Reciprocal Rank Fusion
        limit=limit,
    )

    return [{'score': r.score, **r.payload} for r in results.points]
```

---

### Step 9: Entity Boosting

Boost results that contain query entities:

```python
if self.entity_extractor and entities_extracted and results:
    for result in results:
        text = result.get('text', '')
        entity_score, matched = self.entity_extractor.score_chunk_relevance(
            rewritten_query, text
        )
        result['entity_boost'] = entity_score
        # Slight boost for entity matches (10% max)
        result['score'] = result.get('score', 0) * (1 + 0.1 * entity_score)
```

---

### Step 10: Reranking

**File:** `backend/retrieval/reranker.py`

Uses Cohere's `rerank-v3.5` model:

```python
class CohereReranker:
    def rerank_with_metadata(self, query: str, documents: List[Dict],
                             top_n: int, max_per_paper: int) -> List[Dict]:
        """Rerank and deduplicate by paper_id."""
        # First rerank all documents
        response = self.client.rerank(
            query=query,
            documents=[doc['text'] for doc in documents],
            model="rerank-v3.5",
            top_n=len(documents)
        )

        # Deduplicate by paper_id (max N chunks per paper)
        paper_counts = {}
        deduplicated = []

        for result in response.results:
            doc = documents[result.index].copy()
            doc['rerank_score'] = result.relevance_score
            paper_id = doc.get('paper_id', 'unknown')

            if paper_counts.get(paper_id, 0) < max_per_paper:
                deduplicated.append(doc)
                paper_counts[paper_id] = paper_counts.get(paper_id, 0) + 1

            if len(deduplicated) >= top_n:
                break

        return deduplicated
```

---

### Step 11: Parent Chunk Expansion

For fine chunks, fetch parent section for additional context:

```python
def _expand_fine_chunks(self, chunks: List[Dict]) -> List[Dict]:
    """Expand fine chunks with parent section context."""
    # Collect parent IDs needed
    parent_ids = [c['parent_chunk_id'] for c in chunks
                  if c.get('chunk_type') == 'fine' and c.get('parent_chunk_id')]

    # Batch retrieve parents
    parent_chunks = self.store.get_chunks_by_ids(parent_ids)
    parent_by_id = {p['_chunk_id']: p for p in parent_chunks}

    # Attach parent context
    for chunk in chunks:
        if chunk.get('chunk_type') == 'fine':
            parent = parent_by_id.get(chunk.get('parent_chunk_id'))
            if parent:
                chunk['parent_context'] = parent.get('text', '')[:500]

    return chunks
```

---

### Step 12: Answer Generation

Uses Claude with query-type-specific prompts:

```python
ANSWER_PROMPTS = {
    QueryType.FACTUAL: """You are a research assistant answering factual questions...
Based on the retrieved sources, provide a direct, accurate answer.
Include specific values, definitions, or mechanisms when available.
Cite sources using [Source N] format.

Question: {query}

Retrieved Sources:
{sources}

Provide a clear, factual answer:""",

    QueryType.METHODS: """You are a research methods expert...
Include specific details like reagents, conditions, and equipment.
...""",

    # ... type-specific prompts for each QueryType
}

def _generate_answer(self, query: str, query_type: QueryType,
                     sources: List[Dict]) -> str:
    sources_text = self._format_sources(sources)
    prompt_template = ANSWER_PROMPTS.get(query_type, ANSWER_PROMPTS[QueryType.GENERAL])
    prompt = prompt_template.format(query=query, sources=sources_text)

    response = retry_with_exponential_backoff(lambda:
        self.anthropic.messages.create(
            model=self.claude_model,
            max_tokens=2048,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
    )

    return response.content[0].text
```

---

### Step 13: Citation Verification

**File:** `backend/retrieval/citation_verifier.py`

Verifies that LLM citations match source content:

```python
class CitationVerifier:
    def verify_response(self, answer: str, sources: List[Dict]) -> VerificationResult:
        """Verify all citations in the answer."""
        # Extract citations: [Source 1], [Source 2], etc.
        citations = self.extract_citations(answer)

        checks = []
        for source_id, claims in citations.items():
            source = sources[source_id - 1]  # 1-indexed

            for claim in claims:
                is_valid, confidence, explanation = self._verify_claim(
                    claim, source['text']
                )
                checks.append(CitationCheck(
                    citation_id=source_id,
                    claim=claim,
                    source_text=source['text'][:500],
                    is_valid=is_valid,
                    confidence=confidence,
                    explanation=explanation,
                ))

        valid_count = sum(1 for c in checks if c.is_valid)
        return VerificationResult(
            total_citations=len(checks),
            valid_citations=valid_count,
            overall_confidence=valid_count / len(checks) if checks else 1.0,
            checks=checks,
            is_verified=valid_count / len(checks) >= 0.7 if checks else True,
        )
```

---

### Step 14: Conversation Memory Update

Store query and response for future reference resolution:

```python
if self.conversation_memory:
    self.conversation_memory.add_user_message(query)
    self.conversation_memory.add_assistant_message(answer, sources=expanded_sources)
```

---

## Component Reference

| Component | File | Purpose |
|-----------|------|---------|
| `EnhancedPDFProcessor` | `preprocessing/pdf_processor.py` | PDF extraction with MinerU |
| `PaperChunker` | `preprocessing/chunker.py` | Multi-type semantic chunking |
| `VoyageEmbedder` | `retrieval/embedder.py` | Dense embeddings (voyage-3-large) |
| `BM25Vectorizer` | `retrieval/bm25.py` | Sparse vectors for hybrid search |
| `QdrantStore` | `retrieval/qdrant_store.py` | Vector storage and search |
| `QueryClassifier` | `retrieval/query_classifier.py` | LLM-based query classification |
| `QueryExpander` | `retrieval/query_expander.py` | Domain synonym expansion |
| `QueryRewriter` | `retrieval/query_rewriter.py` | Spelling correction |
| `EntityExtractor` | `retrieval/entity_extractor.py` | Scientific entity extraction |
| `HyDEEmbedder` | `retrieval/hyde.py` | Hypothetical document embeddings |
| `CohereReranker` | `retrieval/reranker.py` | Neural reranking |
| `CitationVerifier` | `retrieval/citation_verifier.py` | Citation validation |
| `ConversationMemory` | `retrieval/conversation_memory.py` | Multi-turn context |
| `RAGCache` | `retrieval/cache.py` | LRU caching layer |
| `QueryEngine` | `retrieval/query_engine.py` | Full pipeline orchestration |

---

## Configuration

**Environment Variables** (via `.env`):

```bash
# API Keys
VOYAGE_API_KEY=...
COHERE_API_KEY=...
ANTHROPIC_API_KEY=...

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=research_papers

# Embedding
EMBEDDING_MODEL=voyage-3-large
EMBEDDING_DIMENSION=1024

# Chunking
ABSTRACT_MAX_TOKENS=400
SECTION_MAX_TOKENS=3000
FINE_CHUNK_TOKENS=500
FINE_CHUNK_OVERLAP=100

# PDF Source
PDF_SOURCE_DIR=data/sample_papers
```

**QueryEngine Options:**

```python
QueryEngine(
    embedder=embedder,
    reranker=reranker,
    store=store,
    anthropic_client=anthropic_client,
    claude_model="claude-opus-4-5-20251101",
    enable_classification=True,      # LLM query classification
    enable_expansion=True,           # Synonym expansion
    enable_caching=True,             # LRU caching
    enable_hyde=False,               # Hypothetical document embeddings
    enable_query_rewriting=True,     # Spelling correction
    enable_entity_extraction=True,   # Entity boosting
    enable_citation_verification=False,  # Citation checks
    enable_conversation_memory=True, # Multi-turn context
    enable_hybrid_search=True,       # Dense + BM25
)
```
