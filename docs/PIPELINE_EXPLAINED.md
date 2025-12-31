# Understanding the Research Paper RAG Pipeline

This document explains how the Research Paper Agent works, step by step. It covers both the **Indexing Pipeline** (how papers get processed and stored) and the **Query Pipeline** (how questions get answered).

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Part 1: Indexing Pipeline](#part-1-indexing-pipeline-how-papers-get-prepared)
   - [Step 1: PDF Extraction](#step-1-pdf-extraction)
   - [Step 2: Section Detection](#step-2-section-detection)
   - [Step 3: Chunking](#step-3-chunking)
   - [Step 4: Creating Embeddings](#step-4-creating-embeddings)
   - [Step 5: Storing in the Vector Database](#step-5-storing-in-the-vector-database)
   - [Step 6: Building the Keyword Index](#step-6-building-the-keyword-index-bm25)
3. [Part 2: Query Pipeline](#part-2-query-pipeline-how-questions-get-answered)
   - [Step 0: Resolving References](#step-0-resolving-references-from-conversation)
   - [Step 1: Query Cleanup](#step-1-query-cleanup-and-spelling-correction)
   - [Step 2: Entity Extraction](#step-2-entity-extraction)
   - [Step 3: Query Classification](#step-3-query-classification)
   - [Step 4: Query Expansion](#step-4-query-expansion)
   - [Step 5: Creating the Search Embedding](#step-5-creating-the-search-embedding)
   - [Step 6: Searching the Database](#step-6-searching-the-database)
   - [Step 7: Reranking Results](#step-7-reranking-results)
   - [Step 8: Generating the Answer](#step-8-generating-the-answer)
   - [Step 9: Verifying Citations](#step-9-verifying-citations)
4. [Supporting Systems](#supporting-systems)
   - [Caching](#caching)
   - [Conversation Memory](#conversation-memory)
5. [Configuration Reference](#configuration-reference)

---

## The Big Picture

Think of this system like a research librarian with a photographic memory:

1. **Indexing** = The librarian reads all papers, takes organized notes, and files them away
2. **Querying** = When you ask a question, the librarian finds relevant notes, reviews them, and gives you a well-sourced answer

The magic happens in *how* the papers are organized and *how* questions are understood.

---

# Part 1: Indexing Pipeline (How Papers Get Prepared)

The indexing pipeline transforms raw PDF files into searchable, organized knowledge. This happens once per paper and enables fast retrieval later.

```
PDF File → Extract Text → Detect Sections → Create Chunks → Generate Embeddings → Store in Database
```

---

## Step 1: PDF Extraction

**What happens:** The system reads the PDF and extracts all text, tables, figures, and formulas.

**Why this matters:** PDFs are complex documents. Text might be in columns, tables can span pages, and formulas are images, not text. We need to extract everything accurately before we can search it.

**How it works:**

The system uses **MinerU**, a state-of-the-art PDF extraction tool that combines multiple AI models:

| Component | What It Does |
|-----------|--------------|
| **DocLayout-YOLO** | Detects the layout - where are the columns, headers, figures, tables? |
| **StructEqTable** | Understands table structure - which cells belong together? |
| **UniMERNet** | Recognizes mathematical formulas and converts them to text |
| **OCR** | Handles scanned documents or text embedded in images |

The extraction produces:
- **Full text** - All readable content in order
- **Markdown** - Structured text with headers and formatting preserved
- **Tables** - Extracted as LaTeX or markdown for accurate representation
- **Figures** - Metadata about images (location, captions)
- **Captions** - All figure and table captions for separate indexing

**Fallback mechanism:** If MinerU fails (timeout, memory issues), the system falls back to a simpler extraction using pypdfium2 that gets basic text without the fancy layout detection.

---

## Step 2: Section Detection

**What happens:** The system identifies the logical sections of the paper (Abstract, Introduction, Methods, Results, Discussion, etc.)

**Why this matters:** Different types of questions need different sections:
- "How was the experiment done?" → Methods section
- "What did they find?" → Results section
- "Why does this matter?" → Introduction/Discussion
- "What are the limitations?" → Discussion section

Without section awareness, the system would treat all text equally and might return irrelevant content.

**How it works:**

The section detector uses pattern matching with scientific paper conventions:

1. **Combined sections first** (highest priority):
   - "Results and Discussion"
   - "Materials and Methods"

2. **Standard sections**:
   - Introduction, Methods, Results, Discussion, Conclusion

3. **Chemistry-specific sections** (domain awareness):
   - Synthesis, Characterization, Biological Evaluation
   - Molecular Docking, Structure-Activity Relationship

4. **Subsections**:
   - Numbered sections like "2.1 Cell Culture"
   - These inherit their parent section's category

The algorithm walks through the document, identifies headers based on formatting patterns (short lines, specific keywords), and marks where each section begins and ends.

**Special handling for abstracts:**
The abstract is extracted separately because it's almost always present and provides the best overview of the entire paper. The detector looks for the word "Abstract" and captures everything until the Introduction begins.

---

## Step 3: Chunking

**What happens:** The system breaks down the paper into smaller, searchable pieces called "chunks."

**Why this matters:**

Imagine searching a library where each book is one giant entry versus one where each book is indexed by chapter, paragraph, and topic. The second approach lets you find exactly what you need.

Papers are too long to search as single units, but too much splitting loses context. The solution: **multiple chunk types** optimized for different purposes.

**The Six Chunk Types:**

| Chunk Type | What It Contains | Size Limit | Best For |
|------------|------------------|------------|----------|
| **ABSTRACT** | The complete abstract as one piece | 400 tokens | "What is this paper about?" |
| **SECTION** | Full logical sections (Methods, Results, etc.) | 3000 tokens | Questions needing broad context |
| **FINE** | Small paragraphs with context headers | 500 tokens | Specific factual questions |
| **CAPTION** | Figure and table captions | No limit | "What does Figure 3 show?" |
| **TABLE** | Extracted table content | No limit | "What were the IC50 values?" |
| **FULL** | Entire paper represented as one vector | N/A | "Find similar papers" |

**How Fine Chunks Work (the most sophisticated type):**

Fine chunks are the workhorse for precise retrieval. Here's how they're created:

1. **Paragraph detection**: Identify natural paragraph breaks (double newlines, indentation)
2. **Sentence awareness**: Never split mid-sentence (uses pattern matching for sentence endings)
3. **Size targeting**: Aim for ~500 tokens per chunk
4. **Context headers**: Each fine chunk starts with its section name in brackets, like `[Methods] The cells were cultured...`
5. **Parent linking**: Each fine chunk remembers which section chunk it came from (used for context expansion later)
6. **Overlap**: Consecutive chunks share ~100 tokens at boundaries to prevent information loss at splits

**Example transformation:**

Original Methods section (2000 tokens) →
- Section chunk: Full 2000-token section
- Fine chunk 1: `[Methods] The compounds were synthesized using...` (500 tokens)
- Fine chunk 2: `[Methods] Purification was performed by HPLC...` (500 tokens)
- Fine chunk 3: `[Methods] Cell viability was assessed...` (500 tokens)
- Fine chunk 4: `[Methods] Statistical analysis was performed...` (500 tokens)

---

## Step 4: Creating Embeddings

**What happens:** Each chunk is converted into a numerical representation (a "vector") that captures its meaning.

**Why this matters:**

You can't search text by meaning directly. The word "synthesis" and "preparation" mean similar things, but string matching won't find that connection. Embeddings solve this by representing meaning as numbers:

- Similar meanings → vectors point in similar directions
- Different meanings → vectors point in different directions

This enables **semantic search**: finding content by what it *means*, not just what words it contains.

**How it works:**

The system uses **Voyage AI's voyage-3-large model**, which produces 1024-dimensional vectors:

1. **Batch processing**: Chunks are grouped (128 at a time) for efficiency
2. **API calls**: Each batch is sent to Voyage AI's embedding service
3. **Rate limiting**: The system respects API limits (2000 requests/minute) with automatic retry on overload
4. **Result caching**: Embeddings are saved to avoid re-computing them

**Special case - The FULL chunk:**

The FULL chunk represents the entire paper as a single vector. But papers are too long for direct embedding (would exceed token limits). The solution is **mean pooling**:

1. Take the embeddings of all section chunks
2. Average them together (element-wise mean)
3. Normalize to unit length

This preserves information from the whole paper without truncation. It's used for paper-level similarity: "Find papers like this one."

---

## Step 5: Storing in the Vector Database

**What happens:** All chunks and their embeddings are stored in Qdrant, a vector database optimized for similarity search.

**Why this matters:**

Regular databases are designed for exact lookups: "Find customer with ID 12345." Vector databases are designed for similarity lookups: "Find the 10 most similar documents to this one."

Qdrant uses specialized algorithms (HNSW) to search millions of vectors in milliseconds.

**How it works:**

**Single collection design:**
All chunk types go into one collection called "research_papers." They're differentiated by metadata, not separate tables. This simplifies queries and allows cross-type searches.

**What's stored for each chunk:**

| Field | Purpose |
|-------|---------|
| `id` | Unique identifier (deterministic UUID from chunk content) |
| `vector` | The 1024-dimensional embedding |
| `bm25` | Sparse vector for keyword matching (see next section) |
| `chunk_type` | Which of the 6 types this is |
| `paper_id` | Which paper this came from |
| `section_name` | Normalized section name (methods, results, etc.) |
| `text` | The actual chunk content |
| `parent_chunk_id` | For fine chunks, which section chunk contains this |
| `title`, `authors`, `year` | Paper metadata |

**Payload indices:**
The database creates indices on `chunk_type`, `paper_id`, and `section_name` for fast filtering. This means queries like "search only in Methods sections" are efficient.

---

## Step 6: Building the Keyword Index (BM25)

**What happens:** Alongside the semantic vectors, a keyword-based index is built using BM25 (a classic information retrieval algorithm).

**Why this matters:**

Semantic search is powerful but has blind spots:
- Exact terms: "IC50" means something specific; semantic search might return "potency" which isn't quite right
- Numbers and codes: "AB-12345" should match exactly
- Rare words: Uncommon scientific terms might not be well-represented in embeddings

BM25 fills these gaps with old-school keyword matching.

**How it works:**

**BM25 basics:**
- Counts how often query terms appear in each document
- Weights rare terms higher than common ones (IDF - Inverse Document Frequency)
- Normalizes for document length (longer documents don't automatically rank higher)

**Sparse vector representation:**
Unlike dense embeddings (1024 numbers for everything), BM25 produces sparse vectors:
- Only includes dimensions for words that actually appear
- Each dimension corresponds to a vocabulary term
- Values reflect term importance (TF-IDF style)

**IDF cache:**
The system maintains a cache of document frequencies. When new papers are indexed, this cache is updated incrementally rather than recalculated from scratch.

---

# Part 2: Query Pipeline (How Questions Get Answered)

When you ask a question, it goes through up to 14 processing steps before you get an answer. Each step improves the quality of the final response.

```
Question → Understand → Search → Rank → Generate → Verify → Answer
```

---

## Step 0: Resolving References from Conversation

**What happens:** If you're in a multi-turn conversation, the system resolves pronouns and references from previous messages.

**Why this matters:**

Consider this conversation:
- You: "Tell me about the synthesis method in the LL-37 paper"
- System: *provides answer*
- You: "What about its antimicrobial activity?"

Without reference resolution, "its" has no meaning. The system needs to understand you mean LL-37.

**How it works:**

The conversation memory component tracks:
- **Papers mentioned** in previous turns
- **Key entities** (compounds, methods, etc.) discussed
- **Recent questions and answers**

When a new query comes in:
1. Detect pronouns: "it", "this", "the paper", "the method", etc.
2. Look up what they refer to from conversation history
3. Substitute to create a clear, standalone query

**Example resolution:**
- Original: "What is its IC50?"
- Context: Previous answer discussed LL-37
- Resolved: "What is the IC50 of LL-37?"

---

## Step 1: Query Cleanup and Spelling Correction

**What happens:** The system fixes typos, expands acronyms, and prepares the query for search.

**Why this matters:**

Scientific terms are hard to spell. A search for "phosphorlyation" should still find results about phosphorylation. And "PCR" should connect to documents that say "polymerase chain reaction."

**How it works:**

**Spelling correction:**
A dictionary of 50+ common scientific misspellings catches errors:
- "chromotography" → "chromatography"
- "flourescence" → "fluorescence"
- "protien" → "protein"
- "sythesis" → "synthesis"

**Acronym expansion:**
Common acronyms are expanded (the original is kept for exact matching):
- "PCR" → "polymerase chain reaction"
- "HPLC" → "high performance liquid chromatography"
- "IC50" → "half maximal inhibitory concentration"

**Query decomposition (for complex queries):**
Multi-part questions get broken into sub-queries:
- "Compare the efficacy of compound A and compound B" becomes:
  - "efficacy of compound A"
  - "efficacy of compound B"
  - "comparison of compound A and compound B"

Each sub-query can be searched separately, then results combined.

---

## Step 2: Entity Extraction

**What happens:** The system identifies scientific entities in your question: chemicals, proteins, methods, organisms, and metrics.

**Why this matters:**

Knowing *what* you're asking about helps in two ways:
1. **Filtering**: Can narrow search to chunks containing those entities
2. **Boosting**: Can rank higher the results that mention the same entities

**How it works:**

The entity extractor uses pattern matching for different categories:

| Category | Pattern Examples | Example Matches |
|----------|------------------|-----------------|
| **Chemicals** | Drug codes (AB-12345), chemical suffixes (-ol, -ine, -ate) | "Doxorubicin", "LL-37", "KF-12345" |
| **Proteins** | Gene symbols, peptide names | "BRCA1", "TP53", "caspase-3" |
| **Methods** | Technique names | "HPLC", "SRS microscopy", "Western blot" |
| **Organisms** | Species, cell lines | "E. coli", "HeLa cells", "mouse" |
| **Metrics** | Measurement types | "IC50", "EC50", "Ki", "10 nM" |

**Example extraction:**
Query: "What is the IC50 of doxorubicin against HeLa cells?"
- Chemicals: ["doxorubicin"]
- Metrics: ["IC50"]
- Organisms: ["HeLa cells"]

---

## Step 3: Query Classification

**What happens:** The system determines what *type* of question you're asking, which determines the best search strategy.

**Why this matters:**

Different questions need different approaches:
- A factual question ("What is the melting point?") needs precise, short answers → search fine chunks
- A methods question ("How was it synthesized?") needs procedural detail → search the Methods section
- A comparative question ("How does X compare to Y?") needs breadth → search across many papers

**The Eight Query Types:**

| Type | Description | Example Question | Search Strategy |
|------|-------------|------------------|-----------------|
| **FACTUAL** | Specific facts, values, definitions | "What is the IC50 of compound X?" | Fine chunks, tables, captions; 50 candidates |
| **METHODS** | Technical procedures, protocols | "How were the peptides synthesized?" | Section + fine chunks filtered to Methods; 50 candidates |
| **FRAMING** | How to position research | "How do I argue this work is novel?" | Abstract + section chunks; 30 candidates |
| **SUMMARY** | Summarize papers or findings | "What are the main findings?" | Abstract + section + full; 20 candidates |
| **COMPARATIVE** | Compare approaches | "How does HPLC compare to LC-MS?" | Abstract + section; 100 candidates for breadth |
| **NOVELTY** | Prior work, research gaps | "Has this been done before?" | Abstract + section; 50 candidates |
| **LIMITATIONS** | Constraints, caveats | "What are the limitations?" | Section + abstract filtered to Discussion; 50 candidates |
| **GENERAL** | Uncategorized | "Tell me about this compound" | All chunk types; 50 candidates |

**How classification works:**

The system sends your query to Claude (Sonnet model) with a carefully designed prompt that includes:
- Definitions of each query type
- Examples of questions in each category
- Instructions to output: type, confidence (0-1), reasoning

If the LLM is unavailable or confidence is low (<0.6), a fallback heuristic kicks in:
- Contains "how" + method words → METHODS
- Contains "compare", "versus", "difference" → COMPARATIVE
- Contains "limitation", "weakness", "caveat" → LIMITATIONS
- Otherwise → GENERAL

---

## Step 4: Query Expansion

**What happens:** The system adds synonyms and related terms to your query.

**Why this matters:**

Scientific language has many equivalent terms:
- "LL-37" = "LL37" = "cathelicidin" = "hCAP18"
- "stapled peptide" = "hydrocarbon-stapled peptide" = "i,i+4 staple"

If you search for "LL-37" but the paper says "cathelicidin," you might miss relevant results. Query expansion catches these synonyms.

**How it works:**

A domain-specific synonym dictionary maps canonical terms to aliases:

```
"LL-37" → ["LL37", "cathelicidin", "CAMP", "hCAP18"]
"SRS microscopy" → ["stimulated Raman", "SRS imaging", "coherent Raman"]
"stapled peptide" → ["hydrocarbon-stapled", "i,i+4 staple", "i,i+7 staple"]
```

When a query term matches a dictionary entry, related terms are appended:
- Original: "LL-37 antimicrobial activity"
- Expanded: "LL-37 antimicrobial activity (related terms: cathelicidin, hCAP18)"

The expansion preserves the original terms while adding alternatives.

---

## Step 5: Creating the Search Embedding

**What happens:** The processed query is converted into an embedding vector for similarity search.

**Why this matters:**

To search a vector database, you need a query vector that can be compared against stored document vectors. Similar meanings → similar vectors → high similarity scores.

**Two approaches:**

**Standard embedding:**
Just convert the query text to a vector using Voyage AI:
```
"LL-37 antimicrobial activity" → [0.12, -0.34, 0.56, ...]
```

**HyDE (Hypothetical Document Embeddings):**
A more sophisticated approach that often improves retrieval:

1. Generate a hypothetical answer to the question using Claude (Haiku model)
2. Combine the question and hypothetical answer
3. Embed the combined text

**Why HyDE works:**

Your query is short: "What is the IC50 of LL-37?"

But actual paper content looks different: "The antimicrobial peptide LL-37 demonstrated potent activity against S. aureus with an IC50 of 2.3 μM in our MIC assay..."

The hypothetical answer bridges this gap by generating text that's closer to what papers actually say, making the embedding more effective at finding relevant content.

**Example HyDE process:**
1. Query: "What is the IC50 of LL-37?"
2. Hypothetical: "The IC50 of LL-37 against [pathogen] was determined to be approximately X μM using [assay type]. This indicates [interpretation of potency]..."
3. Combined: Query + Hypothetical → embedded together

---

## Step 6: Searching the Database

**What happens:** The system searches Qdrant for chunks most similar to your query.

**Why this matters:**

This is where the rubber meets the road. All that indexing and query processing leads to this: finding the most relevant content from all indexed papers.

**Two search modes:**

**Dense (semantic) search:**
- Compare query embedding against all chunk embeddings
- Rank by cosine similarity (how much vectors point in the same direction)
- Fast thanks to Qdrant's HNSW algorithm

**Hybrid search (dense + sparse):**
Combines semantic understanding with keyword matching:

1. **Dense search**: Run semantic similarity search → get top candidates
2. **Sparse search**: Run BM25 keyword search → get top candidates
3. **Fusion**: Combine rankings using RRF (Reciprocal Rank Fusion)

**How RRF works:**
```
RRF_score = 1/(60 + dense_rank) + 1/(60 + sparse_rank)
```

A document ranked #1 in both gets: 1/61 + 1/61 = 0.033
A document ranked #1 in dense, #10 in sparse gets: 1/61 + 1/70 = 0.030
A document ranked #10 in both gets: 1/70 + 1/70 = 0.029

Documents that rank well in *both* methods rise to the top.

**Filtering:**

Based on the query classification, filters are applied:
- **Chunk type filter**: Only search certain chunk types (e.g., "fine", "table")
- **Section filter**: Only search certain sections (e.g., "methods", "discussion")
- **Paper filter**: If user specifies papers, restrict to those

---

## Step 7: Reranking Results

**What happens:** The initial search results are re-scored by a specialized reranking model.

**Why this matters:**

Initial retrieval is fast but imprecise. It returns 50-100 candidates that *might* be relevant. Reranking uses a more sophisticated model to determine which ones are *actually* relevant to your specific question.

**How it works:**

**Cohere Rerank:**
The system uses Cohere's rerank-v3.5 model, which is specifically trained for relevance ranking:

1. Send query + all candidate texts to Cohere
2. Get back relevance scores (0-1) for each
3. Re-sort by these scores
4. Take top N (typically 10-15)

**Why reranking works better:**
Initial search uses embedding similarity (query vector vs document vector). This is fast but limited—both are converted to fixed-size vectors independently.

Reranking uses a **cross-encoder**: it processes query AND document together, allowing it to understand their relationship more deeply. It can catch nuances like:
- "LL-37 kills bacteria" is relevant to "antimicrobial activity of LL-37"
- "LL-37 is produced by humans" is NOT relevant to "antimicrobial activity of LL-37"

**Deduplication:**
Papers often have multiple relevant chunks. Without deduplication, results might be dominated by one paper:
- Chunk 1 from Paper A (score 0.95)
- Chunk 2 from Paper A (score 0.93)
- Chunk 3 from Paper A (score 0.91)
- Chunk 1 from Paper B (score 0.89)
- ...

The system enforces `max_per_paper` (typically 3-5), ensuring diversity:
- Chunk 1 from Paper A (score 0.95)
- Chunk 1 from Paper B (score 0.89)
- Chunk 1 from Paper C (score 0.85)
- Chunk 2 from Paper A (score 0.93)
- ...

**Parent chunk expansion:**
For fine chunks, the system can fetch their parent section chunk to provide more context to the LLM when generating the answer.

---

## Step 8: Generating the Answer

**What happens:** Claude reads the retrieved chunks and generates a well-sourced answer to your question.

**Why this matters:**

Raw chunks aren't the final answer—they're evidence. The LLM synthesizes information from multiple sources, handles nuance, and presents it clearly.

**How it works:**

**Query-type-specific prompts:**
Different question types get different instructions:

| Type | Key Instructions |
|------|------------------|
| **FACTUAL** | "Provide a direct, accurate answer. Include specific values when available." |
| **METHODS** | "Describe the protocol in detail. Include reagents, conditions, and equipment." |
| **FRAMING** | "Provide a Recommended Positioning Framework with angles to argue novelty." |
| **COMPARATIVE** | "Create a structured comparison highlighting similarities, differences, and trade-offs." |
| **NOVELTY** | "Assess prior work and identify research gaps." |
| **LIMITATIONS** | "Discuss constraints and caveats with balance and credibility." |

**Context assembly:**
The prompt includes:
1. System instructions (query-type-specific)
2. Retrieved sources, numbered [Source 1], [Source 2], etc.
3. Conversation history (if multi-turn)
4. The user's question

**Citation format:**
The LLM is instructed to cite sources using [Source N] format, enabling verification in the next step.

**Retry logic:**
If the API fails (rate limit, server error), the system retries with exponential backoff: wait 1s, then 2s, then 4s, up to 30s between retries.

---

## Step 9: Verifying Citations

**What happens:** The system checks that citations in the answer actually match their sources.

**Why this matters:**

LLMs can "hallucinate"—they might cite a source that doesn't actually support what they said. Citation verification catches these errors.

**How it works:**

**Step 1: Extract citations**
Find all `[Source N]` references and associate them with the claims they support.

**Step 2: Verify each claim**
For each citation:
1. Get the sentence/claim that cites [Source N]
2. Get the text of Source N
3. Check if the claim is supported

**Verification methods:**

**Basic (keyword overlap):**
- Extract key terms from the claim (remove stop words)
- Check what percentage appear in the source
- High overlap (>60%) = likely valid
- Low overlap (<30%) = suspicious

**LLM-based (optional, more accurate):**
- Send claim + source to Claude
- Ask: "Does this source support this claim?"
- Get verdict: SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED

**Handling issues:**
- If >70% of citations are valid → mark response as "verified"
- If invalid citations found → add warnings to the response
- Severe issues can trigger answer regeneration

---

# Supporting Systems

## Caching

**What it does:** Stores computed results to avoid redundant API calls.

**Why it matters:** Embedding a query costs money and time. If someone asks the same question twice, why compute everything again?

**Three cache types:**

| Cache | What's Stored | TTL | Size |
|-------|---------------|-----|------|
| **Embedding** | Query → vector | 24 hours | 500 entries |
| **Search** | Query + filters → results | 10 minutes | 200 entries |
| **HyDE** | Query → hypothetical answer | 30 minutes | 100 entries |

**Why short search TTL?**
When new papers are indexed, search results might change. The 10-minute TTL ensures fresh results without excessive recomputation.

**Cache invalidation:**
When papers are indexed, the search cache is cleared. The embedding cache is kept (embeddings don't depend on indexed content).

---

## Conversation Memory

**What it does:** Tracks conversation history for multi-turn interactions.

**Why it matters:** Follow-up questions like "What about its toxicity?" need context from previous turns.

**What's tracked:**
- **Message history**: Last 10 turns (user + assistant messages)
- **Paper context**: Which papers have been discussed
- **Key entities**: Important terms mentioned
- **Timestamps**: When each message occurred

**Context limits:**
- Maximum 10 turns (older messages are trimmed)
- Maximum 2000 tokens for history included in prompts
- First 2 messages are always kept (provides grounding)

**Reference resolution patterns:**
- "it", "this" → last mentioned entity
- "the paper", "this paper" → last mentioned paper title
- "the method" → last mentioned technique

---

# Configuration Reference

## Indexing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `abstract_max_tokens` | 400 | Maximum tokens for abstract chunks |
| `section_max_tokens` | 3000 | Maximum tokens for section chunks |
| `fine_chunk_tokens` | 500 | Target size for fine chunks |
| `fine_chunk_overlap` | 100 | Token overlap between consecutive fine chunks |
| `embedding_model` | voyage-3-large | Model for generating embeddings |
| `embedding_dimension` | 1024 | Vector dimensions |

## Query Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_classification` | true | Use LLM to classify queries |
| `enable_expansion` | true | Add synonyms to queries |
| `enable_hyde` | false | Use hypothetical document embeddings |
| `enable_hybrid_search` | false | Combine dense + sparse search |
| `enable_query_rewriting` | true | Fix spelling, expand acronyms |
| `enable_entity_extraction` | true | Extract scientific entities |
| `enable_citation_verification` | false | Verify LLM citations |
| `enable_conversation_memory` | true | Track multi-turn context |
| `enable_caching` | true | Cache embeddings and results |

## Models Used

| Purpose | Model | Why This Model |
|---------|-------|----------------|
| Answer generation | Claude Opus 4.5 | Most capable, best for synthesis |
| Query classification | Claude Sonnet 4 | Good balance of speed/quality |
| HyDE generation | Claude Haiku | Fast, cheap, sufficient quality |
| Query rewriting | Claude Haiku | Fast for simple transformations |
| Embeddings | Voyage voyage-3-large | Best scientific text embeddings |
| Reranking | Cohere rerank-v3.5 | Best available reranker |

---

## Summary

The Research Paper RAG pipeline is designed for **accuracy** and **relevance** in scientific contexts:

1. **Smart chunking** creates multiple representations optimized for different query types
2. **Hybrid search** combines semantic understanding with exact keyword matching
3. **Query classification** selects the best retrieval strategy for each question type
4. **Reranking** ensures the most relevant content rises to the top
5. **Citation verification** catches LLM hallucinations
6. **Conversation memory** enables natural multi-turn interactions

Each component can be enabled/disabled independently, allowing the system to be tuned for different use cases—from fast, simple lookups to thorough, verified research assistance.
