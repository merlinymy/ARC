# Research Paper RAG System - Complete Planning Document

## Project Context

- **Domain**: Biochemistry, chemistry, biology, and related topics
- **Scale**: ~22,000 papers
- **Source Format**: PDFs
- **Users**: Single user
- **Priority**: Accuracy above all else

---

## Revised Architecture (Simplified for Single User)

### Why Simplify?

With low query volume and one user, latency and cost aren't concerns. Eliminating complexity reduces failure points and improves accuracy.

### Recommended Stack

| Component      | Recommendation                      | Notes                                                       |
| -------------- | ----------------------------------- | ----------------------------------------------------------- |
| Vector DB      | Qdrant                              | Handles metadata filtering, no need for separate PostgreSQL |
| Embeddings     | Voyage-3-large                      | Strong scientific text performance                          |
| Reranker       | Cohere Rerank or bge-reranker-v2-m3 | **Critical for accuracy**                                   |
| LLM            | Claude                              | Answer generation + verification                            |
| PDF Extraction | Nougat or Marker                    | Scientific-specific extraction                              |

### Chunk Strategy (6 Types, Single Collection)

| Chunk Type | Size                             | Purpose                            | Expected Count       |
| ---------- | -------------------------------- | ---------------------------------- | -------------------- |
| Abstract   | ~200 tokens                      | Paper discovery                    | 22k                  |
| Section    | ~2000 tokens                     | Literature review, context         | ~88k (4/paper avg)   |
| Fine       | ~500 tokens, **100-150 overlap** | Specific fact retrieval            | ~220-330k            |
| Full       | Mean-pooled or truncated         | Paper similarity, holistic queries | 22k                  |
| Caption    | Variable (~50-200 tokens)        | Figure/table discovery             | ~66-110k (3-5/paper) |
| Table      | Variable                         | Quantitative data retrieval        | ~22-44k (1-2/paper)  |

**Why include full paper embeddings?**

- Paper similarity queries: "Find papers similar to this one"
- Holistic retrieval when answer spans multiple sections
- Ranking papers (not chunks) by relevance
- Building reading lists

**Full paper embedding approach:**

- **Option A:** Truncate to Voyage's limit (~16k tokens) - simpler but loses end content
- **Option B (Recommended):** Mean pooling - embed each section, average vectors - preserves everything including conclusions

**Key metadata per chunk:**

```json
{
  "paper_id": "string",
  "chunk_type": "abstract|section|fine|full|caption|table",
  "section_name": "string",
  "parent_chunk_id": "string (for fine chunks)",
  "figure_id": "string (for caption/table chunks, e.g. 'Fig3', 'Table2')",
  "title": "string",
  "year": "int",
  "authors": ["string"],
  "text": "string"
}
```

### Query Workflow (No Router)

```
User Query
    ↓
Embed query (Voyage)
    ↓
Search ALL chunk types in parallel (top 50 each)
    ↓
Merge results + Rerank → top 15
    ↓
Expand fine chunks → pull parent section for context
    ↓
Deduplicate by paper_id
    ↓
Generate answer (Claude)
    ↓
Self-verify: "Does cited chunk support this claim?"
    ↓
Return answer with citations
```

---

## Critical Domain-Specific Issues

### 1. Chemical/Biological Entity Problems

**The Problem:** Embeddings don't handle scientific nomenclature well.

Examples of missed connections:

- "dopamine synthesis" → chunks mentioning only "L-DOPA" or "tyrosine hydroxylase"
- "CRISPR" → "Cas9", "guide RNA", "sgRNA"
- "breast cancer gene" → "BRCA1", "BRCA2"

**Potential Solutions:**

- [ ] Build synonym/alias lookup table
- [ ] Use PubChem, UniProt, ChEBI for entity mappings
- [ ] Expand queries with known synonyms before embedding
- [ ] Store normalized entity names in metadata for exact-match filtering

### 2. Notation Inconsistencies

**The Problem:** Same concepts written differently across papers.

Examples:

- IC50, IC₅₀, IC-50
- µM, uM, micromolar
- α-helix, alpha-helix, a-helix
- CO₂, CO2

**Potential Solutions:**

- [ ] Normalize during indexing (pick canonical forms)
- [ ] Regex-based cleanup in post-processing
- [ ] Store both original and normalized text

### 3. Methods Sections

**The Problem:** Critical experimental details (concentrations, protocols, equipment) live in Methods. Often the most valuable content for replication questions.

**Potential Solutions:**

- [ ] Ensure Methods aren't deprioritized in chunking
- [ ] Consider separate "protocol" chunk type
- [ ] Extract structured data: temperatures, concentrations, durations

### 4. Figures and Tables

**The Problem:** Papers reference "see Figure 3" but actual data is in the figure/table, not extractable text. In biochemistry/chemistry papers, figures often contain:

- Experimental results (gel images, spectra, microscopy)
- Quantitative data (bar charts, dose-response curves)
- Molecular structures and pathways
- Statistical comparisons

**Extraction Strategies:**

#### A. Figure Captions (Minimum Viable)

- Extract all figure/table captions as separate chunks
- Captions usually summarize what the figure shows
- Store with metadata: `{chunk_type: "caption", figure_id: "Fig3", paper_id: "..."}`
- Low effort, catches ~60% of figure-related queries

#### B. Table Content Extraction

- Use specialized table extractors: **Camelot**, **Tabula**, or **Table Transformer** (Microsoft)
- Convert tables to structured markdown or JSON
- Store as separate chunks with column headers preserved
- Critical for: IC50 values, experimental conditions, compound properties

```markdown
# Example: Extracted table as chunk

| Compound | IC50 (nM) | Selectivity |
| -------- | --------- | ----------- |
| ABC-123  | 3.2       | >100x       |
| XYZ-456  | 0.8       | 50x         |
```

#### C. Figure Image Analysis (Advanced)

- Extract figure images from PDF
- Use vision models (Claude, GPT-4V) to describe scientific figures
- Generate text descriptions of charts, structures, gels
- Store descriptions as searchable chunks

**Implementation approach:**

```python
# For each figure image:
description = vision_model.describe(
    image,
    prompt="Describe this scientific figure. Include: "
           "type of visualization, key findings, "
           "axis labels, trends, and any numerical values visible."
)
# Store description + caption together
```

#### D. Cross-Reference Linking

- When text says "see Figure 3", link to the figure caption/description chunk
- Store figure references in chunk metadata
- During retrieval, automatically pull referenced figures

**Recommended Approach for Your System:**

| Priority | Action                              | Effort | Value                           |
| -------- | ----------------------------------- | ------ | ------------------------------- |
| 1        | Extract all captions                | Low    | High                            |
| 2        | Extract tables with Camelot/Tabula  | Medium | High (for quantitative queries) |
| 3        | Vision descriptions for key figures | High   | Medium (diminishing returns)    |
| 4        | Cross-reference linking             | Medium | Medium                          |

**Start with #1 and #2.** Add vision-based extraction only if you find queries failing because critical data is locked in figure images.

**Questions to Answer:**

- [ ] Are you extracting figure captions? → **Yes, always do this**
- [ ] Are you extracting table contents? → **Yes, use Camelot or Tabula**
- [ ] How will you handle "data not in text" scenarios? → **Caption + table extraction covers most cases; vision model for remainder**

---

## PDF Extraction Pipeline

### The Core Challenge

Scientific PDFs are notoriously difficult:

- Chemical structures render as images
- Tables become interleaved text
- Multi-column layouts merge incorrectly
- Greek letters and subscripts corrupt
- Equations often unreadable

### Recommended Tools (in order of preference for scientific content)

1. **Nougat** (Meta)

   - Pros: Designed for academic papers, handles equations, preserves structure
   - Cons: Slow
   - Best for: Accuracy-critical extraction

2. **Marker**

   - Pros: Faster, good multi-column handling
   - Cons: Less accurate on complex equations
   - Best for: Bulk processing with acceptable quality

3. **PyMuPDF + custom layout analysis**
   - Pros: Full control
   - Cons: Significant development effort
   - Best for: When you need specific customization

**Avoid:** Raw PyPDF2, pdfminer for scientific content

### Quality Validation Checklist

Before full indexing, manually check extraction on 20 diverse papers:

- [ ] Methods sections: Are concentrations/units correct?
- [ ] Tables: Do rows/columns make sense?
- [ ] Chemical names: Are they intact or corrupted?
- [ ] Greek letters: α, β, γ preserved?
- [ ] Subscripts/superscripts: CO₂, H₂O, x², correct?
- [ ] Multi-column: Text flows correctly?
- [ ] References: Extracting properly?
- [ ] Figure captions: Present and associated correctly?

### Post-Processing Steps

```python
# Suggested post-processing pipeline
def clean_extracted_text(text):
    # 1. Normalize whitespace
    # 2. Fix common Unicode issues
    # 3. Standardize chemical notation
    # 4. Normalize units (µM → uM or vice versa)
    # 5. Fix Greek letter corruption
    # 6. Remove extraction artifacts
    pass
```

---

## Storage Estimates (Revised)

| Item         | Estimate  | Notes                         |
| ------------ | --------- | ----------------------------- |
| Raw text     | ~1 GB     | Depends on extraction quality |
| Vectors      | ~2-2.5 GB | Includes all 6 chunk types    |
| Total chunks | ~440-550k | All chunk types combined      |
| Qdrant RAM   | ~6-10 GB  | With HNSW index               |

**Chunk breakdown:**

- Abstracts: 22k
- Sections: ~88k
- Fine chunks: ~220-330k
- Full paper: 22k
- Figure/table captions: ~66-110k (3-5 per paper)
- Tables (structured): ~22-44k (1-2 per paper)

---

## Implementation Phases

### Phase 1: Validation (Do This First)

**Goal:** Prove the pipeline works before scaling

- [ ] Select 50 representative papers (diverse topics, years, journals)
- [ ] Extract with chosen tool, manually validate quality
- [ ] Create 50 test queries you actually care about
- [ ] Index the 50 papers
- [ ] Run test queries, evaluate retrieval quality
- [ ] Identify failure modes

**Key Questions to Answer:**

1. What's your extraction error rate?
2. Which query types fail?
3. Are entity/synonym issues significant?

### Phase 2: MVP (Minimal Viable Pipeline)

- [ ] PDF extraction pipeline with quality checks
- [ ] Abstract + fine chunk indexing
- [ ] Figure/table caption extraction
- [ ] Table content extraction (Camelot/Tabula)
- [ ] Basic Qdrant setup with metadata
- [ ] Reranker integration
- [ ] Claude answer generation
- [ ] Simple feedback logging (thumbs up/down)

### Phase 3: Full System

- [ ] Add section chunks
- [ ] Add full paper embeddings (mean-pooled)
- [ ] Parent chunk expansion
- [ ] Citation verification step
- [ ] Entity synonym expansion (if Phase 1 showed it's needed)
- [ ] Notation normalization

### Phase 4: Refinements

- [ ] Tune based on feedback log patterns
- [ ] Vision-based figure descriptions (if caption extraction insufficient)
- [ ] Consider hybrid search (sparse + dense) if keyword matching proves important
- [ ] Citation graph features (optional)
- [ ] Cross-reference linking (figure mentions → figure chunks)

---

## Open Questions You Need to Answer

### PDF Extraction

1. Which extraction tool will you use?
2. Do you have compute for Nougat (slow but accurate) or need Marker (faster)?
3. What's your plan for tables with critical data?
4. Will you extract figure captions?

### Entity Handling

5. How significant is the synonym problem in your domain?
6. Will you build a synonym table? Which ontologies (PubChem, UniProt, ChEBI)?
7. What notation normalization rules do you need?

### Retrieval

8. How many chunks to retrieve before reranking? (suggested: 50)
9. How many final chunks to send to Claude? (suggested: 10-15)
10. Will you implement parent chunk expansion?

### Evaluation

11. What are your 50 test queries?
12. How will you measure accuracy? (manual review? specific benchmarks?)
13. What's "good enough" retrieval performance?

### Infrastructure

14. Where will Qdrant run? (local, cloud)
15. Voyage API or self-hosted embeddings?
16. Cohere Rerank API or local reranker model?

### Feedback Loop

17. How will you log queries and feedback?
18. How often will you review failure cases?

---

## Quick Reference: Key Decisions

| Decision              | Recommendation          | Reasoning                                            |
| --------------------- | ----------------------- | ---------------------------------------------------- |
| Query routing         | Don't do it             | Adds complexity, classification errors hurt accuracy |
| PostgreSQL            | Skip for now            | Qdrant metadata filtering sufficient at 22k papers   |
| Chunk overlap         | 100-150 tokens          | Prevents split sentences, storage is cheap           |
| Reranker              | Required                | Biggest accuracy gain for minimal effort             |
| Self-verification     | Recommended             | Catches hallucinations                               |
| Synonym handling      | Test first              | May or may not be significant in your corpus         |
| Full paper embeddings | Yes, mean-pooled        | Enables similarity search, holistic retrieval        |
| Figure captions       | Always extract          | Low effort, high value for scientific papers         |
| Table extraction      | Yes, use Camelot/Tabula | Critical for quantitative data queries               |

---

## Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Voyage AI](https://www.voyageai.com/)
- [Nougat (Meta)](https://github.com/facebookresearch/nougat)
- [Marker](https://github.com/VikParuchuri/marker)
- [PubChem](https://pubchem.ncbi.nlm.nih.gov/) - chemical synonyms
- [UniProt](https://www.uniprot.org/) - protein identifiers
- [ChEBI](https://www.ebi.ac.uk/chebi/) - chemical entities
- [Camelot](https://camelot-py.readthedocs.io/) - table extraction from PDFs
- [Tabula](https://tabula.technology/) - table extraction tool
- [Table Transformer](https://github.com/microsoft/table-transformer) - Microsoft's ML-based table detection

---

## Next Steps

1. **Immediate:** Run extraction test on 20 papers, validate quality
2. **This week:** Define your 50 test queries
3. **Before building:** Answer the open questions above
4. **First milestone:** Working retrieval on 50 papers with manual accuracy check
