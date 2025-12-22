# Research Paper RAG System - Complete Planning Document (Revised)

## Project Context

- **Domain**: Biochemistry, chemistry, biology (Raman imaging, peptide research, ERα inhibitors)
- **Scale**: ~22,000 papers
- **Source Format**: PDFs
- **Users**: Single user (researcher)
- **Priority**: Accuracy above all else
- **Primary Use Cases**: Writing assistance, framing/positioning, literature synthesis (not just factual retrieval)

---

## Understanding the Query Types

**Critical Insight:** Based on actual user questions, this system must handle multiple query types that require different retrieval and generation strategies:

| Query Type | Example | Retrieval Strategy | Generation Strategy |
|------------|---------|-------------------|---------------------|
| **Factual** | "What is the difference between CC50 and IC50?" | Fine chunks, tables | Direct answer with citations |
| **Framing/Positioning** | "How do I frame Raman imaging as a platform?" | Abstracts, sections from multiple papers | Synthesize rhetorical strategies |
| **Methods Writing** | "How should I describe Affinity FPLC vs Ion-Exchange FPLC?" | Methods sections specifically | Technical writing assistance |
| **Paper Summary** | "Summarize this paper's key findings" | Full paper, all sections | Structured summary |
| **Novelty Justification** | "What is the strongest defensible novelty claim?" | Related work across corpus | Comparative analysis |
| **Controls/Validation** | "How do I justify the tag doesn't disrupt bioactivity?" | Methods, results with controls | Argument construction |
| **Limitations** | "How do I explain localization without colocalization markers?" | Discussion sections | Balanced interpretation |

---

## Revised Architecture

### Recommended Stack

| Component      | Recommendation                      | Notes                                                       |
| -------------- | ----------------------------------- | ----------------------------------------------------------- |
| Vector DB      | Qdrant                              | Handles metadata filtering, no need for separate PostgreSQL |
| Embeddings     | Voyage-3-large                      | Strong scientific text performance                          |
| Reranker       | Cohere Rerank v3.5                  | **Critical for accuracy**                                   |
| LLM            | Claude                              | Answer generation + verification + query classification     |
| PDF Extraction | Marker (primary) or Nougat          | Scientific-specific extraction                              |

### Chunk Strategy (6 Types, Single Collection)

| Chunk Type | Size                             | Purpose                            | Expected Count       |
| ---------- | -------------------------------- | ---------------------------------- | -------------------- |
| Abstract   | ~200-300 tokens                  | Paper discovery, framing queries   | 22k                  |
| Section    | ~2000 tokens                     | Literature review, context         | ~88k (4/paper avg)   |
| Fine       | ~500 tokens, **100-150 overlap** | Specific fact retrieval            | ~220-330k            |
| Full       | Mean-pooled                      | Paper similarity, holistic queries | 22k                  |
| Caption    | Variable (~50-200 tokens)        | Figure/table discovery             | ~66-110k (3-5/paper) |
| Table      | Variable                         | Quantitative data retrieval        | ~22-44k (1-2/paper)  |

### Enhanced Metadata Schema

```json
{
  "paper_id": "string",
  "chunk_type": "abstract|section|fine|full|caption|table",
  "section_name": "string (normalized: introduction|methods|results|discussion|conclusion)",
  "parent_chunk_id": "string (for fine chunks)",
  "figure_id": "string (for caption/table chunks)",
  "title": "string",
  "year": "int",
  "authors": ["string"],
  "text": "string",
  "project_tag": "string (e.g., 'ERα_inhibitors', 'LL37_imaging', 'biofilm')",
  "research_area": "string (e.g., 'peptide_imaging', 'protein_purification')"
}
```

**New fields:**
- `project_tag`: User-defined project/manuscript grouping
- `research_area`: High-level research theme for filtering

---

## Query Workflow (With Classification)

```
User Query
    ↓
Query Classification (Claude) → query_type + retrieval_strategy
    ↓
Query Expansion (domain synonyms)
    ↓
Embed expanded query (Voyage)
    ↓
Strategy-Based Retrieval:
  - Factual: Fine chunks + Tables (top 50)
  - Framing: Abstracts + Sections (top 30 each)
  - Methods: Methods sections only (top 50)
  - Summary: All chunks from target paper
  - Comparative: Abstracts across corpus (top 100)
    ↓
Rerank → top 15-25 (varies by query type)
    ↓
Expand fine chunks → pull parent section for context
    ↓
Deduplicate by paper_id (configurable max per paper)
    ↓
Generate answer (Claude) with query-type-specific prompt
    ↓
Self-verify: "Does cited chunk support this claim?"
    ↓
Return answer with citations
```

### Query Classification Prompt

```
Classify this research question into one of these categories:
- FACTUAL: Specific facts, definitions, values (IC50, mechanisms)
- FRAMING: How to position, describe, or justify research for publication
- METHODS: Technical protocols, procedures, experimental details
- SUMMARY: Summarize a specific paper or set of findings
- COMPARATIVE: Compare methods, approaches, or findings across papers
- NOVELTY: Assess what's new, different, or defensible as a contribution
- LIMITATIONS: How to discuss constraints, caveats, or missing data

Also identify:
- Key domain entities (gene names, proteins, techniques, chemicals)
- Whether this needs single-paper focus or cross-corpus synthesis

Question: {query}
```

---

## Domain-Specific Entity Handling

### Domain Synonym Dictionary (Required for Phase 1)

Based on actual user queries, build synonyms for your specific domain:

```python
DOMAIN_SYNONYMS = {
    # Estrogen Receptor
    "ERα": ["estrogen receptor alpha", "ER-alpha", "ESR1", "estrogen receptor"],
    "ERα LBD": ["estrogen receptor ligand binding domain", "ER LBD"],
    "Y537S": ["Y537S mutation", "tyrosine 537 serine"],
    "D538G": ["D538G mutation", "aspartate 538 glycine"],
    
    # Imaging Techniques
    "SRS": ["stimulated Raman scattering", "coherent Raman", "SRS imaging"],
    "Raman": ["Raman spectroscopy", "Raman imaging", "vibrational imaging"],
    
    # Peptides
    "LL-37": ["LL37", "cathelicidin", "CAP18", "CAMP", "antimicrobial peptide"],
    "AMP": ["antimicrobial peptide", "antimicrobial peptides", "host defense peptide"],
    "stapled peptide": ["stapled peptides", "hydrocarbon-stapled", "macrocyclic peptide"],
    
    # Tags and Labels  
    "alkyne tag": ["alkyne", "diyne", "Raman tag", "vibrational tag", "clickable tag"],
    "FITC": ["fluorescein", "fluorescent label", "fluorescent tag"],
    
    # Assays and Measurements
    "IC50": ["IC-50", "IC₅₀", "half-maximal inhibitory concentration"],
    "CC50": ["CC-50", "CC₅₀", "half-maximal cytotoxic concentration"],
    "EC50": ["EC-50", "EC₅₀", "half-maximal effective concentration"],
    
    # Methods
    "FPLC": ["fast protein liquid chromatography", "protein chromatography"],
    "affinity chromatography": ["affinity FPLC", "affinity purification", "His-tag purification"],
    "ion exchange": ["ion-exchange FPLC", "IEX", "anion exchange", "cation exchange"],
    
    # Biological Contexts
    "biofilm": ["biofilms", "bacterial biofilm", "biofilm matrix"],
    "planktonic": ["planktonic cells", "free-floating bacteria"],
    "breast cancer": ["hormone-resistant breast cancer", "ER+ breast cancer", "BRCA"],
}
```

### Query Expansion Function

```python
def expand_query(query: str, synonyms: dict) -> str:
    """Expand query with domain synonyms."""
    expanded_terms = []
    query_lower = query.lower()
    
    for canonical, aliases in synonyms.items():
        # Check if canonical or any alias appears in query
        all_terms = [canonical.lower()] + [a.lower() for a in aliases]
        for term in all_terms:
            if term in query_lower:
                # Add all synonyms that aren't already in query
                for alias in aliases:
                    if alias.lower() not in query_lower:
                        expanded_terms.append(alias)
                break
    
    if expanded_terms:
        return f"{query} (related: {', '.join(expanded_terms[:5])})"
    return query
```

---

## Notation Normalization

### Normalization Rules (Apply During Indexing)

```python
NORMALIZATION_RULES = [
    # Greek letters
    (r'α|alpha|α', 'α'),
    (r'β|beta|β', 'β'),
    (r'γ|gamma|γ', 'γ'),
    (r'δ|delta|δ', 'δ'),
    (r'μ|mu|µ', 'μ'),
    
    # Subscripts (normalize to standard form)
    (r'CO₂|CO2|CO_2', 'CO2'),
    (r'H₂O|H2O|H_2O', 'H2O'),
    
    # Units
    (r'µM|μM|uM|micromolar', 'μM'),
    (r'µg|μg|ug|microgram', 'μg'),
    (r'µL|μL|uL|microliter', 'μL'),
    
    # IC50 variants
    (r'IC50|IC₅₀|IC-50|IC 50', 'IC50'),
    (r'EC50|EC₅₀|EC-50|EC 50', 'EC50'),
    (r'CC50|CC₅₀|CC-50|CC 50', 'CC50'),
]
```

---

## Section Detection Patterns

### Enhanced Patterns for Scientific Papers

```python
SECTION_PATTERNS = [
    # Standard sections
    (r'^(?:1\.?\s*)?(?:introduction|background)', "introduction", 1),
    (r'^(?:2\.?\s*)?(?:materials?\s*(?:and|&)\s*methods?|methods?|experimental)', "methods", 1),
    (r'^(?:3\.?\s*)?(?:results?)', "results", 1),
    (r'^(?:4\.?\s*)?(?:discussion)', "discussion", 1),
    (r'^(?:5\.?\s*)?(?:conclusion|conclusions|summary)', "conclusion", 1),
    
    # Combined sections (common in chemistry)
    (r'^results?\s*(?:and|&)\s*discussion', "results_discussion", 1),
    
    # Other sections
    (r'^abstract', "abstract", 1),
    (r'^(?:references?|bibliography|literature\s*cited)', "references", 1),
    (r'^(?:acknowledg|funding|support)', "acknowledgments", 1),
    (r'^(?:supplementa|supporting\s*information|appendix|SI)', "supplementary", 1),
    
    # Chemistry-specific
    (r'^(?:synthesis|synthetic\s*procedures?)', "synthesis", 1),
    (r'^(?:characterization|compound\s*characterization)', "characterization", 1),
    (r'^(?:biological\s*evaluation|bioactivity)', "bioactivity", 1),
    
    # Numbered subsections
    (r'^\d+\.\d+\.?\s+', "subsection", 2),
]
```

---

## Full Paper Embedding: Mean Pooling Implementation

```python
def compute_full_paper_embedding(
    section_chunks: List[Chunk],
    embedder: VoyageEmbedder
) -> List[float]:
    """
    Compute mean-pooled embedding for full paper.
    
    This preserves information from all sections including conclusions,
    unlike truncation which loses end content.
    """
    # Get section texts (exclude references, acknowledgments)
    valid_sections = [
        c for c in section_chunks 
        if c.section_name not in ('references', 'acknowledgments', 'supplementary')
    ]
    
    if not valid_sections:
        return None
    
    # Embed each section
    texts = [c.text for c in valid_sections]
    embeddings = embedder.embed_documents(texts)
    
    # Mean pool with L2 normalization
    import numpy as np
    embeddings_array = np.array(embeddings)
    mean_embedding = np.mean(embeddings_array, axis=0)
    
    # Normalize to unit length
    norm = np.linalg.norm(mean_embedding)
    if norm > 0:
        mean_embedding = mean_embedding / norm
    
    return mean_embedding.tolist()
```

---

## Evaluation Strategy

### Query Type Distribution for Testing

Based on actual usage patterns, test queries should be distributed as:

| Category | Count | Priority |
|----------|-------|----------|
| Framing/positioning | 12 | High |
| Methods writing | 8 | High |
| Factual/definition | 8 | Medium |
| Controls/validation | 6 | High |
| Novelty justification | 6 | High |
| Paper summarization | 5 | Medium |
| Limitations | 5 | Medium |
| **Total** | **50** | |

### Evaluation Metrics by Query Type

| Query Type | Primary Metric | Secondary Metrics |
|------------|---------------|-------------------|
| Factual | Topic coverage, entity recall | Citation accuracy |
| Framing | Human rating (1-5) | Rhetorical coherence |
| Methods | Section recall (methods chunks) | Technical accuracy |
| Summary | Coverage of key findings | Conciseness |
| Comparative | Papers retrieved diversity | Balance of perspectives |
| Novelty | Cross-corpus coverage | Defensibility rating |
| Limitations | Balanced presentation | Acknowledgment of gaps |

### Human Evaluation Template

For framing/positioning queries (where automated metrics fail):

```
Query: "How do I frame Raman imaging as a platform rather than a one-off demonstration?"

Answer Quality (1-5):
  1 = Unhelpful/wrong
  2 = Partially relevant but misses key points
  3 = Adequate but generic
  4 = Good, addresses the specific framing challenge
  5 = Excellent, provides actionable rhetorical strategy

Retrieval Quality (1-5):
  1 = Retrieved irrelevant content
  2 = Some relevant papers but missed key ones
  3 = Adequate coverage
  4 = Good coverage of relevant literature
  5 = Excellent, found the most relevant papers for this question

Notes: _______________
```

---

## Storage Estimates (Revised)

| Item         | Estimate  | Notes                         |
| ------------ | --------- | ----------------------------- |
| Raw text     | ~1 GB     | Depends on extraction quality |
| Vectors      | ~2-2.5 GB | Includes all 6 chunk types    |
| Total chunks | ~440-550k | All chunk types combined      |
| Qdrant RAM   | ~6-10 GB  | With HNSW index               |

---

## Implementation Phases (Revised)

### Phase 1: Validation (Do This First)

**Goal:** Prove the pipeline works on 50 papers with your actual query types

- [ ] Select 50 representative papers (include ERα, LL-37, Raman imaging papers)
- [ ] Build initial domain synonym dictionary (20-30 key terms)
- [ ] Extract with Marker, manually validate quality on 10 papers
- [ ] Create 50 test queries from actual usage patterns (see CSV)
- [ ] Implement query classification
- [ ] Index the 50 papers
- [ ] Run test queries with human evaluation for framing queries
- [ ] Identify failure modes

**Key Questions to Answer:**

1. Does query classification improve retrieval relevance?
2. Which query types fail most often?
3. Is synonym expansion helping or adding noise?

### Phase 2: MVP

- [ ] Full PDF extraction pipeline with quality checks
- [ ] All 6 chunk types with proper metadata
- [ ] Query classification + type-specific retrieval
- [ ] Domain synonym expansion
- [ ] Reranker integration
- [ ] Type-specific answer generation prompts
- [ ] Feedback logging (thumbs up/down + query type)

### Phase 3: Full System

- [ ] Full corpus indexing (22k papers)
- [ ] Project/research area tagging
- [ ] Parent chunk expansion
- [ ] Citation verification step
- [ ] Notation normalization during indexing
- [ ] Enhanced section detection for chemistry papers

### Phase 4: Refinements

- [ ] Tune retrieval strategies by query type
- [ ] Expand synonym dictionary based on failures
- [ ] Vision-based figure descriptions (if needed)
- [ ] Hybrid search (sparse + dense) for entity queries
- [ ] User preference learning

---

## Quick Reference: Key Decisions (Updated)

| Decision              | Recommendation          | Reasoning                                            |
| --------------------- | ----------------------- | ---------------------------------------------------- |
| Query classification  | Yes, use Claude         | Different query types need different retrieval       |
| Query expansion       | Yes, domain synonyms    | Scientific nomenclature varies significantly         |
| Chunk overlap         | 100-150 tokens          | Prevents split sentences                             |
| Reranker              | Required                | Biggest accuracy gain for minimal effort             |
| Self-verification     | Recommended             | Catches hallucinations                               |
| Full paper embeddings | Yes, mean-pooled        | Preserves conclusion content                         |
| Figure captions       | Always extract          | Low effort, high value                               |
| Table extraction      | Yes, use pdfplumber     | Critical for IC50/CC50 queries                       |
| Section normalization | Yes                     | "Results and Discussion" → "results_discussion"      |
| Human evaluation      | Required for framing    | Automated metrics don't capture rhetorical quality   |

---

## Test Queries (From Actual Usage)

### High Priority - Framing/Positioning

1. "How do I frame Raman imaging as an enabling platform rather than a single-use technique?"
2. "How should I position Raman imaging in biofilms as fundamentally different from planktonic cell imaging?"
3. "How do I frame Y537S and D538G targeting as clinically motivated but mechanistically rigorous?"
4. "How can PET and Raman imaging be presented as complementary rather than redundant?"

### High Priority - Controls/Validation

5. "How do I justify that diyne-girder or alkyne tags do not disrupt biological activity?"
6. "How do I describe controls so reviewers see Raman imaging as quantitative enough for biology journals?"
7. "How do I argue that fluorescent tags alter permeability while vibrational tags preserve native behavior?"

### High Priority - Methods Writing

8. "How should I describe Affinity FPLC vs Ion-Exchange FPLC in a paper rather than vendor language?"
9. "How do I write a technical but concise justification for purifying ERα LBD WT, D538G, and Y537S?"

### Medium Priority - Factual

10. "What is the difference between CC50 and IC50?"
11. "What enzymes are involved in dopamine synthesis?" (tests synonym expansion)

### Medium Priority - Limitations

12. "How do I explain peptide uptake and localization without fluorescent colocalization markers?"
13. "What are the limitations of this study that the authors do not explicitly discuss?"

---

## Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Voyage AI](https://www.voyageai.com/)
- [Marker](https://github.com/VikParuchuri/marker)
- [Nougat (Meta)](https://github.com/facebookresearch/nougat)
- [PubChem](https://pubchem.ncbi.nlm.nih.gov/) - chemical synonyms
- [UniProt](https://www.uniprot.org/) - protein identifiers
- [ChEBI](https://www.ebi.ac.uk/chebi/) - chemical entities

---

## Next Steps

1. **Immediate:** Build domain synonym dictionary (20-30 terms from your papers)
2. **This week:** Create 50 test queries from actual usage (use the CSV provided)
3. **Before building:** Implement query classification prompt and test manually
4. **First milestone:** Working retrieval on 50 papers with human evaluation
