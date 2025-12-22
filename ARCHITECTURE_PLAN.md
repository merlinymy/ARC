# Research Paper RAG System - Architecture & Design Document

## Project Overview

**Goal**: Build a conversational interface for A'Lester to chat with his research paper collection using Claude + RAG (Retrieval-Augmented Generation).

**Scale**: 22,401 PDF research papers

**Requirements**:
- PhD professor-level accuracy
- Comprehensive coverage
- Cloud-hosted service
- Long initial setup is acceptable (days is fine)
- Interest in exploring local LLMs

---

## System Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────┐
│                     COMPLETE SYSTEM                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────────────┐         │
│  │  FRONTEND (React + TypeScript)              │         │
│  │  - User types question                      │         │
│  │  - Displays answers + citations             │         │
│  └──────────────┬──────────────────────────────┘         │
│                 │ HTTP Request                           │
│                 ▼                                        │
│  ┌────────────────────────────────────────────┐         │
│  │  BACKEND (Python - FastAPI/LlamaIndex)      │         │
│  │  - Receives user question                   │         │
│  │  - Orchestrates the RAG pipeline            │         │
│  │  - Returns response to frontend             │         │
│  └──────┬─────────────────────┬─────────────────┘        │
│         │                     │                          │
│         │ 1. Query            │ 3. Send prompt           │
│         ▼                     ▼                          │
│  ┌─────────────────┐   ┌──────────────────┐            │
│  │ VECTOR DATABASE │   │   LLM (Claude)   │            │
│  │ (Qdrant/Chroma) │   │  - Generates     │            │
│  │ - Stores        │   │    answer        │            │
│  │   embeddings    │   │  - Has vision    │            │
│  │ - Returns       │   │    capability    │            │
│  │   relevant      │   │                  │            │
│  │   chunks        │   │                  │            │
│  └─────────────────┘   └──────────────────┘            │
│         ▲                                                │
│         │                                                │
└─────────┼────────────────────────────────────────────────┘
          │
  ┌───────┴──────────────────────────────────┐
  │  EMBEDDING MODEL (preprocessing)         │
  │  - Converts PDF text → vectors           │
  │  - Runs during initial setup             │
  │  - Runs when adding new papers           │
  │  - SAME model used for questions         │
  └──────────────────────────────────────────┘
```

### User Query Flow

When user asks: "What is quantum entanglement?"

1. User types question in React frontend
2. Frontend sends HTTP request to Python backend
3. Backend converts question to embedding (using same model as papers)
4. Backend queries vector database: "find chunks similar to this embedding"
5. Vector database returns top 10-20 relevant paper chunks
6. Backend sends to Claude: "Here are excerpts from papers + question"
7. If chunks reference figures/tables, includes images via Claude Vision
8. Claude generates answer with citations
9. Backend returns response to frontend
10. User sees answer with paper citations

---

## Component Analysis & Decisions

### 1. Vector Database Comparison

**Estimated storage needs:**
- 22,401 papers × ~1,500 chunks/paper = ~33 million chunks
- Each chunk with embedding ≈ 6KB
- **Total: ~198GB of vector data**

| Database | Cost | Hosting | Pros | Cons |
|----------|------|---------|------|------|
| **Chroma** (self-hosted) | $0 (or $30-40/mo cloud server) | Local or cloud | Free, simple, good for prototyping | Less scalable |
| **Qdrant** (self-hosted) | $0 (or $30-40/mo cloud server) | Local or cloud | Fast, scalable, great performance | Requires setup |
| **FAISS** | $0 | Local only | Very fast, free | No cloud features, complex |
| **Pinecone** | ~$70-100/mo | Managed cloud | No server management | Expensive |
| **Weaviate Cloud** | ~$50-80/mo | Managed cloud | Good features | Costs money |

**Hosting clarification:**
- **Self-hosted on your computer**: $0/month, but computer must stay on
- **Self-hosted on cloud server**: $30-40/month for DigitalOcean/AWS server, accessible 24/7
- **Managed service**: More expensive but less setup

**✅ RECOMMENDED: Qdrant (self-hosted on cloud server)**
- Cost: ~$40/month for VPS
- Fast, scalable, perfect for this scale
- Full control

---

### 2. Embedding Strategy Comparison

**Cost & Quality Analysis:**

| Model | Quality Score | Cost (1-time) | Cost (per query) | Notes |
|-------|---------------|---------------|------------------|-------|
| **Voyage AI** | ~59.4 NDCG@10 | $132 | $0.02/1K queries | Best for RAG, handles academic text well |
| **OpenAI large** | ~54.9 | $143 | $0.02/1K queries | Good general purpose |
| **OpenAI small** | ~49.0 | $22 | $0.003/1K queries | Cheaper, decent quality |
| **bge-large-en-v1.5** | ~54.1 | $0 (or ~$10-20 GPU time) | $0 | Best open-source option |
| **e5-mistral-7b** | ~56.9 | $0 (or ~$20 GPU time) | $0 | Excellent quality, needs GPU |

**Quality difference:**
- Voyage AI is ~5-10% better at retrieval than open-source
- For PhD-level accuracy, this might matter
- But good chunking strategy can compensate

**✅ RECOMMENDED: Two options**
- **Option A (Best quality)**: Voyage AI - $132 one-time + negligible per-query cost
- **Option B (Best value)**: bge-large-en-v1.5 - Free, start here and upgrade if needed

**CRITICAL RULE**: Question embeddings MUST use same model as paper embeddings!
- Papers and questions must be in the same vector space
- Cannot mix models (e.g., papers in Voyage, questions in bge)

---

### 3. LLM Strategy

| Option | Cost | Pros | Cons |
|--------|------|------|------|
| **Claude 3.5 Sonnet API** | ~$3 per 1M input tokens | Best accuracy, vision capability, PhD-level reasoning | Ongoing API costs (~$50-200/mo) |
| **Local Llama 3.1 70B** | $0 (need GPU) | Free after setup, privacy | Need expensive GPU, slightly lower quality |
| **Local Mixtral 8x7B** | $0 (need GPU) | Free, good quality | Setup complexity |

**✅ RECOMMENDED: Start with Claude API**
- Best accuracy for PhD-level responses
- Vision capability crucial for graphs/tables
- Can always add local LLM later as backup

---

### 4. Backend Framework

**✅ RECOMMENDED: LlamaIndex + FastAPI**
- **LlamaIndex**: Better for academic/complex RAG than LangChain
- **FastAPI**: Modern, fast, Python async support
- Both are most popular in their categories

---

### 5. Frontend

**✅ RECOMMENDED: React + TypeScript**
- Per your requirements
- Add Tailwind CSS for styling
- Support streaming responses
- Display citations with links to source papers

---

## Multimodal Content Handling

### The Challenge

Research papers contain:
- **Text**: Body paragraphs, abstracts
- **Tables**: Experimental results, comparisons
- **Figures**: Graphs, diagrams, plots
- **Equations**: Mathematical formulas

### Solution Strategy

#### Phase 1: MVP (Recommended Start)

**Text + Tables:**
- Use `pymupdf4llm` for extraction (preserves table structure as markdown)
- Chunk text with table data included

**Figures/Graphs:**
1. Extract images from PDFs using PyMuPDF
2. Store images with references in chunks
3. During retrieval, if chunk mentions "Figure X", include actual image
4. Send image to Claude Vision along with text
5. Claude reads and interprets the graph!

**Example:**
```python
# Chunk contains: "As shown in Figure 2, accuracy improved..."
# System extracts Figure 2 as image
# Sends to Claude:
#   - Text context from papers
#   - Image of Figure 2
#   - User's question
# Claude can see and interpret the graph
```

#### Phase 2: Enhanced (If needed)

**Pre-process figures:**
1. Use Claude Vision during initial setup
2. Generate text descriptions of all figures
3. Embed descriptions
4. Better searchability for visual content

**Multimodal embeddings:**
- Use Jina AI or Nomic Embed for image+text embeddings
- More advanced retrieval

**✅ RECOMMENDED: Start with Phase 1**
- Claude Vision handles most cases perfectly
- Can enhance later if accuracy insufficient

---

## Adding New Papers

**Question: Does adding new papers require re-embedding everything?**

**Answer: NO! Only embed the new papers.**

Vector databases are **additive**:

**Initial setup (one-time):**
```
22,401 papers → Extract → Chunk → Embed → Store
(Takes hours/days)
```

**Adding 10 new papers later:**
```
10 new papers → Extract → Chunk → Embed → Add to existing DB
(Takes minutes)
```

The 22,401 existing embeddings remain untouched. Just insert new ones.

---

## Cost Summary

### One-Time Costs
| Item | Option A (Premium) | Option B (Budget) |
|------|-------------------|-------------------|
| **Embeddings** | Voyage AI: $132 | bge-large: $20 GPU time |
| **Initial Setup** | Server setup: $0 | Server setup: $0 |
| **TOTAL ONE-TIME** | **~$132** | **~$20** |

### Monthly Costs
| Item | Estimated Cost |
|------|----------------|
| **Cloud Server** (Qdrant hosting) | $30-40 |
| **Claude API** (varies by usage) | $50-200 |
| **Embedding new papers** | ~$1-2 |
| **Question embeddings** | <$1 |
| **TOTAL MONTHLY** | **$80-240** |

**Note**: Could reduce monthly costs by running everything on local computer ($0/mo) but less accessible.

---

## Recommended Tech Stack

### Final Recommendation

```yaml
Backend:
  Framework: FastAPI
  RAG Orchestration: LlamaIndex
  Language: Python 3.11+

Vector Database:
  Primary: Qdrant (self-hosted on cloud)
  Hosting: DigitalOcean/AWS ($40/mo)

Embeddings:
  Initial Choice: bge-large-en-v1.5 (free, test first)
  Upgrade Option: Voyage AI ($132 if accuracy needed)
  Critical: Same model for papers AND questions

LLM:
  Primary: Claude 3.5 Sonnet (API)
  Future: Local Llama 3.1 70B (optional)

Frontend:
  Framework: React + TypeScript
  Styling: Tailwind CSS
  Features: Streaming, citations, image display

Document Processing:
  Text/Tables: pymupdf4llm
  Figures: PyMuPDF for image extraction
  Vision: Claude Vision API for interpreting images
```

---

## Implementation Phases

### Phase 1: Core RAG System (Weeks 1-2)
1. Set up document processing pipeline
   - Extract text from 22K PDFs
   - Chunk documents intelligently
   - Extract basic tables as markdown
2. Set up embedding system
   - Choose embedding model (bge-large to start)
   - Embed all chunks
3. Set up vector database (Qdrant)
4. Basic backend (FastAPI + LlamaIndex)
5. Test retrieval quality

### Phase 2: LLM Integration (Week 3)
1. Integrate Claude API
2. Build RAG pipeline
3. Implement citation tracking
4. Test answer quality

### Phase 3: Multimodal Support (Week 4)
1. Extract figures/images from PDFs
2. Link images to chunks
3. Integrate Claude Vision
4. Test with graph-heavy papers

### Phase 4: Frontend (Week 5)
1. React + TypeScript setup
2. Chat interface
3. Citation display
4. Image rendering
5. Streaming responses

### Phase 5: Deployment & Testing (Week 6)
1. Deploy to cloud server
2. End-to-end testing
3. Performance optimization
4. User acceptance testing

### Phase 6: Enhancements (Future)
1. Consider Voyage AI embeddings
2. Add local LLM option
3. Fine-tune retrieval
4. Advanced features (filtering, export, etc.)

---

## Key Technical Decisions

### ✅ Decisions Made
1. **Database**: Qdrant (self-hosted on cloud) - $40/mo
2. **Embeddings**: Start with bge-large-en-v1.5, upgrade to Voyage if needed
3. **LLM**: Claude 3.5 Sonnet API
4. **Backend**: Python + FastAPI + LlamaIndex
5. **Frontend**: React + TypeScript
6. **Multimodal**: Phase 1 approach (Claude Vision)

### 🤔 Still To Decide
1. Exact chunking strategy (size, overlap)
2. Number of chunks to retrieve per query
3. Cloud provider (DigitalOcean vs AWS vs others)
4. Specific UI design
5. Authentication/access control

---

## Questions & Clarifications

### Q: Why pay for server if self-hosted?
**A**: "Self-hosted" means YOU control the software (vs managed service). You can:
- Run on YOUR computer: $0/mo (but must stay on)
- Run on cloud server YOU rent: $30-40/mo (accessible 24/7)

### Q: Same embedding model for questions?
**A**: YES! Papers and questions must use same model. They need to exist in the same vector space for similarity search to work.

### Q: Re-embed when adding papers?
**A**: NO! Vector DB is additive. Only embed new papers, insert into existing DB.

### Q: How to handle tables/graphs?
**A**:
- Tables: Extract as markdown, include in text chunks
- Graphs: Extract as images, send to Claude Vision with text context
- Claude can read and interpret visual content!

---

## Next Steps

1. Set up development environment
2. Start document processing pipeline
3. Test embedding quality with sample papers
4. Build prototype RAG system
5. Iterate based on accuracy testing

---

*Document created: 2025-12-15*
*Project: Research Paper RAG System for A'Lester*
*Scale: 22,401 research papers*
