# Research Paper RAG System

A conversational AI system for chatting with research papers using RAG (Retrieval-Augmented Generation).

## Tech Stack

- **PDF Processing**: MinerU (deep learning-based extraction)
- **Embeddings**: Voyage AI (`voyage-3-large`)
- **Vector DB**: Qdrant (self-hosted via Docker)
- **Reranking**: Cohere
- **LLM**: Claude 3.5 Sonnet (Anthropic)
- **Backend**: Python 3.10+, FastAPI

## Project Structure

```
researchPaperAgent/
├── backend/
│   ├── preprocessing/     # PDF processing, chunking, section detection
│   ├── retrieval/         # Embedder, Qdrant store, query engine
│   ├── evaluation/        # RAG evaluation scripts
│   ├── api/               # FastAPI application
│   ├── index_papers.py    # Main indexing script
│   ├── test_rag.py        # Interactive RAG testing
│   └── requirements.txt
├── docker-compose.yml     # Qdrant container config
└── README.md
```

## Mac Setup Guide

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3) or Intel
- Python 3.10+
- Docker Desktop
- External drive with PDF papers (optional)

### Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd researchPaperAgent
```

### Step 2: Set Up Python Environment

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

MinerU will automatically download ML models on first run (~2GB).

### Step 4: Configure Environment Variables

Create `.env` file in `/backend/`:

```bash
# API Keys (required)
ANTHROPIC_API_KEY=sk-ant-...
VOYAGE_API_KEY=pa-...
COHERE_API_KEY=...

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=research_papers

# Paths - adjust to your setup
PDF_SOURCE_DIR=/Volumes/YourDrive/papers
PROCESSED_DATA_DIR=/Volumes/YourDrive/processed_data

# Processing Settings
CHUNK_SIZE=512
CHUNK_OVERLAP=128

# RAG Settings
ENABLE_QUERY_CLASSIFICATION=false  # Use hybrid retrieval
ENABLE_QUERY_EXPANSION=false

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

### Step 5: Set Up Qdrant (Vector Database)

If storing Qdrant data on external drive, update `docker-compose.yml`:

```yaml
volumes:
  - /Volumes/YourDrive/qdrant_data:/qdrant/storage
```

Start Qdrant:

```bash
cd ..  # Back to project root
docker-compose up -d
```

Verify it's running:

```bash
curl http://localhost:6333/health
# Should return: {"status":"ok"}
```

### Step 6: Copy Data Files

Copy your PDF papers to the `PDF_SOURCE_DIR` path specified in `.env`.

If copying from another Mac, use `-L` flag to resolve symlinks:

```bash
rsync -avL --progress /path/to/source/papers/ /Volumes/YourDrive/papers/
```

### Step 7: Index Papers

Test with a few papers first:

```bash
cd backend
python index_papers.py --reset --limit 10
```

If successful, run full indexing:

```bash
python index_papers.py --reset
```

Indexing features:
- **Checkpointing**: Safe to interrupt with Ctrl+C, will resume
- **Rate limiting**: Handles Voyage API limits automatically
- **Progress tracking**: Shows real-time progress

### Step 8: Test the RAG System

Interactive mode:

```bash
python test_rag.py
```

Single query:

```bash
python test_rag.py "What methods are used for protein purification?"
```

### Step 9: Start the API Server

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at `http://localhost:8000`

## Usage

### Indexing Commands

```bash
# Index all papers (resumes from checkpoint)
python index_papers.py

# Index with limit (for testing)
python index_papers.py --limit 100

# Reset and start fresh
python index_papers.py --reset

# Retry failed papers
python index_papers.py --retry-failed

# Use specific directory
python index_papers.py --papers-dir /path/to/papers
```

### Testing RAG

```bash
# Interactive mode
python test_rag.py

# Single query
python test_rag.py "What is the role of LL37 in antimicrobial activity?"
```

### Running Evaluation

The W5 retrieval harness lives in `backend/evaluation/`. It measures retrieval
only — it never generates an answer.

```bash
cd backend

# Everything: retrieve -> pool -> judge -> score. Re-runs are free; every
# retrieval, embedding and relevance judgment is cached to disk.
python -m evaluation.run_w5

# Score an existing run without spending anything
python -m evaluation.run_w5 --stages score --metric ndcg@10

# Preview judging cost before spending
python -m evaluation.run_w5 --stages pool judge --estimate-only

# Rebuild the golden query set from backend/data/app.db, or verify it still matches
python -m evaluation.build_query_set
python -m evaluation.build_query_set --check
```

Durable assets: `evaluation/golden_queries_v1.json` (50 real user queries) and
`evaluation/qrels_v1.json` (pooled graded relevance labels). Baselines are
written to `evaluation/results/baseline_<date>.json`.

This replaces the former `backend/run_evaluation.py`, which required a full
answer-generation pass to report retrieval quality and scored against
hand-invented expected topics rather than relevance judgments.

## Retrieval Strategy

The system uses a **hybrid retrieval approach**:

| Query Type | Strategy |
|------------|----------|
| Methods queries | Targeted retrieval (section filtering) |
| Limitations queries | Targeted retrieval (section filtering) |
| All other queries | Universal retrieval (all chunk types) |

This provides the best balance of accuracy and latency.

## Chunk Types

| Type | Description | Use Case |
|------|-------------|----------|
| `abstract` | Full paper abstract | Overview queries |
| `section` | Logical sections (intro, methods, results) | Detailed queries |
| `fine` | 500-token overlapping chunks | Precise factual queries |
| `caption` | Figure/table captions | Visual content queries |
| `table` | Extracted table content | Data queries |
| `full` | Mean-pooled paper embedding | Paper-level similarity |

## Troubleshooting

### MinerU Model Download Issues

If models fail to download, set:

```bash
export HF_ENDPOINT=https://huggingface.co
```

### Qdrant Connection Issues

Check if container is running:

```bash
docker ps
docker logs qdrant
```

### macOS AppleDouble Files

Files starting with `._` are automatically filtered during indexing.

### External Drive Issues

Ensure the drive is mounted before starting:

```bash
ls /Volumes/YourDrive
```

## Cost Estimates

### Voyage AI Embeddings

- **Model**: voyage-3-large
- **Price**: $0.06 per 1M tokens
- **22K papers**: ~$15-20

### Anthropic Claude

- **Model**: claude-3-5-sonnet
- **Price**: ~$3 per 1M input tokens, ~$15 per 1M output tokens
- **Per query**: ~$0.01-0.02

## License

MIT
