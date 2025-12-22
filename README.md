# Research Paper RAG System

A conversational AI system for chatting with research papers using RAG (Retrieval-Augmented Generation).

## Project Structure

```
researchPaperAgent/
├── backend/              # Python FastAPI backend
│   ├── preprocessing/    # PDF processing scripts
│   ├── api/             # FastAPI application
│   └── requirements.txt
├── frontend/            # React + TypeScript frontend
├── resource/            # Research papers (22K+ PDFs)
├── docs/               # Documentation
└── docker-compose.yml  # Docker setup for Qdrant
```

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, LlamaIndex
- **Frontend**: React, TypeScript, Tailwind CSS
- **Vector DB**: Qdrant (self-hosted)
- **Embeddings**: Voyage AI API
- **LLM**: Claude 3.5 Sonnet (Anthropic API)

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (for Qdrant)

### 1. Set up the backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Start Qdrant

```bash
docker-compose up -d
```

### 4. Process PDFs (one-time)

```bash
cd backend/preprocessing
python process_pdfs.py
```

### 5. Start the backend

```bash
cd backend
uvicorn api.main:app --reload
```

### 6. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

## Documentation

- [Architecture Plan](ARCHITECTURE_PLAN.md)
- [Server Requirements](SERVER_REQUIREMENTS.md)

## Development Workflow

1. Test with a subset of papers first (10-100 papers)
2. Validate quality of retrieval and responses
3. Process all 22K papers when ready
4. Deploy to production

## License

MIT
