# Quick Start Guide

Get the Research Paper RAG system up and running in minutes!

## Step 1: Set Up Backend Environment

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure Environment Variables

```bash
# Copy example env file
cp .env.example .env

# Edit .env file with your API keys
nano .env  # or use your preferred editor
```

**Required API keys:**
- `ANTHROPIC_API_KEY`: Get from https://console.anthropic.com/
- `VOYAGE_API_KEY`: Get from https://www.voyageai.com/

## Step 3: Start Qdrant Vector Database

```bash
# From project root directory
docker-compose up -d

# Verify it's running
curl http://localhost:6333/health
```

Expected response: `{"title":"qdrant - vector search engine","version":"..."}`

## Step 4: Test PDF Processing (with 1-2 papers)

```bash
cd backend/preprocessing

# Process just 5 PDFs for testing
python process_pdfs.py --test

# Or specify a custom limit
python process_pdfs.py --limit 10
```

This will:
- Extract text from PDFs
- Chunk the content
- Generate embeddings via Voyage API
- Store in Qdrant

**Expected time:**
- 5 PDFs: ~2-5 minutes
- 100 PDFs: ~30-60 minutes
- 22K PDFs: ~24-48 hours

## Step 5: Start the Backend API

```bash
cd backend/api

# Start the FastAPI server
python main.py

# Or use uvicorn directly
uvicorn main:app --reload
```

Backend will be available at: http://localhost:8000

**Test the API:**
```bash
# Check health
curl http://localhost:8000/health

# Check stats
curl http://localhost:8000/stats

# Test a query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is quantum entanglement?"}'
```

## Step 6: Set Up Frontend (Optional for now)

```bash
cd frontend

# Initialize React + TypeScript project
npm create vite@latest . -- --template react-ts

# Install dependencies
npm install

# Install additional packages
npm install @tanstack/react-query axios tailwindcss

# Start development server
npm run dev
```

Frontend will be available at: http://localhost:5173

## Testing Your Setup

### 1. Test Qdrant Connection

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
print(client.get_collections())
```

### 2. Test Voyage API

```python
import voyageai

client = voyageai.Client(api_key="your_key_here")
result = client.embed(
    texts=["Hello world"],
    model="voyage-2"
)
print(f"Embedding dimension: {len(result.embeddings[0])}")
```

### 3. Test Claude API

```python
from anthropic import Anthropic

client = Anthropic(api_key="your_key_here")
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=100,
    messages=[{"role": "user", "content": "Say hello!"}]
)
print(message.content[0].text)
```

## Common Issues

### Qdrant won't start
```bash
# Check if port is already in use
lsof -i :6333

# Stop existing container
docker stop qdrant
docker rm qdrant

# Restart
docker-compose up -d
```

### PDF processing fails
- Make sure PDF path in `.env` is correct
- Check that PDFs are readable (not corrupted)
- Verify Voyage API key is valid
- Check API rate limits

### Import errors
```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

## Next Steps

1. **Test with subset**: Start with 10-100 papers to validate quality
2. **Tune parameters**: Adjust `chunk_size`, `chunk_overlap`, `top_k`
3. **Build frontend**: Create a nice UI for A'Lester
4. **Process all papers**: When ready, process all 22K papers
5. **Deploy**: Move to production server

## Development Workflow

```bash
# Terminal 1: Qdrant
docker-compose up

# Terminal 2: Backend
cd backend/api
source ../venv/bin/activate
python main.py

# Terminal 3: Frontend (when ready)
cd frontend
npm run dev

# Terminal 4: Process new papers as needed
cd backend/preprocessing
source ../venv/bin/activate
python process_pdfs.py --limit 10
```

## Monitoring

Check processing progress:
```bash
# Watch logs
tail -f backend/preprocessing/processing.log

# Check Qdrant stats
curl http://localhost:6333/collections/research_papers
```

## Cost Tracking

For testing with 100 papers:
- Voyage AI embeddings: ~$0.60
- Claude API (10 queries): ~$0.50
- **Total**: ~$1.10

For all 22K papers:
- Voyage AI embeddings: ~$132 (one-time)
- Claude API: $3 per million tokens (~$50-200/month)

## Help

If you run into issues:
1. Check the logs in `backend/api/main.py`
2. Verify all environment variables in `.env`
3. Test each component separately (Qdrant, Voyage, Claude)
4. Start with a small subset of papers

Ready to start? Run:
```bash
docker-compose up -d && cd backend && source venv/bin/activate && python preprocessing/process_pdfs.py --test
```
