"""FastAPI application for the Research Paper RAG system."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import settings
from qdrant_client import QdrantClient
import voyageai
from anthropic import Anthropic

app = FastAPI(
    title="Research Paper RAG API",
    description="API for querying research papers using RAG",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
voyage_client = voyageai.Client(api_key=settings.voyage_api_key)
anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


class QueryRequest(BaseModel):
    """Request model for chat queries."""
    question: str
    top_k: int = 10
    temperature: float = 0.7


class Source(BaseModel):
    """Source citation model."""
    paper_title: str
    paper_id: str
    page_number: int
    chunk_text: str
    relevance_score: float


class QueryResponse(BaseModel):
    """Response model for chat queries."""
    answer: str
    sources: List[Source]
    question: str


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Research Paper RAG API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        # Check Qdrant connection
        collections = qdrant.get_collections()

        # Check if our collection exists
        collection_names = [c.name for c in collections.collections]
        has_collection = settings.qdrant_collection_name in collection_names

        if has_collection:
            collection_info = qdrant.get_collection(settings.qdrant_collection_name)
            points_count = collection_info.points_count
        else:
            points_count = 0

        return {
            "status": "healthy",
            "qdrant_connected": True,
            "collection_exists": has_collection,
            "total_vectors": points_count
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_papers(request: QueryRequest):
    """Query the research papers and get an answer."""
    try:
        # 1. Embed the question using Voyage AI
        question_embedding = voyage_client.embed(
            texts=[request.question],
            model=settings.embedding_model,
            input_type="query"
        ).embeddings[0]

        # 2. Search Qdrant for relevant chunks
        search_results = qdrant.search(
            collection_name=settings.qdrant_collection_name,
            query_vector=question_embedding,
            limit=request.top_k
        )

        if not search_results:
            return QueryResponse(
                answer="I couldn't find any relevant information in the research papers to answer your question.",
                sources=[],
                question=request.question
            )

        # 3. Prepare context from search results
        context_parts = []
        sources = []

        for idx, result in enumerate(search_results):
            payload = result.payload

            context_parts.append(
                f"[Source {idx + 1}] {payload['paper_title']}\n"
                f"Page {payload['page_number']}\n"
                f"{payload['text']}\n"
            )

            sources.append(Source(
                paper_title=payload['paper_title'],
                paper_id=payload['paper_id'],
                page_number=payload['page_number'],
                chunk_text=payload['text'][:500] + "..." if len(payload['text']) > 500 else payload['text'],
                relevance_score=result.score
            ))

        context = "\n\n".join(context_parts)

        # 4. Create prompt for Claude
        prompt = f"""You are a PhD-level research assistant helping to answer questions about research papers.

Based on the following excerpts from research papers, please answer the user's question. Be precise, cite specific sources when making claims, and acknowledge if the information is insufficient to fully answer the question.

Context from papers:
{context}

User's question: {request.question}

Please provide a comprehensive, accurate answer based on the context above. If you reference specific information, mention which source it comes from (e.g., "According to Source 1...").
"""

        # 5. Get response from Claude
        message = anthropic_client.messages.create(
            model=settings.claude_model,
            max_tokens=settings.max_tokens,
            temperature=request.temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = message.content[0].text

        return QueryResponse(
            answer=answer,
            sources=sources,
            question=request.question
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get statistics about the database."""
    try:
        collection_info = qdrant.get_collection(settings.qdrant_collection_name)

        return {
            "collection_name": settings.qdrant_collection_name,
            "total_vectors": collection_info.points_count,
            "vector_dimension": settings.embedding_dimension,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.claude_model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
