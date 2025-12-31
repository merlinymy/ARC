"""FastAPI application for the Research Paper RAG system."""

import asyncio
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import List, Optional, Callable

from fastapi import FastAPI, HTTPException, Depends, Request, Response, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
import json
from datetime import datetime
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import settings
from dependencies import (
    get_dependencies,
    get_qdrant_client,
    get_query_engine,
    get_paper_library_service,
)
from qdrant_client import QdrantClient
from retrieval.query_engine import QueryEngine
from retrieval.qdrant_store import QdrantStore
from services.paper_library import PaperLibraryService
from services.chat_service import ChatService
from services.memory_service import MemoryService
from services.upload_queue import UploadQueueService
from database import async_session_maker
from database import get_async_session, init_db, create_default_user, User
from auth import verify_password, create_access_token, get_current_user, get_current_user_optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Annotated

logger = logging.getLogger(__name__)


# =============================================================================
# Rate Limiting
# =============================================================================
class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for given client."""
        async with self._lock:
            now = time.time()
            window_start = now - 60  # 1 minute window

            # Clean old requests
            self.requests[client_id] = [
                ts for ts in self.requests[client_id] if ts > window_start
            ]

            if len(self.requests[client_id]) >= self.requests_per_minute:
                return False

            self.requests[client_id].append(now)
            return True

    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client."""
        now = time.time()
        window_start = now - 60
        recent = [ts for ts in self.requests.get(client_id, []) if ts > window_start]
        return max(0, self.requests_per_minute - len(recent))


rate_limiter = RateLimiter(requests_per_minute=settings.rate_limit_per_minute if hasattr(settings, 'rate_limit_per_minute') else 60)


# =============================================================================
# Graceful Shutdown
# =============================================================================
class GracefulShutdown:
    """Manages graceful shutdown with in-flight request tracking."""

    def __init__(self):
        self.is_shutting_down = False
        self.active_requests = 0
        self._lock = asyncio.Lock()

    async def start_request(self) -> bool:
        """Register a new request. Returns False if shutting down."""
        async with self._lock:
            if self.is_shutting_down:
                return False
            self.active_requests += 1
            return True

    async def end_request(self):
        """Mark a request as complete."""
        async with self._lock:
            self.active_requests -= 1

    async def initiate_shutdown(self, timeout: float = 30.0):
        """Initiate graceful shutdown, waiting for in-flight requests."""
        async with self._lock:
            self.is_shutting_down = True

        logger.info(f"Graceful shutdown initiated, waiting for {self.active_requests} active requests...")

        start = time.time()
        while self.active_requests > 0 and (time.time() - start) < timeout:
            await asyncio.sleep(0.1)

        if self.active_requests > 0:
            logger.warning(f"Shutdown timeout reached with {self.active_requests} requests still active")
        else:
            logger.info("All requests completed, shutting down cleanly")


shutdown_handler = GracefulShutdown()

# Global upload queue service (initialized in lifespan)
upload_queue_service: Optional[UploadQueueService] = None


def get_upload_queue() -> UploadQueueService:
    """Get the upload queue service."""
    if upload_queue_service is None:
        raise HTTPException(status_code=503, detail="Upload queue service not initialized")
    return upload_queue_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    global upload_queue_service

    # Startup: dependencies are lazily initialized on first access
    logger.info("Starting Research Paper RAG API...")

    # Initialize database and create default user
    await init_db()
    await create_default_user()
    logger.info("Database initialized")

    # Initialize and start upload queue service
    try:
        paper_library = get_paper_library_service()
        upload_queue_service = UploadQueueService(
            session_maker=async_session_maker,
            paper_library=paper_library,
            max_workers=2,
        )
        await upload_queue_service.start()
        logger.info("Upload queue service started")
    except Exception as e:
        logger.error(f"Failed to start upload queue service: {e}")

    # Note: Don't set up signal handlers here - uvicorn handles SIGINT/SIGTERM
    # and setting our own handlers interferes with --reload mode

    yield

    # Shutdown: wait for in-flight requests and clean up resources
    await shutdown_handler.initiate_shutdown()

    # Stop upload queue service
    if upload_queue_service:
        await upload_queue_service.stop()
        logger.info("Upload queue service stopped")

    logger.info("Shutting down Research Paper RAG API...")
    get_dependencies().close()


app = FastAPI(
    title="Research Paper RAG API",
    description="API for querying research papers using RAG",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_middleware(request: Request, call_next: Callable) -> Response:
    """Middleware for request tracing, rate limiting, and graceful shutdown."""
    # Generate correlation ID
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

    # Check if shutting down
    if not await shutdown_handler.start_request():
        return Response(
            content='{"detail": "Service is shutting down"}',
            status_code=503,
            media_type="application/json",
            headers={"X-Request-ID": request_id}
        )

    try:
        # Rate limiting (use client IP or API key as identifier)
        client_id = request.headers.get("X-API-Key") or request.client.host if request.client else "unknown"

        if not await rate_limiter.is_allowed(client_id):
            await shutdown_handler.end_request()
            return Response(
                content='{"detail": "Rate limit exceeded. Try again later."}',
                status_code=429,
                media_type="application/json",
                headers={
                    "X-Request-ID": request_id,
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": "60"
                }
            )

        # Log request with correlation ID
        logger.info(f"[{request_id}] {request.method} {request.url.path}")

        # Process request
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        # Add tracing headers to response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        response.headers["X-RateLimit-Remaining"] = str(rate_limiter.get_remaining(client_id))

        logger.info(f"[{request_id}] Completed in {duration:.3f}s with status {response.status_code}")

        return response
    finally:
        await shutdown_handler.end_request()


class QueryRequest(BaseModel):
    """Request model for chat queries."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The question to ask about the research papers"
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to retrieve"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature for response generation"
    )
    paper_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of paper IDs to filter search results. If None or empty, searches all papers."
    )
    max_chunks_per_paper: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Maximum chunks per paper in results. If None, system decides based on context."
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation ID to persist chat history. If provided, messages will be saved to the database."
    )
    query_type: Optional[str] = Field(
        default=None,
        description="Optional query type override. If None, auto-detected. Valid values: factual, methods, summary, comparative, novelty, limitations, general, framing"
    )
    enable_hyde: Optional[bool] = Field(
        default=None,
        description="Enable HyDE (Hypothetical Document Embedding). If None, uses system default."
    )
    enable_expansion: Optional[bool] = Field(
        default=None,
        description="Enable query expansion with synonyms. If None, uses system default."
    )
    enable_citation_check: Optional[bool] = Field(
        default=None,
        description="Enable citation verification. If None, uses system default."
    )


class Source(BaseModel):
    """Source citation model."""
    paper_title: str
    paper_id: str
    section_name: Optional[str] = None
    subsection_name: Optional[str] = None
    chunk_type: str
    chunk_text: str
    relevance_score: float


class QueryResponse(BaseModel):
    """Response model for chat queries."""
    answer: str
    sources: List[Source]
    question: str
    query_type: str
    expanded_query: str
    retrieval_count: int
    reranked_count: int
    warnings: List[str] = []


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Research Paper RAG API",
        "version": "1.0.0",
        "status": "running"
    }


async def check_voyage_health() -> dict:
    """Check Voyage AI connectivity."""
    try:
        import voyageai
        client = voyageai.Client(api_key=settings.voyage_api_key)
        # Minimal embedding call to verify connectivity
        client.embed(["health check"], model=settings.embedding_model, truncation=True)
        return {"status": "healthy", "error": None}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def check_cohere_health() -> dict:
    """Check Cohere connectivity.

    Note: We don't make actual API calls here to avoid consuming rate limits.
    With free tier (10 req/min), health checks would quickly exhaust the quota.
    Instead, we just verify the API key is configured.
    """
    if not settings.cohere_api_key:
        return {"status": "unhealthy", "error": "COHERE_API_KEY not configured"}
    return {"status": "healthy", "error": None}


async def check_anthropic_health() -> dict:
    """Check Anthropic connectivity."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        # Use a minimal API call - count tokens is lightweight
        client.messages.count_tokens(
            model=settings.claude_model,
            messages=[{"role": "user", "content": "test"}]
        )
        return {"status": "healthy", "error": None}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/health")
async def health(qdrant: QdrantClient = Depends(get_qdrant_client)):
    """Health check endpoint - checks all service dependencies."""
    services = {}
    overall_healthy = True

    # Check Qdrant
    try:
        collections = qdrant.get_collections()
        collection_names = [c.name for c in collections.collections]
        has_collection = settings.qdrant_collection_name in collection_names

        if has_collection:
            collection_info = qdrant.get_collection(settings.qdrant_collection_name)
            points_count = collection_info.points_count
        else:
            points_count = 0

        services["qdrant"] = {
            "status": "healthy",
            "connected": True,
            "collection_exists": has_collection,
            "total_vectors": points_count
        }
    except Exception as e:
        services["qdrant"] = {"status": "unhealthy", "error": str(e)}
        overall_healthy = False

    # Check external AI services concurrently
    voyage_result, cohere_result, anthropic_result = await asyncio.gather(
        check_voyage_health(),
        check_cohere_health(),
        check_anthropic_health(),
        return_exceptions=True
    )

    # Process Voyage result
    if isinstance(voyage_result, Exception):
        services["voyage_ai"] = {"status": "unhealthy", "error": str(voyage_result)}
        overall_healthy = False
    else:
        services["voyage_ai"] = voyage_result
        if voyage_result["status"] == "unhealthy":
            overall_healthy = False

    # Process Cohere result
    if isinstance(cohere_result, Exception):
        services["cohere"] = {"status": "unhealthy", "error": str(cohere_result)}
        overall_healthy = False
    else:
        services["cohere"] = cohere_result
        if cohere_result["status"] == "unhealthy":
            overall_healthy = False

    # Process Anthropic result
    if isinstance(anthropic_result, Exception):
        services["anthropic"] = {"status": "unhealthy", "error": str(anthropic_result)}
        overall_healthy = False
    else:
        services["anthropic"] = anthropic_result
        if anthropic_result["status"] == "unhealthy":
            overall_healthy = False

    response = {
        "status": "healthy" if overall_healthy else "degraded",
        "services": services
    }

    if not overall_healthy:
        raise HTTPException(status_code=503, detail=response)

    return response


@app.get("/health/quick")
async def health_quick():
    """Quick health check - only verifies the API is responding."""
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
async def query_papers(
    request: QueryRequest,
    query_engine: QueryEngine = Depends(get_query_engine),
    current_user: Optional[User] = Depends(get_current_user_optional),
    session: AsyncSession = Depends(get_async_session),
):
    """Query the research papers using the full RAG pipeline.

    This endpoint uses the complete QueryEngine pipeline including:
    - Query classification and rewriting
    - Query expansion with domain synonyms
    - Hybrid vector search
    - Reranking with entity boosting
    - Query-type-specific answer generation

    If conversation_id is provided and user is authenticated, messages will be
    persisted to the database for chat history.
    """
    try:
        # Use the full QueryEngine pipeline
        result = query_engine.query(
            request.question,
            paper_ids=request.paper_ids,
            max_chunks_per_paper=request.max_chunks_per_paper,
            query_type_override=request.query_type,
            enable_hyde_override=request.enable_hyde,
            enable_expansion_override=request.enable_expansion,
            enable_citation_check_override=request.enable_citation_check,
        )

        # Convert sources to API response format
        sources = []
        for source in result.sources:
            sources.append(Source(
                paper_title=source.get('title', 'Unknown'),
                paper_id=source.get('paper_id', ''),
                section_name=source.get('section_name'),
                subsection_name=source.get('subsection_name'),
                chunk_type=source.get('chunk_type', 'unknown'),
                chunk_text=source.get('text', '')[:500] + "..." if len(source.get('text', '')) > 500 else source.get('text', ''),
                relevance_score=source.get('score', 0.0),
            ))

        # Persist messages to database if conversation_id is provided
        if request.conversation_id and current_user:
            try:
                chat_service = ChatService(session)

                # Check if conversation exists, create if not
                conversation = await chat_service.get_conversation(
                    request.conversation_id, current_user.id, include_messages=False
                )
                if conversation is None:
                    # Auto-create conversation with first message as title
                    title = request.question[:100] + "..." if len(request.question) > 100 else request.question
                    await chat_service.create_conversation(
                        request.conversation_id, current_user.id, title=title
                    )

                # Save user message
                await chat_service.add_message(
                    conversation_id=request.conversation_id,
                    user_id=current_user.id,
                    role="user",
                    content=request.question,
                )

                # Save assistant message with metadata
                await chat_service.add_message(
                    conversation_id=request.conversation_id,
                    user_id=current_user.id,
                    role="assistant",
                    content=result.answer,
                    metadata={
                        "query_type": result.query_type.value,
                        "sources": [s.model_dump() for s in sources],
                        "retrieval_count": result.retrieval_count,
                        "reranked_count": result.reranked_count,
                    }
                )
            except Exception as e:
                # Log but don't fail the request if message persistence fails
                logger.warning(f"Failed to persist chat messages: {e}")

        return QueryResponse(
            answer=result.answer,
            sources=sources,
            question=request.question,
            query_type=result.query_type.value,
            expanded_query=result.expanded_query,
            retrieval_count=result.retrieval_count,
            reranked_count=result.reranked_count,
            warnings=result.warnings,
        )

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/query/stream")
async def query_papers_stream(
    request: QueryRequest,
    query_engine: QueryEngine = Depends(get_query_engine),
    current_user: Optional[User] = Depends(get_current_user_optional),
    session: AsyncSession = Depends(get_async_session),
):
    """Stream query progress using Server-Sent Events.

    Emits progress events as the RAG pipeline executes:
    - rewriting: Query rewriting/correction
    - entities: Entity extraction
    - classification: Query type classification
    - expansion: Query expansion with synonyms
    - hyde: HyDE embedding (if enabled)
    - retrieval: Document retrieval
    - reranking: Result reranking
    - generation: Answer generation
    - verification: Citation verification
    - complete: Final result

    If conversation_id is provided and user is authenticated, messages will be
    persisted to the database for chat history.
    """

    async def event_generator():
        progress_events = []
        result_holder = [None]
        error_holder = [None]

        def progress_callback(step: str, data: dict):
            """Capture progress events."""
            progress_events.append({"step": step, "data": data})

        # Run query in a thread to not block
        import concurrent.futures

        def run_query():
            try:
                result = query_engine.query(
                    request.question,
                    paper_ids=request.paper_ids,
                    max_chunks_per_paper=request.max_chunks_per_paper,
                    progress_callback=progress_callback,
                    query_type_override=request.query_type,
                    enable_hyde_override=request.enable_hyde,
                    enable_expansion_override=request.enable_expansion,
                    enable_citation_check_override=request.enable_citation_check,
                )
                result_holder[0] = result
            except Exception as e:
                error_holder[0] = str(e)

        # Start query execution in background thread
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(run_query)

        # Stream progress events as they come in
        last_sent = 0
        while not future.done() or last_sent < len(progress_events):
            # Send any new progress events
            while last_sent < len(progress_events):
                event = progress_events[last_sent]
                yield f"data: {json.dumps({'type': 'progress', 'step': event['step'], 'data': event['data']})}\n\n"
                last_sent += 1

            if not future.done():
                await asyncio.sleep(0.05)  # Small delay to avoid busy waiting

        # Check for errors
        if error_holder[0]:
            yield f"data: {json.dumps({'type': 'error', 'message': error_holder[0]})}\n\n"
            return

        # Send final result
        result = result_holder[0]
        if result:
            sources = []
            for source in result.sources:
                sources.append({
                    "paper_title": source.get('title', 'Unknown'),
                    "paper_id": source.get('paper_id', ''),
                    "section_name": source.get('section_name'),
                    "subsection_name": source.get('subsection_name'),
                    "chunk_type": source.get('chunk_type', 'unknown'),
                    "chunk_text": source.get('text', '')[:500] + "..." if len(source.get('text', '')) > 500 else source.get('text', ''),
                    "relevance_score": source.get('score', 0.0),
                })

            final_data = {
                "type": "complete",
                "answer": result.answer,
                "sources": sources,
                "question": request.question,
                "query_type": result.query_type.value,
                "expanded_query": result.expanded_query,
                "retrieval_count": result.retrieval_count,
                "reranked_count": result.reranked_count,
                "warnings": result.warnings,
            }
            yield f"data: {json.dumps(final_data)}\n\n"

            # Persist messages to database if conversation_id is provided
            if request.conversation_id and current_user:
                try:
                    chat_service = ChatService(session)

                    # Check if conversation exists, create if not
                    conversation = await chat_service.get_conversation(
                        request.conversation_id, current_user.id, include_messages=False
                    )
                    if conversation is None:
                        # Auto-create conversation with first message as title
                        title = request.question[:100] + "..." if len(request.question) > 100 else request.question
                        await chat_service.create_conversation(
                            request.conversation_id, current_user.id, title=title
                        )

                    # Save user message
                    await chat_service.add_message(
                        conversation_id=request.conversation_id,
                        user_id=current_user.id,
                        role="user",
                        content=request.question,
                    )

                    # Save assistant message with metadata
                    await chat_service.add_message(
                        conversation_id=request.conversation_id,
                        user_id=current_user.id,
                        role="assistant",
                        content=result.answer,
                        metadata={
                            "query_type": result.query_type.value,
                            "sources": sources,
                            "retrieval_count": result.retrieval_count,
                            "reranked_count": result.reranked_count,
                        }
                    )
                except Exception as e:
                    # Log but don't fail the stream if message persistence fails
                    logger.warning(f"Failed to persist chat messages: {e}")

        executor.shutdown(wait=False)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@app.get("/stats")
async def get_stats(
    qdrant: QdrantClient = Depends(get_qdrant_client),
    query_engine: QueryEngine = Depends(get_query_engine),
):
    """Get statistics about the database and pipeline."""
    try:
        collection_info = qdrant.get_collection(settings.qdrant_collection_name)

        return {
            "collection_name": settings.qdrant_collection_name,
            "total_vectors": collection_info.points_count,
            "vector_dimension": settings.embedding_dimension,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.claude_model,
            "cache_stats": query_engine.get_cache_stats(),
            "conversation_stats": query_engine.get_conversation_stats(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.post("/conversation/clear")
async def clear_conversation(query_engine: QueryEngine = Depends(get_query_engine)):
    """Clear conversation memory to start a fresh session."""
    query_engine.clear_conversation()
    return {"status": "ok", "message": "Conversation memory cleared"}


# =============================================================================
# Paper Library Endpoints
# =============================================================================

class ChunkStats(BaseModel):
    """Chunk statistics by type."""
    abstract: int = 0
    section: int = 0
    fine: int = 0
    table: int = 0
    caption: int = 0
    full: int = 0


class PaperResponse(BaseModel):
    """Response model for a paper."""
    paper_id: str
    title: str
    authors: List[str]
    year: Optional[int] = None
    filename: str
    page_count: int
    chunk_count: int
    chunk_stats: dict
    indexed_at: Optional[datetime] = None
    status: str
    error_message: Optional[str] = None
    pdf_url: str


class PaperListResponse(BaseModel):
    """Response model for paper list with pagination."""
    papers: List[PaperResponse]
    total: int
    offset: int = 0
    limit: int = 50
    has_more: bool = False


class UploadResponse(BaseModel):
    """Response model for paper upload."""
    paper_id: str
    filename: str
    status: str
    message: str
    chunks: int = 0


class DeleteResponse(BaseModel):
    """Response model for paper deletion."""
    paper_id: str
    pdf_deleted: bool
    chunks_deleted: int
    message: str


class PaperSearchResult(BaseModel):
    """A single paper search result with relevance score and preview."""
    paper_id: str
    title: str
    authors: List[str]
    year: Optional[int] = None
    filename: str
    relevance_score: float
    chunk_count: int
    status: str
    pdf_url: str
    # Preview of best matching chunk
    preview_text: Optional[str] = None
    preview_section: Optional[str] = None
    preview_subsection: Optional[str] = None
    preview_chunk_type: Optional[str] = None


class PaperSearchResponse(BaseModel):
    """Response model for paper search."""
    results: List[PaperSearchResult]
    query: str
    total: int


class CheckDuplicatesRequest(BaseModel):
    """Request model for checking duplicate papers."""
    hashes: List[str] = Field(..., description="List of SHA256 file hashes to check")


class DuplicateInfo(BaseModel):
    """Info about a duplicate paper."""
    hash: str
    paper_id: str
    title: Optional[str] = None


class CheckDuplicatesResponse(BaseModel):
    """Response model for duplicate check."""
    duplicates: List[DuplicateInfo]
    unique_count: int
    duplicate_count: int


@app.post("/papers/check-duplicates", response_model=CheckDuplicatesResponse)
async def check_duplicate_papers(
    request: CheckDuplicatesRequest,
    library: PaperLibraryService = Depends(get_paper_library_service),
):
    """Check if papers with given file hashes already exist in the library.

    Used by batch upload to skip duplicate files before uploading.
    """
    try:
        result = library.check_duplicate_hashes(request.hashes)

        duplicates = []
        for file_hash, paper_id in result.items():
            if paper_id:
                # Get paper title for display
                paper = library.get_paper(paper_id)
                duplicates.append(DuplicateInfo(
                    hash=file_hash,
                    paper_id=paper_id,
                    title=paper.title if paper else None,
                ))

        return CheckDuplicatesResponse(
            duplicates=duplicates,
            unique_count=len(request.hashes) - len(duplicates),
            duplicate_count=len(duplicates),
        )
    except Exception as e:
        logger.error(f"Duplicate check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Duplicate check failed: {str(e)}")


@app.post("/papers/backfill-hashes")
async def backfill_paper_hashes(
    library: PaperLibraryService = Depends(get_paper_library_service),
):
    """Backfill file hashes for existing papers.

    Run this once after enabling duplicate detection to populate hashes
    for papers that were indexed before this feature was added.
    """
    try:
        result = library.backfill_file_hashes()
        return {
            "status": "success",
            "processed": result["processed"],
            "skipped": result["skipped"],
            "failed": result["failed"],
            "message": f"Processed {result['processed']} papers, skipped {result['skipped']}, failed {result['failed']}",
        }
    except Exception as e:
        logger.error(f"Hash backfill failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Backfill failed: {str(e)}")


@app.get("/papers/search", response_model=PaperSearchResponse)
async def search_papers(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    library: PaperLibraryService = Depends(get_paper_library_service),
):
    """Semantic search for papers using natural language queries.

    Searches paper-level embeddings to find the most relevant papers.
    """
    try:
        results = library.search_papers(query=q, limit=limit)
        search_results = [
            PaperSearchResult(
                paper_id=r['paper_id'],
                title=r['title'],
                authors=r['authors'],
                year=r['year'],
                filename=r['filename'],
                relevance_score=r['relevance_score'],
                chunk_count=r['chunk_count'],
                status=r['status'],
                pdf_url=f"/papers/{r['paper_id']}/pdf",
                preview_text=r.get('preview_text'),
                preview_section=r.get('preview_section'),
                preview_subsection=r.get('preview_subsection'),
                preview_chunk_type=r.get('preview_chunk_type'),
            )
            for r in results
        ]
        return PaperSearchResponse(
            results=search_results,
            query=q,
            total=len(search_results),
        )
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Paper search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/papers", response_model=PaperListResponse)
async def list_papers(
    offset: int = Query(0, ge=0, description="Number of papers to skip"),
    limit: int = Query(50, ge=1, le=100, description="Maximum papers to return"),
    library: PaperLibraryService = Depends(get_paper_library_service),
):
    """List papers in the library with pagination."""
    try:
        result = library.list_papers(offset=offset, limit=limit)
        paper_responses = [
            PaperResponse(
                paper_id=p.paper_id,
                title=p.title,
                authors=p.authors,
                year=p.year,
                filename=p.filename,
                page_count=p.page_count,
                chunk_count=p.chunk_count,
                chunk_stats=p.chunk_stats,
                indexed_at=p.indexed_at,
                status=p.status,
                error_message=p.error_message,
                pdf_url=f"/papers/{p.paper_id}/pdf",
            )
            for p in result.papers
        ]
        return PaperListResponse(
            papers=paper_responses,
            total=result.total,
            offset=result.offset,
            limit=result.limit,
            has_more=result.has_more,
        )
    except Exception as e:
        logger.error(f"Failed to list papers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing papers: {str(e)}")


@app.get("/papers/{paper_id}", response_model=PaperResponse)
async def get_paper(
    paper_id: str,
    library: PaperLibraryService = Depends(get_paper_library_service),
):
    """Get details of a specific paper."""
    try:
        paper = library.get_paper(paper_id)
        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}")

        return PaperResponse(
            paper_id=paper.paper_id,
            title=paper.title,
            authors=paper.authors,
            year=paper.year,
            filename=paper.filename,
            page_count=paper.page_count,
            chunk_count=paper.chunk_count,
            chunk_stats=paper.chunk_stats,
            indexed_at=paper.indexed_at,
            status=paper.status,
            error_message=paper.error_message,
            pdf_url=f"/papers/{paper.paper_id}/pdf",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get paper {paper_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting paper: {str(e)}")


@app.get("/papers/{paper_id}/pdf")
async def get_paper_pdf(
    paper_id: str,
    download: bool = False,
    library: PaperLibraryService = Depends(get_paper_library_service),
):
    """Serve the PDF file for a paper.

    Args:
        paper_id: The paper ID
        download: If True, force download. If False (default), display inline.
    """
    try:
        pdf_path = library.get_pdf_path(paper_id)
        if not pdf_path or not pdf_path.exists():
            raise HTTPException(status_code=404, detail=f"PDF not found for paper: {paper_id}")

        # inline = display in browser, attachment = force download
        disposition = "attachment" if download else "inline"
        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            headers={"Content-Disposition": f'{disposition}; filename="{pdf_path.name}"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve PDF for {paper_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error serving PDF: {str(e)}")


@app.delete("/papers/{paper_id}", response_model=DeleteResponse)
async def delete_paper(
    paper_id: str,
    library: PaperLibraryService = Depends(get_paper_library_service),
):
    """Delete a paper and all its associated data."""
    try:
        result = library.delete_paper(paper_id)
        return DeleteResponse(
            paper_id=paper_id,
            pdf_deleted=result["pdf_deleted"],
            chunks_deleted=result["chunks_deleted"],
            message=f"Deleted paper {paper_id}: {result['chunks_deleted']} chunks removed",
        )
    except Exception as e:
        logger.error(f"Failed to delete paper {paper_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting paper: {str(e)}")


@app.post("/papers/upload", response_model=UploadResponse)
async def upload_paper(
    file: UploadFile = File(...),
    library: PaperLibraryService = Depends(get_paper_library_service),
):
    """Upload a PDF and index it.

    This is a synchronous upload that waits for indexing to complete.
    For large files, consider using the streaming endpoint.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Check file size
    max_size = settings.max_upload_size_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_upload_size_mb}MB"
        )

    try:
        # Save the uploaded file
        pdf_path = library.save_uploaded_file(content, file.filename)

        # Index the paper
        result = library.index_paper(pdf_path)

        if result["success"]:
            return UploadResponse(
                paper_id=result["paper_id"],
                filename=result["filename"],
                status="indexed",
                message=f"Successfully indexed {result['chunks']} chunks",
                chunks=result["chunks"],
            )
        else:
            return UploadResponse(
                paper_id=result["paper_id"],
                filename=result["filename"],
                status="error",
                message=result.get("error", "Unknown error during indexing"),
                chunks=0,
            )

    except Exception as e:
        logger.error(f"Failed to upload paper: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error uploading paper: {str(e)}")


@app.post("/papers/upload/stream")
async def upload_paper_stream(
    file: UploadFile = File(...),
    library: PaperLibraryService = Depends(get_paper_library_service),
):
    """Upload a PDF and stream indexing progress.

    Returns Server-Sent Events with progress updates during indexing.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Check file size
    max_size = settings.max_upload_size_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_upload_size_mb}MB"
        )

    async def event_generator():
        progress_events = []
        result_holder = [None]
        error_holder = [None]

        def progress_callback(step: str, data: dict):
            """Capture progress events."""
            progress_events.append({"step": step, "data": data})

        import concurrent.futures

        def run_indexing():
            try:
                # Save the uploaded file
                pdf_path = library.save_uploaded_file(content, file.filename)

                # Index with progress callback
                result = library.index_paper(pdf_path, progress_callback=progress_callback)
                result_holder[0] = result
            except Exception as e:
                error_holder[0] = str(e)

        # Start indexing in background thread
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(run_indexing)

        # Stream progress events
        last_sent = 0
        while not future.done() or last_sent < len(progress_events):
            while last_sent < len(progress_events):
                event = progress_events[last_sent]
                yield f"data: {json.dumps({'type': 'progress', 'step': event['step'], 'data': event['data']})}\n\n"
                last_sent += 1

            if not future.done():
                await asyncio.sleep(0.1)

        # Check for errors
        if error_holder[0]:
            yield f"data: {json.dumps({'type': 'error', 'message': error_holder[0]})}\n\n"
            return

        # Send final result
        result = result_holder[0]
        if result:
            final_data = {
                "type": "complete",
                "paper_id": result["paper_id"],
                "filename": result["filename"],
                "status": "indexed" if result["success"] else "error",
                "message": f"Successfully indexed {result['chunks']} chunks" if result["success"] else result.get("error", "Unknown error"),
                "chunks": result.get("chunks", 0),
            }
            yield f"data: {json.dumps(final_data)}\n\n"

        executor.shutdown(wait=False)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# =============================================================================
# Batch Upload Endpoints
# =============================================================================

class BatchUploadInitRequest(BaseModel):
    """Request to initialize batch upload."""
    filenames: List[str] = Field(..., min_length=1, max_length=20)


class UploadTaskResponse(BaseModel):
    """Response for a single upload task."""
    task_id: str = Field(..., alias="taskId")
    batch_id: str = Field(..., alias="batchId")
    filename: str
    paper_id: Optional[str] = Field(None, alias="paperId")
    status: str
    current_step: Optional[str] = Field(None, alias="currentStep")
    progress_percent: int = Field(..., alias="progressPercent")
    error_message: Optional[str] = Field(None, alias="errorMessage")
    priority: int = 0
    file_size: int = Field(..., alias="fileSize")
    created_at: Optional[str] = Field(None, alias="createdAt")

    model_config = {"populate_by_name": True}


class BatchUploadInitResponse(BaseModel):
    """Response after initializing batch upload."""
    batch_id: str = Field(..., alias="batchId")
    tasks: List[UploadTaskResponse]

    model_config = {"populate_by_name": True}


class BatchStatusResponse(BaseModel):
    """Status of all tasks in a batch."""
    batch_id: str = Field(..., alias="batchId")
    tasks: List[UploadTaskResponse]
    total: int
    completed: int
    failed: int
    in_progress: int = Field(..., alias="inProgress")
    pending: int

    model_config = {"populate_by_name": True}


@app.post("/papers/upload/batch/init", response_model=BatchUploadInitResponse)
async def init_batch_upload(
    request: BatchUploadInitRequest,
    queue: UploadQueueService = Depends(get_upload_queue),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """Initialize a batch upload session.

    Creates upload tasks for each file in the batch. Files should be uploaded
    individually using the /papers/upload/batch/{batch_id}/file endpoint.
    """
    if len(request.filenames) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per batch")

    # Validate all filenames are PDFs
    for filename in request.filenames:
        if not filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"Only PDF files are allowed: {filename}")

    batch_id = await queue.create_batch(user_id=current_user.id if current_user else None)

    tasks = []
    for i, filename in enumerate(request.filenames):
        # Higher index = lower priority (first file has highest priority)
        priority = len(request.filenames) - i
        task = await queue.add_task(
            batch_id=batch_id,
            filename=filename,
            file_size=0,  # Will be updated when file is uploaded
            user_id=current_user.id if current_user else None,
            priority=priority,
        )
        tasks.append(UploadTaskResponse(
            taskId=task.id,
            batchId=task.batch_id,
            filename=task.filename,
            paperId=task.paper_id,
            status=task.status,
            currentStep=task.current_step,
            progressPercent=task.progress_percent,
            errorMessage=task.error_message,
            priority=task.priority,
            fileSize=task.file_size,
            createdAt=task.created_at.isoformat() if task.created_at else None,
        ))

    return BatchUploadInitResponse(batchId=batch_id, tasks=tasks)


@app.post("/papers/upload/batch/{batch_id}/file")
async def upload_batch_file(
    batch_id: str,
    task_id: str = Query(..., alias="taskId"),
    file: UploadFile = File(...),
    queue: UploadQueueService = Depends(get_upload_queue),
    library: PaperLibraryService = Depends(get_paper_library_service),
):
    """Upload a single file for a batch task.

    The task_id must match a task created in init_batch_upload.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Check file size
    max_size = settings.max_upload_size_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_upload_size_mb}MB"
        )

    # Verify task exists and belongs to this batch
    task = await queue.get_task(task_id)
    if not task or task['batchId'] != batch_id:
        raise HTTPException(status_code=404, detail="Task not found")

    # Save the file
    pdf_path = library.save_uploaded_file(content, file.filename)

    # Update task with file path and size
    await queue.set_task_file_path(task_id, pdf_path)

    # Update file size in database
    async with queue.session_maker() as session:
        from database.models import UploadTask
        db_task = await session.get(UploadTask, task_id)
        if db_task:
            db_task.file_size = len(content)
            await session.commit()

    return {"status": "ok", "taskId": task_id, "fileSize": len(content)}


@app.post("/papers/upload/batch/{batch_id}/start")
async def start_batch_processing(
    batch_id: str,
    queue: UploadQueueService = Depends(get_upload_queue),
):
    """Start processing all uploaded files in the batch.

    Files will be processed by the queue service in priority order.
    Use the SSE endpoint to monitor progress.
    """
    status = await queue.get_batch_status(batch_id)
    if status['total'] == 0:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Check if there are any files ready to process
    pending = status['pending']
    if pending == 0:
        raise HTTPException(status_code=400, detail="No pending files to process")

    return {
        "status": "ok",
        "batchId": batch_id,
        "message": f"Started processing {pending} files",
    }


@app.get("/papers/upload/batch/{batch_id}/stream")
async def stream_batch_progress(
    batch_id: str,
    queue: UploadQueueService = Depends(get_upload_queue),
):
    """Stream progress updates for all tasks in a batch via SSE.

    Events:
    - task_progress: Progress update for a single task
    - task_complete: A task finished successfully
    - task_error: A task failed
    - batch_complete: All tasks in the batch are done
    """
    import asyncio
    from queue import Queue as ThreadQueue

    # Thread-safe queue for events
    event_queue: ThreadQueue = ThreadQueue()

    def on_progress(event: dict):
        event_queue.put(event)

    # Register callback
    queue.register_progress_callback(batch_id, on_progress)

    async def event_generator():
        try:
            # Send initial status
            status = await queue.get_batch_status(batch_id)
            yield f"data: {json.dumps({'type': 'status', **status})}\n\n"

            # Stream events
            while True:
                try:
                    # Check for new events (non-blocking)
                    while not event_queue.empty():
                        event = event_queue.get_nowait()
                        yield f"data: {json.dumps(event)}\n\n"

                        # Exit if batch is complete
                        if event.get('type') == 'batch_complete':
                            return

                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error in SSE stream: {e}")
                    break
        finally:
            queue.unregister_progress_callback(batch_id, on_progress)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/papers/upload/batch/{batch_id}/status", response_model=BatchStatusResponse)
async def get_batch_status(
    batch_id: str,
    queue: UploadQueueService = Depends(get_upload_queue),
):
    """Get current status of all tasks in a batch."""
    status = await queue.get_batch_status(batch_id)
    if status['total'] == 0:
        raise HTTPException(status_code=404, detail="Batch not found")

    return BatchStatusResponse(
        batchId=status['batchId'],
        tasks=[
            UploadTaskResponse(
                taskId=t['taskId'],
                batchId=t['batchId'],
                filename=t['filename'],
                paperId=t.get('paperId'),
                status=t['status'],
                currentStep=t.get('currentStep'),
                progressPercent=t['progressPercent'],
                errorMessage=t.get('errorMessage'),
                priority=t.get('priority', 0),
                fileSize=t.get('fileSize', 0),
                createdAt=t.get('createdAt'),
            )
            for t in status['tasks']
        ],
        total=status['total'],
        completed=status['completed'],
        failed=status['failed'],
        inProgress=status['inProgress'],
        pending=status['pending'],
    )


@app.post("/papers/upload/batch/{batch_id}/task/{task_id}/retry")
async def retry_upload_task(
    batch_id: str,
    task_id: str,
    queue: UploadQueueService = Depends(get_upload_queue),
):
    """Retry a failed upload task."""
    task = await queue.get_task(task_id)
    if not task or task['batchId'] != batch_id:
        raise HTTPException(status_code=404, detail="Task not found")

    success = await queue.retry_task(task_id)
    if not success:
        raise HTTPException(status_code=400, detail="Task cannot be retried (not in error state or file missing)")

    return {"status": "ok", "taskId": task_id, "message": "Task queued for retry"}


@app.delete("/papers/upload/batch/{batch_id}/task/{task_id}")
async def cancel_upload_task(
    batch_id: str,
    task_id: str,
    queue: UploadQueueService = Depends(get_upload_queue),
):
    """Cancel a pending upload task."""
    task = await queue.get_task(task_id)
    if not task or task['batchId'] != batch_id:
        raise HTTPException(status_code=404, detail="Task not found")

    success = await queue.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=400, detail="Task cannot be cancelled (already processing or complete)")

    return {"status": "ok", "taskId": task_id, "message": "Task cancelled"}


@app.put("/papers/upload/batch/{batch_id}/task/{task_id}/priority")
async def set_upload_task_priority(
    batch_id: str,
    task_id: str,
    priority: int = Query(..., ge=0, le=100),
    queue: UploadQueueService = Depends(get_upload_queue),
):
    """Set the priority of a pending upload task.

    Higher priority tasks are processed first.
    """
    task = await queue.get_task(task_id)
    if not task or task['batchId'] != batch_id:
        raise HTTPException(status_code=404, detail="Task not found")

    success = await queue.set_task_priority(task_id, priority)
    if not success:
        raise HTTPException(status_code=400, detail="Task priority cannot be changed (not pending)")

    return {"status": "ok", "taskId": task_id, "priority": priority}


# =============================================================================
# Auth Endpoints
# =============================================================================

class LoginRequest(BaseModel):
    """Request model for login."""
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    """Response model for login."""
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    """Response model for user info."""
    id: int
    username: str
    created_at: datetime


@app.post("/auth/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    session: Annotated[AsyncSession, Depends(get_async_session)],
):
    """Login with username and password."""
    # Find user
    result = await session.execute(
        select(User).where(User.username == request.username)
    )
    user = result.scalar_one_or_none()

    if user is None or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password",
        )

    # Create token
    access_token = create_access_token(data={"sub": user.username})
    return TokenResponse(access_token=access_token)


@app.get("/auth/me", response_model=UserResponse)
async def get_me(
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Get current user info."""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        created_at=current_user.created_at,
    )


# =============================================================================
# Conversation Endpoints
# =============================================================================

class MessageResponse(BaseModel):
    """Response model for a message."""
    id: int
    role: str
    content: str
    metadata: Optional[dict] = None
    created_at: datetime


class ConversationResponse(BaseModel):
    """Response model for a conversation."""
    id: str
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse] = []


class ConversationListResponse(BaseModel):
    """Response model for conversation list."""
    conversations: List[ConversationResponse]
    total: int


class CreateConversationRequest(BaseModel):
    """Request model for creating a conversation."""
    id: str = Field(..., description="UUID for the conversation")
    title: Optional[str] = None


class UpdateConversationRequest(BaseModel):
    """Request model for updating a conversation."""
    title: Optional[str] = None


class AddMessageRequest(BaseModel):
    """Request model for adding a message."""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)
    metadata: Optional[dict] = None


@app.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
):
    """List all conversations for the current user."""
    chat_service = ChatService(session)
    conversations = await chat_service.list_conversations(current_user.id)
    return ConversationListResponse(
        conversations=[
            ConversationResponse(
                id=c.id,
                title=c.title,
                created_at=c.created_at,
                updated_at=c.updated_at,
            )
            for c in conversations
        ],
        total=len(conversations),
    )


@app.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    request: CreateConversationRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
):
    """Create a new conversation."""
    chat_service = ChatService(session)
    conversation = await chat_service.create_conversation(
        conversation_id=request.id,
        user_id=current_user.id,
        title=request.title,
    )
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
):
    """Get a conversation with its messages."""
    chat_service = ChatService(session)
    conversation = await chat_service.get_conversation(
        conversation_id=conversation_id,
        user_id=current_user.id,
        include_messages=True,
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[
            MessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                metadata=m.message_metadata,
                created_at=m.created_at,
            )
            for m in conversation.messages
        ],
    )


@app.put("/conversations/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: str,
    request: UpdateConversationRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
):
    """Update a conversation's title."""
    chat_service = ChatService(session)
    conversation = await chat_service.update_conversation(
        conversation_id=conversation_id,
        user_id=current_user.id,
        title=request.title,
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
):
    """Delete a conversation."""
    chat_service = ChatService(session)
    deleted = await chat_service.delete_conversation(
        conversation_id=conversation_id,
        user_id=current_user.id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"status": "ok", "message": "Conversation deleted"}


@app.post("/conversations/{conversation_id}/messages", response_model=MessageResponse)
async def add_message(
    conversation_id: str,
    request: AddMessageRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
):
    """Add a message to a conversation."""
    chat_service = ChatService(session)
    try:
        message = await chat_service.add_message(
            conversation_id=conversation_id,
            user_id=current_user.id,
            role=request.role,
            content=request.content,
            metadata=request.metadata,
        )
        return MessageResponse(
            id=message.id,
            role=message.role,
            content=message.content,
            metadata=message.message_metadata,
            created_at=message.created_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# =============================================================================
# Memory Endpoints
# =============================================================================

class MemoryResponse(BaseModel):
    """Response model for a memory fact."""
    id: int
    fact: str
    category: Optional[str] = None
    created_at: datetime


class MemoryListResponse(BaseModel):
    """Response model for memory list."""
    memories: List[MemoryResponse]
    total: int


class AddMemoryRequest(BaseModel):
    """Request model for adding a memory."""
    fact: str = Field(..., min_length=1)
    category: Optional[str] = None


@app.get("/memory", response_model=MemoryListResponse)
async def list_memories(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
    category: Optional[str] = None,
):
    """List all memory facts for the current user."""
    memory_service = MemoryService(session)
    memories = await memory_service.list_memories(current_user.id, category)
    return MemoryListResponse(
        memories=[
            MemoryResponse(
                id=m.id,
                fact=m.fact,
                category=m.category,
                created_at=m.created_at,
            )
            for m in memories
        ],
        total=len(memories),
    )


@app.post("/memory", response_model=MemoryResponse)
async def add_memory(
    request: AddMemoryRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
):
    """Add a new memory fact."""
    memory_service = MemoryService(session)
    memory = await memory_service.add_memory(
        user_id=current_user.id,
        fact=request.fact,
        category=request.category,
    )
    return MemoryResponse(
        id=memory.id,
        fact=memory.fact,
        category=memory.category,
        created_at=memory.created_at,
    )


@app.delete("/memory/{memory_id}")
async def delete_memory(
    memory_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
):
    """Delete a memory fact."""
    memory_service = MemoryService(session)
    deleted = await memory_service.delete_memory(memory_id, current_user.id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")

    return {"status": "ok", "message": "Memory deleted"}


@app.get("/memory/context")
async def get_memory_context(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
):
    """Get formatted memory context for LLM prompts."""
    memory_service = MemoryService(session)
    context = await memory_service.get_memory_context(current_user.id)
    return {"context": context}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
