import type {
  QueryResponse,
  HealthStatus,
  StatsResponse,
  Paper,
  PaperListResponse,
  UploadResponse,
  DeleteResponse,
  UploadProgressEvent,
  BatchUploadInitResponse,
  BatchStatusResponse,
  BatchUploadSSEEvent,
  UploadTask,
} from '../types';
import { getAuthToken } from '../context/AuthContext';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api';

// Helper to get auth headers
function getAuthHeaders(): HeadersInit {
  const token = getAuthToken();
  if (token) {
    return { Authorization: `Bearer ${token}` };
  }
  return {};
}

// Pipeline step types for streaming progress
export type PipelineStep =
  | 'rewriting'
  | 'entities'
  | 'classification'
  | 'expansion'
  | 'hyde'
  | 'retrieval'
  | 'reranking'
  | 'generation'
  | 'answer_chunk'
  | 'answer_complete'
  | 'citation_verified'
  | 'verification';

export interface ProgressEvent {
  type: 'progress';
  step: PipelineStep;
  data: Record<string, unknown>;
}

// New event types for streaming LLM response
export interface AnswerChunkEvent {
  type: 'progress';
  step: 'answer_chunk';
  data: {
    chunk: string;
  };
}

export interface AnswerCompleteEvent {
  type: 'progress';
  step: 'answer_complete';
  data: {
    answer: string;
  };
}

export interface CitationVerifiedEvent {
  type: 'progress';
  step: 'citation_verified';
  data: {
    citation_id: number;
    claim: string;
    confidence: number;
    is_valid: boolean;
    explanation: string;
  };
}

export interface CompleteEvent {
  type: 'complete';
  answer: string;
  sources: QueryResponse['sources'];
  question: string;
  query_type: string;
  expanded_query: string;
  retrieval_count: number;
  reranked_count: number;
  warnings: string[];
  citation_checks: QueryResponse['citation_checks'];
}

export interface ErrorEvent {
  type: 'error';
  message: string;
}

export type StreamEvent = ProgressEvent | AnswerChunkEvent | AnswerCompleteEvent | CitationVerifiedEvent | CompleteEvent | ErrorEvent;

class ApiError extends Error {
  status: number;
  details?: unknown;

  constructor(status: number, message: string, details?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    // Read as text first, then try to parse as JSON
    const text = await response.text();
    let details;
    try {
      details = JSON.parse(text);
    } catch {
      details = text;
    }
    throw new ApiError(
      response.status,
      details?.detail || `Request failed with status ${response.status}`,
      details
    );
  }
  return response.json();
}

// Query the research papers
export async function queryPapers(
  question: string,
  options?: {
    topK?: number;
    temperature?: number;
    paperIds?: string[];
    maxChunksPerPaper?: number | 'auto';
    conversationId?: string;
    queryType?: string;
    enableHyde?: boolean;
    enableExpansion?: boolean;
    enableCitationCheck?: boolean;
  }
): Promise<QueryResponse & { latency?: number }> {
  const startTime = performance.now();

  const response = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
    body: JSON.stringify({
      question,
      top_k: options?.topK ?? 10,
      temperature: options?.temperature ?? 0.7,
      paper_ids: options?.paperIds?.length ? options.paperIds : null,
      max_chunks_per_paper: options?.maxChunksPerPaper === 'auto' ? null : options?.maxChunksPerPaper,
      conversation_id: options?.conversationId ?? null,
      query_type: options?.queryType === 'auto' ? null : options?.queryType ?? null,
      enable_hyde: options?.enableHyde ?? null,
      enable_expansion: options?.enableExpansion ?? null,
      enable_citation_check: options?.enableCitationCheck ?? null,
    }),
  });

  const data = await handleResponse<QueryResponse>(response);

  // Add latency to the response
  const latency = performance.now() - startTime;
  return { ...data, latency };
}

// Stream query with real-time progress updates
export async function queryPapersStream(
  question: string,
  onProgress: (event: StreamEvent) => void,
  options?: {
    topK?: number;
    temperature?: number;
    paperIds?: string[];
    maxChunksPerPaper?: number | 'auto';
    conversationId?: string;
    queryType?: string;
    enableHyde?: boolean;
    enableExpansion?: boolean;
    enableCitationCheck?: boolean;
  }
): Promise<void> {
  const response = await fetch(`${API_BASE}/query/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
    body: JSON.stringify({
      question,
      top_k: options?.topK ?? 10,
      temperature: options?.temperature ?? 0.7,
      paper_ids: options?.paperIds?.length ? options.paperIds : null,
      max_chunks_per_paper: options?.maxChunksPerPaper === 'auto' ? null : options?.maxChunksPerPaper,
      conversation_id: options?.conversationId ?? null,
      query_type: options?.queryType === 'auto' ? null : options?.queryType ?? null,
      enable_hyde: options?.enableHyde ?? null,
      enable_expansion: options?.enableExpansion ?? null,
      enable_citation_check: options?.enableCitationCheck ?? null,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    let details;
    try {
      details = JSON.parse(text);
    } catch {
      details = text;
    }
    throw new ApiError(
      response.status,
      details?.detail || `Request failed with status ${response.status}`,
      details
    );
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('No response body');
  }

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();

    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Process complete SSE messages
    const lines = buffer.split('\n');
    buffer = lines.pop() || ''; // Keep incomplete line in buffer

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const jsonStr = line.slice(6).trim();
        if (jsonStr) {
          try {
            const event = JSON.parse(jsonStr) as StreamEvent;
            onProgress(event);
          } catch (e) {
            console.error('Failed to parse SSE event:', e, jsonStr);
          }
        }
      }
    }
  }
}

// Get health status
export async function getHealth(): Promise<HealthStatus> {
  const response = await fetch(`${API_BASE}/health`);
  return handleResponse<HealthStatus>(response);
}

// Quick health check
export async function getQuickHealth(): Promise<{ status: string }> {
  const response = await fetch(`${API_BASE}/health/quick`);
  return handleResponse<{ status: string }>(response);
}

// Get statistics
export async function getStats(): Promise<StatsResponse> {
  const response = await fetch(`${API_BASE}/stats`);
  return handleResponse<StatsResponse>(response);
}

// Clear conversation
export async function clearConversation(): Promise<{ status: string; message: string }> {
  const response = await fetch(`${API_BASE}/conversation/clear`, {
    method: 'POST',
  });
  return handleResponse<{ status: string; message: string }>(response);
}

// =============================================================================
// Paper Library API
// =============================================================================

// Helper to convert API response to Paper type
function apiPaperToPaper(apiPaper: {
  paper_id: string;
  title: string;
  authors: string[];
  year?: number;
  filename: string;
  page_count: number;
  chunk_count: number;
  chunk_stats: Record<string, number>;
  indexed_at?: string;
  status: string;
  error_message?: string;
  pdf_url: string;
}): Paper {
  return {
    id: apiPaper.paper_id,
    title: apiPaper.title,
    authors: apiPaper.authors,
    year: apiPaper.year,
    filename: apiPaper.filename,
    pageCount: apiPaper.page_count,
    chunkCount: apiPaper.chunk_count,
    chunkStats: apiPaper.chunk_stats,
    indexedAt: apiPaper.indexed_at ? new Date(apiPaper.indexed_at) : undefined,
    status: apiPaper.status as Paper['status'],
    errorMessage: apiPaper.error_message,
    pdfUrl: apiPaper.pdf_url,
  };
}

// Get papers with pagination
export async function getPapers(
  offset: number = 0,
  limit: number = 50
): Promise<PaperListResponse> {
  const params = new URLSearchParams({
    offset: offset.toString(),
    limit: limit.toString(),
  });
  const response = await fetch(`${API_BASE}/papers?${params}`);
  const data = await handleResponse<{
    papers: Array<{
      paper_id: string;
      title: string;
      authors: string[];
      year?: number;
      filename: string;
      page_count: number;
      chunk_count: number;
      chunk_stats: Record<string, number>;
      indexed_at?: string;
      status: string;
      error_message?: string;
      pdf_url: string;
    }>;
    total: number;
    offset: number;
    limit: number;
    has_more: boolean;
  }>(response);

  return {
    papers: data.papers.map(apiPaperToPaper),
    total: data.total,
    offset: data.offset,
    limit: data.limit,
    hasMore: data.has_more,
  };
}

// Get a single paper
export async function getPaper(paperId: string): Promise<Paper> {
  const response = await fetch(`${API_BASE}/papers/${paperId}`);
  const data = await handleResponse<{
    paper_id: string;
    title: string;
    authors: string[];
    year?: number;
    filename: string;
    page_count: number;
    chunk_count: number;
    chunk_stats: Record<string, number>;
    indexed_at?: string;
    status: string;
    error_message?: string;
    pdf_url: string;
  }>(response);

  return apiPaperToPaper(data);
}

// Get PDF URL for a paper
export function getPdfUrl(paperId: string): string {
  return `${API_BASE}/papers/${paperId}/pdf`;
}

// Delete a paper
export async function deletePaper(paperId: string): Promise<DeleteResponse> {
  const response = await fetch(`${API_BASE}/papers/${paperId}`, {
    method: 'DELETE',
  });
  return handleResponse<DeleteResponse>(response);
}

// Search result type
export interface PaperSearchResult {
  id: string;
  title: string;
  authors: string[];
  year?: number;
  filename: string;
  relevanceScore: number;
  chunkCount: number;
  status: string;
  pdfUrl: string;
  // Preview of best matching chunk
  previewText?: string;
  previewSection?: string;
  previewSubsection?: string;
  previewChunkType?: string;
}

// Semantic search for papers
export async function searchPapers(
  query: string,
  limit: number = 25,
  offset: number = 0
): Promise<{ results: PaperSearchResult[]; query: string; total: number; hasMore: boolean }> {
  const params = new URLSearchParams({
    q: query,
    limit: limit.toString(),
    offset: offset.toString(),
  });
  const response = await fetch(`${API_BASE}/papers/search?${params}`);
  const data = await handleResponse<{
    results: Array<{
      paper_id: string;
      title: string;
      authors: string[];
      year?: number;
      filename: string;
      relevance_score: number;
      chunk_count: number;
      status: string;
      pdf_url: string;
      preview_text?: string;
      preview_section?: string;
      preview_subsection?: string;
      preview_chunk_type?: string;
    }>;
    query: string;
    total: number;
    has_more: boolean;
  }>(response);

  return {
    results: data.results.map((r) => ({
      id: r.paper_id,
      title: r.title,
      authors: r.authors,
      year: r.year,
      filename: r.filename,
      relevanceScore: r.relevance_score,
      chunkCount: r.chunk_count,
      status: r.status,
      pdfUrl: r.pdf_url,
      previewText: r.preview_text,
      previewSection: r.preview_section,
      previewSubsection: r.preview_subsection,
      previewChunkType: r.preview_chunk_type,
    })),
    query: data.query,
    total: data.total,
    hasMore: data.has_more,
  };
}

// Upload a paper (synchronous)
export async function uploadPaper(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/papers/upload`, {
    method: 'POST',
    body: formData,
  });

  return handleResponse<UploadResponse>(response);
}

// Upload a paper with streaming progress
export async function uploadPaperStream(
  file: File,
  onProgress: (event: UploadProgressEvent) => void
): Promise<void> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/papers/upload/stream`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    let details;
    try {
      details = JSON.parse(text);
    } catch {
      details = text;
    }
    throw new ApiError(
      response.status,
      details?.detail || `Request failed with status ${response.status}`,
      details
    );
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('No response body');
  }

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();

    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Process complete SSE messages
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const jsonStr = line.slice(6).trim();
        if (jsonStr) {
          try {
            const event = JSON.parse(jsonStr) as UploadProgressEvent;
            onProgress(event);
          } catch (e) {
            console.error('Failed to parse SSE event:', e, jsonStr);
          }
        }
      }
    }
  }
}

// =============================================================================
// Duplicate Detection
// =============================================================================

export interface DuplicateInfo {
  hash: string;
  paper_id: string;
  title?: string;
}

export interface CheckDuplicatesResponse {
  duplicates: DuplicateInfo[];
  unique_count: number;
  duplicate_count: number;
}

// Compute SHA256 hash of a file using Web Crypto API
export async function computeFileHash(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

// Check which file hashes already exist in the library
export async function checkDuplicates(hashes: string[]): Promise<CheckDuplicatesResponse> {
  const response = await fetch(`${API_BASE}/papers/check-duplicates`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
    body: JSON.stringify({ hashes }),
  });
  return handleResponse<CheckDuplicatesResponse>(response);
}

// Backfill hashes for existing papers (admin operation)
export async function backfillPaperHashes(): Promise<{
  status: string;
  processed: number;
  skipped: number;
  failed: number;
  message: string;
}> {
  const response = await fetch(`${API_BASE}/papers/backfill-hashes`, {
    method: 'POST',
    headers: getAuthHeaders(),
  });
  return handleResponse(response);
}

// =============================================================================
// Batch Upload API
// =============================================================================

// Initialize batch upload - creates tasks for each file
export async function initBatchUpload(filenames: string[]): Promise<BatchUploadInitResponse> {
  const response = await fetch(`${API_BASE}/papers/upload/batch/init`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
    body: JSON.stringify({ filenames }),
  });
  return handleResponse<BatchUploadInitResponse>(response);
}

// Upload a single file for a batch task
export async function uploadBatchFile(
  batchId: string,
  taskId: string,
  file: File
): Promise<{ status: string; taskId: string; fileSize: number }> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(
    `${API_BASE}/papers/upload/batch/${batchId}/file?taskId=${encodeURIComponent(taskId)}`,
    {
      method: 'POST',
      headers: getAuthHeaders(),
      body: formData,
    }
  );
  return handleResponse(response);
}

// Start processing all uploaded files in a batch
export async function startBatchProcessing(
  batchId: string
): Promise<{ status: string; batchId: string; message: string }> {
  const response = await fetch(`${API_BASE}/papers/upload/batch/${batchId}/start`, {
    method: 'POST',
    headers: getAuthHeaders(),
  });
  return handleResponse(response);
}

// Get batch status (polling alternative to SSE)
export async function getBatchStatus(batchId: string): Promise<BatchStatusResponse> {
  const response = await fetch(`${API_BASE}/papers/upload/batch/${batchId}/status`, {
    headers: getAuthHeaders(),
  });
  return handleResponse<BatchStatusResponse>(response);
}

// Stream batch progress via SSE
export function streamBatchProgress(
  batchId: string,
  onEvent: (event: BatchUploadSSEEvent) => void,
  onError?: (error: Error) => void
): { close: () => void } {
  let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
  let isClosed = false;

  const startStream = async () => {
    try {
      const response = await fetch(`${API_BASE}/papers/upload/batch/${batchId}/stream`, {
        headers: getAuthHeaders(),
      });

      if (!response.ok) {
        throw new ApiError(response.status, `Stream failed with status ${response.status}`);
      }

      reader = response.body?.getReader() || null;
      if (!reader) {
        throw new Error('No response body');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (!isClosed) {
        const { done, value } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const jsonStr = line.slice(6).trim();
            if (jsonStr) {
              try {
                const event = JSON.parse(jsonStr) as BatchUploadSSEEvent;
                onEvent(event);

                // Close stream if batch is complete
                if (event.type === 'batch_complete') {
                  isClosed = true;
                  break;
                }
              } catch (e) {
                console.error('Failed to parse SSE event:', e, jsonStr);
              }
            }
          }
        }
      }
    } catch (error) {
      if (!isClosed && onError) {
        onError(error instanceof Error ? error : new Error(String(error)));
      }
    }
  };

  startStream();

  return {
    close: () => {
      isClosed = true;
      if (reader) {
        reader.cancel();
      }
    },
  };
}

// Cancel a pending upload task
export async function cancelUploadTask(
  batchId: string,
  taskId: string
): Promise<{ status: string; taskId: string; message: string }> {
  const response = await fetch(`${API_BASE}/papers/upload/batch/${batchId}/task/${taskId}`, {
    method: 'DELETE',
    headers: getAuthHeaders(),
  });
  return handleResponse(response);
}

// Retry a failed upload task
export async function retryUploadTask(
  batchId: string,
  taskId: string
): Promise<{ status: string; taskId: string; message: string }> {
  const response = await fetch(`${API_BASE}/papers/upload/batch/${batchId}/task/${taskId}/retry`, {
    method: 'POST',
    headers: getAuthHeaders(),
  });
  return handleResponse(response);
}

// Set priority of a pending task
export async function setUploadTaskPriority(
  batchId: string,
  taskId: string,
  priority: number
): Promise<{ status: string; taskId: string; priority: number }> {
  const response = await fetch(
    `${API_BASE}/papers/upload/batch/${batchId}/task/${taskId}/priority?priority=${priority}`,
    {
      method: 'PUT',
      headers: getAuthHeaders(),
    }
  );
  return handleResponse(response);
}

// =============================================================================
// Conversation API
// =============================================================================

export interface ConversationMessage {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  metadata?: Record<string, unknown>;
  created_at: string;
}

export interface Conversation {
  id: string;
  title?: string;
  created_at: string;
  updated_at: string;
  messages?: ConversationMessage[];
}

export async function getConversations(): Promise<{ conversations: Conversation[]; total: number }> {
  const response = await fetch(`${API_BASE}/conversations`, {
    headers: getAuthHeaders(),
  });
  return handleResponse(response);
}

export async function getConversation(id: string): Promise<Conversation> {
  const response = await fetch(`${API_BASE}/conversations/${id}`, {
    headers: getAuthHeaders(),
  });
  return handleResponse(response);
}

export async function createConversation(id: string, title?: string): Promise<Conversation> {
  const response = await fetch(`${API_BASE}/conversations`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
    body: JSON.stringify({ id, title }),
  });
  return handleResponse(response);
}

export async function updateConversation(id: string, title: string): Promise<Conversation> {
  const response = await fetch(`${API_BASE}/conversations/${id}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
    body: JSON.stringify({ title }),
  });
  return handleResponse(response);
}

export async function deleteConversation(id: string): Promise<void> {
  const response = await fetch(`${API_BASE}/conversations/${id}`, {
    method: 'DELETE',
    headers: getAuthHeaders(),
  });
  await handleResponse(response);
}

export async function addMessage(
  conversationId: string,
  role: 'user' | 'assistant',
  content: string,
  metadata?: Record<string, unknown>
): Promise<ConversationMessage> {
  const response = await fetch(`${API_BASE}/conversations/${conversationId}/messages`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
    body: JSON.stringify({ role, content, metadata }),
  });
  return handleResponse(response);
}

// =============================================================================
// Memory API
// =============================================================================

export interface Memory {
  id: number;
  fact: string;
  category?: string;
  created_at: string;
}

export async function getMemories(category?: string): Promise<{ memories: Memory[]; total: number }> {
  const params = category ? `?category=${encodeURIComponent(category)}` : '';
  const response = await fetch(`${API_BASE}/memory${params}`, {
    headers: getAuthHeaders(),
  });
  return handleResponse(response);
}

export async function addMemory(fact: string, category?: string): Promise<Memory> {
  const response = await fetch(`${API_BASE}/memory`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
    body: JSON.stringify({ fact, category }),
  });
  return handleResponse(response);
}

export async function deleteMemory(id: number): Promise<void> {
  const response = await fetch(`${API_BASE}/memory/${id}`, {
    method: 'DELETE',
    headers: getAuthHeaders(),
  });
  await handleResponse(response);
}

export async function getMemoryContext(): Promise<{ context: string }> {
  const response = await fetch(`${API_BASE}/memory/context`, {
    headers: getAuthHeaders(),
  });
  return handleResponse(response);
}

// Export all API functions
export const api = {
  queryPapers,
  queryPapersStream,
  getHealth,
  getQuickHealth,
  getStats,
  clearConversation,
  getPapers,
  getPaper,
  getPdfUrl,
  deletePaper,
  searchPapers,
  uploadPaper,
  uploadPaperStream,
  // Duplicate detection
  computeFileHash,
  checkDuplicates,
  backfillPaperHashes,
  // Batch upload
  initBatchUpload,
  uploadBatchFile,
  startBatchProcessing,
  getBatchStatus,
  streamBatchProgress,
  cancelUploadTask,
  retryUploadTask,
  setUploadTaskPriority,
  // Conversations
  getConversations,
  getConversation,
  createConversation,
  updateConversation,
  deleteConversation,
  addMessage,
  // Memory
  getMemories,
  addMemory,
  deleteMemory,
  getMemoryContext,
};

export { ApiError };
