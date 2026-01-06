// Query Types matching backend classification
export type QueryType =
  | 'factual'
  | 'framing'
  | 'methods'
  | 'summary'
  | 'comparative'
  | 'novelty'
  | 'limitations'
  | 'general';

// Chunk types from the document
export type ChunkType = 'abstract' | 'section' | 'fine' | 'table' | 'caption' | 'full';

// Entity types
export interface Entity {
  text: string;
  type: 'chemical' | 'protein' | 'method' | 'organism' | 'metric' | 'other';
}

// Source from the backend
export interface Source {
  paper_title: string;
  paper_id: string;
  section_name: string | null;
  subsection_name: string | null;
  chunk_type: ChunkType;
  chunk_text: string;
  relevance_score: number;
}

// Citation verification check result
export interface CitationCheck {
  citation_id: number;
  claim: string;
  confidence: number;
  is_valid: boolean;
  explanation: string;
}

// Query response from backend
export interface QueryResponse {
  answer: string;
  sources: Source[];
  question: string;
  query_type: string;
  expanded_query: string;
  retrieval_count: number;
  reranked_count: number;
  warnings: string[];
  citation_checks: CitationCheck[];
}

// Message in a conversation
export interface Message {
  id: string;
  type: 'query' | 'response';
  content: string;
  timestamp: Date;
  metadata?: {
    queryType?: QueryType;
    expandedQuery?: string;
    entities?: Entity[];
    sources?: Source[];
    citationScore?: number;
    retrievalCount?: number;
    rerankedCount?: number;
    latency?: number;
    warnings?: string[];
    citationChecks?: CitationCheck[];
  };
}

// Conversation
export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

// Paper in the library
export interface Paper {
  id: string;
  title: string;
  authors: string[];
  year?: number;
  filename: string;
  pageCount: number;
  indexedAt?: Date;
  chunkCount: number;
  chunkStats: Record<string, number>;
  status: 'indexed' | 'indexing' | 'error' | 'pending';
  errorMessage?: string;
  pdfUrl: string;
  progress?: number;
}

// Paper list response from API with pagination
export interface PaperListResponse {
  papers: Paper[];
  total: number;
  offset: number;
  limit: number;
  hasMore: boolean;
}

// Upload progress event for SSE
export interface UploadProgressEvent {
  type: 'progress' | 'complete' | 'error';
  step?: string;
  data?: Record<string, unknown>;
  message?: string;
  paper_id?: string;
  filename?: string;
  status?: string;
  chunks?: number;
}

// Upload response
export interface UploadResponse {
  paper_id: string;
  filename: string;
  status: string;
  message: string;
  chunks: number;
}

// Batch upload types
export type UploadTaskStatus =
  | 'pending'
  | 'uploading'
  | 'processing'
  | 'extracting'
  | 'embedding'
  | 'indexing'
  | 'complete'
  | 'error';

export interface UploadTask {
  taskId: string;
  batchId: string;
  filename: string;
  paperId?: string;
  status: UploadTaskStatus;
  currentStep?: string;
  progressPercent: number;
  errorMessage?: string;
  priority: number;
  fileSize: number;
  createdAt?: string;
  file?: File; // Local reference before upload
}

export interface BatchUpload {
  batchId: string;
  tasks: UploadTask[];
  isMinimized: boolean;
  createdAt: Date;
  skippedDuplicates?: number;
}

export interface BatchUploadInitResponse {
  batchId: string;
  tasks: UploadTask[];
}

export interface BatchStatusResponse {
  batchId: string;
  tasks: UploadTask[];
  total: number;
  completed: number;
  failed: number;
  inProgress: number;
  pending: number;
}

// SSE events for batch upload
export interface BatchUploadSSEEvent {
  type: 'status' | 'task_progress' | 'task_complete' | 'task_error' | 'batch_complete';
  taskId?: string;
  batchId?: string;
  status?: string;
  currentStep?: string;
  progressPercent?: number;
  paperId?: string;
  chunks?: number;
  errorMessage?: string;
  succeeded?: number;
  failed?: number;
  total?: number;
  // For initial status event
  tasks?: UploadTask[];
  completed?: number;
  inProgress?: number;
  pending?: number;
}

// Delete response
export interface DeleteResponse {
  paper_id: string;
  pdf_deleted: boolean;
  chunks_deleted: number;
  message: string;
}

// Query options for advanced settings
export interface QueryOptions {
  queryType: QueryType | 'auto';
  topK: number;
  temperature: number;
  paperFilter: string[];
  sectionFilter: string | null;
  enableHyde: boolean;
  enableExpansion: boolean;
  enableCitationCheck: boolean;
  maxChunksPerPaper: number | 'auto'; // 'auto' lets system decide based on context
}

// Health status for services
export interface ServiceHealth {
  status: 'healthy' | 'unhealthy' | 'degraded';
  connected?: boolean;
  collection_exists?: boolean;
  total_vectors?: number;
  error?: string;
}

export interface HealthStatus {
  status: 'healthy' | 'unhealthy' | 'degraded';
  services: {
    qdrant?: ServiceHealth;
    voyage_ai?: ServiceHealth;
    cohere?: ServiceHealth;
    anthropic?: ServiceHealth;
  };
}

// Cache statistics
export interface CacheStats {
  embedding_cache?: {
    hits: number;
    misses: number;
    size: number;
    max_size: number;
  };
  search_cache?: {
    hits: number;
    misses: number;
    size: number;
    max_size: number;
  };
  hyde_cache?: {
    hits: number;
    misses: number;
    size: number;
    max_size: number;
  };
}

// Analytics statistics for dashboard
export interface EntityStat {
  name: string;
  count: number;
}

export interface AnalyticsStats {
  query_type_distribution: Record<string, number>;
  citation_stats: {
    avg_score: number;
    verified_rate: number;
    partial_rate: number;
    failed_rate: number;
    total_checked: number;
  };
  latency_stats: {
    query_processing_ms: number;
    embedding_ms: number;
    retrieval_ms: number;
    reranking_ms: number;
    generation_ms: number;
    total_avg_ms: number;
  };
  entity_stats: {
    chemicals: EntityStat[];
    proteins: EntityStat[];
    methods: EntityStat[];
    organisms: EntityStat[];
    metrics: EntityStat[];
  };
  total_queries: number;
}

// Statistics response
export interface StatsResponse {
  collection_name: string;
  total_vectors: number;
  vector_dimension: number;
  embedding_model: string;
  llm_model: string;
  cache_stats: CacheStats;
  conversation_stats: {
    turns: number;
    papers_discussed: string[];
  };
  analytics?: AnalyticsStats;
}

// Pipeline step for visualization
export interface PipelineStep {
  step: number;
  name: string;
  duration: number;
  cached?: boolean;
  details?: string;
  status: 'completed' | 'skipped' | 'pending' | 'failed';
  error?: string;
}

// Pipeline step names for streaming progress
export type PipelineStepName =
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

// Streaming citation verification result
export interface StreamingCitationCheck {
  citation_id: number;
  claim: string;
  confidence: number;
  is_valid: boolean;
  explanation: string;
}

// Step info for display
export interface PipelineStepInfo {
  name: PipelineStepName;
  label: string;
  description: string;
  status: 'pending' | 'active' | 'completed' | 'skipped' | 'failed';
  data?: Record<string, unknown>;
  error?: string;
}

// All pipeline steps for streaming display
export const PIPELINE_STEPS: Array<{ name: PipelineStepName; label: string; description: string }> = [
  { name: 'rewriting', label: 'Query Rewriting', description: 'Correcting spelling and formatting' },
  { name: 'entities', label: 'Entity Extraction', description: 'Identifying key terms' },
  { name: 'classification', label: 'Classification', description: 'Determining query type' },
  { name: 'expansion', label: 'Query Expansion', description: 'Adding synonyms' },
  { name: 'hyde', label: 'HyDE Embedding', description: 'Generating hypothetical document' },
  { name: 'retrieval', label: 'Retrieval', description: 'Searching documents' },
  { name: 'reranking', label: 'Reranking', description: 'Ranking by relevance' },
  { name: 'generation', label: 'Generation', description: 'Creating answer' },
  { name: 'verification', label: 'Verification', description: 'Checking citations' },
];

// Streaming state for progressive response building
export interface StreamingState {
  messageId: string;
  conversationId: string;
  content: string;
  citationChecks: CitationCheck[];
  isStreaming: boolean;
}

// App state
export interface AppState {
  // Conversations
  conversations: Conversation[];
  activeConversationId: string | null;

  // Papers
  papers: Paper[];
  indexingQueue: Paper[];
  // Pagination state for papers
  totalPapers: number;
  hasMorePapers: boolean;
  isLoadingMorePapers: boolean;

  // Query state
  currentQuery: string;
  queryOptions: QueryOptions;
  isLoading: boolean;
  streamingResponse: string | null;
  pipelineProgress: PipelineStepInfo[] | null;

  // System
  health: HealthStatus | null;
  stats: StatsResponse | null;

  // UI
  sidebarOpen: boolean;
  theme: 'light' | 'dark';
  pipelineExpanded: boolean;
  activePage: 'chat' | 'analytics' | 'settings' | 'library';
  selectedPaperId: string | null;
  viewingPdfId: string | null;

  // Batch upload
  activeBatchUpload: BatchUpload | null;
  isUploadPanelOpen: boolean;
  isUploadPanelMinimized: boolean;

  // Streaming state for LLM response
  streamingState: StreamingState | null;
}

// Default query options
export const DEFAULT_QUERY_OPTIONS: QueryOptions = {
  queryType: 'auto',
  topK: 15,
  temperature: 0.3,
  paperFilter: [],
  sectionFilter: null,
  enableHyde: true,
  enableExpansion: true,
  enableCitationCheck: true,
  maxChunksPerPaper: 'auto',
};
