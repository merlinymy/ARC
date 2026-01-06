import {
  createContext,
  useContext,
  useReducer,
  useCallback,
  useEffect,
  type ReactNode,
} from 'react';
import type {
  AppState,
  Conversation,
  Message,
  QueryOptions,
  QueryType,
  HealthStatus,
  StatsResponse,
  PipelineStepInfo,
  PipelineStepName,
  Paper,
  BatchUpload,
  UploadTask,
  BatchUploadSSEEvent,
  CitationCheck,
} from '../types';
import { DEFAULT_QUERY_OPTIONS, PIPELINE_STEPS } from '../types';
import { api, queryPapersStream, type StreamEvent } from '../services/api';

// Generate unique ID
const generateId = () => crypto.randomUUID();

// Check if we're on desktop (lg breakpoint = 1024px)
const isDesktop = typeof window !== 'undefined' && window.innerWidth >= 1024;

// Initial state
const initialState: AppState = {
  conversations: [],
  activeConversationId: null,
  papers: [],
  indexingQueue: [],
  totalPapers: 0,
  hasMorePapers: false,
  isLoadingMorePapers: false,
  currentQuery: '',
  queryOptions: DEFAULT_QUERY_OPTIONS,
  isLoading: false,
  streamingResponse: null,
  pipelineProgress: null,
  health: null,
  stats: null,
  sidebarOpen: isDesktop, // Open on desktop, closed on mobile
  theme: (typeof window !== 'undefined' && localStorage.getItem('theme') as 'light' | 'dark') || 'light',
  pipelineExpanded: false,
  activePage: 'chat',
  selectedPaperId: null,
  viewingPdfId: null,
  // Batch upload
  activeBatchUpload: null,
  isUploadPanelOpen: false,
  isUploadPanelMinimized: false,
  // Streaming state
  streamingState: null,
};

// Action types
type Action =
  | { type: 'SET_THEME'; payload: 'light' | 'dark' }
  | { type: 'TOGGLE_SIDEBAR' }
  | { type: 'SET_SIDEBAR_OPEN'; payload: boolean }
  | { type: 'SET_ACTIVE_PAGE'; payload: 'chat' | 'analytics' | 'settings' | 'library' }
  | { type: 'SET_CURRENT_QUERY'; payload: string }
  | { type: 'SET_QUERY_OPTIONS'; payload: Partial<QueryOptions> }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_HEALTH'; payload: HealthStatus | null }
  | { type: 'SET_STATS'; payload: StatsResponse | null }
  | { type: 'CREATE_CONVERSATION'; payload: Conversation }
  | { type: 'SET_ACTIVE_CONVERSATION'; payload: string | null }
  | { type: 'ADD_MESSAGE'; payload: { conversationId: string; message: Message } }
  | { type: 'UPDATE_CONVERSATION_TITLE'; payload: { id: string; title: string } }
  | { type: 'DELETE_CONVERSATION'; payload: string }
  | { type: 'CLEAR_CONVERSATIONS' }
  | { type: 'LOAD_CONVERSATIONS'; payload: Conversation[] }
  | { type: 'SET_CONVERSATION_MESSAGES'; payload: { id: string; messages: Message[] } }
  | { type: 'TOGGLE_PIPELINE_EXPANDED' }
  | { type: 'SET_SELECTED_PAPER'; payload: string | null }
  | { type: 'SET_PIPELINE_PROGRESS'; payload: PipelineStepInfo[] | null }
  | { type: 'UPDATE_PIPELINE_STEP'; payload: { step: PipelineStepName; status: PipelineStepInfo['status']; data?: Record<string, unknown> } }
  // Paper library actions
  | { type: 'SET_PAPERS'; payload: { papers: Paper[]; total: number; hasMore: boolean } }
  | { type: 'APPEND_PAPERS'; payload: { papers: Paper[]; hasMore: boolean } }
  | { type: 'SET_LOADING_MORE_PAPERS'; payload: boolean }
  | { type: 'ADD_PAPER'; payload: Paper }
  | { type: 'UPDATE_PAPER'; payload: { id: string; updates: Partial<Paper> } }
  | { type: 'REMOVE_PAPER'; payload: string }
  | { type: 'SET_VIEWING_PDF'; payload: string | null }
  // Batch upload actions
  | { type: 'START_BATCH_UPLOAD'; payload: BatchUpload }
  | { type: 'UPDATE_UPLOAD_TASK'; payload: { taskId: string; updates: Partial<UploadTask> } }
  | { type: 'SET_UPLOAD_PANEL_OPEN'; payload: boolean }
  | { type: 'SET_UPLOAD_PANEL_MINIMIZED'; payload: boolean }
  | { type: 'CANCEL_UPLOAD_TASK'; payload: string }
  | { type: 'CLEAR_BATCH_UPLOAD' }
  // Streaming actions
  | { type: 'START_STREAMING'; payload: { messageId: string; conversationId: string } }
  | { type: 'APPEND_STREAMING_CHUNK'; payload: string }
  | { type: 'ADD_STREAMING_CITATION'; payload: CitationCheck }
  | { type: 'STOP_STREAMING' }
  | { type: 'UPDATE_MESSAGE_CONTENT'; payload: { conversationId: string; messageId: string; content: string } }
  | { type: 'UPDATE_MESSAGE_CITATIONS'; payload: { conversationId: string; messageId: string; citationChecks: CitationCheck[] } }
  | { type: 'UPDATE_MESSAGE'; payload: { conversationId: string; message: Message } };

// Reducer
function appReducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'SET_THEME':
      localStorage.setItem('theme', action.payload);
      return { ...state, theme: action.payload };

    case 'TOGGLE_SIDEBAR':
      return { ...state, sidebarOpen: !state.sidebarOpen };

    case 'SET_SIDEBAR_OPEN':
      return { ...state, sidebarOpen: action.payload };

    case 'SET_ACTIVE_PAGE':
      return { ...state, activePage: action.payload };

    case 'SET_CURRENT_QUERY':
      return { ...state, currentQuery: action.payload };

    case 'SET_QUERY_OPTIONS':
      return { ...state, queryOptions: { ...state.queryOptions, ...action.payload } };

    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };

    case 'SET_HEALTH':
      return { ...state, health: action.payload };

    case 'SET_STATS':
      return { ...state, stats: action.payload };

    case 'CREATE_CONVERSATION':
      return {
        ...state,
        conversations: [action.payload, ...state.conversations],
        activeConversationId: action.payload.id,
      };

    case 'SET_ACTIVE_CONVERSATION':
      return { ...state, activeConversationId: action.payload };

    case 'ADD_MESSAGE': {
      return {
        ...state,
        conversations: state.conversations.map((conv) =>
          conv.id === action.payload.conversationId
            ? {
                ...conv,
                messages: [...conv.messages, action.payload.message],
                updatedAt: new Date(),
              }
            : conv
        ),
      };
    }

    case 'UPDATE_CONVERSATION_TITLE':
      return {
        ...state,
        conversations: state.conversations.map((conv) =>
          conv.id === action.payload.id
            ? { ...conv, title: action.payload.title }
            : conv
        ),
      };

    case 'DELETE_CONVERSATION':
      return {
        ...state,
        conversations: state.conversations.filter((c) => c.id !== action.payload),
        activeConversationId:
          state.activeConversationId === action.payload
            ? state.conversations[0]?.id ?? null
            : state.activeConversationId,
      };

    case 'CLEAR_CONVERSATIONS':
      return { ...state, conversations: [], activeConversationId: null };

    case 'LOAD_CONVERSATIONS':
      return {
        ...state,
        conversations: action.payload,
        activeConversationId: action.payload.length > 0 ? action.payload[0].id : null,
      };

    case 'SET_CONVERSATION_MESSAGES':
      return {
        ...state,
        conversations: state.conversations.map((conv) =>
          conv.id === action.payload.id
            ? { ...conv, messages: action.payload.messages }
            : conv
        ),
      };

    case 'TOGGLE_PIPELINE_EXPANDED':
      return { ...state, pipelineExpanded: !state.pipelineExpanded };

    case 'SET_SELECTED_PAPER':
      return { ...state, selectedPaperId: action.payload };

    case 'SET_PIPELINE_PROGRESS':
      return { ...state, pipelineProgress: action.payload };

    case 'UPDATE_PIPELINE_STEP':
      if (!state.pipelineProgress) return state;
      return {
        ...state,
        pipelineProgress: state.pipelineProgress.map((step) =>
          step.name === action.payload.step
            ? { ...step, status: action.payload.status, data: action.payload.data }
            : step
        ),
      };

    // Paper library reducers
    case 'SET_PAPERS':
      return {
        ...state,
        papers: action.payload.papers,
        totalPapers: action.payload.total,
        hasMorePapers: action.payload.hasMore,
      };

    case 'APPEND_PAPERS':
      return {
        ...state,
        papers: [...state.papers, ...action.payload.papers],
        hasMorePapers: action.payload.hasMore,
        isLoadingMorePapers: false,
      };

    case 'SET_LOADING_MORE_PAPERS':
      return { ...state, isLoadingMorePapers: action.payload };

    case 'ADD_PAPER':
      return { ...state, papers: [action.payload, ...state.papers] };

    case 'UPDATE_PAPER':
      return {
        ...state,
        papers: state.papers.map((p) =>
          p.id === action.payload.id ? { ...p, ...action.payload.updates } : p
        ),
      };

    case 'REMOVE_PAPER':
      return {
        ...state,
        papers: state.papers.filter((p) => p.id !== action.payload),
        viewingPdfId: state.viewingPdfId === action.payload ? null : state.viewingPdfId,
      };

    case 'SET_VIEWING_PDF':
      return { ...state, viewingPdfId: action.payload };

    // Batch upload reducers
    case 'START_BATCH_UPLOAD':
      return {
        ...state,
        activeBatchUpload: action.payload,
        isUploadPanelOpen: true,
        isUploadPanelMinimized: false,
      };

    case 'UPDATE_UPLOAD_TASK':
      if (!state.activeBatchUpload) return state;
      return {
        ...state,
        activeBatchUpload: {
          ...state.activeBatchUpload,
          tasks: state.activeBatchUpload.tasks.map((task) =>
            task.taskId === action.payload.taskId
              ? { ...task, ...action.payload.updates }
              : task
          ),
        },
      };

    case 'SET_UPLOAD_PANEL_OPEN':
      return { ...state, isUploadPanelOpen: action.payload };

    case 'SET_UPLOAD_PANEL_MINIMIZED':
      return { ...state, isUploadPanelMinimized: action.payload };

    case 'CANCEL_UPLOAD_TASK':
      if (!state.activeBatchUpload) return state;
      return {
        ...state,
        activeBatchUpload: {
          ...state.activeBatchUpload,
          tasks: state.activeBatchUpload.tasks.map((task) =>
            task.taskId === action.payload
              ? { ...task, status: 'error' as const, errorMessage: 'Cancelled' }
              : task
          ),
        },
      };

    case 'CLEAR_BATCH_UPLOAD':
      return {
        ...state,
        activeBatchUpload: null,
        isUploadPanelOpen: false,
        isUploadPanelMinimized: false,
      };

    // Streaming reducers
    case 'START_STREAMING':
      return {
        ...state,
        streamingState: {
          messageId: action.payload.messageId,
          conversationId: action.payload.conversationId,
          content: '',
          citationChecks: [],
          isStreaming: true,
        },
      };

    case 'APPEND_STREAMING_CHUNK':
      if (!state.streamingState) return state;
      return {
        ...state,
        streamingState: {
          ...state.streamingState,
          content: state.streamingState.content + action.payload,
        },
        // Also update the message in the conversation
        conversations: state.conversations.map((conv) =>
          conv.id === state.streamingState?.conversationId
            ? {
                ...conv,
                messages: conv.messages.map((msg) =>
                  msg.id === state.streamingState?.messageId
                    ? { ...msg, content: state.streamingState.content + action.payload }
                    : msg
                ),
              }
            : conv
        ),
      };

    case 'ADD_STREAMING_CITATION':
      if (!state.streamingState) return state;
      return {
        ...state,
        streamingState: {
          ...state.streamingState,
          citationChecks: [...state.streamingState.citationChecks, action.payload],
        },
        // Also update the message metadata in the conversation
        conversations: state.conversations.map((conv) =>
          conv.id === state.streamingState?.conversationId
            ? {
                ...conv,
                messages: conv.messages.map((msg) =>
                  msg.id === state.streamingState?.messageId
                    ? {
                        ...msg,
                        metadata: {
                          ...msg.metadata,
                          citationChecks: [...(state.streamingState?.citationChecks || []), action.payload],
                        },
                      }
                    : msg
                ),
              }
            : conv
        ),
      };

    case 'STOP_STREAMING':
      return {
        ...state,
        streamingState: null,
      };

    case 'UPDATE_MESSAGE_CONTENT':
      return {
        ...state,
        conversations: state.conversations.map((conv) =>
          conv.id === action.payload.conversationId
            ? {
                ...conv,
                messages: conv.messages.map((msg) =>
                  msg.id === action.payload.messageId
                    ? { ...msg, content: action.payload.content }
                    : msg
                ),
              }
            : conv
        ),
      };

    case 'UPDATE_MESSAGE_CITATIONS':
      return {
        ...state,
        conversations: state.conversations.map((conv) =>
          conv.id === action.payload.conversationId
            ? {
                ...conv,
                messages: conv.messages.map((msg) =>
                  msg.id === action.payload.messageId
                    ? {
                        ...msg,
                        metadata: {
                          ...msg.metadata,
                          citationChecks: action.payload.citationChecks,
                        },
                      }
                    : msg
                ),
              }
            : conv
        ),
      };

    case 'UPDATE_MESSAGE':
      return {
        ...state,
        conversations: state.conversations.map((conv) =>
          conv.id === action.payload.conversationId
            ? {
                ...conv,
                messages: conv.messages.map((msg) =>
                  msg.id === action.payload.message.id
                    ? action.payload.message
                    : msg
                ),
              }
            : conv
        ),
      };

    default:
      return state;
  }
}

// Context
interface AppContextValue {
  state: AppState;
  dispatch: React.Dispatch<Action>;
  // Convenience actions
  setTheme: (theme: 'light' | 'dark') => void;
  toggleSidebar: () => void;
  setActivePage: (page: 'chat' | 'analytics' | 'settings' | 'library') => void;
  createNewConversation: () => string;
  submitQuery: (query: string) => Promise<void>;
  clearConversation: () => Promise<void>;
  refreshHealth: () => Promise<void>;
  refreshStats: () => Promise<void>;
  // Conversation actions
  refreshConversations: () => Promise<void>;
  selectConversation: (id: string) => Promise<void>;
  // Paper library actions
  refreshPapers: () => Promise<void>;
  loadMorePapers: () => Promise<void>;
  deletePaper: (paperId: string) => Promise<void>;
  setViewingPdf: (paperId: string | null) => void;
  // Batch upload actions
  startBatchUpload: (files: File[]) => Promise<void>;
  cancelUploadTask: (taskId: string) => Promise<void>;
  retryUploadTask: (taskId: string) => Promise<void>;
  openUploadPanel: () => void;
  closeUploadPanel: () => void;
  minimizeUploadPanel: () => void;
  maximizeUploadPanel: () => void;
}

const AppContext = createContext<AppContextValue | null>(null);

// Provider component
export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  // Apply theme to document
  useEffect(() => {
    if (state.theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [state.theme]);

  // Convenience actions
  const setTheme = useCallback((theme: 'light' | 'dark') => {
    dispatch({ type: 'SET_THEME', payload: theme });
  }, []);

  const toggleSidebar = useCallback(() => {
    dispatch({ type: 'TOGGLE_SIDEBAR' });
  }, []);

  const setActivePage = useCallback((page: 'chat' | 'analytics' | 'settings' | 'library') => {
    dispatch({ type: 'SET_ACTIVE_PAGE', payload: page });
  }, []);

  const createNewConversation = useCallback(() => {
    const id = generateId();
    const conversation: Conversation = {
      id,
      title: 'New Conversation',
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    dispatch({ type: 'CREATE_CONVERSATION', payload: conversation });
    return id;
  }, []);

  const submitQuery = useCallback(async (query: string) => {
    if (!query.trim()) return;

    const startTime = performance.now();
    dispatch({ type: 'SET_LOADING', payload: true });

    // Initialize pipeline progress
    const initialProgress: PipelineStepInfo[] = PIPELINE_STEPS.map((step) => ({
      ...step,
      status: 'pending' as const,
    }));
    dispatch({ type: 'SET_PIPELINE_PROGRESS', payload: initialProgress });

    // Ensure we have an active conversation
    let conversationId = state.activeConversationId;
    if (!conversationId) {
      conversationId = generateId();
      const conversation: Conversation = {
        id: conversationId,
        title: query.slice(0, 50) + (query.length > 50 ? '...' : ''),
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      };
      dispatch({ type: 'CREATE_CONVERSATION', payload: conversation });
    }

    // Add query message
    const queryMessage: Message = {
      id: generateId(),
      type: 'query',
      content: query,
      timestamp: new Date(),
    };
    dispatch({
      type: 'ADD_MESSAGE',
      payload: { conversationId, message: queryMessage },
    });

    // Create a placeholder response message for streaming
    const responseMessageId = generateId();
    const placeholderMessage: Message = {
      id: responseMessageId,
      type: 'response',
      content: '',
      timestamp: new Date(),
      metadata: {
        citationChecks: [], // Will be populated as citations are verified
      },
    };
    dispatch({
      type: 'ADD_MESSAGE',
      payload: { conversationId, message: placeholderMessage },
    });

    // Start streaming state
    dispatch({
      type: 'START_STREAMING',
      payload: { messageId: responseMessageId, conversationId },
    });

    try {
      let completedSteps: Set<PipelineStepName> = new Set();
      let streamedContent = '';
      const streamedCitations: CitationCheck[] = [];

      await queryPapersStream(
        query,
        (event: StreamEvent) => {
          if (event.type === 'progress') {
            const stepName = event.step as PipelineStepName;
            // Cast data to Record for flexible property access
            const data = event.data as Record<string, unknown>;

            // Handle streaming answer chunks
            if (stepName === 'answer_chunk' && data?.chunk) {
              const chunk = data.chunk as string;
              streamedContent += chunk;
              dispatch({ type: 'APPEND_STREAMING_CHUNK', payload: chunk });
              return;
            }

            // Handle citation verification results
            if (stepName === 'citation_verified' && data) {
              const citationCheck: CitationCheck = {
                citation_id: data.citation_id as number,
                claim: data.claim as string,
                confidence: data.confidence as number,
                is_valid: data.is_valid as boolean,
                explanation: data.explanation as string,
              };
              streamedCitations.push(citationCheck);
              dispatch({ type: 'ADD_STREAMING_CITATION', payload: citationCheck });
              return;
            }

            // Skip answer_complete - we already have the content from chunks
            if (stepName === 'answer_complete') {
              return;
            }

            // Mark this step as active and previous steps as completed
            PIPELINE_STEPS.forEach((s) => {
              if (s.name === stepName) {
                dispatch({
                  type: 'UPDATE_PIPELINE_STEP',
                  payload: { step: s.name, status: 'active', data },
                });
              } else if (completedSteps.has(s.name)) {
                // Already completed
              } else {
                // Check if this step comes before the current one
                const currentIdx = PIPELINE_STEPS.findIndex((p) => p.name === stepName);
                const thisIdx = PIPELINE_STEPS.findIndex((p) => p.name === s.name);
                if (thisIdx < currentIdx) {
                  completedSteps.add(s.name);
                  dispatch({
                    type: 'UPDATE_PIPELINE_STEP',
                    payload: { step: s.name, status: data?.skipped ? 'skipped' : 'completed' },
                  });
                }
              }
            });

            // Mark step completed when we get data (not just "starting")
            if (data && data.status !== 'starting') {
              completedSteps.add(stepName);
              // Determine status: failed if success is explicitly false, skipped if marked, otherwise completed
              let status: 'completed' | 'skipped' | 'failed' = 'completed';
              if (data.success === false) {
                status = 'failed';
              } else if (data.skipped) {
                status = 'skipped';
              }
              dispatch({
                type: 'UPDATE_PIPELINE_STEP',
                payload: {
                  step: stepName,
                  status,
                  data
                },
              });
            }
          } else if (event.type === 'complete') {
            const latency = performance.now() - startTime;

            // Stop streaming state
            dispatch({ type: 'STOP_STREAMING' });

            // Mark all remaining steps as completed
            dispatch({ type: 'SET_PIPELINE_PROGRESS', payload: null });

            // Update the existing response message with final content and metadata
            // Use streamed content if available, otherwise fall back to event.answer
            const finalContent = streamedContent || event.answer;
            const finalCitations = streamedCitations.length > 0 ? streamedCitations : event.citation_checks;

            // Update the message with complete data
            dispatch({
              type: 'UPDATE_MESSAGE',
              payload: {
                conversationId: conversationId!,
                message: {
                  id: responseMessageId,
                  type: 'response',
                  content: finalContent,
                  timestamp: new Date(),
                  metadata: {
                    queryType: event.query_type as QueryType,
                    expandedQuery: event.expanded_query,
                    sources: event.sources,
                    retrievalCount: event.retrieval_count,
                    rerankedCount: event.reranked_count,
                    latency,
                    warnings: event.warnings,
                    citationChecks: finalCitations,
                  },
                },
              },
            });

            // Update conversation title if it's the first query
            const conversation = state.conversations.find((c) => c.id === conversationId);
            if (conversation?.title === 'New Conversation') {
              dispatch({
                type: 'UPDATE_CONVERSATION_TITLE',
                payload: { id: conversationId!, title: query.slice(0, 50) + (query.length > 50 ? '...' : '') },
              });
            }
          } else if (event.type === 'error') {
            dispatch({ type: 'STOP_STREAMING' });
            throw new Error(event.message);
          }
        },
        {
          topK: state.queryOptions.topK,
          temperature: state.queryOptions.temperature,
          paperIds: state.queryOptions.paperFilter,
          maxChunksPerPaper: state.queryOptions.maxChunksPerPaper,
          conversationId: conversationId,
          queryType: state.queryOptions.queryType,
          enableHyde: state.queryOptions.enableHyde,
          enableExpansion: state.queryOptions.enableExpansion,
          enableCitationCheck: state.queryOptions.enableCitationCheck,
        }
      );
    } catch (error) {
      console.error('Query failed:', error);
      // Add error message
      const errorMessage: Message = {
        id: generateId(),
        type: 'response',
        content: `Error: ${error instanceof Error ? error.message : 'Failed to process query'}`,
        timestamp: new Date(),
      };
      dispatch({
        type: 'ADD_MESSAGE',
        payload: { conversationId, message: errorMessage },
      });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
      dispatch({ type: 'SET_CURRENT_QUERY', payload: '' });
      dispatch({ type: 'SET_PIPELINE_PROGRESS', payload: null });
      // Refresh stats to update query count and cache hit rate
      try {
        const stats = await api.getStats();
        dispatch({ type: 'SET_STATS', payload: stats });
      } catch (error) {
        console.error('Failed to refresh stats:', error);
      }
    }
  }, [state.activeConversationId, state.conversations, state.queryOptions]);

  const clearConversation = useCallback(async () => {
    try {
      await api.clearConversation();
      if (state.activeConversationId) {
        dispatch({ type: 'DELETE_CONVERSATION', payload: state.activeConversationId });
      }
    } catch (error) {
      console.error('Failed to clear conversation:', error);
    }
  }, [state.activeConversationId]);

  const refreshHealth = useCallback(async () => {
    try {
      const health = await api.getHealth();
      dispatch({ type: 'SET_HEALTH', payload: health });
    } catch (error) {
      console.error('Failed to fetch health:', error);
      dispatch({
        type: 'SET_HEALTH',
        payload: { status: 'unhealthy', services: {} },
      });
    }
  }, []);

  const refreshStats = useCallback(async () => {
    try {
      const stats = await api.getStats();
      dispatch({ type: 'SET_STATS', payload: stats });
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  }, []);

  const refreshConversations = useCallback(async () => {
    try {
      const response = await api.getConversations();
      // Load conversations without messages first (fast)
      const conversations: Conversation[] = response.conversations.map((conv) => ({
        id: conv.id,
        title: conv.title || 'Untitled',
        messages: [],
        createdAt: new Date(conv.created_at),
        updatedAt: new Date(conv.updated_at),
      }));
      dispatch({ type: 'LOAD_CONVERSATIONS', payload: conversations });
    } catch (error) {
      // Silently fail if not authenticated - conversations will be local only
      console.debug('Failed to fetch conversations (user may not be authenticated):', error);
    }
  }, []);

  const selectConversation = useCallback(async (id: string) => {
    dispatch({ type: 'SET_ACTIVE_CONVERSATION', payload: id });

    // Check if messages are already loaded
    const conv = state.conversations.find((c) => c.id === id);
    if (conv && conv.messages.length > 0) {
      return; // Already loaded
    }

    // Fetch full conversation with messages
    try {
      const fullConv = await api.getConversation(id);
      const messages: Message[] = (fullConv.messages || []).map((msg) => ({
        id: String(msg.id),
        type: msg.role === 'user' ? 'query' : 'response',
        content: msg.content,
        timestamp: new Date(msg.created_at),
        metadata: msg.metadata as Message['metadata'],
      }));
      dispatch({ type: 'SET_CONVERSATION_MESSAGES', payload: { id, messages } });
    } catch (error) {
      console.error('Failed to load conversation messages:', error);
    }
  }, [state.conversations]);

  // Paper library actions
  const refreshPapers = useCallback(async () => {
    try {
      const response = await api.getPapers(0, 50);
      dispatch({
        type: 'SET_PAPERS',
        payload: {
          papers: response.papers,
          total: response.total,
          hasMore: response.hasMore,
        },
      });
    } catch (error) {
      console.error('Failed to fetch papers:', error);
    }
  }, []);

  const loadMorePapers = useCallback(async () => {
    if (state.isLoadingMorePapers || !state.hasMorePapers) return;

    dispatch({ type: 'SET_LOADING_MORE_PAPERS', payload: true });
    try {
      const offset = state.papers.length;
      const response = await api.getPapers(offset, 50);
      dispatch({
        type: 'APPEND_PAPERS',
        payload: {
          papers: response.papers,
          hasMore: response.hasMore,
        },
      });
    } catch (error) {
      console.error('Failed to load more papers:', error);
      dispatch({ type: 'SET_LOADING_MORE_PAPERS', payload: false });
    }
  }, [state.isLoadingMorePapers, state.hasMorePapers, state.papers.length]);

  const deletePaperAction = useCallback(async (paperId: string) => {
    try {
      await api.deletePaper(paperId);
      dispatch({ type: 'REMOVE_PAPER', payload: paperId });
      // Refresh stats after deletion
      refreshStats();
    } catch (error) {
      console.error('Failed to delete paper:', error);
      throw error;
    }
  }, [refreshStats]);

  const setViewingPdf = useCallback((paperId: string | null) => {
    dispatch({ type: 'SET_VIEWING_PDF', payload: paperId });
  }, []);

  // Batch upload actions
  const startBatchUpload = useCallback(async (files: File[]) => {
    if (files.length === 0) return;
    if (files.length > 20) {
      console.error('Maximum 20 files per batch');
      return;
    }

    try {
      // Step 1: Compute file hashes to check for duplicates
      console.log(`Computing hashes for ${files.length} files...`);
      const fileHashes = await Promise.all(
        files.map(async (file) => ({
          file,
          hash: await api.computeFileHash(file),
        }))
      );

      // Step 2: Check which hashes already exist
      const hashes = fileHashes.map((fh) => fh.hash);
      const duplicateCheck = await api.checkDuplicates(hashes);

      // Step 3: Filter out duplicates
      const duplicateHashes = new Set(duplicateCheck.duplicates.map((d) => d.hash));
      const uniqueFiles = fileHashes
        .filter((fh) => !duplicateHashes.has(fh.hash))
        .map((fh) => fh.file);

      // Log skipped duplicates
      if (duplicateCheck.duplicate_count > 0) {
        const skippedNames = duplicateCheck.duplicates
          .map((d) => d.title || 'Unknown')
          .join(', ');
        console.log(
          `Skipped ${duplicateCheck.duplicate_count} duplicate(s): ${skippedNames}`
        );
      }

      // If all files are duplicates, don't proceed
      if (uniqueFiles.length === 0) {
        console.log('All files are duplicates, nothing to upload');
        return;
      }

      console.log(`Uploading ${uniqueFiles.length} unique files (${duplicateCheck.duplicate_count} duplicates skipped)`);

      // Initialize batch on backend with only unique files
      const filenames = uniqueFiles.map((f) => f.name);
      const initResponse = await api.initBatchUpload(filenames);

      // Create batch upload state with file references
      const batch: BatchUpload = {
        batchId: initResponse.batchId,
        tasks: initResponse.tasks.map((task, index) => ({
          ...task,
          file: uniqueFiles[index],
        })),
        isMinimized: false,
        createdAt: new Date(),
        skippedDuplicates: duplicateCheck.duplicate_count,
      };

      dispatch({ type: 'START_BATCH_UPLOAD', payload: batch });

      // Add papers to library with pending status
      for (const task of batch.tasks) {
        const pendingPaper: Paper = {
          id: task.taskId, // Use task ID temporarily until we get paper ID
          title: task.filename.replace('.pdf', ''),
          authors: [],
          filename: task.filename,
          pageCount: 0,
          chunkCount: 0,
          chunkStats: {},
          status: 'pending',
          pdfUrl: '',
          progress: 0,
        };
        dispatch({ type: 'ADD_PAPER', payload: pendingPaper });
      }

      // Start SSE stream for progress updates
      api.streamBatchProgress(
        initResponse.batchId,
        (event: BatchUploadSSEEvent) => {
          if (event.type === 'task_progress' && event.taskId) {
            dispatch({
              type: 'UPDATE_UPLOAD_TASK',
              payload: {
                taskId: event.taskId,
                updates: {
                  status: (event.status as UploadTask['status']) || 'processing',
                  currentStep: event.currentStep,
                  progressPercent: event.progressPercent || 0,
                  paperId: event.paperId,
                },
              },
            });

            // Update paper in library
            dispatch({
              type: 'UPDATE_PAPER',
              payload: {
                id: event.taskId,
                updates: {
                  status: 'indexing',
                  progress: event.progressPercent,
                },
              },
            });
          } else if (event.type === 'task_complete' && event.taskId) {
            dispatch({
              type: 'UPDATE_UPLOAD_TASK',
              payload: {
                taskId: event.taskId,
                updates: {
                  status: 'complete',
                  progressPercent: 100,
                  paperId: event.paperId,
                },
              },
            });

            // Refresh papers to get the real paper data
            refreshPapers();
          } else if (event.type === 'task_error' && event.taskId) {
            dispatch({
              type: 'UPDATE_UPLOAD_TASK',
              payload: {
                taskId: event.taskId,
                updates: {
                  status: 'error',
                  errorMessage: event.errorMessage,
                },
              },
            });

            dispatch({
              type: 'UPDATE_PAPER',
              payload: {
                id: event.taskId,
                updates: {
                  status: 'error',
                  errorMessage: event.errorMessage,
                },
              },
            });
          } else if (event.type === 'batch_complete') {
            // Batch complete - refresh papers list
            refreshPapers();
            refreshStats();
          }
        },
        (error) => {
          console.error('Batch upload stream error:', error);
        }
      );

      // Upload files in parallel (2 at a time)
      const concurrency = 2;

      for (let i = 0; i < batch.tasks.length; i += concurrency) {
        const chunk = batch.tasks.slice(i, i + concurrency);
        const chunkPromises = chunk.map(async (task) => {
          if (!task.file) return;

          dispatch({
            type: 'UPDATE_UPLOAD_TASK',
            payload: {
              taskId: task.taskId,
              updates: { status: 'uploading' },
            },
          });

          try {
            await api.uploadBatchFile(batch.batchId, task.taskId, task.file);
          } catch (error) {
            console.error(`Failed to upload ${task.filename}:`, error);
            dispatch({
              type: 'UPDATE_UPLOAD_TASK',
              payload: {
                taskId: task.taskId,
                updates: {
                  status: 'error',
                  errorMessage: error instanceof Error ? error.message : 'Upload failed',
                },
              },
            });
          }
        });

        await Promise.all(chunkPromises);
      }

      // Start processing after all files are uploaded
      await api.startBatchProcessing(batch.batchId);

    } catch (error) {
      console.error('Failed to start batch upload:', error);
    }
  }, [refreshPapers, refreshStats]);

  const cancelUploadTaskAction = useCallback(async (taskId: string) => {
    const batch = state.activeBatchUpload;
    if (!batch) return;

    try {
      await api.cancelUploadTask(batch.batchId, taskId);
      dispatch({ type: 'CANCEL_UPLOAD_TASK', payload: taskId });
    } catch (error) {
      console.error('Failed to cancel upload task:', error);
    }
  }, [state.activeBatchUpload]);

  const retryUploadTaskAction = useCallback(async (taskId: string) => {
    const batch = state.activeBatchUpload;
    if (!batch) return;

    try {
      await api.retryUploadTask(batch.batchId, taskId);
      dispatch({
        type: 'UPDATE_UPLOAD_TASK',
        payload: {
          taskId,
          updates: {
            status: 'pending',
            errorMessage: undefined,
            progressPercent: 0,
          },
        },
      });
    } catch (error) {
      console.error('Failed to retry upload task:', error);
    }
  }, [state.activeBatchUpload]);

  const openUploadPanel = useCallback(() => {
    dispatch({ type: 'SET_UPLOAD_PANEL_OPEN', payload: true });
    dispatch({ type: 'SET_UPLOAD_PANEL_MINIMIZED', payload: false });
  }, []);

  const closeUploadPanel = useCallback(() => {
    dispatch({ type: 'SET_UPLOAD_PANEL_OPEN', payload: false });
  }, []);

  const minimizeUploadPanel = useCallback(() => {
    dispatch({ type: 'SET_UPLOAD_PANEL_MINIMIZED', payload: true });
  }, []);

  const maximizeUploadPanel = useCallback(() => {
    dispatch({ type: 'SET_UPLOAD_PANEL_MINIMIZED', payload: false });
  }, []);

  // Initial data load - run once when site opens
  useEffect(() => {
    refreshHealth();
    refreshStats();
    refreshPapers();
    refreshConversations();
  }, [refreshHealth, refreshStats, refreshPapers, refreshConversations]);

  const value: AppContextValue = {
    state,
    dispatch,
    setTheme,
    toggleSidebar,
    setActivePage,
    createNewConversation,
    submitQuery,
    clearConversation,
    refreshHealth,
    refreshStats,
    refreshConversations,
    selectConversation,
    refreshPapers,
    loadMorePapers,
    deletePaper: deletePaperAction,
    setViewingPdf,
    // Batch upload
    startBatchUpload,
    cancelUploadTask: cancelUploadTaskAction,
    retryUploadTask: retryUploadTaskAction,
    openUploadPanel,
    closeUploadPanel,
    minimizeUploadPanel,
    maximizeUploadPanel,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

// Custom hook to use the context
export function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
}
