import { useEffect, useRef, useCallback } from 'react';
import { MessageSquare, Sparkles, Check, Loader2, SkipForward, AlertTriangle } from 'lucide-react';
import { useApp } from '../context/AppContext';
import { ResponseCard } from './ResponseCard';
import type { PipelineStepInfo } from '../types';

// Threshold in pixels - if user is within this distance from bottom, auto-scroll
const SCROLL_THRESHOLD = 150;

// Pipeline step status icon
function StepIcon({ status }: { status: PipelineStepInfo['status'] }) {
  switch (status) {
    case 'completed':
      return <Check className="w-3.5 h-3.5 text-green-500" />;
    case 'active':
      return <Loader2 className="w-3.5 h-3.5 text-blue-500 animate-spin" />;
    case 'skipped':
      return <SkipForward className="w-3.5 h-3.5 text-gray-400" />;
    case 'failed':
      return <AlertTriangle className="w-3.5 h-3.5 text-amber-500" />;
    default:
      return <div className="w-3.5 h-3.5 rounded-full border-2 border-gray-300 dark:border-gray-600" />;
  }
}

// Pipeline progress display component
function PipelineProgress({ steps }: { steps: PipelineStepInfo[] }) {
  const activeStep = steps.find((s) => s.status === 'active');

  return (
    <div className="p-4 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 animate-fade-in">
      {/* Current step indicator */}
      {activeStep && (
        <div className="flex items-center gap-3 mb-4 pb-4 border-b border-gray-100 dark:border-gray-700">
          <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
          <div>
            <div className="font-medium text-gray-900 dark:text-gray-100">
              {activeStep.label}
            </div>
            <div className="text-sm text-gray-500 dark:text-gray-400">
              {activeStep.description}
            </div>
          </div>
        </div>
      )}

      {/* Steps list */}
      <div className="grid grid-cols-3 sm:grid-cols-5 gap-2">
        {steps.map((step) => (
          <div
            key={step.name}
            className={`flex flex-col items-center p-2 rounded-lg transition-colors ${
              step.status === 'active'
                ? 'bg-blue-50 dark:bg-blue-900/20'
                : step.status === 'completed'
                ? 'bg-green-50 dark:bg-green-900/10'
                : step.status === 'failed'
                ? 'bg-amber-50 dark:bg-amber-900/20'
                : step.status === 'skipped'
                ? 'bg-gray-50 dark:bg-gray-800/50'
                : ''
            }`}
          >
            <StepIcon status={step.status} />
            <span
              className={`text-xs mt-1 text-center ${
                step.status === 'active'
                  ? 'text-blue-600 dark:text-blue-400 font-medium'
                  : step.status === 'completed'
                  ? 'text-green-600 dark:text-green-400'
                  : step.status === 'failed'
                  ? 'text-amber-600 dark:text-amber-400'
                  : step.status === 'skipped'
                  ? 'text-gray-500 dark:text-gray-400'
                  : 'text-gray-500 dark:text-gray-400'
              }`}
            >
              {step.label}
            </span>
            {step.status === 'skipped' && (
              <span className="text-xs text-gray-400 dark:text-gray-500">
                skipped
              </span>
            )}
            {step.status === 'failed' && (
              <span className="text-xs text-amber-500 dark:text-amber-400">
                failed
              </span>
            )}
            {/* Show step data if available */}
            {step.data && step.status === 'completed' && (
              <span className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">
                {typeof step.data.count === 'number' && `${step.data.count}`}
                {typeof step.data.type === 'string' && `${step.data.type}`}
                {Array.isArray(step.data.found) && step.data.found.length > 0 && `${step.data.found.length}`}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export function ConversationThread() {
  const { state } = useApp();
  const { conversations, activeConversationId, isLoading, pipelineProgress, streamingState } = state;
  const scrollRef = useRef<HTMLDivElement>(null);
  const isNearBottomRef = useRef(true);
  const lastMessageCountRef = useRef(0);

  const activeConversation = conversations.find(
    (c) => c.id === activeConversationId
  );

  // Check if user is near bottom of scroll container
  const checkIfNearBottom = useCallback(() => {
    if (!scrollRef.current) return true;
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    return scrollHeight - scrollTop - clientHeight < SCROLL_THRESHOLD;
  }, []);

  // Track scroll position to know if user scrolled up
  useEffect(() => {
    const container = scrollRef.current;
    if (!container) return;

    const handleScroll = () => {
      isNearBottomRef.current = checkIfNearBottom();
    };

    container.addEventListener('scroll', handleScroll, { passive: true });
    return () => container.removeEventListener('scroll', handleScroll);
  }, [checkIfNearBottom]);

  // Auto-scroll only when a new message is added (not during streaming content updates)
  useEffect(() => {
    const messageCount = activeConversation?.messages.length ?? 0;
    const isNewMessage = messageCount > lastMessageCountRef.current;
    lastMessageCountRef.current = messageCount;

    // Always scroll to bottom when a new message is added
    if (isNewMessage && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      isNearBottomRef.current = true;
    }
  }, [activeConversation?.messages.length]);

  // During streaming, only scroll if user was already near bottom
  useEffect(() => {
    if (streamingState?.isStreaming && isNearBottomRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [streamingState?.content]);

  // Empty state
  if (!activeConversation || activeConversation.messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="text-center max-w-md">
          <div className="w-16 h-16 mx-auto mb-6 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center">
            <Sparkles className="w-8 h-8 text-white" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-3">
            A'Lester's Research Chatbot
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            Ask questions about your indexed research papers. Get accurate
            answers with source citations and full RAG pipeline transparency.
          </p>
        </div>
      </div>
    );
  }

  // Group messages into query-response pairs
  const messagePairs: { query: typeof activeConversation.messages[0]; response: typeof activeConversation.messages[0] | null }[] =
    [];
  for (let i = 0; i < activeConversation.messages.length; i += 2) {
    const query = activeConversation.messages[i];
    const response = activeConversation.messages[i + 1] ?? null;
    if (query.type === 'query') {
      messagePairs.push({ query, response });
    }
  }

  return (
    <div ref={scrollRef} className="h-full overflow-y-auto p-4 md:p-6">
      <div className="max-w-4xl mx-auto space-y-8">
        {messagePairs.map(({ query, response }) => (
          <div key={query.id}>
            {response ? (
              <ResponseCard queryMessage={query} responseMessage={response} />
            ) : (
              /* Loading state for pending response with pipeline progress */
              <div className="animate-fade-in">
                <div className="mb-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-xl border border-blue-200 dark:border-blue-800">
                  <div className="flex items-start gap-3">
                    <div className="p-2 bg-blue-100 dark:bg-blue-800 rounded-lg">
                      <MessageSquare className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                    </div>
                    <p className="text-gray-900 dark:text-gray-100 font-medium">
                      {query.content}
                    </p>
                  </div>
                </div>

                {/* Pipeline progress or simple loading */}
                {pipelineProgress ? (
                  <PipelineProgress steps={pipelineProgress} />
                ) : (
                  <div className="p-4 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
                    <div className="flex items-center gap-3">
                      <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                      <span className="text-gray-600 dark:text-gray-400">
                        Processing your query...
                      </span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}

        {/* Show loading state when isLoading but no pending message */}
        {isLoading && messagePairs.length > 0 && messagePairs[messagePairs.length - 1].response && (
          pipelineProgress ? (
            <PipelineProgress steps={pipelineProgress} />
          ) : (
            <div className="p-4 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 animate-fade-in">
              <div className="flex items-center gap-3">
                <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                <span className="text-gray-600 dark:text-gray-400">
                  Processing your query...
                </span>
              </div>
            </div>
          )
        )}
      </div>
    </div>
  );
}
