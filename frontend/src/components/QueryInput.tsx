import { useState, useRef, useEffect, type KeyboardEvent } from 'react';
import {
  Send,
  ChevronDown,
  ChevronUp,
  Sparkles,
  Zap,
  Shield,
  Loader2,
  Trash2,
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { InfoTooltip } from './Tooltip';
import type { QueryType } from '../types';

const QUERY_TYPES: { value: QueryType | 'auto'; label: string }[] = [
  { value: 'auto', label: 'Auto-detect (Recommended)' },
  { value: 'factual', label: 'Factual' },
  { value: 'framing', label: 'Framing' },
  { value: 'methods', label: 'Methods' },
  { value: 'summary', label: 'Summary' },
  { value: 'comparative', label: 'Comparative' },
  { value: 'novelty', label: 'Novelty' },
  { value: 'limitations', label: 'Limitations' },
  { value: 'general', label: 'General' },
];

const TOP_K_OPTIONS = [5, 10, 15, 20, 30, 50];
const TEMPERATURE_OPTIONS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0];

export function QueryInput() {
  const { state, dispatch, submitQuery, clearConversation } = useApp();
  const { currentQuery, queryOptions, isLoading } = state;
  const [showAdvanced, setShowAdvanced] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea with a max of 104px (about 4 lines)
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = '32px';
      const scrollHeight = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = `${Math.min(scrollHeight, 104)}px`;
    }
  }, [currentQuery]);

  const handleSubmit = () => {
    if (currentQuery.trim() && !isLoading) {
      submitQuery(currentQuery);
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 p-4 max-h-[60vh] overflow-y-auto shrink-0">
      <div className="max-w-4xl mx-auto">
        {/* Main input area */}
        <div className="flex items-center gap-2 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-xl bg-white dark:bg-gray-800 focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-transparent">
          <textarea
            ref={textareaRef}
            value={currentQuery}
            onChange={(e) =>
              dispatch({ type: 'SET_CURRENT_QUERY', payload: e.target.value })
            }
            onKeyDown={handleKeyDown}
            placeholder="Ask about your research papers..."
            className="flex-1 py-1 bg-transparent text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none resize-none min-h-8 max-h-26 scrollbar-none"
            rows={1}
            disabled={isLoading}
          />
          <div className="flex items-center gap-1 shrink-0">
            <span className="text-xs text-gray-400 dark:text-gray-500 mr-2 hidden sm:inline">
              {navigator.platform.includes('Mac') ? '⌘' : 'Ctrl'}+Enter
            </span>
            <button
              onClick={handleSubmit}
              disabled={!currentQuery.trim() || isLoading}
              className="p-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 dark:disabled:bg-gray-600 text-white rounded-lg transition-colors disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>
        </div>

        {/* Advanced options toggle */}
        <div className="flex items-center justify-between mt-3">
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
          >
            {showAdvanced ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
            Advanced Options
          </button>

          <button
            onClick={clearConversation}
            className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 hover:text-red-600 dark:hover:text-red-400 transition-colors"
          >
            <Trash2 className="w-4 h-4" />
            Clear Conversation
          </button>
        </div>

        {/* Advanced options panel */}
        {showAdvanced && (
          <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 animate-fade-in">
            {/* Query Type */}
            <div className="mb-4">
              <label className="flex items-center gap-1.5 text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Query Type
                <InfoTooltip content="Select how the system should interpret your question. Auto-detect analyzes your query to choose the best approach automatically." />
              </label>
              <div className="flex flex-wrap gap-2">
                {QUERY_TYPES.map(({ value, label }) => (
                  <button
                    key={value}
                    onClick={() =>
                      dispatch({ type: 'SET_QUERY_OPTIONS', payload: { queryType: value } })
                    }
                    className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                      queryOptions.queryType === value
                        ? 'bg-blue-600 text-white'
                        : 'bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 hover:border-blue-400'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            {/* Parameters */}
            <div className="grid grid-cols-3 gap-4 mb-4">
              <div>
                <label className="flex items-center gap-1.5 text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Results (Top-K)
                  <InfoTooltip content="Number of document chunks to retrieve. Higher values provide more context but may include less relevant results. Recommended: 10-20 for focused queries, 30+ for comprehensive research." />
                </label>
                <select
                  value={queryOptions.topK}
                  onChange={(e) =>
                    dispatch({
                      type: 'SET_QUERY_OPTIONS',
                      payload: { topK: parseInt(e.target.value) },
                    })
                  }
                  className="w-full px-3 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {TOP_K_OPTIONS.map((k) => (
                    <option key={k} value={k}>
                      {k}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="flex items-center gap-1.5 text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Temperature
                  <InfoTooltip content="Controls response creativity. Lower values (0.0-0.3) give precise, factual answers. Higher values (0.5-1.0) allow more creative, varied responses. Use low for citations, higher for summaries." />
                </label>
                <select
                  value={queryOptions.temperature}
                  onChange={(e) =>
                    dispatch({
                      type: 'SET_QUERY_OPTIONS',
                      payload: { temperature: parseFloat(e.target.value) },
                    })
                  }
                  className="w-full px-3 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {TEMPERATURE_OPTIONS.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="flex items-center gap-1.5 text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Chunks/Paper
                  <InfoTooltip content="Maximum chunks per paper in results. Auto: System decides based on context (more for single-paper queries). Low (3-5): Better diversity across papers, faster responses. High (15-25): Deeper analysis of each paper, more comprehensive but may include redundant info." />
                </label>
                <select
                  value={queryOptions.maxChunksPerPaper}
                  onChange={(e) =>
                    dispatch({
                      type: 'SET_QUERY_OPTIONS',
                      payload: { maxChunksPerPaper: e.target.value === 'auto' ? 'auto' : parseInt(e.target.value) },
                    })
                  }
                  className="w-full px-3 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="auto">Auto</option>
                  <option value="3">3 (Fast, diverse)</option>
                  <option value="5">5</option>
                  <option value="10">10</option>
                  <option value="15">15</option>
                  <option value="20">20</option>
                  <option value="25">25 (Deep analysis)</option>
                </select>
              </div>
            </div>

            {/* Feature toggles */}
            <div>
              <label className="flex items-center gap-1.5 text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Features
                <InfoTooltip content="Toggle advanced RAG pipeline features to customize how your query is processed." />
              </label>
              <div className="flex flex-wrap gap-4">
                <div className="flex items-center gap-2">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={queryOptions.enableHyde}
                      onChange={(e) =>
                        dispatch({
                          type: 'SET_QUERY_OPTIONS',
                          payload: { enableHyde: e.target.checked },
                        })
                      }
                      className="w-4 h-4 text-blue-600 bg-white dark:bg-gray-700 border-gray-300 dark:border-gray-600 rounded focus:ring-blue-500"
                    />
                    <Sparkles className="w-4 h-4 text-purple-500" />
                    <span className="text-sm text-gray-700 dark:text-gray-300">HyDE</span>
                  </label>
                  <InfoTooltip content="Hypothetical Document Embedding: Generates a hypothetical answer first, then uses it to find similar real content. Improves retrieval for complex or abstract queries." />
                </div>

                <div className="flex items-center gap-2">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={queryOptions.enableExpansion}
                      onChange={(e) =>
                        dispatch({
                          type: 'SET_QUERY_OPTIONS',
                          payload: { enableExpansion: e.target.checked },
                        })
                      }
                      className="w-4 h-4 text-blue-600 bg-white dark:bg-gray-700 border-gray-300 dark:border-gray-600 rounded focus:ring-blue-500"
                    />
                    <Zap className="w-4 h-4 text-amber-500" />
                    <span className="text-sm text-gray-700 dark:text-gray-300">Query Expansion</span>
                  </label>
                  <InfoTooltip content="Expands your query with synonyms and related terms to improve recall. Helps find relevant documents that use different terminology." />
                </div>

                <div className="flex items-center gap-2">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={queryOptions.enableCitationCheck}
                      onChange={(e) =>
                        dispatch({
                          type: 'SET_QUERY_OPTIONS',
                          payload: { enableCitationCheck: e.target.checked },
                        })
                      }
                      className="w-4 h-4 text-blue-600 bg-white dark:bg-gray-700 border-gray-300 dark:border-gray-600 rounded focus:ring-blue-500"
                    />
                    <Shield className="w-4 h-4 text-green-500" />
                    <span className="text-sm text-gray-700 dark:text-gray-300">Citation Check</span>
                  </label>
                  <InfoTooltip content="Verifies that claims in the generated answer are supported by the source documents. Reduces hallucinations and improves trustworthiness." />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
