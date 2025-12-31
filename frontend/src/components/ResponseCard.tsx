import { useState } from 'react';
import {
  Search,
  CheckCircle,
  ChevronDown,
  ChevronUp,
  Clock,
  Database,
  Filter,
  Tag,
  AlertTriangle,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { SourceCard } from './SourceCard';
import type { Message } from '../types';

interface ResponseCardProps {
  queryMessage: Message;
  responseMessage: Message;
}

export function ResponseCard({ queryMessage, responseMessage }: ResponseCardProps) {
  const [showAllSources, setShowAllSources] = useState(false);
  const [showPipeline, setShowPipeline] = useState(false);

  const { metadata } = responseMessage;
  // Sort sources by relevance score (highest first)
  const sources = [...(metadata?.sources ?? [])].sort(
    (a, b) => (b.relevance_score ?? 0) - (a.relevance_score ?? 0)
  );
  const visibleSources = showAllSources ? sources : sources.slice(0, 3);
  const hiddenCount = sources.length - 3;

  return (
    <div className="animate-fade-in">
      {/* Query Card */}
      <div className="mb-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-xl border border-blue-200 dark:border-blue-800">
        <div className="flex items-start gap-3">
          <div className="p-2 bg-blue-100 dark:bg-blue-800 rounded-lg">
            <Search className="w-5 h-5 text-blue-600 dark:text-blue-400" />
          </div>
          <div className="flex-1">
            <p className="text-gray-900 dark:text-gray-100 font-medium">
              {queryMessage.content}
            </p>

            {metadata?.queryType && (
              <div className="flex items-center gap-4 mt-2 text-sm">
                <div className="flex items-center gap-1.5 text-gray-600 dark:text-gray-400">
                  <Tag className="w-4 h-4" />
                  <span>
                    Query Type:{' '}
                    <span className="font-medium text-blue-600 dark:text-blue-400 uppercase">
                      {metadata.queryType}
                    </span>
                  </span>
                </div>

                {metadata.expandedQuery && (
                  <div className="flex items-center gap-1.5 text-gray-500 dark:text-gray-500">
                    <span className="truncate max-w-xs">
                      Expanded: "{metadata.expandedQuery}"
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Answer Card */}
      <div className="p-4 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
        <div className="markdown-content">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {responseMessage.content}
          </ReactMarkdown>
        </div>

        {/* Citation Verification */}
        {metadata?.citationScore !== undefined && (
          <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-2 text-sm">
              <CheckCircle className="w-4 h-4 text-green-500" />
              <span className="text-gray-600 dark:text-gray-400">
                Citation Verification:{' '}
                <span className="font-medium text-green-600 dark:text-green-400">
                  {Math.round(metadata.citationScore * 100)}% Trustworthy
                </span>
              </span>
            </div>
          </div>
        )}

        {/* Pipeline Warnings */}
        {metadata?.warnings && metadata.warnings.length > 0 && (
          <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
            <div className="p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
              {metadata.warnings.map((warning, index) => (
                <div key={index} className="flex items-center gap-2 text-sm">
                  <AlertTriangle className="w-4 h-4 text-amber-500 shrink-0" />
                  <span className="text-amber-700 dark:text-amber-400">
                    {warning}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Sources */}
      {sources.length > 0 && (
        <div className="mt-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
              Sources ({sources.length} retrieved)
            </h4>
          </div>

          <div className="space-y-2">
            {visibleSources.map((source, index) => (
              <SourceCard key={index} source={source} index={index} />
            ))}
          </div>

          {hiddenCount > 0 && (
            <button
              onClick={() => setShowAllSources(!showAllSources)}
              className="mt-3 w-full py-2 text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition-colors"
            >
              {showAllSources
                ? 'Show less'
                : `Show ${hiddenCount} more source${hiddenCount > 1 ? 's' : ''}...`}
            </button>
          )}
        </div>
      )}

      {/* Pipeline Details */}
      <div className="mt-4">
        <button
          onClick={() => setShowPipeline(!showPipeline)}
          className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
        >
          {showPipeline ? (
            <ChevronUp className="w-4 h-4" />
          ) : (
            <ChevronDown className="w-4 h-4" />
          )}
          Pipeline Details
        </button>

        {showPipeline && (
          <div className="mt-3 p-4 bg-gray-50 dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 animate-fade-in">
            <div className="flex flex-wrap gap-4 text-sm">
              {metadata?.retrievalCount !== undefined && (
                <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                  <Database className="w-4 h-4" />
                  <span>
                    Retrieved: {metadata.retrievalCount}
                  </span>
                </div>
              )}

              {metadata?.rerankedCount !== undefined && (
                <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                  <Filter className="w-4 h-4" />
                  <span>
                    Reranked: {metadata.rerankedCount}
                  </span>
                </div>
              )}

              {sources.length > 0 && (
                <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                  <CheckCircle className="w-4 h-4" />
                  <span>Cited: {sources.length}</span>
                </div>
              )}

              {metadata?.latency !== undefined && (
                <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                  <Clock className="w-4 h-4" />
                  <span>
                    Latency: {(metadata.latency / 1000).toFixed(2)}s
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
