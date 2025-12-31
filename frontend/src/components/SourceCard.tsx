import { useState } from 'react';
import {
  ChevronDown,
  ChevronUp,
  FileText,
  Copy,
  ExternalLink,
  Check,
} from 'lucide-react';
import type { Source, ChunkType } from '../types';
import { useApp } from '../context/AppContext';

interface SourceCardProps {
  source: Source;
  index: number;
}

const CHUNK_TYPE_COLORS: Record<ChunkType, string> = {
  abstract: 'bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-400',
  section: 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400',
  fine: 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400',
  table: 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400',
  caption: 'bg-pink-100 dark:bg-pink-900/30 text-pink-700 dark:text-pink-400',
  full: 'bg-gray-100 dark:bg-gray-900/30 text-gray-700 dark:text-gray-400',
};

const CHUNK_TYPE_LABELS: Record<ChunkType, string> = {
  abstract: 'Abstract',
  section: 'Section',
  fine: 'Fine chunk',
  table: 'Table',
  caption: 'Caption',
  full: 'Full document',
};

export function SourceCard({ source, index }: SourceCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const { setViewingPdf } = useApp();

  const scorePercent = Math.round(source.relevance_score * 100);
  const chunkType = (source.chunk_type as ChunkType) || 'section';

  const handleCopy = async () => {
    await navigator.clipboard.writeText(source.chunk_text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden bg-white dark:bg-gray-800">
      {/* Header */}
      <div
        className="p-3 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-start gap-3">
          {/* Index and score */}
          <div className="flex flex-col items-center shrink-0">
            <span className="text-sm font-semibold text-gray-500 dark:text-gray-400">
              [{index + 1}]
            </span>
            <div className="mt-1 w-12 h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 rounded-full transition-all"
                style={{ width: `${scorePercent}%` }}
              />
            </div>
            <span className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">
              {source.relevance_score.toFixed(2)}
            </span>
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <FileText className="w-4 h-4 text-gray-400 shrink-0" />
              <span className="font-medium text-gray-900 dark:text-gray-100 truncate">
                {source.paper_title}
              </span>
            </div>

            {source.section_name && (
              <div className="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
                § {source.section_name}
                {source.subsection_name && (
                  <span className="text-gray-400 dark:text-gray-500">
                    {' › '}{source.subsection_name}
                  </span>
                )}
              </div>
            )}

            <div className="flex items-center gap-2 mt-2">
              <span
                className={`text-xs px-2 py-0.5 rounded-full ${CHUNK_TYPE_COLORS[chunkType]}`}
              >
                {CHUNK_TYPE_LABELS[chunkType]}
              </span>
            </div>
          </div>

          {/* Expand toggle */}
          <button className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors">
            {expanded ? (
              <ChevronUp className="w-4 h-4 text-gray-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-gray-400" />
            )}
          </button>
        </div>
      </div>

      {/* Expanded content */}
      {expanded && (
        <div className="border-t border-gray-200 dark:border-gray-700 p-3 bg-gray-50 dark:bg-gray-900">
          <div className="text-sm text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 p-3 rounded-lg border border-gray-200 dark:border-gray-700 max-h-48 overflow-y-auto">
            <p className="whitespace-pre-wrap">{source.chunk_text}</p>
          </div>

          <div className="flex items-center gap-2 mt-3">
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleCopy();
              }}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg transition-colors"
            >
              {copied ? (
                <Check className="w-4 h-4 text-green-500" />
              ) : (
                <Copy className="w-4 h-4" />
              )}
              {copied ? 'Copied!' : 'Copy'}
            </button>

            <button
              onClick={(e) => {
                e.stopPropagation();
                setViewingPdf(source.paper_id);
              }}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg transition-colors"
            >
              <ExternalLink className="w-4 h-4" />
              View in Paper
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
