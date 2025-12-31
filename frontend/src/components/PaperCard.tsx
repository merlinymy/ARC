import { useState } from 'react';
import { FileText, Eye, Trash2, AlertCircle, Loader2, Calendar, Layers, User, Quote, Check } from 'lucide-react';
import { useApp } from '../context/AppContext';
import type { Paper } from '../types';

interface PaperCardProps {
  paper: Paper;
  viewMode: 'grid' | 'list';
  // Optional preview for search results
  preview?: {
    text: string;
    section?: string;
    subsection?: string;
    chunkType?: string;
  };
  searchQuery?: string;
  // Selection support
  isSelected?: boolean;
  onToggleSelect?: (paperId: string) => void;
}

// Highlight search terms in text
function highlightText(text: string, query: string): React.ReactNode {
  if (!query.trim()) return text;

  // Split query into words and escape regex special chars
  const words = query.split(/\s+/).filter(w => w.length > 2);
  if (words.length === 0) return text;

  const pattern = new RegExp(`(${words.map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi');
  const parts = text.split(pattern);

  return parts.map((part, i) =>
    pattern.test(part) ? (
      <mark key={i} className="bg-yellow-200 dark:bg-yellow-700/50 text-inherit rounded px-0.5">
        {part}
      </mark>
    ) : (
      part
    )
  );
}

export function PaperCard({ paper, viewMode, preview, searchQuery, isSelected, onToggleSelect }: PaperCardProps) {
  const { deletePaper, setViewingPdf } = useApp();
  const selectable = onToggleSelect !== undefined;
  const [isDeleting, setIsDeleting] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  const handleDelete = async () => {
    setIsDeleting(true);
    try {
      await deletePaper(paper.id);
    } catch (error) {
      console.error('Failed to delete paper:', error);
    } finally {
      setIsDeleting(false);
      setShowDeleteConfirm(false);
    }
  };

  const handleViewPdf = () => {
    setViewingPdf(paper.id);
  };

  const statusColors = {
    indexed: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
    indexing: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
    error: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
    pending: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
  };

  const handleCardClick = () => {
    if (selectable && paper.status === 'indexed') {
      onToggleSelect(paper.id);
    }
  };

  if (viewMode === 'list') {
    return (
      <div
        onClick={handleCardClick}
        className={`flex items-center gap-4 p-4 bg-white dark:bg-gray-800 rounded-lg border transition-shadow ${
          selectable && paper.status === 'indexed' ? 'cursor-pointer' : ''
        } ${
          isSelected
            ? 'border-blue-500 dark:border-blue-400 ring-1 ring-blue-500 dark:ring-blue-400'
            : 'border-gray-200 dark:border-gray-700 hover:shadow-md'
        }`}
      >
        {/* Selection checkbox */}
        {selectable && (
          <button
            onClick={(e) => { e.stopPropagation(); onToggleSelect(paper.id); }}
            className={`shrink-0 w-6 h-6 rounded border-2 flex items-center justify-center transition-colors ${
              isSelected
                ? 'bg-blue-600 border-blue-600 text-white'
                : 'border-gray-300 dark:border-gray-600 hover:border-blue-400'
            }`}
          >
            {isSelected && <Check className="w-4 h-4" />}
          </button>
        )}

        {/* Icon */}
        <div className="shrink-0 w-12 h-12 flex items-center justify-center bg-blue-100 dark:bg-blue-900/30 rounded-lg">
          <FileText className="w-6 h-6 text-blue-600 dark:text-blue-400" />
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <h3 className="font-medium text-gray-900 dark:text-gray-100 truncate">
            {paper.title}
          </h3>
          <div className="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400 mt-1">
            {paper.authors.length > 0 && (
              <span className="flex items-center gap-1">
                <User className="w-4 h-4" />
                {paper.authors.slice(0, 2).join(', ')}
                {paper.authors.length > 2 && ` +${paper.authors.length - 2}`}
              </span>
            )}
            {paper.year && (
              <span className="flex items-center gap-1">
                <Calendar className="w-4 h-4" />
                {paper.year}
              </span>
            )}
            <span className="flex items-center gap-1">
              <Layers className="w-4 h-4" />
              {paper.chunkCount} chunks
            </span>
          </div>
          {/* Preview for list view */}
          {preview?.text && (
            <div className="mt-2 p-2 bg-purple-50 dark:bg-purple-900/20 rounded border border-purple-100 dark:border-purple-800">
              <div className="flex items-center gap-1 text-xs text-purple-600 dark:text-purple-400 mb-1">
                <Quote className="w-3 h-3" />
                {preview.section && (
                  <span>
                    § {preview.section}
                    {preview.subsection && <span className="opacity-70"> › {preview.subsection}</span>}
                  </span>
                )}
              </div>
              <p className="text-xs text-gray-700 dark:text-gray-300 line-clamp-2">
                {searchQuery ? highlightText(preview.text, searchQuery) : preview.text}
              </p>
            </div>
          )}
        </div>

        {/* Status */}
        <span className={`px-2 py-1 rounded-full text-xs font-medium ${statusColors[paper.status]}`}>
          {paper.status === 'indexing' && <Loader2 className="w-3 h-3 inline mr-1 animate-spin" />}
          {paper.status === 'error' && <AlertCircle className="w-3 h-3 inline mr-1" />}
          {paper.status}
        </span>

        {/* Actions */}
        <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
          <button
            onClick={handleViewPdf}
            className="p-2 text-gray-500 hover:text-blue-600 dark:text-gray-400 dark:hover:text-blue-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
            title="View PDF"
          >
            <Eye className="w-5 h-5" />
          </button>
          {showDeleteConfirm ? (
            <div className="flex items-center gap-1">
              <button
                onClick={handleDelete}
                disabled={isDeleting}
                className="px-2 py-1 text-xs bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
              >
                {isDeleting ? 'Deleting...' : 'Confirm'}
              </button>
              <button
                onClick={() => setShowDeleteConfirm(false)}
                className="px-2 py-1 text-xs bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-300 dark:hover:bg-gray-600"
              >
                Cancel
              </button>
            </div>
          ) : (
            <button
              onClick={() => setShowDeleteConfirm(true)}
              className="p-2 text-gray-500 hover:text-red-600 dark:text-gray-400 dark:hover:text-red-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
              title="Delete paper"
            >
              <Trash2 className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>
    );
  }

  // Grid view
  return (
    <div
      onClick={handleCardClick}
      className={`flex flex-col p-4 bg-white dark:bg-gray-800 rounded-lg border transition-shadow ${
        selectable && paper.status === 'indexed' ? 'cursor-pointer' : ''
      } ${
        isSelected
          ? 'border-blue-500 dark:border-blue-400 ring-1 ring-blue-500 dark:ring-blue-400'
          : 'border-gray-200 dark:border-gray-700 hover:shadow-md'
      }`}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          {/* Selection checkbox */}
          {selectable && (
            <button
              onClick={(e) => { e.stopPropagation(); onToggleSelect(paper.id); }}
              className={`shrink-0 w-5 h-5 rounded border-2 flex items-center justify-center transition-colors ${
                isSelected
                  ? 'bg-blue-600 border-blue-600 text-white'
                  : 'border-gray-300 dark:border-gray-600 hover:border-blue-400'
              }`}
            >
              {isSelected && <Check className="w-3 h-3" />}
            </button>
          )}
          <div className="w-10 h-10 flex items-center justify-center bg-blue-100 dark:bg-blue-900/30 rounded-lg shrink-0">
            <FileText className="w-5 h-5 text-blue-600 dark:text-blue-400" />
          </div>
        </div>
        <span className={`px-2 py-1 rounded-full text-xs font-medium ${statusColors[paper.status]}`}>
          {paper.status === 'indexing' && <Loader2 className="w-3 h-3 inline mr-1 animate-spin" />}
          {paper.status === 'error' && <AlertCircle className="w-3 h-3 inline mr-1" />}
          {paper.status}
        </span>
      </div>

      {/* Title */}
      <h3 className="font-medium text-gray-900 dark:text-gray-100 line-clamp-2 mb-2">
        {paper.title}
      </h3>

      {/* Authors */}
      {paper.authors.length > 0 && (
        <p className="text-sm text-gray-500 dark:text-gray-400 truncate mb-2">
          {paper.authors.slice(0, 2).join(', ')}
          {paper.authors.length > 2 && ` +${paper.authors.length - 2}`}
        </p>
      )}

      {/* Preview for search results */}
      {preview?.text && (
        <div className="mb-3 p-2 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-100 dark:border-purple-800">
          <div className="flex items-center gap-1 text-xs text-purple-600 dark:text-purple-400 mb-1">
            <Quote className="w-3 h-3" />
            {preview.section && (
              <span>
                § {preview.section}
                {preview.subsection && <span className="opacity-70"> › {preview.subsection}</span>}
              </span>
            )}
          </div>
          <p className="text-xs text-gray-700 dark:text-gray-300 line-clamp-3 leading-relaxed">
            {searchQuery ? highlightText(preview.text, searchQuery) : preview.text}
          </p>
        </div>
      )}

      {/* Meta */}
      <div className="flex items-center gap-3 text-xs text-gray-400 dark:text-gray-500 mt-auto mb-3">
        {paper.year && (
          <span className="flex items-center gap-1">
            <Calendar className="w-3 h-3" />
            {paper.year}
          </span>
        )}
        <span className="flex items-center gap-1">
          <Layers className="w-3 h-3" />
          {paper.chunkCount} chunks
        </span>
      </div>

      {/* Error message */}
      {paper.status === 'error' && paper.errorMessage && (
        <p className="text-xs text-red-600 dark:text-red-400 mb-3 line-clamp-2">
          {paper.errorMessage}
        </p>
      )}

      {/* Actions */}
      <div className="flex items-center gap-2 pt-3 border-t border-gray-100 dark:border-gray-700" onClick={(e) => e.stopPropagation()}>
        <button
          onClick={handleViewPdf}
          className="flex-1 flex items-center justify-center gap-2 px-3 py-2 text-sm text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-lg transition-colors"
        >
          <Eye className="w-4 h-4" />
          View PDF
        </button>
        {showDeleteConfirm ? (
          <div className="flex items-center gap-1">
            <button
              onClick={handleDelete}
              disabled={isDeleting}
              className="px-3 py-2 text-sm bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50"
            >
              {isDeleting ? '...' : 'Yes'}
            </button>
            <button
              onClick={() => setShowDeleteConfirm(false)}
              className="px-3 py-2 text-sm bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600"
            >
              No
            </button>
          </div>
        ) : (
          <button
            onClick={() => setShowDeleteConfirm(true)}
            className="p-2 text-gray-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors"
            title="Delete paper"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
}
