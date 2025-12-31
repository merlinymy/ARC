import { X, FileText, Calendar, User, ChevronRight, ExternalLink, RefreshCw, Trash2 } from 'lucide-react';
import type { Paper } from '../types';

interface PaperDetailModalProps {
  paper: Paper;
  onClose: () => void;
}

export function PaperDetailModal({ paper, onClose }: PaperDetailModalProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto bg-white dark:bg-gray-900 rounded-2xl shadow-2xl">
        {/* Header */}
        <div className="sticky top-0 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 px-6 py-4 flex items-start justify-between">
          <div className="flex items-start gap-3">
            <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
              <FileText className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                {paper.title}
              </h2>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
                {paper.filename}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Metadata */}
          <div>
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
              Metadata
            </h3>
            <div className="grid grid-cols-2 gap-4">
              {paper.authors && paper.authors.length > 0 && (
                <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                  <User className="w-4 h-4" />
                  <span>{paper.authors.join(', ')}</span>
                </div>
              )}
              {paper.year && (
                <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                  <Calendar className="w-4 h-4" />
                  <span>{paper.year}</span>
                </div>
              )}
              {paper.pageCount && (
                <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                  <FileText className="w-4 h-4" />
                  <span>{paper.pageCount} pages</span>
                </div>
              )}
              {paper.indexedAt && (
                <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                  <Calendar className="w-4 h-4" />
                  <span>
                    Indexed: {new Date(paper.indexedAt).toLocaleDateString()}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Chunk Statistics */}
          <div>
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
              Chunk Statistics
            </h3>
            <div className="bg-gray-50 dark:bg-gray-800 rounded-xl p-4">
              <div className="space-y-3">
                {[
                  { type: 'Abstract', count: 1, color: 'bg-violet-500' },
                  { type: 'Sections', count: 12, color: 'bg-blue-500' },
                  { type: 'Fine', count: 78, color: 'bg-emerald-500' },
                  { type: 'Tables', count: 6, color: 'bg-amber-500' },
                  { type: 'Captions', count: 8, color: 'bg-pink-500' },
                  { type: 'Full', count: 1, color: 'bg-gray-500' },
                ].map((item) => (
                  <div key={item.type} className="flex items-center gap-3">
                    <span className="w-20 text-sm text-gray-600 dark:text-gray-400">
                      {item.type}
                    </span>
                    <div className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${item.color} rounded-full transition-all`}
                        style={{ width: `${Math.min(100, item.count)}%` }}
                      />
                    </div>
                    <span className="w-8 text-sm text-gray-500 dark:text-gray-400 text-right">
                      {item.count}
                    </span>
                  </div>
                ))}
              </div>
              <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700 flex justify-between text-sm">
                <span className="text-gray-600 dark:text-gray-400">
                  Total: {paper.chunkCount ?? 106} chunks
                </span>
                <span className="text-gray-500 dark:text-gray-500">
                  ~45,000 tokens
                </span>
              </div>
            </div>
          </div>

          {/* Sections */}
          <div>
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
              Sections
            </h3>
            <div className="bg-gray-50 dark:bg-gray-800 rounded-xl divide-y divide-gray-200 dark:divide-gray-700">
              {[
                { name: 'Abstract', expanded: false },
                { name: 'Introduction', expanded: false },
                {
                  name: 'Methods',
                  expanded: true,
                  children: [
                    'Peptide Synthesis (3 fine chunks)',
                    'Antimicrobial Assays (4 fine chunks)',
                    'Cell Culture (2 fine chunks)',
                  ],
                },
                { name: 'Results', expanded: false },
                { name: 'Discussion', expanded: false },
                { name: 'Conclusion', expanded: false },
                { name: 'References', expanded: false },
              ].map((section) => (
                <div key={section.name}>
                  <button className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-gray-100 dark:hover:bg-gray-700/50 transition-colors">
                    <div className="flex items-center gap-2">
                      <ChevronRight
                        className={`w-4 h-4 text-gray-400 transition-transform ${
                          section.expanded ? 'rotate-90' : ''
                        }`}
                      />
                      <span className="text-sm text-gray-700 dark:text-gray-300">
                        {section.name}
                      </span>
                    </div>
                  </button>
                  {section.expanded && section.children && (
                    <div className="pl-10 pb-2 space-y-1">
                      {section.children.map((child) => (
                        <div
                          key={child}
                          className="text-sm text-gray-500 dark:text-gray-400 py-1"
                        >
                          • {child}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="sticky bottom-0 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700 px-6 py-4 flex items-center justify-between">
          <button className="flex items-center gap-2 px-4 py-2 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors">
            <Trash2 className="w-4 h-4" />
            Delete from Library
          </button>

          <div className="flex items-center gap-2">
            <button className="flex items-center gap-2 px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors">
              <RefreshCw className="w-4 h-4" />
              Re-index
            </button>
            <button className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
              <ExternalLink className="w-4 h-4" />
              View Original PDF
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
