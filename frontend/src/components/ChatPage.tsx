import { FileText, X } from 'lucide-react';
import { ConversationThread } from './ConversationThread';
import { QueryInput } from './QueryInput';
import { useApp } from '../context/AppContext';

export function ChatPage() {
  const { state, dispatch } = useApp();
  const { papers, queryOptions } = state;
  const selectedPaperIds = queryOptions.paperFilter;

  // Get selected paper objects (may not find all if they came from search results)
  const selectedPapers = papers.filter(p => selectedPaperIds.includes(p.id));
  const selectedCount = selectedPaperIds.length;

  const clearPaperFilter = () => {
    dispatch({ type: 'SET_QUERY_OPTIONS', payload: { paperFilter: [] } });
  };

  return (
    <div className="flex-1 flex flex-col bg-gray-50 dark:bg-gray-900 min-h-0">
      <div className="flex-1 min-h-0 overflow-hidden">
        <ConversationThread />
      </div>

      {/* Paper filter indicator */}
      {selectedCount > 0 && (
        <div className="px-4 py-2 bg-blue-50 dark:bg-blue-900/20 border-t border-blue-100 dark:border-blue-800">
          <div className="max-w-4xl mx-auto flex items-center gap-2">
            <FileText className="w-4 h-4 text-blue-600 dark:text-blue-400 shrink-0" />
            <span className="text-sm text-blue-700 dark:text-blue-300">
              Searching in:{' '}
              <span className="font-medium">
                {selectedCount === 1 && selectedPapers.length === 1
                  ? selectedPapers[0]?.title
                  : `${selectedCount} paper${selectedCount > 1 ? 's' : ''}`}
              </span>
            </span>
            {selectedPapers.length > 1 && selectedPapers.length <= 3 && (
              <span className="text-xs text-blue-500 dark:text-blue-400 truncate">
                ({selectedPapers.map(p => p.title).join(', ')})
              </span>
            )}
            <button
              onClick={clearPaperFilter}
              className="ml-auto p-1 hover:bg-blue-100 dark:hover:bg-blue-800 rounded transition-colors"
              title="Clear filter and search all papers"
            >
              <X className="w-4 h-4 text-blue-600 dark:text-blue-400" />
            </button>
          </div>
        </div>
      )}

      <QueryInput />
    </div>
  );
}
