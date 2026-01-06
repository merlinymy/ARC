import {
  Plus,
  MessageSquare,
  FileText,
  Trash2,
  ChevronRight,
  Database,
  Clock,
  BarChart2,
  X,
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { MinimizedUploadWidget } from './MinimizedUploadWidget';
import type { Conversation } from '../types';

function formatDate(date: Date): string {
  const now = new Date();
  const messageDate = new Date(date);

  // Normalize to start of day for accurate calendar day comparison
  const nowDay = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const messageDay = new Date(messageDate.getFullYear(), messageDate.getMonth(), messageDate.getDate());

  const diffMs = nowDay.getTime() - messageDay.getTime();
  const diffDays = Math.round(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays <= 0) return 'Today';
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return `${diffDays} days ago`;
  return messageDate.toLocaleDateString();
}

function groupConversationsByDate(conversations: Conversation[]): Record<string, Conversation[]> {
  const groups: Record<string, Conversation[]> = {};

  conversations.forEach((conv) => {
    const dateKey = formatDate(conv.updatedAt);
    if (!groups[dateKey]) {
      groups[dateKey] = [];
    }
    groups[dateKey].push(conv);
  });

  return groups;
}

export function Sidebar() {
  const { state, dispatch, createNewConversation, setActivePage, toggleSidebar, selectConversation } = useApp();
  const { conversations, activeConversationId, activePage, stats } = state;

  const groupedConversations = groupConversationsByDate(conversations);

  return (
    <aside className="w-full h-full border-r border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 flex flex-col">
      {/* Mobile close button */}
      <div className="flex items-center justify-between p-4 lg:hidden border-b border-gray-200 dark:border-gray-700">
        <span className="font-semibold text-gray-900 dark:text-gray-100">Menu</span>
        <button
          onClick={toggleSidebar}
          className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
        >
          <X className="w-5 h-5 text-gray-500" />
        </button>
      </div>

      {/* New Conversation Button */}
      <div className="p-4">
        <button
          onClick={() => {
            createNewConversation();
            setActivePage('chat');
            // Close sidebar on mobile after action
            if (window.innerWidth < 1024) {
              toggleSidebar();
            }
          }}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors font-medium"
        >
          <Plus className="w-5 h-5" />
          New Conversation
        </button>
      </div>

      {/* Chat History */}
      <div className="flex-1 overflow-y-auto px-2">
        <div className="px-2 py-2 text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
          Chat History
        </div>

        {conversations.length === 0 ? (
          <div className="px-4 py-8 text-center text-gray-500 dark:text-gray-400 text-sm">
            No conversations yet.
            <br />
            Start by asking a question!
          </div>
        ) : (
          Object.entries(groupedConversations).map(([date, convs]) => (
            <div key={date} className="mb-4">
              <div className="px-2 py-1 text-xs text-gray-400 dark:text-gray-500">
                {date}
              </div>
              {convs.map((conv) => (
                <div
                  key={conv.id}
                  role="button"
                  tabIndex={0}
                  onClick={() => {
                    selectConversation(conv.id);
                    setActivePage('chat');
                    // Close sidebar on mobile after action
                    if (window.innerWidth < 1024) {
                      toggleSidebar();
                    }
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      selectConversation(conv.id);
                      setActivePage('chat');
                    }
                  }}
                  className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-colors text-left group cursor-pointer ${
                    activeConversationId === conv.id && activePage === 'chat'
                      ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                      : 'hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300'
                  }`}
                >
                  <MessageSquare className="w-4 h-4 shrink-0" />
                  <span className="truncate text-sm flex-1">{conv.title}</span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      dispatch({ type: 'DELETE_CONVERSATION', payload: conv.id });
                    }}
                    className="opacity-0 group-hover:opacity-100 p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded transition-all"
                  >
                    <Trash2 className="w-3.5 h-3.5 text-gray-400 hover:text-red-500" />
                  </button>
                </div>
              ))}
            </div>
          ))
        )}
      </div>

      {/* Paper Library Section */}
      <div className="border-t border-gray-200 dark:border-gray-700">
        <button
          onClick={() => {
            setActivePage('library');
            if (window.innerWidth < 1024) {
              toggleSidebar();
            }
          }}
          className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
        >
          <div className="flex items-center gap-2 text-gray-700 dark:text-gray-300">
            <FileText className="w-5 h-5" />
            <span className="font-medium">Paper Library</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-400 dark:text-gray-500">
              {state.totalPapers || state.papers?.length || 0} papers
            </span>
            <ChevronRight className="w-4 h-4 text-gray-400" />
          </div>
        </button>
        {/* Currently viewing paper */}
        {state.viewingPdfId && (() => {
          const viewingPaper = state.papers.find(p => p.id === state.viewingPdfId);
          return viewingPaper ? (
            <div className="px-4 py-2 bg-blue-50 dark:bg-blue-900/20 border-t border-gray-200 dark:border-gray-700">
              <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Currently viewing:</div>
              <div className="text-sm text-blue-700 dark:text-blue-300 truncate font-medium">
                {viewingPaper.title}
              </div>
            </div>
          ) : null;
        })()}
      </div>

      {/* Quick Stats */}
      <div className="border-t border-gray-200 dark:border-gray-700 p-4">
        <div className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">
          Quick Stats
        </div>
        <div className="space-y-2 text-sm">
          <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
            <Database className="w-4 h-4" />
            <span>{stats?.total_vectors?.toLocaleString() ?? 0} chunks indexed</span>
          </div>
          <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
            <MessageSquare className="w-4 h-4" />
            <span>{conversations.length} conversations</span>
          </div>
          <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
            <Clock className="w-4 h-4" />
            <span>{stats?.conversation_stats?.turns ?? 0} queries this session</span>
          </div>
          {stats?.cache_stats && (
            <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
              <BarChart2 className="w-4 h-4" />
              <span>
                {Math.round(
                  ((stats.cache_stats.embedding_cache?.hits ?? 0) /
                    Math.max(
                      1,
                      (stats.cache_stats.embedding_cache?.hits ?? 0) +
                        (stats.cache_stats.embedding_cache?.misses ?? 0)
                    )) *
                    100
                )}
                % cache hit rate
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Minimized Upload Widget */}
      <div className="border-t border-gray-200 dark:border-gray-700 p-3">
        <MinimizedUploadWidget />
      </div>
    </aside>
  );
}
