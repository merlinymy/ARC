import { useState, useMemo, useEffect, useCallback, useRef } from 'react';
import { Search, Upload, Grid, List, SortAsc, SortDesc, FileText, Loader2, MessageSquare, X } from 'lucide-react';
import { useApp } from '../context/AppContext';
import { PaperCard } from './PaperCard';
import { BatchUploadPanel } from './BatchUploadPanel';
import { searchPapers, type PaperSearchResult } from '../services/api';

type SortField = 'title' | 'year' | 'chunkCount' | 'relevance';
type SortOrder = 'asc' | 'desc';
type ViewMode = 'grid' | 'list';

export function LibraryPage() {
  const { state, dispatch, refreshPapers, loadMorePapers, setActivePage, createNewConversation, openUploadPanel } = useApp();
  const { papers, totalPapers, hasMorePapers, isLoadingMorePapers } = state;

  const [searchQuery, setSearchQuery] = useState('');
  const [sortField, setSortField] = useState<SortField>('relevance');
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [semanticResults, setSemanticResults] = useState<PaperSearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [selectedPaperIds, setSelectedPaperIds] = useState<Set<string>>(new Set());

  // Search pagination state
  const [totalSearchResults, setTotalSearchResults] = useState(0);
  const [hasMoreSearchResults, setHasMoreSearchResults] = useState(false);
  const [isLoadingMoreSearchResults, setIsLoadingMoreSearchResults] = useState(false);

  // Sentinel ref for infinite scroll
  const sentinelRef = useRef<HTMLDivElement>(null);

  // Infinite scroll with IntersectionObserver
  useEffect(() => {
    // Only enable infinite scroll when not searching
    if (searchQuery.trim()) return;

    const sentinel = sentinelRef.current;
    if (!sentinel) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasMorePapers && !isLoadingMorePapers) {
          loadMorePapers();
        }
      },
      { threshold: 0.1, rootMargin: '100px' }
    );

    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [searchQuery, hasMorePapers, isLoadingMorePapers, loadMorePapers]);

  const togglePaperSelection = useCallback((paperId: string) => {
    setSelectedPaperIds(prev => {
      const next = new Set(prev);
      if (next.has(paperId)) {
        next.delete(paperId);
      } else {
        next.add(paperId);
      }
      return next;
    });
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedPaperIds(new Set());
  }, []);

  const handleChatWithSelected = useCallback(() => {
    createNewConversation();
    dispatch({
      type: 'SET_QUERY_OPTIONS',
      payload: { paperFilter: Array.from(selectedPaperIds) },
    });
    setActivePage('chat');
    clearSelection();
  }, [selectedPaperIds, dispatch, setActivePage, clearSelection, createNewConversation]);

  // Debounced semantic search
  const performSemanticSearch = useCallback(async (query: string) => {
    if (!query.trim()) {
      setSemanticResults([]);
      setTotalSearchResults(0);
      setHasMoreSearchResults(false);
      return;
    }

    setIsSearching(true);
    setSearchError(null);

    try {
      const response = await searchPapers(query, 25, 0);
      setSemanticResults(response.results);
      setTotalSearchResults(response.total);
      setHasMoreSearchResults(response.hasMore);
    } catch (err) {
      setSearchError(err instanceof Error ? err.message : 'Search failed');
      setSemanticResults([]);
      setTotalSearchResults(0);
      setHasMoreSearchResults(false);
    } finally {
      setIsSearching(false);
    }
  }, []);

  // Load more search results
  const loadMoreSearchResults = useCallback(async () => {
    if (!searchQuery.trim() || isLoadingMoreSearchResults || !hasMoreSearchResults) return;

    setIsLoadingMoreSearchResults(true);
    try {
      const response = await searchPapers(searchQuery, 25, semanticResults.length);
      setSemanticResults(prev => [...prev, ...response.results]);
      setHasMoreSearchResults(response.hasMore);
    } catch (err) {
      setSearchError(err instanceof Error ? err.message : 'Failed to load more results');
    } finally {
      setIsLoadingMoreSearchResults(false);
    }
  }, [searchQuery, semanticResults.length, isLoadingMoreSearchResults, hasMoreSearchResults]);

  // Debounce semantic search
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      performSemanticSearch(searchQuery);
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [searchQuery, performSemanticSearch]);

  // Convert semantic results to Paper-like objects for display
  const semanticPapers = useMemo(() => {
    return semanticResults.map((r) => ({
      id: r.id,
      title: r.title,
      authors: r.authors,
      year: r.year,
      filename: r.filename,
      pageCount: 0,
      chunkCount: r.chunkCount,
      chunkStats: {},
      status: r.status as 'indexed' | 'indexing' | 'error' | 'pending',
      pdfUrl: r.pdfUrl,
      relevanceScore: r.relevanceScore,
      previewText: r.previewText,
      previewSection: r.previewSection,
      previewSubsection: r.previewSubsection,
      previewChunkType: r.previewChunkType,
    }));
  }, [semanticResults]);

  // Filter and sort papers
  const filteredPapers = useMemo(() => {
    if (searchQuery.trim()) {
      if (sortField === 'relevance') {
        return sortOrder === 'desc' ? semanticPapers : [...semanticPapers].reverse();
      }
      const sorted = [...semanticPapers];
      sorted.sort((a, b) => {
        let comparison = 0;
        switch (sortField) {
          case 'title':
            comparison = a.title.localeCompare(b.title);
            break;
          case 'year':
            comparison = (a.year || 0) - (b.year || 0);
            break;
          case 'chunkCount':
            comparison = a.chunkCount - b.chunkCount;
            break;
        }
        return sortOrder === 'asc' ? comparison : -comparison;
      });
      return sorted;
    }

    const result = [...papers];
    result.sort((a, b) => {
      let comparison = 0;
      switch (sortField) {
        case 'title':
          comparison = a.title.localeCompare(b.title);
          break;
        case 'year':
          comparison = (a.year || 0) - (b.year || 0);
          break;
        case 'chunkCount':
          comparison = a.chunkCount - b.chunkCount;
          break;
      }
      return sortOrder === 'asc' ? comparison : -comparison;
    });

    return result;
  }, [papers, searchQuery, sortField, sortOrder, semanticPapers]);

  const toggleSortOrder = () => {
    setSortOrder((prev) => (prev === 'asc' ? 'desc' : 'asc'));
  };

  // Display count: show total when browsing, search results when searching
  const displayCount = searchQuery.trim() ? totalSearchResults : totalPapers;
  const loadedCount = searchQuery.trim() ? semanticResults.length : papers.length;

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="p-6 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <FileText className="w-8 h-8 text-blue-600 dark:text-blue-400" />
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                Paper Library
              </h1>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                {searchQuery.trim()
                  ? `${loadedCount} of ${displayCount} papers found`
                  : `${loadedCount} of ${displayCount} papers loaded`
                }
              </p>
            </div>
          </div>
          <button
            onClick={openUploadPanel}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors font-medium"
          >
            <Upload className="w-5 h-5" />
            Upload Papers
          </button>
        </div>

        {/* Search and Controls */}
        <div className="flex flex-col sm:flex-row gap-4">
          {/* Search input */}
          <div className="flex-1 relative">
            {isSearching ? (
              <Loader2 className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-purple-500 animate-spin" />
            ) : (
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            )}
            <input
              type="text"
              placeholder='Search papers: "SERS imaging techniques", "nanoparticle synthesis"...'
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-purple-300 dark:border-purple-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
          </div>

          {/* Sort */}
          <div className="flex items-center gap-2">
            <select
              value={sortField}
              onChange={(e) => setSortField(e.target.value as SortField)}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {searchQuery.trim() && (
                <option value="relevance">Sort by Relevance</option>
              )}
              <option value="title">Sort by Title</option>
              <option value="year">Sort by Year</option>
              <option value="chunkCount">Sort by Chunks</option>
            </select>
            <button
              onClick={toggleSortOrder}
              className="p-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              title={sortOrder === 'asc' ? 'Ascending' : 'Descending'}
            >
              {sortOrder === 'asc' ? (
                <SortAsc className="w-5 h-5 text-gray-600 dark:text-gray-400" />
              ) : (
                <SortDesc className="w-5 h-5 text-gray-600 dark:text-gray-400" />
              )}
            </button>
          </div>

          {/* View Mode */}
          <div className="flex items-center border border-gray-300 dark:border-gray-600 rounded-lg overflow-hidden">
            <button
              onClick={() => setViewMode('grid')}
              className={`p-2 ${
                viewMode === 'grid'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-600'
              } transition-colors`}
              title="Grid view"
            >
              <Grid className="w-5 h-5" />
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`p-2 ${
                viewMode === 'list'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-600'
              } transition-colors`}
              title="List view"
            >
              <List className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Search info/error */}
        {searchQuery.trim() && (
          <div className="mt-4">
            {searchError ? (
              <div className="text-sm text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 px-3 py-2 rounded-lg">
                Search error: {searchError}
              </div>
            ) : semanticResults.length > 0 ? (
              <div className="text-sm text-purple-600 dark:text-purple-400">
                Showing {semanticResults.length} of {totalSearchResults} papers matching "{searchQuery}"
              </div>
            ) : !isSearching ? (
              <div className="text-sm text-gray-500 dark:text-gray-400">
                No papers found for "{searchQuery}"
              </div>
            ) : null}
          </div>
        )}
      </div>

      {/* Paper Grid/List */}
      <div className="flex-1 overflow-y-auto p-6">
        {filteredPapers.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-gray-500 dark:text-gray-400">
            {papers.length === 0 && !searchQuery.trim() ? (
              <>
                <FileText className="w-16 h-16 mb-4 opacity-50" />
                <p className="text-lg font-medium">No papers yet</p>
                <p className="text-sm">Upload your first PDF to get started</p>
                <button
                  onClick={openUploadPanel}
                  className="mt-4 flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                >
                  <Upload className="w-4 h-4" />
                  Upload Papers
                </button>
              </>
            ) : (
              <>
                <Search className="w-16 h-16 mb-4 opacity-50" />
                <p className="text-lg font-medium">No papers found</p>
                <p className="text-sm">Try a different search query</p>
              </>
            )}
          </div>
        ) : (
          <>
            <div
              className={
                viewMode === 'grid'
                  ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4'
                  : 'flex flex-col gap-3'
              }
            >
              {filteredPapers.map((paper) => (
                <PaperCard
                  key={paper.id}
                  paper={paper}
                  viewMode={viewMode}
                  preview={
                    searchQuery.trim() && 'previewText' in paper && paper.previewText
                      ? {
                          text: paper.previewText,
                          section: paper.previewSection,
                          subsection: paper.previewSubsection,
                          chunkType: paper.previewChunkType,
                        }
                      : undefined
                  }
                  searchQuery={searchQuery.trim() ? searchQuery : undefined}
                  isSelected={selectedPaperIds.has(paper.id)}
                  onToggleSelect={paper.status === 'indexed' ? togglePaperSelection : undefined}
                />
              ))}
            </div>

            {/* Load more for search results */}
            {searchQuery.trim() && (
              <div className="h-20 flex items-center justify-center mt-4">
                {isLoadingMoreSearchResults ? (
                  <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400">
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span className="text-sm">Loading more results...</span>
                  </div>
                ) : hasMoreSearchResults ? (
                  <button
                    onClick={loadMoreSearchResults}
                    className="px-4 py-2 text-sm text-purple-600 dark:text-purple-400 hover:underline"
                  >
                    Load more results ({totalSearchResults - semanticResults.length} remaining)
                  </button>
                ) : semanticResults.length > 0 ? (
                  <span className="text-sm text-gray-400 dark:text-gray-500">
                    All {totalSearchResults} matching papers shown
                  </span>
                ) : null}
              </div>
            )}

            {/* Infinite scroll sentinel for browsing */}
            {!searchQuery.trim() && (
              <div
                ref={sentinelRef}
                className="h-20 flex items-center justify-center mt-4"
              >
                {isLoadingMorePapers ? (
                  <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400">
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span className="text-sm">Loading more papers...</span>
                  </div>
                ) : hasMorePapers ? (
                  <button
                    onClick={loadMorePapers}
                    className="px-4 py-2 text-sm text-blue-600 dark:text-blue-400 hover:underline"
                  >
                    Load more papers
                  </button>
                ) : papers.length > 0 ? (
                  <span className="text-sm text-gray-400 dark:text-gray-500">
                    All {totalPapers} papers loaded
                  </span>
                ) : null}
              </div>
            )}
          </>
        )}
      </div>

      {/* Floating action bar when papers are selected */}
      {selectedPaperIds.size > 0 && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-3 px-4 py-3 bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 rounded-full shadow-lg z-40">
          <span className="text-sm font-medium">
            {selectedPaperIds.size} paper{selectedPaperIds.size > 1 ? 's' : ''} selected
          </span>
          <div className="w-px h-5 bg-gray-600 dark:bg-gray-400" />
          <button
            onClick={handleChatWithSelected}
            className="flex items-center gap-2 px-4 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-full text-sm font-medium transition-colors"
          >
            <MessageSquare className="w-4 h-4" />
            Chat about {selectedPaperIds.size === 1 ? 'this paper' : 'these papers'}
          </button>
          <button
            onClick={clearSelection}
            className="p-1.5 hover:bg-gray-700 dark:hover:bg-gray-300 rounded-full transition-colors"
            title="Clear selection"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Batch Upload Panel */}
      <BatchUploadPanel />
    </div>
  );
}
