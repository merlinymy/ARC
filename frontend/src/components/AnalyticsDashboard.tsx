import {
  PieChart,
  Clock,
  TrendingUp,
  Zap,
  AlertCircle,
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { InfoTooltip } from './Tooltip';
import type { EntityStat } from '../types';

// Query type colors for the distribution chart
const QUERY_TYPE_COLORS: Record<string, string> = {
  FACTUAL: 'bg-blue-500',
  FRAMING: 'bg-indigo-500',
  METHODS: 'bg-emerald-500',
  SUMMARY: 'bg-purple-500',
  COMPARATIVE: 'bg-amber-500',
  NOVELTY: 'bg-pink-500',
  LIMITATIONS: 'bg-red-500',
  GENERAL: 'bg-gray-500',
};

export function AnalyticsDashboard() {
  const { state } = useApp();
  const { stats } = state;
  const cacheStats = stats?.cache_stats;
  const analytics = stats?.analytics;

  // Calculate cache hit rates
  const embeddingHitRate = cacheStats?.embedding_cache
    ? Math.round(
        (cacheStats.embedding_cache.hits /
          Math.max(1, cacheStats.embedding_cache.hits + cacheStats.embedding_cache.misses)) *
          100
      )
    : 0;

  const searchHitRate = cacheStats?.search_cache
    ? Math.round(
        (cacheStats.search_cache.hits /
          Math.max(1, cacheStats.search_cache.hits + cacheStats.search_cache.misses)) *
          100
      )
    : 0;

  const hydeHitRate = cacheStats?.hyde_cache
    ? Math.round(
        (cacheStats.hyde_cache.hits /
          Math.max(1, cacheStats.hyde_cache.hits + cacheStats.hyde_cache.misses)) *
          100
      )
    : 0;

  // Calculate query type distribution percentages
  const queryTypeDistribution = analytics?.query_type_distribution ?? {};
  const totalQueries = analytics?.total_queries ?? 0;
  const queryTypeItems = Object.entries(queryTypeDistribution)
    .map(([type, count]) => ({
      type,
      count,
      percent: totalQueries > 0 ? Math.round((count / totalQueries) * 100) : 0,
      color: QUERY_TYPE_COLORS[type] || 'bg-gray-500',
    }))
    .filter((item) => item.count > 0)
    .sort((a, b) => b.count - a.count);

  // Latency stats
  const latencyStats = analytics?.latency_stats;
  const latencyItems = latencyStats
    ? [
        { step: 'Query Processing', ms: latencyStats.query_processing_ms },
        { step: 'Embedding', ms: latencyStats.embedding_ms },
        { step: 'Retrieval', ms: latencyStats.retrieval_ms },
        { step: 'Reranking', ms: latencyStats.reranking_ms },
        { step: 'Generation', ms: latencyStats.generation_ms },
      ]
    : [];
  const totalLatencyMs = latencyStats?.total_avg_ms ?? 0;

  // Entity stats
  const entityStats = analytics?.entity_stats;

  // Helper to render entity list
  const renderEntityList = (entities: EntityStat[] | undefined) => {
    if (!entities || entities.length === 0) {
      return <div className="text-gray-400 dark:text-gray-500 italic">No data yet</div>;
    }
    return (
      <div className="space-y-1">
        {entities.map((entity) => (
          <div key={entity.name}>
            {entity.name} ({entity.count})
          </div>
        ))}
      </div>
    );
  };

  // Empty state component
  const EmptyState = ({ message }: { message: string }) => (
    <div className="flex flex-col items-center justify-center py-8 text-gray-400 dark:text-gray-500">
      <AlertCircle className="w-8 h-8 mb-2" />
      <p className="text-sm">{message}</p>
    </div>
  );

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            Analytics Dashboard
          </h1>
          {totalQueries > 0 && (
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {totalQueries} queries tracked
            </span>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Query Types Distribution */}
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-2 mb-4">
              <PieChart className="w-5 h-5 text-blue-500" />
              <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                Query Types Distribution
              </h3>
              <InfoTooltip content="Breakdown of query categories. The system classifies each query to optimize retrieval strategy: factual queries focus on specific facts, methods on procedures, summaries on overviews, etc." />
            </div>
            {queryTypeItems.length > 0 ? (
              <div className="space-y-3">
                {queryTypeItems.map((item) => (
                  <div key={item.type} className="flex items-center gap-3">
                    <span className="w-24 text-sm text-gray-600 dark:text-gray-400 uppercase">
                      {item.type}
                    </span>
                    <div className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${item.color} rounded-full`}
                        style={{ width: `${item.percent}%` }}
                      />
                    </div>
                    <span className="w-10 text-sm text-gray-500 dark:text-gray-400 text-right">
                      {item.percent}%
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <EmptyState message="No queries yet" />
            )}
          </div>

          {/* Cache Performance */}
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-2 mb-4">
              <Zap className="w-5 h-5 text-amber-500" />
              <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                Cache Performance
              </h3>
              <InfoTooltip content="Caching reduces API costs and latency by reusing previous computations. Higher hit rates mean faster responses and lower costs." />
            </div>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <div className="flex items-center gap-1 text-gray-600 dark:text-gray-400">
                    Embedding Cache
                    <InfoTooltip content="Stores vector embeddings for previously seen text. Avoids re-calling Voyage AI for repeated queries or document chunks." />
                  </div>
                  <span className="text-gray-900 dark:text-gray-100 font-medium">
                    {embeddingHitRate}% hit
                  </span>
                </div>
                <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500 rounded-full"
                    style={{ width: `${embeddingHitRate}%` }}
                  />
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  {cacheStats?.embedding_cache?.size ?? 0}/{cacheStats?.embedding_cache?.max_size ?? 500} entries
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm mb-1">
                  <div className="flex items-center gap-1 text-gray-600 dark:text-gray-400">
                    Search Cache
                    <InfoTooltip content="Caches vector search results from Qdrant. Identical queries return instantly without database lookup." />
                  </div>
                  <span className="text-gray-900 dark:text-gray-100 font-medium">
                    {searchHitRate}% hit
                  </span>
                </div>
                <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-emerald-500 rounded-full"
                    style={{ width: `${searchHitRate}%` }}
                  />
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  {cacheStats?.search_cache?.size ?? 0}/{cacheStats?.search_cache?.max_size ?? 200} entries
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm mb-1">
                  <div className="flex items-center gap-1 text-gray-600 dark:text-gray-400">
                    HyDE Cache
                    <InfoTooltip content="Stores generated hypothetical documents. Avoids re-generating HyDE content for similar queries, saving LLM API calls." />
                  </div>
                  <span className="text-gray-900 dark:text-gray-100 font-medium">
                    {hydeHitRate}% hit
                  </span>
                </div>
                <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-purple-500 rounded-full"
                    style={{ width: `${hydeHitRate}%` }}
                  />
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  {cacheStats?.hyde_cache?.size ?? 0}/{cacheStats?.hyde_cache?.max_size ?? 100} entries
                </div>
              </div>
            </div>
          </div>

          {/* Latency Breakdown */}
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-2 mb-4">
              <Clock className="w-5 h-5 text-purple-500" />
              <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                Latency Breakdown
              </h3>
              <InfoTooltip content="Average time spent in each pipeline stage. Query Processing includes rewriting and classification. Generation typically takes the longest as it calls the LLM." />
            </div>
            {latencyItems.length > 0 && totalLatencyMs > 0 ? (
              <div className="space-y-3">
                {latencyItems.map((item) => {
                  const maxMs = Math.max(1000, ...latencyItems.map((i) => i.ms));
                  const percent = Math.min(100, (item.ms / maxMs) * 100);
                  return (
                    <div key={item.step} className="flex items-center gap-3">
                      <span className="w-28 text-sm text-gray-600 dark:text-gray-400">
                        {item.step}
                      </span>
                      <div className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-purple-500 rounded-full"
                          style={{ width: `${percent}%` }}
                        />
                      </div>
                      <span className="w-16 text-sm text-gray-500 dark:text-gray-400 text-right">
                        {item.ms}ms
                      </span>
                    </div>
                  );
                })}
                <div className="pt-3 border-t border-gray-200 dark:border-gray-700 flex justify-between">
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Total Avg
                  </span>
                  <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                    {(totalLatencyMs / 1000).toFixed(2)}s
                  </span>
                </div>
              </div>
            ) : (
              <EmptyState message="No latency data yet" />
            )}
          </div>

          {/* Entity Extraction Stats */}
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 md:col-span-2">
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp className="w-5 h-5 text-emerald-500" />
              <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                Entity Extraction Stats
              </h3>
              <InfoTooltip content="Scientific entities extracted from your documents and queries. Used to enhance retrieval by matching domain-specific terminology. Numbers in parentheses indicate occurrence frequency." />
            </div>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div className="p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-lg">🧪</span>
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Chemicals
                  </span>
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  {renderEntityList(entityStats?.chemicals)}
                </div>
              </div>

              <div className="p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-lg">🧬</span>
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Proteins
                  </span>
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  {renderEntityList(entityStats?.proteins)}
                </div>
              </div>

              <div className="p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-lg">🔬</span>
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Methods
                  </span>
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  {renderEntityList(entityStats?.methods)}
                </div>
              </div>

              <div className="p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-lg">🦠</span>
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Organisms
                  </span>
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  {renderEntityList(entityStats?.organisms)}
                </div>
              </div>

              <div className="p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-lg">📏</span>
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Metrics
                  </span>
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  {renderEntityList(entityStats?.metrics)}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
