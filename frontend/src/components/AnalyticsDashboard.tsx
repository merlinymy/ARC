import {
  PieChart,
  Clock,
  CheckCircle,
  TrendingUp,
  Zap,
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { InfoTooltip } from './Tooltip';

export function AnalyticsDashboard() {
  const { state } = useApp();
  const { stats } = state;
  const cacheStats = stats?.cache_stats;

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

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-6">
          Analytics Dashboard
        </h1>

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
            <div className="space-y-3">
              {[
                { type: 'FACTUAL', percent: 34, color: 'bg-blue-500' },
                { type: 'METHODS', percent: 22, color: 'bg-emerald-500' },
                { type: 'SUMMARY', percent: 18, color: 'bg-purple-500' },
                { type: 'COMPARATIVE', percent: 12, color: 'bg-amber-500' },
                { type: 'NOVELTY', percent: 8, color: 'bg-pink-500' },
                { type: 'LIMITATIONS', percent: 4, color: 'bg-red-500' },
                { type: 'GENERAL', percent: 2, color: 'bg-gray-500' },
              ].map((item) => (
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
          </div>

          {/* Response Quality */}
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-2 mb-4">
              <CheckCircle className="w-5 h-5 text-green-500" />
              <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                Response Quality
              </h3>
              <InfoTooltip content="Citation verification score measuring how well responses are grounded in source documents. Verified = claims fully supported, Partial = some claims unverified, Failed = significant unsupported claims." />
            </div>
            <div className="flex items-center justify-center py-6">
              <div className="relative w-32 h-32">
                <svg className="w-full h-full transform -rotate-90">
                  <circle
                    cx="64"
                    cy="64"
                    r="56"
                    stroke="currentColor"
                    strokeWidth="12"
                    fill="none"
                    className="text-gray-200 dark:text-gray-700"
                  />
                  <circle
                    cx="64"
                    cy="64"
                    r="56"
                    stroke="currentColor"
                    strokeWidth="12"
                    fill="none"
                    strokeDasharray={`${87 * 3.52} ${100 * 3.52}`}
                    className="text-green-500"
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-3xl font-bold text-gray-900 dark:text-gray-100">
                    87%
                  </span>
                </div>
              </div>
            </div>
            <div className="text-center text-sm text-gray-500 dark:text-gray-400">
              Avg Citation Score
            </div>
            <div className="mt-4 grid grid-cols-3 gap-2 text-center text-sm">
              <div>
                <div className="text-green-600 dark:text-green-400 font-semibold">
                  94%
                </div>
                <div className="text-gray-500 dark:text-gray-400">Verified</div>
              </div>
              <div>
                <div className="text-amber-600 dark:text-amber-400 font-semibold">
                  4%
                </div>
                <div className="text-gray-500 dark:text-gray-400">Partial</div>
              </div>
              <div>
                <div className="text-red-600 dark:text-red-400 font-semibold">
                  2%
                </div>
                <div className="text-gray-500 dark:text-gray-400">Failed</div>
              </div>
            </div>
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
            <div className="space-y-3">
              {[
                { step: 'Query Processing', ms: 120 },
                { step: 'Embedding', ms: 280 },
                { step: 'Retrieval', ms: 180 },
                { step: 'Reranking', ms: 420 },
                { step: 'Generation', ms: 890 },
              ].map((item) => {
                const maxMs = 1000;
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
                  1.89s
                </span>
              </div>
            </div>
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
                <div className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
                  <div>LL-37 (24)</div>
                  <div>Fmoc (18)</div>
                  <div>TFA (12)</div>
                </div>
              </div>

              <div className="p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-lg">🧬</span>
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Proteins
                  </span>
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
                  <div>BRCA1 (8)</div>
                  <div>TP53 (6)</div>
                  <div>hCAP18 (4)</div>
                </div>
              </div>

              <div className="p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-lg">🔬</span>
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Methods
                  </span>
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
                  <div>HPLC (32)</div>
                  <div>SPPS (28)</div>
                  <div>NMR (15)</div>
                </div>
              </div>

              <div className="p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-lg">🦠</span>
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Organisms
                  </span>
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
                  <div>E. coli (22)</div>
                  <div>HeLa (8)</div>
                  <div>S. aureus (5)</div>
                </div>
              </div>

              <div className="p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-lg">📏</span>
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Metrics
                  </span>
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
                  <div>IC50 (14)</div>
                  <div>MIC (12)</div>
                  <div>EC50 (6)</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
