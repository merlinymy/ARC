import {
  CheckCircle,
  XCircle,
  AlertCircle,
  Database,
  Cloud,
  Cpu,
  Brain,
  Activity,
  RefreshCw,
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { InfoTooltip } from './Tooltip';

export function HealthStatusPanel() {
  const { state, refreshHealth } = useApp();
  const { health, stats } = state;

  const services = health?.services ?? {};

  const getStatusIcon = (status: string | undefined) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'degraded':
        return <AlertCircle className="w-5 h-5 text-amber-500" />;
      default:
        return <XCircle className="w-5 h-5 text-red-500" />;
    }
  };

  const getStatusBadge = (status: string | undefined) => {
    switch (status) {
      case 'healthy':
        return (
          <span className="px-2 py-0.5 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 text-xs rounded-full">
            Healthy
          </span>
        );
      case 'degraded':
        return (
          <span className="px-2 py-0.5 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 text-xs rounded-full">
            Degraded
          </span>
        );
      default:
        return (
          <span className="px-2 py-0.5 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 text-xs rounded-full">
            Unhealthy
          </span>
        );
    }
  };

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            System Health
          </h1>
          <button
            onClick={refreshHealth}
            className="flex items-center gap-2 px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
        </div>

        {/* Overall Status */}
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 mb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div
                className={`w-12 h-12 rounded-full flex items-center justify-center ${
                  health?.status === 'healthy'
                    ? 'bg-green-100 dark:bg-green-900/30'
                    : health?.status === 'degraded'
                    ? 'bg-amber-100 dark:bg-amber-900/30'
                    : 'bg-red-100 dark:bg-red-900/30'
                }`}
              >
                <Activity
                  className={`w-6 h-6 ${
                    health?.status === 'healthy'
                      ? 'text-green-600 dark:text-green-400'
                      : health?.status === 'degraded'
                      ? 'text-amber-600 dark:text-amber-400'
                      : 'text-red-600 dark:text-red-400'
                  }`}
                />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  System Status
                </h2>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {health?.status === 'healthy'
                    ? 'All services are operational'
                    : health?.status === 'degraded'
                    ? 'Some services are experiencing issues'
                    : 'System is experiencing problems'}
                </p>
              </div>
            </div>
            {getStatusBadge(health?.status)}
          </div>
        </div>

        {/* Services Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          {/* Qdrant */}
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                  <Database className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <div className="flex items-center gap-1">
                    <h3 className="font-medium text-gray-900 dark:text-gray-100">
                      Qdrant Vector DB
                    </h3>
                    <InfoTooltip content="High-performance vector database for storing and searching document embeddings. Enables fast semantic similarity search across your research papers." />
                  </div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Vector storage
                  </p>
                </div>
              </div>
              {getStatusIcon(services.qdrant?.status)}
            </div>
            {services.qdrant?.total_vectors !== undefined && (
              <div className="text-sm text-gray-600 dark:text-gray-400">
                {services.qdrant.total_vectors.toLocaleString()} vectors indexed
              </div>
            )}
            {services.qdrant?.error && (
              <div className="mt-2 text-sm text-red-600 dark:text-red-400">
                {services.qdrant.error}
              </div>
            )}
          </div>

          {/* Voyage AI */}
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
                  <Cloud className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                </div>
                <div>
                  <div className="flex items-center gap-1">
                    <h3 className="font-medium text-gray-900 dark:text-gray-100">
                      Voyage AI
                    </h3>
                    <InfoTooltip content="Converts text into dense vector embeddings for semantic search. Uses voyage-3-large model optimized for scientific and technical content." />
                  </div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Embeddings
                  </p>
                </div>
              </div>
              {getStatusIcon(services.voyage_ai?.status)}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Model: {stats?.embedding_model ?? 'voyage-3-large'}
            </div>
            {services.voyage_ai?.error && (
              <div className="mt-2 text-sm text-red-600 dark:text-red-400">
                {services.voyage_ai.error}
              </div>
            )}
          </div>

          {/* Cohere */}
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-amber-100 dark:bg-amber-900/30 rounded-lg">
                  <Cpu className="w-5 h-5 text-amber-600 dark:text-amber-400" />
                </div>
                <div>
                  <div className="flex items-center gap-1">
                    <h3 className="font-medium text-gray-900 dark:text-gray-100">
                      Cohere
                    </h3>
                    <InfoTooltip content="Re-ranks retrieved documents by relevance to improve answer quality. Uses cross-encoder model to score query-document pairs more accurately than initial vector search." />
                  </div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Reranking
                  </p>
                </div>
              </div>
              {getStatusIcon(services.cohere?.status)}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Model: rerank-english-v3.0
            </div>
            {services.cohere?.error && (
              <div className="mt-2 text-sm text-red-600 dark:text-red-400">
                {services.cohere.error}
              </div>
            )}
          </div>

          {/* Anthropic */}
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-emerald-100 dark:bg-emerald-900/30 rounded-lg">
                  <Brain className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                </div>
                <div>
                  <div className="flex items-center gap-1">
                    <h3 className="font-medium text-gray-900 dark:text-gray-100">
                      Anthropic Claude
                    </h3>
                    <InfoTooltip content="Large language model that generates answers from retrieved context. Synthesizes information from multiple sources into coherent, cited responses." />
                  </div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    LLM Generation
                  </p>
                </div>
              </div>
              {getStatusIcon(services.anthropic?.status)}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Model: {stats?.llm_model ?? 'claude-sonnet-4-20250514'}
            </div>
            {services.anthropic?.error && (
              <div className="mt-2 text-sm text-red-600 dark:text-red-400">
                {services.anthropic.error}
              </div>
            )}
          </div>
        </div>

        {/* System Info */}
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center gap-2 mb-4">
            <h3 className="font-semibold text-gray-900 dark:text-gray-100">
              System Information
            </h3>
            <InfoTooltip content="Technical details about the vector database configuration and current session state." />
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <div className="flex items-center gap-1 text-gray-500 dark:text-gray-400">
                Collection
                <InfoTooltip content="Name of the Qdrant collection storing your document vectors. Each collection is an isolated namespace for vector search." />
              </div>
              <div className="font-medium text-gray-900 dark:text-gray-100">
                {stats?.collection_name ?? 'research_papers'}
              </div>
            </div>
            <div>
              <div className="flex items-center gap-1 text-gray-500 dark:text-gray-400">
                Vector Dimension
                <InfoTooltip content="Size of each embedding vector. Higher dimensions capture more semantic nuance but require more storage. 1024 is standard for voyage-3-large." />
              </div>
              <div className="font-medium text-gray-900 dark:text-gray-100">
                {stats?.vector_dimension ?? 1024}
              </div>
            </div>
            <div>
              <div className="flex items-center gap-1 text-gray-500 dark:text-gray-400">
                Total Vectors
                <InfoTooltip content="Total number of document chunks indexed. Each chunk is a searchable segment of text from your research papers." />
              </div>
              <div className="font-medium text-gray-900 dark:text-gray-100">
                {stats?.total_vectors?.toLocaleString() ?? 0}
              </div>
            </div>
            <div>
              <div className="flex items-center gap-1 text-gray-500 dark:text-gray-400">
                Session Turns
                <InfoTooltip content="Number of query-response pairs in the current conversation. Used for conversation memory and context tracking." />
              </div>
              <div className="font-medium text-gray-900 dark:text-gray-100">
                {stats?.conversation_stats?.turns ?? 0}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
