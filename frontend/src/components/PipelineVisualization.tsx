import { CheckCircle, SkipForward, Clock, Zap, X } from 'lucide-react';

interface PipelineStep {
  step: number;
  name: string;
  duration: number;
  cached?: boolean;
  details?: string;
  status: 'completed' | 'skipped' | 'pending';
}

interface PipelineVisualizationProps {
  steps: PipelineStep[];
  totalDuration: number;
  onClose: () => void;
}

const SAMPLE_STEPS: PipelineStep[] = [
  { step: 0, name: 'Conversation Resolution', duration: 2, cached: true, details: 'Resolved "it" to "LL-37 peptide"', status: 'completed' },
  { step: 1, name: 'Query Rewriting', duration: 8, details: 'Corrected: "synthesise" → "synthesize"', status: 'completed' },
  { step: 2, name: 'Entity Extraction', duration: 45, details: 'Found: LL-37 (chemical), SPPS (method)', status: 'completed' },
  { step: 3, name: 'Query Classification', duration: 180, details: 'Type: METHODS (0.94 confidence)', status: 'completed' },
  { step: 4, name: 'Query Expansion', duration: 3, details: 'Added: "solid-phase", "Fmoc chemistry"', status: 'completed' },
  { step: 5, name: 'Strategy Selection', duration: 1, details: 'Chunks: section, fine | Section: methods', status: 'completed' },
  { step: 6, name: 'Cache Lookup', duration: 2, cached: false, details: 'MISS', status: 'skipped' },
  { step: 7, name: 'Query Embedding + HyDE', duration: 320, details: 'Generated hypothetical methods excerpt', status: 'completed' },
  { step: 8, name: 'Hybrid Search', duration: 180, details: 'Dense: 50 results | Sparse: 50 | Fused: 50', status: 'completed' },
  { step: 9, name: 'Entity Boosting', duration: 12, details: 'Boosted 8 chunks with entity matches', status: 'completed' },
  { step: 10, name: 'Reranking', duration: 420, details: '50 → 15 (dedup: max 5 per paper)', status: 'completed' },
  { step: 11, name: 'Parent Expansion', duration: 45, details: 'Expanded 6 fine chunks with parent context', status: 'completed' },
  { step: 12, name: 'Answer Generation', duration: 890, details: 'Claude opus-4.5 | 1,245 tokens', status: 'completed' },
  { step: 13, name: 'Citation Verification', duration: 220, details: '3/3 citations verified (92% trust)', status: 'completed' },
  { step: 14, name: 'Memory Update', duration: 5, details: 'Stored turn 4 with paper context', status: 'completed' },
];

export function PipelineVisualization({ steps = SAMPLE_STEPS, totalDuration = 2333, onClose }: PipelineVisualizationProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative w-full max-w-2xl max-h-[90vh] overflow-hidden bg-white dark:bg-gray-900 rounded-2xl shadow-2xl flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-amber-500" />
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Pipeline Steps
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="space-y-0">
            {steps.map((step, index) => (
              <div key={step.step} className="relative">
                {/* Connector line */}
                {index < steps.length - 1 && (
                  <div className="absolute left-4 top-10 w-0.5 h-full bg-gray-200 dark:bg-gray-700" />
                )}

                {/* Step */}
                <div className="flex items-start gap-4">
                  {/* Icon */}
                  <div className="relative z-10">
                    {step.status === 'completed' ? (
                      <div className="w-8 h-8 bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center">
                        <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
                      </div>
                    ) : step.status === 'skipped' ? (
                      <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center">
                        <SkipForward className="w-5 h-5 text-blue-500 dark:text-blue-400" />
                      </div>
                    ) : (
                      <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center">
                        <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse" />
                      </div>
                    )}
                  </div>

                  {/* Content */}
                  <div className="flex-1 pb-6">
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-400 dark:text-gray-500">
                          Step {step.step}
                        </span>
                        <span className="font-medium text-gray-900 dark:text-gray-100">
                          {step.name}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 text-sm">
                        <span className="text-gray-500 dark:text-gray-400">
                          {step.duration}ms
                        </span>
                        {step.cached !== undefined && (
                          <span
                            className={`px-2 py-0.5 rounded text-xs ${
                              step.cached
                                ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                                : 'bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400'
                            }`}
                          >
                            {step.cached ? 'cached' : 'MISS'}
                          </span>
                        )}
                      </div>
                    </div>

                    {step.details && (
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        → {step.details}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
              <Clock className="w-4 h-4" />
              <span>Total: {totalDuration}ms</span>
            </div>
            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 bg-green-500 rounded-full" />
                <span className="text-gray-600 dark:text-gray-400">Completed</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 bg-blue-500 rounded-full" />
                <span className="text-gray-600 dark:text-gray-400">Skipped</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
