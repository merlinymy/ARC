import { Upload, ChevronUp, CheckCircle, AlertCircle } from 'lucide-react';
import { useApp } from '../context/AppContext';

export function MinimizedUploadWidget() {
  const { state, maximizeUploadPanel } = useApp();
  const { activeBatchUpload, isUploadPanelMinimized } = state;

  // Only show when minimized and there's an active batch
  if (!isUploadPanelMinimized || !activeBatchUpload) {
    return null;
  }

  const tasks = activeBatchUpload.tasks;
  const completedCount = tasks.filter((t) => t.status === 'complete').length;
  const failedCount = tasks.filter((t) => t.status === 'error').length;
  const totalCount = tasks.length;
  // Calculate overall progress
  const totalProgress = tasks.reduce((sum, t) => sum + t.progressPercent, 0);
  const overallProgress = totalCount > 0 ? Math.round(totalProgress / totalCount) : 0;

  // Find current processing file
  const currentTask = tasks.find(
    (t) => !['complete', 'error', 'pending'].includes(t.status)
  );

  const isComplete = completedCount + failedCount === totalCount;
  const hasErrors = failedCount > 0;

  return (
    <button
      onClick={maximizeUploadPanel}
      className={`w-full p-3 rounded-lg border transition-all hover:shadow-md ${
        isComplete
          ? hasErrors
            ? 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800'
            : 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
          : 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800'
      }`}
    >
      <div className="flex items-center gap-3">
        {/* Progress circle */}
        <div className="relative flex-shrink-0">
          <svg className="w-10 h-10 -rotate-90">
            <circle
              cx="20"
              cy="20"
              r="16"
              fill="none"
              stroke="currentColor"
              strokeWidth="3"
              className="text-gray-200 dark:text-gray-700"
            />
            <circle
              cx="20"
              cy="20"
              r="16"
              fill="none"
              stroke="currentColor"
              strokeWidth="3"
              strokeDasharray={`${overallProgress} 100`}
              strokeLinecap="round"
              className={
                isComplete
                  ? hasErrors
                    ? 'text-amber-500'
                    : 'text-green-500'
                  : 'text-blue-500'
              }
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            {isComplete ? (
              hasErrors ? (
                <AlertCircle className="h-4 w-4 text-amber-500" />
              ) : (
                <CheckCircle className="h-4 w-4 text-green-500" />
              )
            ) : (
              <Upload className="h-4 w-4 text-blue-500 animate-pulse" />
            )}
          </div>
        </div>

        {/* Text content */}
        <div className="flex-1 min-w-0 text-left">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
              {isComplete
                ? hasErrors
                  ? 'Upload complete with errors'
                  : 'Upload complete'
                : 'Uploading papers'}
            </p>
            <ChevronUp className="h-4 w-4 text-gray-400 flex-shrink-0" />
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
            {isComplete ? (
              <>
                {completedCount} complete
                {failedCount > 0 && <>, {failedCount} failed</>}
              </>
            ) : currentTask ? (
              <>
                {currentTask.filename.slice(0, 25)}
                {currentTask.filename.length > 25 && '...'}
              </>
            ) : (
              `${completedCount}/${totalCount} complete`
            )}
          </p>
        </div>
      </div>

      {/* Progress bar */}
      {!isComplete && (
        <div className="mt-2 h-1 w-full bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-500 transition-all duration-300 ease-out"
            style={{ width: `${overallProgress}%` }}
          />
        </div>
      )}
    </button>
  );
}
