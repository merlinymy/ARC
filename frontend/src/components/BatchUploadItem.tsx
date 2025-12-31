import { FileText, X, RotateCcw, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import type { UploadTask } from '../types';

interface BatchUploadItemProps {
  task: UploadTask;
  onCancel: (taskId: string) => void;
  onRetry: (taskId: string) => void;
}

const statusConfig = {
  pending: {
    color: 'text-gray-500',
    bgColor: 'bg-gray-100 dark:bg-gray-700',
    progressColor: 'bg-gray-300',
    label: 'Waiting...',
  },
  uploading: {
    color: 'text-blue-500',
    bgColor: 'bg-blue-50 dark:bg-blue-900/20',
    progressColor: 'bg-blue-500',
    label: 'Uploading...',
  },
  processing: {
    color: 'text-blue-500',
    bgColor: 'bg-blue-50 dark:bg-blue-900/20',
    progressColor: 'bg-blue-500',
    label: 'Processing...',
  },
  extracting: {
    color: 'text-blue-500',
    bgColor: 'bg-blue-50 dark:bg-blue-900/20',
    progressColor: 'bg-blue-500',
    label: 'Extracting text...',
  },
  embedding: {
    color: 'text-purple-500',
    bgColor: 'bg-purple-50 dark:bg-purple-900/20',
    progressColor: 'bg-purple-500',
    label: 'Generating embeddings...',
  },
  indexing: {
    color: 'text-indigo-500',
    bgColor: 'bg-indigo-50 dark:bg-indigo-900/20',
    progressColor: 'bg-indigo-500',
    label: 'Indexing...',
  },
  complete: {
    color: 'text-green-500',
    bgColor: 'bg-green-50 dark:bg-green-900/20',
    progressColor: 'bg-green-500',
    label: 'Complete',
  },
  error: {
    color: 'text-red-500',
    bgColor: 'bg-red-50 dark:bg-red-900/20',
    progressColor: 'bg-red-500',
    label: 'Failed',
  },
};

export function BatchUploadItem({ task, onCancel, onRetry }: BatchUploadItemProps) {
  const config = statusConfig[task.status];
  const isProcessing = ['uploading', 'processing', 'extracting', 'embedding', 'indexing'].includes(task.status);
  const canCancel = ['pending', 'uploading'].includes(task.status);
  const canRetry = task.status === 'error';

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className={`rounded-lg p-3 ${config.bgColor} transition-colors`}>
      <div className="flex items-start gap-3">
        {/* File icon */}
        <div className={`flex-shrink-0 ${config.color}`}>
          <FileText className="h-5 w-5" />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {/* Filename and size */}
          <div className="flex items-center justify-between gap-2">
            <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
              {task.filename}
            </p>
            {task.fileSize > 0 && (
              <span className="text-xs text-gray-500 dark:text-gray-400 flex-shrink-0">
                {formatFileSize(task.fileSize)}
              </span>
            )}
          </div>

          {/* Status and progress */}
          <div className="mt-1 flex items-center gap-2">
            {isProcessing && (
              <Loader2 className={`h-3 w-3 animate-spin ${config.color}`} />
            )}
            {task.status === 'complete' && (
              <CheckCircle className="h-3 w-3 text-green-500" />
            )}
            {task.status === 'error' && (
              <AlertCircle className="h-3 w-3 text-red-500" />
            )}
            <span className={`text-xs ${config.color}`}>
              {task.currentStep || config.label}
            </span>
            {task.progressPercent > 0 && task.status !== 'complete' && task.status !== 'error' && (
              <span className="text-xs text-gray-500 dark:text-gray-400">
                {task.progressPercent}%
              </span>
            )}
          </div>

          {/* Progress bar */}
          {task.status !== 'pending' && (
            <div className="mt-2 h-1.5 w-full bg-gray-200 dark:bg-gray-600 rounded-full overflow-hidden">
              <div
                className={`h-full ${config.progressColor} transition-all duration-300 ease-out`}
                style={{ width: `${task.progressPercent}%` }}
              />
            </div>
          )}

          {/* Error message */}
          {task.status === 'error' && task.errorMessage && (
            <p className="mt-1 text-xs text-red-600 dark:text-red-400 truncate">
              {task.errorMessage}
            </p>
          )}
        </div>

        {/* Actions */}
        <div className="flex-shrink-0 flex items-center gap-1">
          {canRetry && (
            <button
              onClick={() => onRetry(task.taskId)}
              className="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
              title="Retry"
            >
              <RotateCcw className="h-4 w-4" />
            </button>
          )}
          {canCancel && (
            <button
              onClick={() => onCancel(task.taskId)}
              className="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
              title="Cancel"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
