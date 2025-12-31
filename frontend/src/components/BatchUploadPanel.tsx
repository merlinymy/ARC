import { useCallback, useState } from 'react';
import { X, Minus, Upload, FileText } from 'lucide-react';
import { useApp } from '../context/AppContext';
import { BatchUploadItem } from './BatchUploadItem';

export function BatchUploadPanel() {
  const {
    state,
    startBatchUpload,
    cancelUploadTask,
    retryUploadTask,
    closeUploadPanel,
    minimizeUploadPanel,
  } = useApp();

  const [isDragOver, setIsDragOver] = useState(false);
  const [showCloseConfirm, setShowCloseConfirm] = useState(false);

  const { activeBatchUpload, isUploadPanelOpen, isUploadPanelMinimized } = state;

  // All hooks must be called before any conditional returns
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragOver(false);

      const files = Array.from(e.dataTransfer.files).filter(
        (f) => f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf')
      );

      if (files.length > 0) {
        if (files.length > 20) {
          alert('Maximum 20 files per batch. Please select fewer files.');
          return;
        }
        startBatchUpload(files);
      }
    },
    [startBatchUpload]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files || []);
      if (files.length > 0) {
        if (files.length > 20) {
          alert('Maximum 20 files per batch. Please select fewer files.');
          return;
        }
        startBatchUpload(files);
      }
      // Reset input so the same files can be selected again
      e.target.value = '';
    },
    [startBatchUpload]
  );

  // Don't render if panel is not open or is minimized
  if (!isUploadPanelOpen || isUploadPanelMinimized) {
    return null;
  }

  const hasActiveUploads = activeBatchUpload?.tasks.some(
    (t) => !['complete', 'error'].includes(t.status)
  );

  const completedCount = activeBatchUpload?.tasks.filter((t) => t.status === 'complete').length ?? 0;
  const failedCount = activeBatchUpload?.tasks.filter((t) => t.status === 'error').length ?? 0;
  const totalCount = activeBatchUpload?.tasks.length ?? 0;
  const skippedCount = activeBatchUpload?.skippedDuplicates ?? 0;

  const handleClose = () => {
    if (hasActiveUploads) {
      setShowCloseConfirm(true);
    } else {
      closeUploadPanel();
    }
  };

  const confirmClose = () => {
    setShowCloseConfirm(false);
    closeUploadPanel();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl w-full max-w-lg mx-4 max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-2">
            <Upload className="h-5 w-5 text-blue-500" />
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Upload Papers
            </h2>
            {totalCount > 0 && (
              <span className="text-sm text-gray-500 dark:text-gray-400">
                ({completedCount}/{totalCount} complete{failedCount > 0 ? `, ${failedCount} failed` : ''}{skippedCount > 0 ? `, ${skippedCount} skipped` : ''})
              </span>
            )}
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={minimizeUploadPanel}
              className="p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
              title="Minimize to sidebar"
            >
              <Minus className="h-5 w-5" />
            </button>
            <button
              onClick={handleClose}
              className="p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
              title="Close"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {/* Drop zone - show when no active batch */}
          {!activeBatchUpload && (
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
                isDragOver
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
              }`}
            >
              <div className="flex flex-col items-center gap-3">
                <div className={`p-3 rounded-full ${isDragOver ? 'bg-blue-100 dark:bg-blue-800' : 'bg-gray-100 dark:bg-gray-700'}`}>
                  <FileText className={`h-8 w-8 ${isDragOver ? 'text-blue-500' : 'text-gray-400'}`} />
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    Drop PDF files here
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    or click to browse (max 20 files, use Ctrl/Cmd+Click to select multiple)
                  </p>
                </div>
                <label className="cursor-pointer">
                  <span className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-500 hover:bg-blue-600 text-white text-sm font-medium transition-colors">
                    <Upload className="h-4 w-4" />
                    Select Files
                  </span>
                  <input
                    type="file"
                    accept=".pdf,application/pdf"
                    multiple={true}
                    onChange={handleFileSelect}
                    className="sr-only"
                  />
                </label>
              </div>
            </div>
          )}

          {/* Task list */}
          {activeBatchUpload && activeBatchUpload.tasks.length > 0 && (
            <div className="space-y-2">
              {/* Add more files button */}
              {!hasActiveUploads && (
                <label className="block mb-4 cursor-pointer">
                  <div
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    className={`border-2 border-dashed rounded-lg p-4 text-center transition-colors ${
                      isDragOver
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                        : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
                    }`}
                  >
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      <span className="text-blue-500 font-medium">Click to add more files</span> or drag and drop
                    </p>
                  </div>
                  <input
                    type="file"
                    accept=".pdf,application/pdf"
                    multiple={true}
                    onChange={handleFileSelect}
                    className="sr-only"
                  />
                </label>
              )}

              {/* Task items */}
              {activeBatchUpload.tasks.map((task) => (
                <BatchUploadItem
                  key={task.taskId}
                  task={task}
                  onCancel={cancelUploadTask}
                  onRetry={retryUploadTask}
                />
              ))}
            </div>
          )}
        </div>

        {/* Footer with info */}
        {activeBatchUpload && hasActiveUploads && (
          <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50 rounded-b-xl">
            <p className="text-xs text-gray-500 dark:text-gray-400 text-center">
              You can minimize this panel and continue using the app while files upload
            </p>
          </div>
        )}
      </div>

      {/* Close confirmation dialog */}
      {showCloseConfirm && (
        <div className="fixed inset-0 z-60 flex items-center justify-center bg-black/50">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl p-6 max-w-sm mx-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
              Uploads in progress
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              Some files are still being uploaded. Are you sure you want to close? You can also minimize to continue uploads in the background.
            </p>
            <div className="flex gap-2 justify-end">
              <button
                onClick={() => setShowCloseConfirm(false)}
                className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setShowCloseConfirm(false);
                  minimizeUploadPanel();
                }}
                className="px-4 py-2 text-sm font-medium text-blue-600 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-lg transition-colors"
              >
                Minimize
              </button>
              <button
                onClick={confirmClose}
                className="px-4 py-2 text-sm font-medium text-white bg-red-500 hover:bg-red-600 rounded-lg transition-colors"
              >
                Close Anyway
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
