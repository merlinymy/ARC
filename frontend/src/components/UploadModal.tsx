import { useState, useCallback, useRef } from 'react';
import { X, Upload, FileText, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { uploadPaperStream } from '../services/api';
import type { UploadProgressEvent } from '../types';

interface UploadModalProps {
  onClose: () => void;
  onComplete: () => void;
}

type UploadStatus = 'idle' | 'uploading' | 'processing' | 'complete' | 'error';

interface UploadState {
  status: UploadStatus;
  progress: string;
  message: string;
  filename: string;
}

export function UploadModal({ onClose, onComplete }: UploadModalProps) {
  const [uploadState, setUploadState] = useState<UploadState>({
    status: 'idle',
    progress: '',
    message: '',
    filename: '',
  });
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setUploadState({
        status: 'error',
        progress: '',
        message: 'Only PDF files are allowed',
        filename: file.name,
      });
      return;
    }

    setUploadState({
      status: 'uploading',
      progress: 'Uploading...',
      message: '',
      filename: file.name,
    });

    try {
      await uploadPaperStream(file, (event: UploadProgressEvent) => {
        if (event.type === 'progress') {
          setUploadState((prev) => ({
            ...prev,
            status: 'processing',
            progress: event.step || 'Processing...',
            message: event.data?.message as string || '',
          }));
        } else if (event.type === 'complete') {
          setUploadState((prev) => ({
            ...prev,
            status: 'complete',
            progress: 'Complete',
            message: event.message || `Successfully indexed ${event.chunks} chunks`,
          }));
        } else if (event.type === 'error') {
          setUploadState((prev) => ({
            ...prev,
            status: 'error',
            progress: '',
            message: event.message || 'Upload failed',
          }));
        }
      });
    } catch (error) {
      setUploadState((prev) => ({
        ...prev,
        status: 'error',
        progress: '',
        message: error instanceof Error ? error.message : 'Upload failed',
      }));
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);

      const file = e.dataTransfer.files[0];
      if (file) {
        handleFile(file);
      }
    },
    [handleFile]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        handleFile(file);
      }
    },
    [handleFile]
  );

  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  const handleDone = () => {
    if (uploadState.status === 'complete') {
      onComplete();
    } else {
      onClose();
    }
  };

  const handleUploadAnother = () => {
    setUploadState({
      status: 'idle',
      progress: '',
      message: '',
      filename: '',
    });
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
      <div className="w-full max-w-md bg-white dark:bg-gray-800 rounded-lg shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700">
          <h2 className="font-semibold text-gray-900 dark:text-gray-100">
            Upload PDF
          </h2>
          <button
            onClick={onClose}
            className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {uploadState.status === 'idle' && (
            <>
              {/* Drop Zone */}
              <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                  isDragOver
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                    : 'border-gray-300 dark:border-gray-600 hover:border-blue-400 dark:hover:border-blue-500'
                }`}
              >
                <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400 dark:text-gray-500" />
                <p className="text-gray-600 dark:text-gray-300 mb-2">
                  Drag and drop your PDF here
                </p>
                <p className="text-sm text-gray-400 dark:text-gray-500 mb-4">or</p>
                <button
                  onClick={handleBrowseClick}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                >
                  Browse Files
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf"
                  onChange={handleFileInput}
                  className="hidden"
                />
              </div>
              <p className="text-xs text-gray-400 dark:text-gray-500 mt-3 text-center">
                Maximum file size: 50MB
              </p>
            </>
          )}

          {(uploadState.status === 'uploading' || uploadState.status === 'processing') && (
            <div className="text-center py-8">
              <Loader2 className="w-12 h-12 mx-auto mb-4 text-blue-600 animate-spin" />
              <div className="flex items-center justify-center gap-2 mb-2">
                <FileText className="w-4 h-4 text-gray-400" />
                <span className="text-sm text-gray-600 dark:text-gray-300 truncate max-w-xs">
                  {uploadState.filename}
                </span>
              </div>
              <p className="text-blue-600 dark:text-blue-400 font-medium">
                {uploadState.progress}
              </p>
              {uploadState.message && (
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
                  {uploadState.message}
                </p>
              )}
            </div>
          )}

          {uploadState.status === 'complete' && (
            <div className="text-center py-8">
              <CheckCircle className="w-12 h-12 mx-auto mb-4 text-green-500" />
              <div className="flex items-center justify-center gap-2 mb-2">
                <FileText className="w-4 h-4 text-gray-400" />
                <span className="text-sm text-gray-600 dark:text-gray-300 truncate max-w-xs">
                  {uploadState.filename}
                </span>
              </div>
              <p className="text-green-600 dark:text-green-400 font-medium mb-2">
                Upload Complete
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                {uploadState.message}
              </p>
            </div>
          )}

          {uploadState.status === 'error' && (
            <div className="text-center py-8">
              <AlertCircle className="w-12 h-12 mx-auto mb-4 text-red-500" />
              {uploadState.filename && (
                <div className="flex items-center justify-center gap-2 mb-2">
                  <FileText className="w-4 h-4 text-gray-400" />
                  <span className="text-sm text-gray-600 dark:text-gray-300 truncate max-w-xs">
                    {uploadState.filename}
                  </span>
                </div>
              )}
              <p className="text-red-600 dark:text-red-400 font-medium mb-2">
                Upload Failed
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                {uploadState.message}
              </p>
            </div>
          )}
        </div>

        {/* Footer */}
        {uploadState.status !== 'idle' && (
          <div className="flex items-center justify-end gap-3 px-4 py-3 border-t border-gray-200 dark:border-gray-700">
            {(uploadState.status === 'complete' || uploadState.status === 'error') && (
              <button
                onClick={handleUploadAnother}
                className="px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
              >
                Upload Another
              </button>
            )}
            <button
              onClick={handleDone}
              disabled={uploadState.status === 'uploading' || uploadState.status === 'processing'}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {uploadState.status === 'complete' ? 'Done' : 'Close'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
