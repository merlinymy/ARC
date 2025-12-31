import { X, Download, ExternalLink } from 'lucide-react';
import { useApp } from '../context/AppContext';
import { getPdfUrl } from '../services/api';

interface PdfViewerProps {
  paperId: string;
}

export function PdfViewer({ paperId }: PdfViewerProps) {
  const { state, setViewingPdf } = useApp();

  const paper = state.papers.find((p) => p.id === paperId);
  const pdfUrl = getPdfUrl(paperId);

  const handleClose = () => {
    setViewingPdf(null);
  };

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = pdfUrl;
    link.download = paper?.filename || 'paper.pdf';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleOpenInNewTab = () => {
    window.open(pdfUrl, '_blank');
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
      <div className="w-full h-full max-w-6xl max-h-[90vh] bg-white dark:bg-gray-800 rounded-lg shadow-xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
          <div className="flex-1 min-w-0">
            <h2 className="font-semibold text-gray-900 dark:text-gray-100 truncate">
              {paper?.title || 'PDF Viewer'}
            </h2>
            {paper && (
              <p className="text-sm text-gray-500 dark:text-gray-400 truncate">
                {paper.filename}
              </p>
            )}
          </div>

          <div className="flex items-center gap-2 ml-4">
            <button
              onClick={handleOpenInNewTab}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
              title="Open in new tab"
            >
              <ExternalLink className="w-5 h-5" />
            </button>
            <button
              onClick={handleDownload}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
              title="Download PDF"
            >
              <Download className="w-5 h-5" />
            </button>
            <button
              onClick={handleClose}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
              title="Close"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* PDF Embed */}
        <div className="flex-1 bg-gray-100 dark:bg-gray-900">
          <iframe
            src={pdfUrl}
            className="w-full h-full"
            title={paper?.title || 'PDF Document'}
          />
        </div>
      </div>
    </div>
  );
}
