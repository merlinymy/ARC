import {
  Sun,
  Moon,
  Settings,
  Activity,
  Menu,
  Wifi,
  WifiOff,
  Library,
  LogOut,
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { useAuth } from '../context/AuthContext';
import { Tooltip } from './Tooltip';

export function Header() {
  const { state, setTheme, toggleSidebar, setActivePage } = useApp();
  const { logout } = useAuth();
  const { theme, health, sidebarOpen } = state;

  const isHealthy = health?.status === 'healthy';
  const isDegraded = health?.status === 'degraded';

  return (
    <header className="h-14 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 flex items-center justify-between px-4 shrink-0">
      {/* Left section */}
      <div className="flex items-center gap-3">
        <button
          onClick={toggleSidebar}
          className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
          aria-label={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
        >
          <Menu className="w-5 h-5 text-gray-600 dark:text-gray-400" />
        </button>

        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">ARC</span>
          </div>
          <span className="font-semibold text-gray-900 dark:text-gray-100 hidden sm:block">
            ARC
          </span>
        </div>
      </div>

      {/* Center section - Status indicators */}
      <div className="flex items-center gap-4">
        <Tooltip
          content={
            isHealthy
              ? 'All backend services (Qdrant, Voyage AI, Cohere, Anthropic) are operational and responding normally.'
              : isDegraded
              ? 'Some services are experiencing issues. The system may have reduced functionality.'
              : 'Unable to connect to backend services. Check your server connection.'
          }
          position="bottom"
        >
          <div
            className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm cursor-help ${
              isHealthy
                ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                : isDegraded
                ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400'
                : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
            }`}
          >
            {isHealthy ? (
              <Wifi className="w-4 h-4" />
            ) : (
              <WifiOff className="w-4 h-4" />
            )}
            <span className="hidden sm:inline">
              {isHealthy ? 'Connected' : isDegraded ? 'Degraded' : 'Disconnected'}
            </span>
          </div>
        </Tooltip>

        {state.stats && (
          <Tooltip
            content="Total number of document chunks indexed in the vector database. Each chunk represents a searchable segment of your research papers."
            position="bottom"
          >
            <div className="hidden md:flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 cursor-help">
              <span>{state.stats.total_vectors.toLocaleString()} vectors</span>
            </div>
          </Tooltip>
        )}
      </div>

      {/* Right section */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => setActivePage('library')}
          className={`p-2 rounded-lg transition-colors ${
            state.activePage === 'library'
              ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
              : 'hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-600 dark:text-gray-400'
          }`}
          aria-label="Paper Library"
        >
          <Library className="w-5 h-5" />
        </button>

        <button
          onClick={() => setActivePage('analytics')}
          className={`p-2 rounded-lg transition-colors ${
            state.activePage === 'analytics'
              ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
              : 'hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-600 dark:text-gray-400'
          }`}
          aria-label="Analytics"
        >
          <Activity className="w-5 h-5" />
        </button>

        <button
          onClick={() => setActivePage('settings')}
          className={`p-2 rounded-lg transition-colors ${
            state.activePage === 'settings'
              ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
              : 'hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-600 dark:text-gray-400'
          }`}
          aria-label="Settings"
        >
          <Settings className="w-5 h-5" />
        </button>

        <div className="w-px h-6 bg-gray-200 dark:bg-gray-700 mx-1" />

        <button
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
        >
          {theme === 'dark' ? (
            <Sun className="w-5 h-5 text-gray-400" />
          ) : (
            <Moon className="w-5 h-5 text-gray-600" />
          )}
        </button>

        <button
          onClick={logout}
          className="p-2 hover:bg-red-100 dark:hover:bg-red-900/30 rounded-lg transition-colors text-gray-600 dark:text-gray-400 hover:text-red-600 dark:hover:text-red-400"
          aria-label="Sign out"
        >
          <LogOut className="w-5 h-5" />
        </button>
      </div>
    </header>
  );
}
