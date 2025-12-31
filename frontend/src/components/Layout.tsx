import type { ReactNode } from 'react';
import { Header } from './Header';
import { Sidebar } from './Sidebar';
import { useApp } from '../context/AppContext';

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const { state, toggleSidebar } = useApp();
  const { sidebarOpen } = state;

  return (
    <div className="h-screen flex flex-col bg-gray-100 dark:bg-gray-950 text-gray-900 dark:text-gray-100">
      <Header />
      <div className="flex flex-1 overflow-hidden relative">
        {/* Mobile overlay backdrop */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 bg-black/50 z-20 lg:hidden"
            onClick={toggleSidebar}
          />
        )}

        {/* Sidebar - overlay on mobile, inline on desktop */}
        <div
          className={`
            fixed inset-y-0 left-0 z-30 w-72 transform transition-transform duration-200 ease-in-out
            lg:relative lg:translate-x-0 lg:z-0
            ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
            top-14 lg:top-0 h-[calc(100vh-3.5rem)] lg:h-full
          `}
        >
          <Sidebar />
        </div>

        {/* Main content */}
        <main className="flex-1 flex flex-col min-h-0 min-w-0">
          {children}
        </main>
      </div>
    </div>
  );
}
