import { useState, type FormEvent } from 'react';
import { Lock, User, AlertCircle, Loader2 } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export function LoginPage() {
  const { login, state, clearError } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !password.trim()) return;
    await login(username.trim(), password);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 px-4">
      <div className="max-w-md w-full space-y-8">
        <div className="text-center">
          <div className="mx-auto h-16 w-16 flex items-center justify-center rounded-full bg-indigo-100 dark:bg-indigo-900">
            <Lock className="h-8 w-8 text-indigo-600 dark:text-indigo-400" />
          </div>
          <h2 className="mt-6 text-3xl font-bold text-gray-900 dark:text-white">
            Research Paper RAG
          </h2>
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
            Sign in to access your research assistant
          </p>
        </div>

        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          {state.error && (
            <div className="flex items-center gap-2 p-3 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-400">
              <AlertCircle className="h-5 w-5 flex-shrink-0" />
              <span className="text-sm">{state.error}</span>
              <button
                type="button"
                onClick={clearError}
                className="ml-auto text-red-500 hover:text-red-700 dark:hover:text-red-300"
              >
                &times;
              </button>
            </div>
          )}

          <div className="space-y-4">
            <div>
              <label htmlFor="username" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Username
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <User className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  id="username"
                  name="username"
                  type="text"
                  autoComplete="username"
                  required
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="block w-full pl-10 pr-3 py-2.5 border border-gray-300 dark:border-gray-600 rounded-lg
                           bg-white dark:bg-gray-800 text-gray-900 dark:text-white
                           placeholder-gray-400 dark:placeholder-gray-500
                           focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent
                           transition-colors"
                  placeholder="Enter your username"
                />
              </div>
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Password
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  id="password"
                  name="password"
                  type="password"
                  autoComplete="current-password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="block w-full pl-10 pr-3 py-2.5 border border-gray-300 dark:border-gray-600 rounded-lg
                           bg-white dark:bg-gray-800 text-gray-900 dark:text-white
                           placeholder-gray-400 dark:placeholder-gray-500
                           focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent
                           transition-colors"
                  placeholder="Enter your password"
                />
              </div>
            </div>
          </div>

          <button
            type="submit"
            disabled={state.isLoading || !username.trim() || !password.trim()}
            className="w-full flex justify-center items-center gap-2 py-2.5 px-4
                     border border-transparent rounded-lg shadow-sm
                     text-white bg-indigo-600 hover:bg-indigo-700
                     focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500
                     disabled:opacity-50 disabled:cursor-not-allowed
                     transition-colors"
          >
            {state.isLoading ? (
              <>
                <Loader2 className="h-5 w-5 animate-spin" />
                Signing in...
              </>
            ) : (
              'Sign in'
            )}
          </button>
        </form>

        <p className="text-center text-xs text-gray-500 dark:text-gray-400">
          Default credentials are configured in your .env file
        </p>
      </div>
    </div>
  );
}
