# Learn React + TypeScript by Building the Research Paper Agent

A hands-on guide to learning React and TypeScript by building a real application.

---

## How This Guide Works

Each lesson introduces **one concept** with:
1. **Concept explanation** - What it is and why it matters
2. **Your task** - What to build
3. **Hints** - If you get stuck
4. **Checkpoint** - How to verify you did it right

Work through each lesson in order. Type everything yourself - don't copy-paste.

---

## Phase 1: Foundation

### Lesson 1: Understanding Your Project Structure

**Concept**: A React + TypeScript project has a specific structure.

Open these files and understand what each does:

```
frontend/
├── src/
│   ├── main.tsx      ← Entry point, mounts React to DOM
│   ├── App.tsx       ← Root component
│   ├── App.css       ← Styles for App
│   └── index.css     ← Global styles
├── index.html        ← HTML template (has <div id="root">)
├── tsconfig.json     ← TypeScript configuration
├── vite.config.ts    ← Build tool configuration
└── package.json      ← Dependencies and scripts
```

**Your Task**:
1. Read `src/main.tsx` - find where React attaches to the DOM
2. Read `src/App.tsx` - this is a React component
3. Run `npm run dev` and open the browser
4. Change the text in `App.tsx` and watch it hot-reload

**Checkpoint**: You can modify App.tsx and see changes in the browser.

---

### Lesson 2: Your First Component

**Concept**: React apps are built from **components** - reusable pieces of UI.

A component is just a function that returns JSX (HTML-like syntax):

```tsx
// This is a React component
function Greeting() {
  return <h1>Hello, World!</h1>;
}
```

**TypeScript addition**: We can type our components:

```tsx
// FC = Function Component (optional but common)
const Greeting: React.FC = () => {
  return <h1>Hello, World!</h1>;
};
```

**Your Task**:
1. Create a new file: `src/components/Header.tsx`
2. Create a simple Header component that displays:
   - The app name: "Research Paper Agent"
   - A subtitle: "Ask questions about your papers"
3. Import and use it in `App.tsx`

**Hints**:
- Create `src/components/` folder first
- Export your component: `export function Header() { ... }`
- Import in App: `import { Header } from './components/Header'`
- Use it like HTML: `<Header />`

**Checkpoint**: Your header appears in the browser.

---

### Lesson 3: Props - Passing Data to Components

**Concept**: **Props** let you pass data from parent to child components.

```tsx
// Define what props the component accepts with TypeScript
interface GreetingProps {
  name: string;
  age?: number;  // ? means optional
}

function Greeting({ name, age }: GreetingProps) {
  return (
    <div>
      <h1>Hello, {name}!</h1>
      {age && <p>You are {age} years old</p>}
    </div>
  );
}

// Usage
<Greeting name="Alice" age={25} />
<Greeting name="Bob" />  // age is optional
```

**Your Task**:
1. Create `src/components/StatusBadge.tsx`
2. It should accept props:
   - `status`: "connected" | "disconnected" | "loading"
   - `label`: string
3. Display different colors based on status:
   - connected = green
   - disconnected = red
   - loading = yellow
4. Use it in Header to show "API Status"

**Hints**:
- Define an interface for your props
- Use inline styles: `style={{ backgroundColor: 'green' }}`
- Or use conditional classes: `className={status === 'connected' ? 'green' : 'red'}`

**Checkpoint**: StatusBadge shows different colors for different statuses.

---

### Lesson 4: State - Making Components Interactive

**Concept**: **State** is data that changes over time. When state changes, React re-renders.

```tsx
import { useState } from 'react';

function Counter() {
  // useState returns [currentValue, setterFunction]
  const [count, setCount] = useState(0);  // 0 is initial value

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>
        Increment
      </button>
    </div>
  );
}
```

**TypeScript**: useState infers types, or you can be explicit:

```tsx
const [count, setCount] = useState<number>(0);
const [name, setName] = useState<string>('');
const [items, setItems] = useState<string[]>([]);
```

**Your Task**:
1. Create `src/components/QueryInput.tsx`
2. Add a textarea for the user's question
3. Use `useState` to track what the user types
4. Add a "Ask" button (doesn't need to do anything yet)
5. Show character count below the textarea

**Hints**:
- `<textarea value={query} onChange={(e) => setQuery(e.target.value)} />`
- `onChange` fires every time the user types
- `e.target.value` is the current text

**Checkpoint**: You can type in the textarea and see the character count update.

---

### Lesson 5: Handling Events

**Concept**: React handles events with camelCase props like `onClick`, `onSubmit`, `onChange`.

```tsx
function Form() {
  const [text, setText] = useState('');

  // Event handler function
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();  // Stop page refresh
    console.log('Submitted:', text);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <button type="submit">Submit</button>
    </form>
  );
}
```

**Common event types in TypeScript**:
- `React.FormEvent` - form submissions
- `React.ChangeEvent<HTMLInputElement>` - input changes
- `React.MouseEvent` - clicks
- `React.KeyboardEvent` - key presses

**Your Task**:
1. Update QueryInput to:
   - Wrap in a `<form>` tag
   - Handle form submission
   - When submitted, log the query to console
   - Clear the input after submission
   - Also submit when user presses Cmd/Ctrl + Enter

**Hints**:
- For keyboard shortcut: `onKeyDown={(e) => { if (e.key === 'Enter' && e.metaKey) ... }}`
- `e.metaKey` is Cmd on Mac, `e.ctrlKey` is Ctrl on Windows

**Checkpoint**: Form logs to console on submit and clears the input.

---

### Lesson 6: Conditional Rendering

**Concept**: Show different UI based on conditions.

```tsx
function UserStatus({ isLoggedIn }: { isLoggedIn: boolean }) {
  // Method 1: Ternary
  return isLoggedIn ? <p>Welcome back!</p> : <p>Please log in</p>;

  // Method 2: && operator (show only if true)
  return isLoggedIn && <p>Welcome back!</p>;

  // Method 3: Early return
  if (!isLoggedIn) {
    return <p>Please log in</p>;
  }
  return <p>Welcome back!</p>;
}
```

**Your Task**:
1. Update QueryInput to show:
   - A "Loading..." state when submitting (use a `isLoading` state)
   - Disable the button and textarea while loading
   - Show "Ask" button text normally, "Asking..." when loading

**Hints**:
- `<button disabled={isLoading}>...</button>`
- `<textarea disabled={isLoading}>...</textarea>`
- Simulate loading: `setIsLoading(true)` then `setTimeout(() => setIsLoading(false), 2000)`

**Checkpoint**: UI shows loading state for 2 seconds after submitting.

---

### Lesson 7: Lists and Keys

**Concept**: Render arrays of data with `.map()`. Each item needs a unique `key`.

```tsx
interface Message {
  id: string;
  text: string;
}

function MessageList({ messages }: { messages: Message[] }) {
  return (
    <ul>
      {messages.map((msg) => (
        <li key={msg.id}>{msg.text}</li>
      ))}
    </ul>
  );
}
```

**Why keys?** React uses keys to track which items changed, added, or removed.

**Your Task**:
1. Create `src/components/ConversationThread.tsx`
2. Define a `Message` interface with: id, type ('query' | 'response'), content, timestamp
3. Create a state array of messages
4. Render each message in a list
5. When QueryInput submits, add a new query message to the list

**Hints**:
- Generate unique IDs: `Date.now().toString()` or `crypto.randomUUID()`
- Style query messages differently from response messages
- Pass a callback from ConversationThread to QueryInput (next lesson covers this better)

**Checkpoint**: You can submit queries and see them appear in a list.

---

### Lesson 8: Lifting State Up

**Concept**: When multiple components need the same state, "lift" it to their common parent.

```tsx
// Parent manages the state
function Parent() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <Display count={count} />
      <Controls onIncrement={() => setCount(c => c + 1)} />
    </div>
  );
}

// Children receive state/callbacks as props
function Display({ count }: { count: number }) {
  return <p>Count: {count}</p>;
}

function Controls({ onIncrement }: { onIncrement: () => void }) {
  return <button onClick={onIncrement}>+1</button>;
}
```

**Your Task**:
1. In `App.tsx`, create state for: `messages` array
2. Pass `messages` to ConversationThread
3. Pass an `onSubmit` callback to QueryInput
4. When QueryInput submits, call `onSubmit` with the query text
5. App adds the message to state

**Checkpoint**: App.tsx manages all conversation state. Components just display/trigger.

---

## Phase 2: Styling

### Lesson 9: Setting Up Tailwind CSS

**Concept**: Tailwind is a utility-first CSS framework. Instead of writing CSS, you use classes.

```tsx
// Traditional CSS
<button className="submit-btn">Submit</button>
// .submit-btn { background: blue; padding: 8px 16px; border-radius: 4px; }

// Tailwind
<button className="bg-blue-500 px-4 py-2 rounded">Submit</button>
```

**Your Task**:
1. Tailwind is already installed. Configure it:

Edit `tailwind.config.js`:
```js
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

2. Replace `src/index.css` with:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

3. Restart the dev server

4. Test it works: add `className="bg-red-500 p-4"` to any element

**Checkpoint**: You see Tailwind styles applied.

---

### Lesson 10: Building a Layout with Tailwind

**Concept**: Common Tailwind patterns for layouts.

```tsx
// Flexbox
<div className="flex">           // horizontal
<div className="flex flex-col">  // vertical
<div className="flex-1">         // grow to fill space

// Grid
<div className="grid grid-cols-3 gap-4">

// Spacing
<div className="p-4">   // padding all sides
<div className="px-4">  // padding left/right
<div className="py-2">  // padding top/bottom
<div className="m-4">   // margin
<div className="space-y-4">  // gap between children

// Sizing
<div className="w-64">      // fixed width
<div className="w-full">    // full width
<div className="h-screen">  // full viewport height
<div className="min-h-screen">

// Colors
<div className="bg-gray-100 text-gray-900">
<div className="bg-blue-500 text-white">
```

**Your Task**:
Create the main layout structure in App.tsx:

```
┌─────────────────────────────────────────┐
│ Header (full width, fixed height)       │
├───────────────┬─────────────────────────┤
│ Sidebar       │ Main Content            │
│ (fixed width) │ (fills remaining)       │
│               │                         │
│               │                         │
└───────────────┴─────────────────────────┘
```

**Hints**:
- Use `h-screen` for full height
- Use `flex` for horizontal layout
- Sidebar: `w-64` (256px) or `w-80` (320px)
- Main: `flex-1` to fill remaining space
- Header: `h-16` (64px)

**Checkpoint**: You have a responsive layout with header, sidebar, and main area.

---

### Lesson 11: Styling Components

**Your Task**: Style these components with Tailwind:

1. **Header**: Dark background, white text, logo on left, status on right
2. **Sidebar**: Light gray background, scrollable if content overflows
3. **QueryInput**:
   - Rounded textarea with border
   - Blue submit button with hover state
   - Disabled state should look different
4. **Message bubbles**:
   - Query = right-aligned, blue background
   - Response = left-aligned, gray background

**Useful Tailwind classes**:
```
// Hover states
hover:bg-blue-600

// Focus states
focus:outline-none focus:ring-2 focus:ring-blue-500

// Transitions
transition-colors duration-200

// Disabled
disabled:opacity-50 disabled:cursor-not-allowed

// Shadows
shadow-sm shadow-md shadow-lg

// Borders
border border-gray-300 rounded-lg
```

**Checkpoint**: Your app looks polished with consistent styling.

---

## Phase 3: Data Fetching & State Management

### Lesson 12: TypeScript Interfaces for API Data

**Concept**: Define types that match your API responses.

Look at your backend's response structure and create matching TypeScript interfaces.

**Your Task**:
1. Create `src/types/index.ts`
2. Define interfaces based on your API:

```tsx
// Match these to your actual backend responses!

export type QueryType =
  | 'FACTUAL'
  | 'METHODS'
  | 'SUMMARY'
  | 'COMPARATIVE'
  | 'NOVELTY'
  | 'LIMITATIONS'
  | 'FRAMING'
  | 'GENERAL';

export interface Source {
  title: string;
  paper_id: string;
  section: string;
  chunk_type: string;
  text: string;
  score: number;
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
  question: string;
  query_type: QueryType;
  expanded_query: string;
  retrieval_count: number;
  reranked_count: number;
}

export interface Message {
  id: string;
  type: 'query' | 'response';
  content: string;
  timestamp: Date;
  // For responses
  sources?: Source[];
  queryType?: QueryType;
  expandedQuery?: string;
}
```

**Checkpoint**: You have type definitions that match your backend.

---

### Lesson 13: Fetching Data with fetch()

**Concept**: Use the Fetch API to call your backend.

```tsx
async function fetchData() {
  try {
    const response = await fetch('http://localhost:8000/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question: 'What is...?' }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Fetch failed:', error);
    throw error;
  }
}
```

**Your Task**:
1. Create `src/api/client.ts`
2. Create functions for each endpoint:
   - `queryPapers(question: string): Promise<QueryResponse>`
   - `getHealth(): Promise<HealthStatus>`
   - `getStats(): Promise<Stats>`
3. Use these functions in your components

**Hints**:
- Create a base URL constant: `const API_BASE = 'http://localhost:8000'`
- Create a helper function for common fetch logic
- Handle errors gracefully

**Checkpoint**: You can call your backend and get real responses.

---

### Lesson 14: useEffect - Side Effects

**Concept**: `useEffect` runs code after render. Used for data fetching, subscriptions, etc.

```tsx
import { useEffect, useState } from 'react';

function HealthStatus() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // This runs after component mounts
    async function checkHealth() {
      try {
        const data = await getHealth();
        setHealth(data);
      } catch (err) {
        setError('Failed to fetch health');
      } finally {
        setLoading(false);
      }
    }

    checkHealth();

    // Optional: poll every 30 seconds
    const interval = setInterval(checkHealth, 30000);

    // Cleanup function - runs when component unmounts
    return () => clearInterval(interval);
  }, []); // Empty array = run once on mount

  if (loading) return <p>Checking...</p>;
  if (error) return <p>Error: {error}</p>;
  return <p>Status: {health?.status}</p>;
}
```

**Dependency array**:
- `[]` - Run once on mount
- `[userId]` - Run when userId changes
- No array - Run after every render (usually wrong!)

**Your Task**:
1. Create a health check component that polls the API
2. Show connection status in the Header
3. Fetch stats on the Analytics page

**Checkpoint**: Health status updates automatically.

---

### Lesson 15: Custom Hooks

**Concept**: Extract reusable logic into custom hooks. Hooks are functions that start with `use`.

```tsx
// src/hooks/useHealth.ts
function useHealth() {
  const [health, setHealth] = useState<Health | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // ... fetch logic
  }, []);

  return { health, loading, error };
}

// Usage in any component
function Header() {
  const { health, loading } = useHealth();
  // ...
}
```

**Your Task**:
1. Create `src/hooks/useQuery.ts` - handles sending queries and tracking state
2. Create `src/hooks/useHealth.ts` - handles health polling
3. Refactor your components to use these hooks

**Checkpoint**: Data fetching logic is reusable across components.

---

### Lesson 16: Global State with Zustand

**Concept**: When state needs to be shared across many components, use a state management library.

```tsx
// src/store/useStore.ts
import { create } from 'zustand';

interface Message {
  id: string;
  content: string;
}

interface AppState {
  messages: Message[];
  addMessage: (message: Message) => void;
  clearMessages: () => void;
}

export const useStore = create<AppState>((set) => ({
  // Initial state
  messages: [],

  // Actions
  addMessage: (message) =>
    set((state) => ({
      messages: [...state.messages, message]
    })),

  clearMessages: () =>
    set({ messages: [] }),
}));

// Usage in components
function MessageList() {
  const messages = useStore((state) => state.messages);
  return /* render messages */;
}

function QueryInput() {
  const addMessage = useStore((state) => state.addMessage);
  const handleSubmit = () => {
    addMessage({ id: '1', content: 'Hello' });
  };
}
```

**Your Task**:
1. Install zustand: `npm install zustand`
2. Create `src/store/useConversationStore.ts` with:
   - `messages` array
   - `activeConversationId`
   - `addMessage`, `clearMessages` actions
3. Create `src/store/useSettingsStore.ts` with:
   - `queryOptions` (topK, temperature, etc.)
   - Actions to update options
4. Refactor App to use stores instead of local state

**Checkpoint**: State persists across component re-renders and is accessible anywhere.

---

## Phase 4: Building Features

### Lesson 17: React Router - Multiple Pages

**Concept**: React Router enables navigation between different views.

```tsx
// src/main.tsx
import { BrowserRouter } from 'react-router-dom';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <BrowserRouter>
    <App />
  </BrowserRouter>
);

// src/App.tsx
import { Routes, Route, Link } from 'react-router-dom';

function App() {
  return (
    <div>
      <nav>
        <Link to="/">Chat</Link>
        <Link to="/analytics">Analytics</Link>
      </nav>

      <Routes>
        <Route path="/" element={<ChatPage />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Routes>
    </div>
  );
}
```

**Your Task**:
1. Set up React Router in main.tsx
2. Create page components: `ChatPage`, `AnalyticsPage`, `SettingsPage`
3. Add navigation links in the Sidebar
4. Move conversation components into ChatPage

**Checkpoint**: You can navigate between pages without full page reload.

---

### Lesson 18: Building the Source Card

**Concept**: Complex components with multiple visual states.

**Your Task**:
Build `src/components/SourceCard.tsx` that displays:

1. Relevance score as a visual bar (0-100%)
2. Paper title
3. Section name with hierarchy (Methods > Synthesis)
4. Chunk type badge with color coding
5. Text preview (truncated)
6. Expandable full text
7. Copy button

**Features to implement**:
- Hover effect
- Expand/collapse animation
- Click to copy text
- Different colors for chunk types

**Hints**:
```tsx
// Truncate text
const preview = text.length > 200 ? text.slice(0, 200) + '...' : text;

// Copy to clipboard
navigator.clipboard.writeText(text);

// Conditional classes with clsx
import clsx from 'clsx';
className={clsx(
  'px-2 py-1 rounded',
  chunkType === 'section' && 'bg-blue-100 text-blue-800',
  chunkType === 'fine' && 'bg-green-100 text-green-800',
)}
```

**Checkpoint**: SourceCard displays all information with expand/copy features.

---

### Lesson 19: Markdown Rendering

**Concept**: Render markdown content (like LLM responses) as formatted HTML.

```tsx
import ReactMarkdown from 'react-markdown';

function Answer({ content }: { content: string }) {
  return (
    <div className="prose prose-sm max-w-none">
      <ReactMarkdown>{content}</ReactMarkdown>
    </div>
  );
}
```

**Your Task**:
1. Install: `npm install react-markdown`
2. For Tailwind typography: `npm install @tailwindcss/typography`
3. Add to tailwind.config.js plugins: `require('@tailwindcss/typography')`
4. Create a response renderer that:
   - Renders markdown
   - Makes `[Source N]` citations clickable
   - Scrolls to corresponding source when clicked

**Hints**:
- Use regex to find citations: `/\[Source \d+\]/g`
- Use custom component renderer in ReactMarkdown for links

**Checkpoint**: LLM responses render with formatting and clickable citations.

---

### Lesson 20: Building the Pipeline Visualization

**Concept**: Visualize the 14-step pipeline with timing data.

**Your Task**:
Build `src/components/PipelineVisualization.tsx`:

1. List all 14 pipeline steps
2. Show status for each: pending, running, completed, skipped
3. Show timing (if available from API)
4. Show details for each step (expanded query, entities found, etc.)
5. Collapsible by default

**This is a stretch goal** - your backend would need to return pipeline timing data.
For now, create a mock version that shows the steps.

**Checkpoint**: Pipeline visualization shows all steps with mock data.

---

### Lesson 21: File Upload

**Concept**: Handle file uploads in React.

```tsx
function FileUpload() {
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      uploadFile(files[0]);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      uploadFile(files[0]);
    }
  };

  return (
    <div
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
      className="border-2 border-dashed p-8"
    >
      <input
        type="file"
        accept=".pdf"
        onChange={handleFileChange}
      />
      <p>Or drag and drop PDF here</p>
    </div>
  );
}
```

**Your Task**:
1. Create `src/components/PaperUpload.tsx`
2. Accept PDF files via click or drag-drop
3. Show upload progress
4. Display success/error state
5. Add to Sidebar

**Note**: Your backend needs an upload endpoint for this to work end-to-end.

**Checkpoint**: You can select/drop files and see them in the UI.

---

## Phase 5: Polish

### Lesson 22: Loading States and Skeletons

**Concept**: Show loading indicators while data fetches.

```tsx
// Skeleton component
function Skeleton({ className }: { className?: string }) {
  return (
    <div
      className={clsx(
        'animate-pulse bg-gray-200 rounded',
        className
      )}
    />
  );
}

// Usage
function SourceCardSkeleton() {
  return (
    <div className="p-4 border rounded">
      <Skeleton className="h-4 w-3/4 mb-2" />
      <Skeleton className="h-3 w-1/2 mb-4" />
      <Skeleton className="h-20 w-full" />
    </div>
  );
}
```

**Your Task**:
1. Create skeleton components for: SourceCard, Message, Stats
2. Show skeletons while data loads
3. Add loading spinner for query submission

**Checkpoint**: App shows smooth loading states instead of blank screens.

---

### Lesson 23: Error Handling

**Concept**: Handle errors gracefully and inform the user.

```tsx
// Error boundary (class component required)
class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  render() {
    if (this.state.hasError) {
      return <p>Something went wrong. Please refresh.</p>;
    }
    return this.props.children;
  }
}

// Inline error handling
function QueryResult() {
  const { data, error, isLoading } = useQuery();

  if (error) {
    return (
      <div className="bg-red-50 text-red-700 p-4 rounded">
        <p>Failed to get response: {error.message}</p>
        <button onClick={retry}>Try Again</button>
      </div>
    );
  }
  // ...
}
```

**Your Task**:
1. Add error states to all data-fetching components
2. Create a reusable ErrorMessage component
3. Handle network errors, API errors, and unexpected errors
4. Add retry functionality

**Checkpoint**: Errors display helpful messages with retry options.

---

### Lesson 24: Dark Mode

**Concept**: Support light and dark themes.

```tsx
// tailwind.config.js
module.exports = {
  darkMode: 'class', // or 'media' for system preference
  // ...
}

// Toggle component
function ThemeToggle() {
  const [dark, setDark] = useState(false);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark);
  }, [dark]);

  return (
    <button onClick={() => setDark(!dark)}>
      {dark ? '☀️' : '🌙'}
    </button>
  );
}

// Usage in components
<div className="bg-white dark:bg-gray-900 text-gray-900 dark:text-white">
```

**Your Task**:
1. Enable dark mode in Tailwind config
2. Create ThemeToggle component
3. Add dark: variants to all your components
4. Save preference to localStorage

**Checkpoint**: Theme toggle works and persists across sessions.

---

### Lesson 25: Keyboard Shortcuts

**Concept**: Add keyboard shortcuts for power users.

```tsx
useEffect(() => {
  const handleKeyDown = (e: KeyboardEvent) => {
    // Cmd/Ctrl + K to focus search
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      inputRef.current?.focus();
    }
  };

  window.addEventListener('keydown', handleKeyDown);
  return () => window.removeEventListener('keydown', handleKeyDown);
}, []);
```

**Your Task**:
1. Add shortcuts:
   - `Cmd+K` - Focus query input
   - `Cmd+Enter` - Submit query
   - `Escape` - Clear input / close modals
   - `Cmd+N` - New conversation

**Checkpoint**: Keyboard shortcuts work throughout the app.

---

## Exercises for Practice

After completing the lessons, try these challenges:

1. **Conversation Persistence**: Save conversations to localStorage
2. **Search History**: Add search/filter for past conversations
3. **Export Feature**: Export conversation as markdown/PDF
4. **Responsive Design**: Make the app work well on mobile
5. **Animations**: Add smooth transitions with Tailwind or Framer Motion
6. **Testing**: Write tests with Vitest and React Testing Library
7. **Performance**: Add React.memo, useMemo, useCallback where appropriate

---

## Resources

### Documentation
- [React Docs](https://react.dev)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [Zustand](https://zustand-demo.pmnd.rs/)
- [React Router](https://reactrouter.com/)

### Recommended Learning Path
1. Complete all lessons above (2-3 weeks)
2. Read React docs "Learn" section
3. Build another small project from scratch
4. Learn testing (Vitest, React Testing Library)
5. Explore advanced patterns (compound components, render props, etc.)

---

## Getting Help

When stuck:
1. Read the error message carefully
2. Check the browser console (F12)
3. Google the exact error message
4. Check React/library documentation
5. Ask me specific questions about what you're trying to do

Good luck! Building this project will teach you real-world React + TypeScript skills.
