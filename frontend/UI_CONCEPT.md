# Research Paper Agent - Frontend UI Concept

## Overview

A sophisticated research assistant interface for querying scientific papers with full visibility into the RAG pipeline. The UI emphasizes transparency, source attribution, and conversational interaction.

---

## Layout Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Header: Logo | Status Indicators | Settings | Theme Toggle                │
├────────────────────────┬────────────────────────────────────────────────────┤
│                        │                                                    │
│   Sidebar              │   Main Content Area                                │
│   ─────────            │   ─────────────────                                │
│                        │                                                    │
│   • New Chat           │   ┌──────────────────────────────────────────┐    │
│   • Chat History       │   │  Conversation Thread                     │    │
│   • Paper Library      │   │  ────────────────────                    │    │
│   • Analytics          │   │                                          │    │
│   • Settings           │   │  [Query + Response pairs with sources]   │    │
│                        │   │                                          │    │
│   ─────────            │   │                                          │    │
│   Indexed Papers       │   └──────────────────────────────────────────┘    │
│   • Paper 1            │                                                    │
│   • Paper 2            │   ┌──────────────────────────────────────────┐    │
│   • ...                │   │  Query Input                             │    │
│                        │   │  [                                    ]  │    │
│                        │   │  [Advanced Options ▼]  [Ask]             │    │
│                        │   └──────────────────────────────────────────┘    │
│                        │                                                    │
├────────────────────────┴────────────────────────────────────────────────────┤
│  Footer: Connection Status | Cache Stats | API Health                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Query Input Panel

```
┌─────────────────────────────────────────────────────────────────┐
│  Ask about your research papers...                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ What methods were used to synthesize the peptides?        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ Advanced Options ─────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Query Type:  ○ Auto-detect (Recommended)                  │ │
│  │               ○ Factual  ○ Methods  ○ Summary              │ │
│  │               ○ Comparative  ○ Novelty  ○ Limitations      │ │
│  │                                                            │ │
│  │  Results:     Top-K [15 ▼]    Temperature [0.3 ▼]          │ │
│  │                                                            │ │
│  │  Filters:     □ Specific papers only [Select...]           │ │
│  │               □ Section filter [Methods ▼]                 │ │
│  │                                                            │ │
│  │  Features:    ☑ HyDE  ☑ Query Expansion  ☑ Citation Check  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  [Clear Conversation]                              [Ask ▶]      │
└─────────────────────────────────────────────────────────────────┘
```

**Features:**
- Auto-expanding textarea
- Query type selector (8 types matching backend classification)
- Configurable top-k and temperature
- Paper/section filters
- Feature toggles (HyDE, expansion, citation verification)
- Keyboard shortcut: Cmd/Ctrl+Enter to submit

---

### 2. Response Card with Sources

```
┌─────────────────────────────────────────────────────────────────┐
│  ┌─ Query ─────────────────────────────────────────────────┐   │
│  │ 🔍 What methods were used to synthesize the peptides?   │   │
│  │                                                         │   │
│  │ Query Type: METHODS  │  Expanded: "peptide synthesis,   │   │
│  │                         solid-phase, SPPS, Fmoc"        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─ Answer ────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │  The peptides were synthesized using **solid-phase      │   │
│  │  peptide synthesis (SPPS)** with Fmoc chemistry         │   │
│  │  [Source 1]. The synthesis was performed on Rink        │   │
│  │  amide resin with a loading of 0.5 mmol/g [Source 2].   │   │
│  │                                                         │   │
│  │  Key steps included:                                    │   │
│  │  1. Deprotection with 20% piperidine [Source 1]         │   │
│  │  2. Coupling using HBTU/DIPEA [Source 2]                │   │
│  │  3. Final cleavage with TFA cocktail [Source 3]         │   │
│  │                                                         │   │
│  │  ─────────────────────────────────────────────────────  │   │
│  │  📊 Citation Verification: 92% Trustworthy              │   │
│  │  ✓ All citations verified against source content        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─ Sources (3 of 12 retrieved) ───────────────────────────┐   │
│  │                                                         │   │
│  │  [1] ████████████████░░░░ 0.94                          │   │
│  │      📄 Antimicrobial Peptide Design Study              │   │
│  │      § Methods > Peptide Synthesis                      │   │
│  │      📦 Section chunk  │  1,245 tokens                  │   │
│  │      ┌───────────────────────────────────────────────┐  │   │
│  │      │ "Peptides were synthesized using standard     │  │   │
│  │      │ Fmoc solid-phase peptide synthesis on Rink    │  │   │
│  │      │ amide resin. Deprotection was achieved..."    │  │   │
│  │      └───────────────────────────────────────────────┘  │   │
│  │      [Expand] [Copy] [View in Paper]                    │   │
│  │                                                         │   │
│  │  [2] ████████████████░░░░ 0.91                          │   │
│  │      📄 Antimicrobial Peptide Design Study              │   │
│  │      § Methods > Materials                              │   │
│  │      📦 Fine chunk  │  Parent: Methods section          │   │
│  │      ┌───────────────────────────────────────────────┐  │   │
│  │      │ "Rink amide resin (0.5 mmol/g loading) was    │  │   │
│  │      │ obtained from Sigma-Aldrich. Coupling was     │  │   │
│  │      │ performed using HBTU and DIPEA in DMF..."     │  │   │
│  │      └───────────────────────────────────────────────┘  │   │
│  │      [Expand] [Copy] [View in Paper]                    │   │
│  │                                                         │   │
│  │  [Show 9 more sources...]                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─ Pipeline Details ──────────────────────────────────────┐   │
│  │  Retrieved: 50 → Reranked: 15 → Cited: 3                │   │
│  │  Cache: HIT (embedding) | MISS (search)                 │   │
│  │  Latency: 2.3s total                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Features:**
- Query metadata (type, expansion terms)
- Markdown-rendered answer with clickable citation links
- Citation verification score with visual indicator
- Collapsible source cards with:
  - Relevance score bar
  - Paper title and section hierarchy
  - Chunk type badge (abstract/section/fine/table/caption)
  - Parent context indicator for fine chunks
  - Preview text with expand option
- Pipeline statistics (retrieval counts, cache hits, latency)

---

### 3. Sidebar - Chat History & Papers

```
┌─────────────────────────────┐
│  🔬 Research Assistant      │
├─────────────────────────────┤
│                             │
│  [+ New Conversation]       │
│                             │
│  ─── Today ───              │
│  💬 Peptide synthesis...    │
│  💬 LL-37 antimicrobial...  │
│                             │
│  ─── Yesterday ───          │
│  💬 SRS microscopy...       │
│  💬 IC50 values for...      │
│                             │
├─────────────────────────────┤
│  📚 Paper Library           │
├─────────────────────────────┤
│                             │
│  [↑ Upload PDFs]            │
│                             │
│  ─── Indexed (12) ───       │
│                             │
│  📄 Antimicrobial Peptide   │
│     Design Study            │
│     ✓ 847 chunks indexed    │
│     [View] [Remove]         │
│                             │
│  📄 SRS Microscopy in       │
│     Drug Delivery           │
│     ✓ 623 chunks indexed    │
│     [View] [Remove]         │
│                             │
│  📄 Protein Engineering     │
│     Methods Review          │
│     ⏳ Indexing... 45%      │
│     [Cancel]                │
│                             │
├─────────────────────────────┤
│  📊 Quick Stats             │
│  • 12 papers indexed        │
│  • 8,432 total chunks       │
│  • 156 queries today        │
│  • 78% cache hit rate       │
└─────────────────────────────┘
```

**Features:**
- Conversation history with search
- Paper library with upload capability
- Real-time indexing progress
- Quick stats overview
- Paper management (view details, remove)

---

### 4. Paper Detail View (Modal/Page)

```
┌─────────────────────────────────────────────────────────────────┐
│  📄 Antimicrobial Peptide Design Study                      [×] │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Metadata                                                       │
│  ─────────                                                      │
│  Authors:    Smith, J., Johnson, A., Williams, R.               │
│  Year:       2024                                               │
│  File:       antimicrobial_peptides_2024.pdf                    │
│  Pages:      24                                                 │
│  Indexed:    Dec 15, 2024 at 3:42 PM                            │
│                                                                 │
│  Chunk Statistics                                               │
│  ────────────────                                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Abstract    ████ 1                                       │  │
│  │  Sections    ████████████████ 12                          │  │
│  │  Fine        ████████████████████████████████████ 78      │  │
│  │  Tables      ████████ 6                                   │  │
│  │  Captions    ██████████ 8                                 │  │
│  │  Full        ████ 1                                       │  │
│  └───────────────────────────────────────────────────────────┘  │
│  Total: 106 chunks  │  ~45,000 tokens                           │
│                                                                 │
│  Sections                                                       │
│  ────────                                                       │
│  ▸ Abstract                                                     │
│  ▸ Introduction                                                 │
│  ▾ Methods                                                      │
│    • Peptide Synthesis (3 fine chunks)                          │
│    • Antimicrobial Assays (4 fine chunks)                       │
│    • Cell Culture (2 fine chunks)                               │
│  ▸ Results                                                      │
│  ▸ Discussion                                                   │
│  ▸ Conclusion                                                   │
│  ▸ References                                                   │
│                                                                 │
│  [View Original PDF]  [Re-index]  [Delete from Library]         │
└─────────────────────────────────────────────────────────────────┘
```

---

### 5. Analytics Dashboard

```
┌─────────────────────────────────────────────────────────────────┐
│  📊 Analytics Dashboard                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─ Query Types Distribution ─────┐  ┌─ Response Quality ─────┐ │
│  │                                │  │                        │ │
│  │  FACTUAL     ████████████ 34%  │  │  Avg Citation Score    │ │
│  │  METHODS     ████████ 22%      │  │  ┌────────────────┐    │ │
│  │  SUMMARY     ██████ 18%        │  │  │     87%        │    │ │
│  │  COMPARATIVE ████ 12%          │  │  └────────────────┘    │ │
│  │  NOVELTY     ███ 8%            │  │                        │ │
│  │  LIMITATIONS ██ 4%             │  │  Verified: 94%         │ │
│  │  GENERAL     ██ 2%             │  │  Partial: 4%           │ │
│  │                                │  │  Failed: 2%            │ │
│  └────────────────────────────────┘  └────────────────────────┘ │
│                                                                 │
│  ┌─ Cache Performance ────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Embedding Cache   ████████████████████░░░░░░ 78% hit      │ │
│  │  Search Cache      ████████████░░░░░░░░░░░░░░ 45% hit      │ │
│  │  HyDE Cache        ██████████████████░░░░░░░░ 62% hit      │ │
│  │                                                            │ │
│  │  Entries: 423/500  │  156/200  │  78/100                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─ Latency Breakdown ────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Query Processing   ████ 120ms                             │ │
│  │  Embedding          ████████ 280ms                         │ │
│  │  Retrieval          ██████ 180ms                           │ │
│  │  Reranking          ████████████ 420ms                     │ │
│  │  Generation         ████████████████████████ 890ms         │ │
│  │  ─────────────────────────────────────────                 │ │
│  │  Total Avg          1.89s                                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─ Entity Extraction Stats ──────────────────────────────────┐ │
│  │                                                            │ │
│  │  Top Entities:                                             │ │
│  │  🧪 Chemicals: LL-37 (24), Fmoc (18), TFA (12)             │ │
│  │  🧬 Proteins: BRCA1 (8), TP53 (6), hCAP18 (4)              │ │
│  │  🔬 Methods: HPLC (32), SPPS (28), NMR (15)                │ │
│  │  🦠 Organisms: E. coli (22), HeLa (8)                      │ │
│  │  📏 Metrics: IC50 (14), MIC (12), EC50 (6)                 │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

### 6. Pipeline Visualization (Expandable)

```
┌─────────────────────────────────────────────────────────────────┐
│  🔄 Pipeline Steps                                      [Hide]  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ✓ Step 0   Conversation Resolution          2ms    [cached]   │
│  │          → Resolved "it" to "LL-37 peptide"                  │
│  │                                                              │
│  ✓ Step 1   Query Rewriting                  8ms               │
│  │          → Corrected: "synthesise" → "synthesize"            │
│  │                                                              │
│  ✓ Step 2   Entity Extraction               45ms               │
│  │          → Found: LL-37 (chemical), SPPS (method)            │
│  │                                                              │
│  ✓ Step 3   Query Classification            180ms              │
│  │          → Type: METHODS (0.94 confidence)                   │
│  │                                                              │
│  ✓ Step 4   Query Expansion                  3ms               │
│  │          → Added: "solid-phase", "Fmoc chemistry"            │
│  │                                                              │
│  ✓ Step 5   Strategy Selection               1ms               │
│  │          → Chunks: section, fine | Section: methods          │
│  │                                                              │
│  ○ Step 6   Cache Lookup                     2ms    [MISS]     │
│  │                                                              │
│  ✓ Step 7   Query Embedding + HyDE          320ms              │
│  │          → Generated hypothetical methods excerpt            │
│  │                                                              │
│  ✓ Step 8   Hybrid Search                   180ms              │
│  │          → Dense: 50 results | Sparse: 50 | Fused: 50        │
│  │                                                              │
│  ✓ Step 9   Entity Boosting                  12ms              │
│  │          → Boosted 8 chunks with entity matches              │
│  │                                                              │
│  ✓ Step 10  Reranking                       420ms              │
│  │          → 50 → 15 (dedup: max 5 per paper)                  │
│  │                                                              │
│  ✓ Step 11  Parent Expansion                 45ms              │
│  │          → Expanded 6 fine chunks with parent context        │
│  │                                                              │
│  ✓ Step 12  Answer Generation               890ms              │
│  │          → Claude opus-4.5 | 1,245 tokens                    │
│  │                                                              │
│  ✓ Step 13  Citation Verification           220ms              │
│  │          → 3/3 citations verified (92% trust)                │
│  │                                                              │
│  ✓ Step 14  Memory Update                    5ms               │
│             → Stored turn 4 with paper context                  │
│                                                                 │
│  ────────────────────────────────────────────────────────────── │
│  Total: 2,333ms                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### 7. Health Status Panel

```
┌─────────────────────────────────────────────────────────────────┐
│  🏥 System Health                                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Services                                                       │
│  ────────                                                       │
│  ✓ Qdrant Vector DB        Connected (12,432 vectors)          │
│  ✓ Voyage AI Embeddings    Active (1,847/3,000,000 TPM)        │
│  ✓ Cohere Reranker         Active (234/10,000 RPM)             │
│  ✓ Anthropic Claude        Active (12,456/100,000 TPM)         │
│                                                                 │
│  Rate Limits                                                    │
│  ───────────                                                    │
│  API Requests:  ████████░░░░░░░░░░░░ 42/60 per minute           │
│  Reset in: 34 seconds                                           │
│                                                                 │
│  Indexing Queue                                                 │
│  ──────────────                                                 │
│  ⏳ 2 papers pending                                            │
│  📄 protein_folding.pdf - Processing page 12/45                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Color Scheme & Design Tokens

```
Theme: Light / Dark mode support

Primary:        #2563EB (Blue - actions, links)
Secondary:      #7C3AED (Purple - highlights)
Success:        #10B981 (Green - verified, healthy)
Warning:        #F59E0B (Amber - partial, pending)
Error:          #EF4444 (Red - failed, errors)

Background:
  Light:        #FFFFFF / #F9FAFB / #F3F4F6
  Dark:         #111827 / #1F2937 / #374151

Text:
  Light:        #111827 / #4B5563 / #9CA3AF
  Dark:         #F9FAFB / #D1D5DB / #6B7280

Chunk Type Colors:
  Abstract:     #8B5CF6 (Violet)
  Section:      #3B82F6 (Blue)
  Fine:         #10B981 (Emerald)
  Table:        #F59E0B (Amber)
  Caption:      #EC4899 (Pink)
  Full:         #6B7280 (Gray)
```

---

## Component Hierarchy

```
App
├── Layout
│   ├── Header
│   │   ├── Logo
│   │   ├── StatusIndicators
│   │   ├── SettingsButton
│   │   └── ThemeToggle
│   │
│   ├── Sidebar
│   │   ├── NewChatButton
│   │   ├── ChatHistory
│   │   ├── PaperLibrary
│   │   │   ├── UploadButton
│   │   │   ├── PaperList
│   │   │   └── IndexingProgress
│   │   └── QuickStats
│   │
│   └── MainContent
│       ├── ConversationThread
│       │   ├── QueryCard
│       │   │   ├── QueryText
│       │   │   ├── QueryMetadata (type, expansion)
│       │   │   └── EntityTags
│       │   │
│       │   └── ResponseCard
│       │       ├── AnswerContent (markdown)
│       │       ├── CitationVerification
│       │       ├── SourceList
│       │       │   └── SourceCard
│       │       │       ├── RelevanceBar
│       │       │       ├── PaperInfo
│       │       │       ├── ChunkTypeBadge
│       │       │       ├── TextPreview
│       │       │       └── Actions
│       │       └── PipelineDetails
│       │
│       └── QueryInput
│           ├── TextArea
│           ├── AdvancedOptions
│           │   ├── QueryTypeSelector
│           │   ├── ParameterSliders
│           │   ├── FilterSelectors
│           │   └── FeatureToggles
│           └── SubmitButton
│
├── Modals
│   ├── PaperDetailModal
│   ├── SettingsModal
│   └── PipelineVisualization
│
└── Pages
    ├── ChatPage (default)
    ├── AnalyticsPage
    └── SettingsPage
```

---

## State Management

```typescript
interface AppState {
  // Conversation
  conversations: Conversation[];
  activeConversationId: string | null;

  // Papers
  papers: Paper[];
  indexingQueue: IndexingJob[];

  // Query
  currentQuery: string;
  queryOptions: QueryOptions;
  isLoading: boolean;
  streamingResponse: string | null;

  // System
  health: HealthStatus;
  cacheStats: CacheStats;

  // UI
  sidebarOpen: boolean;
  theme: 'light' | 'dark';
  pipelineExpanded: boolean;
}

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

interface Message {
  id: string;
  type: 'query' | 'response';
  content: string;
  metadata?: {
    queryType?: QueryType;
    expandedQuery?: string;
    entities?: Entity[];
    sources?: Source[];
    citationScore?: number;
    pipelineStats?: PipelineStats;
  };
  timestamp: Date;
}

interface QueryOptions {
  queryType: QueryType | 'auto';
  topK: number;
  temperature: number;
  paperFilter: string[];
  sectionFilter: string | null;
  enableHyde: boolean;
  enableExpansion: boolean;
  enableCitationCheck: boolean;
}
```

---

## API Integration

```typescript
// Query endpoint
POST /query
Request:  { question, top_k?, temperature? }
Response: { answer, sources, query_type, expanded_query,
            retrieval_count, reranked_count }

// Health check
GET /health
Response: { status, qdrant, voyage, cohere, anthropic }

// Statistics
GET /stats
Response: { papers_count, chunks_count, cache_stats }

// Conversation management
POST /conversation/clear
Response: { success }

// Future: Paper management
POST /papers/upload
GET /papers
DELETE /papers/:id
GET /papers/:id/chunks
```

---

## Responsive Breakpoints

```
Mobile:   < 640px   - Single column, bottom sheet for sources
Tablet:   640-1024px - Collapsible sidebar, stacked layout
Desktop:  > 1024px  - Full three-column layout
Wide:     > 1440px  - Expanded source previews
```

---

## Key Interactions

1. **Query Submission**: Type → (optional) Configure → Submit → Stream response → Show sources
2. **Source Exploration**: Click citation → Scroll to source → Expand → View parent context
3. **Paper Upload**: Drag & drop → Show progress → Update library → Ready to query
4. **Conversation**: Auto-save → Resume anytime → Clear to reset context
5. **Pipeline Inspection**: Click "Pipeline Details" → See all 14 steps with timing

---

## Accessibility

- Keyboard navigation throughout
- Screen reader support with ARIA labels
- High contrast mode support
- Focus indicators on all interactive elements
- Semantic HTML structure
- Reduced motion preference support

---

## Future Enhancements

1. **PDF Viewer**: In-app PDF viewing with chunk highlighting
2. **Collaborative**: Share conversations and papers with team
3. **Export**: Export answers with citations to various formats
4. **Annotations**: Add notes to papers and chunks
5. **Custom Prompts**: User-defined prompt templates per query type
6. **Comparison View**: Side-by-side paper comparison
7. **Citation Graph**: Visualize citation relationships
8. **Batch Queries**: Process multiple questions at once
