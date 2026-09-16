# ARC Improvement Plan — September 2026

Scope: `/Users/merlin/projects/ARC` (backend) and `/Users/merlin/projects/ARCFrontEnd` (frontend).
ARC-Cloud is retired and is not a target. ARC is production, single user.

Driver: user feedback transcript (10 requests, below) plus a full-stack audit of code, live data, and logs.

---

## 1. Audit baseline (measured, not estimated)

| Metric | Value |
|---|---|
| PDFs on volume | 4,861 |
| Papers indexed | 4,792 (5 failed) |
| Chunks in Qdrant | 212,651 |
| Conversations / messages | 46 / 276 |
| `user_memory` rows | **0** |
| Qdrant storage | 2.2 GB |
| Persisted extraction output | **none** (`processed_data/` is empty) |

Chunk composition:

```
caption   93,144  (44%)   abstract   2,232  (46% of papers)
fine      83,946          full       4,284
section   26,758          table          0   <-- zero, corpus-wide
```

Two denominators appear below and they are not interchangeable: **4,792** papers reached Qdrant, while the checkpoint holds metadata for **4,797** (the 4,792 plus 5 that failed after metadata extraction). Chunk- and metadata-derived percentages use 4,797.

- 13.5% of papers have **zero** fine chunks; 14.6% have zero section chunks
- 56.4% of papers have more caption chunks than fine (body) chunks
- `page_count == 0` for **all 4,797** papers
- DOI present on **118 / 4,797** (2.5%); year missing on 1,075; zero authors on 1,142
- Library hygiene: 166 files with `_N.pdf` duplicate suffixes, 32 duplicate titles across 71 papers, 6 papers titled "untitled"
- MinerU succeeded for ~96.9% of papers; the pypdfium2 fallback handled ~3.1%. Extraction is **not** the main problem — the integration around it is.
- Avg assistant message 5,161 chars (max 15,787) vs avg 344-char question

### Root causes

| # | Defect | Location | Consequence |
|---|---|---|---|
| R1 | Reads `item["latex"]`/`item["text"]` for tables; MinerU emits `table_body` | `preprocessing/pdf_processor.py:358` | Zero table chunks. `FACTUAL` strategy queries an empty chunk type. |
| R2 | Caption regex `finditer` over full markdown matches in-text "Figure 3…" mentions | `preprocessing/pdf_processor.py:381-400` | 93k pseudo-caption chunks crowding out real content. MinerU's clean `image_caption`/`table_caption` lists ignored. |
| R3 | Sections truncated to 3–4k tokens, then fine chunks built from the **truncated** text | `preprocessing/chunker.py:371, 394-401` | Tails of long Results/Discussion sections never indexed at any granularity. |
| R4 | `enable_hybrid_search` never passed, defaults `False`; sparse vectors written anyway | `dependencies.py:109-126`, `retrieval/query_engine.py:462` | Dense-only retrieval. BM25 computed and stored for 212k chunks, never queried. |
| R5 | Sources cut to 1,000 chars before the model sees them; parent context = first 500 chars of section | `retrieval/query_engine.py:1559, 1215` | Retrieved evidence discarded pre-generation; parent window is not around the match. |
| R6 | `Chunk.page_numbers` defined in model + payload, never populated; MinerU `page_idx`/`bbox` discarded | `preprocessing/chunker.py` (all constructors), `pdf_processor.py:337-348` | No provenance possible. `PdfViewer.tsx` iframe opens at page 1. |
| R7 | `ConversationMemory` is a process-global singleton, 2,000-token window, not keyed by conversation | `retrieval/query_engine.py:524, 1304`; `dependencies.py:107` | Cross-thread contamination. `selectConversation` never resets it. |
| R8 | Full DB history used only to render UI, never sent to the model | `api/main.py:716-948` | Every turn is near-amnesiac. |
| R9 | `get_memory_context()` never injected into any prompt | `services/memory_service.py:77`, endpoint at `api/main.py:2655` | The ChatGPT-style profile feature is inert. |
| R10 | "Confidence" = LLM self-reported float, or bag-of-words overlap on parse failure; UI keeps the min per source | `retrieval/citation_verifier.py:144-210`, `ResponseCard.tsx:112` | The percentage is not a meaningful quantity. |
| R11 | UI relevance bar shows pre-rerank `score`; `rerank_score` computed and never read | `api/main.py:829`, `retrieval/reranker.py:79` | Displayed relevance is the wrong number. |
| R12 | `response_mode` defaults `detailed`, whose prompts mandate numbered report scaffolds at 32k max_tokens | `retrieval/query_engine.py:199-322, 544` | The "book report" voice. |
| R13 | BM25 IDF cache `doc_count` = 13,979 against 212,953 points (**6.6% coverage**), and `bm25.py:282` bakes `idf` into the **stored document vectors** rather than applying it at query time | `data/bm25_idf_cache.json`, `retrieval/bm25.py:266-283` | Every stored sparse vector is stamped with 6.6%-coverage statistics. A cache rebuild alone would leave query-side IDF disagreeing with doc-side IDF, so fixing this means **re-vectorizing and re-upserting all 212,953 sparse vectors** (zero API cost — computed locally from stored text — but a real batch job). Confirmed live: sparse vectors are present and populated (112 terms on a sampled point), and hybrid already surfaces 7-8 of 15 chunks that dense-only misses *despite* the handicap. |
| R14 | Models are three generations behind, and `temperature` is passed on every call | `config.py:99-103`; `query_engine.py:1346, 1372, 1448`; `query_classifier.py:232, 329`; `api/main.py:354` | ARC **cannot run on a current model at all** — `temperature` is a 400 on Opus 4.7+ and on Sonnet 5. Also pins the stale `web_search_20250305` tool. See §3b.0. |
| R15 | No prompt caching anywhere | all Anthropic call sites | Every turn re-bills the full system prompt and all source text. Compounds badly once W3 feeds real history. See §3b.3. |
| R17 | The `FACTUAL` retrieval strategy searches `["fine", "table", "caption"]` — a chunk type with **zero** members, 93k caption fragments, and neither `section` nor `abstract` | `retrieval/query_classifier.py:54-59` (`RETRIEVAL_STRATEGIES`) | **Measured after the W0 migration: 5 of 8 sampled real user queries classify `FACTUAL`**, so the highest-traffic path is the worst-configured one. Excludes 26,758 `section` chunks of real body text while including a dead filter term. Plausibly the largest single contributor to "it consistently misses things" on the numeric questions the user values most. |
| R16 | Section and abstract detection is pure regex over raw lines — header must start a line, be <100 chars, and survive no markdown stripping | `preprocessing/section_detector.py` (`SECTION_PATTERNS`, `extract_abstract`) | **14.6% of papers get zero section chunks and 13.5% zero fine chunks**, because no section boundary is ever found. Only **46% of papers get an abstract chunk** at all. MinerU already emits `type: "title"` blocks that make this structural rather than heuristic. |

---

## 2. The ten requests

1. Inline **direct quotes** with every citation, findable in the paper
2. Better **recall** — "it consistently misses things"; refine chunking
3. Keep the **numbers** advantage, add "taken from [location]"
4. Make the **confidence percentage** understandable
5. Real **memory** — ongoing ideas linked across sessions, editable user profile
6. **Reference managers** — RIS/BibTeX in/out with dedupe + cleanup, plus live Zotero sync
7. **Conversational voice**, not a book report
8. Explicit **modes** — generate / develop / refine / critique
9. **Writing help** — ideas into grant/paper starts
10. **Citation recommendation** — "writing on this topic, what should I cite from my library"

Decisions taken: page + verbatim quote + click-to-highlight for provenance; RIS/BibTeX file I/O **and** live Zotero API sync; conversational default **with** explicit modes.

Coverage: request 1 → W1+W2+W3 · 2 → W0+W1+W2 · 3 → W1+W2+W3 · 4 → W3 · 5 → W3 · 6 → W4 · 7 → W3 · 8 → W3 · 9 → W6 · 10 → W4. See the convergence matrix in §4 for the full mapping, including which root causes each workstream closes.

---

## 3. 2026 state of the art, and where ARC sits

This table is the literature view. **§3b is the authoritative implementation guidance** — where the two differ, §3b wins, because several of these techniques now ship natively in the platform and do not need building.

| Area | 2026 consensus | ARC today | Adopt via |
|---|---|---|---|
| Retrieval | Hybrid BM25+dense with RRF, then cross-encoder rerank. Recall@5 0.816 vs 0.587 dense-only | All three pieces present; hybrid disabled, rerank score unused | W0, W2 |
| Chunk context | Contextual retrieval (context prepended per chunk) — gains concentrated on **numeric** queries | Minimal `[Section]` header only | §3b.2 |
| Citations | Span-level: every claim carries a verbatim evidence span + doc id | `[Source N]` with no span, no page | §3b.1 — **native Citations API, do not hand-roll** |
| Confidence | Claim extraction + NLI entailment; self-reported confidence deprecated | Self-reported float / keyword overlap | W3 (entailment only; §3b.1 makes quote fidelity free) |
| Memory | Extract→update ledger (Mem0), tiered memory (MemoryOS), decay/consolidation | Global 2k-token buffer; profile table empty | §3b.4 — **native memory tool + profile, not a Mem0 clone** |
| Citation rec. | Benchmarked task (CiteRAG): list-level and position-specific | Not implemented | W4 |
| Extraction | MinerU is the accuracy leader for scientific PDFs and equations | Already MinerU — **keep it**, fix the integration | W1, W2 |
| Agentic loops | Real gains on multi-hop, but some benchmarks show worse scores at ~6× latency | N/A | §3b.5 — opt-in deep modes only |
| Model generation | Current frontier tier for research synthesis | **Three generations behind**, and blocked from upgrading (R14) | §3b.0 |
| Context strategy | Long context replaces retrieval for small document sets | Chunk retrieval for everything | §3b.6 |
| Cost control | Prompt caching as table stakes | None (R15) | §3b.3 |

References: [text-and-table retrieval benchmark](https://arxiv.org/html/2604.01733v1), [hybrid+rerank reference](https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026), [structured inline citation](https://arxiv.org/pdf/2606.07130), [citation extraction](https://zeroentropy.dev/concepts/citation-extraction/), [R2VC calibration](https://arxiv.org/html/2609.11955), [MedRAGChecker](https://arxiv.org/html/2601.06519v2), [persistent memory survey](https://arxiv.org/pdf/2606.30306), [adaptive memory structures](https://arxiv.org/pdf/2602.14038), [CiteRAG](https://arxiv.org/html/2601.14949v1), [parser comparison](https://builderai.tools/blog/pdf-parsing-for-rag-mineru-docling-marker-compared).

---

## 3b. The 2026 platform layer (what to actually adopt)

§1 catalogues defects and §3 surveys the literature. This section is the implementation layer for both, and it is authoritative: where a technique in §3 now ships natively, the instruction here is to **use the platform feature rather than build it**. Read this before implementing Phases 0, 1, 2, 4, or 5.

### 3b.0 The model migration is a code change, not a string swap

ARC runs `claude-opus-4-5-20251101` for generation, `claude-sonnet-4-5-20250929` for classification and web search, `claude-haiku-4-5-20251001` for helpers. Current models are `claude-opus-5` ($5/$25 per MTok, 1M context), `claude-sonnet-5` ($3/$15), `claude-haiku-4-5` ($1/$5).

**Hard blocker:** `temperature` is **removed on Opus 4.7+ and rejected at non-default values on Sonnet 5 — a 400, not a warning.** ARC passes it on every call:

| Call site | Model | Status on target |
|---|---|---|
| `retrieval/query_engine.py:1346, 1372` | Opus (generation) | **400** — delete the field |
| `retrieval/query_engine.py:1448` (`temperature=0.5`) | Sonnet (web search) | **400** — delete |
| `retrieval/query_classifier.py:232, 329` (`temperature=0`) | Sonnet (classifier) | **400** — `0` is non-default |
| `retrieval/citation_verifier.py:191`, `entity_extractor.py:238`, `hyde.py:153` | Haiku 4.5 | Accepted — no break |
| `api/main.py:354`, `UserPreferences.temperature` | user-facing slider | Must be replaced |

So the temperature slider in Settings cannot survive the upgrade. **Replace it with an effort selector** (`output_config: {"effort": "low"|"medium"|"high"|"xhigh"|"max"}`) — an honest mapping, since what the user wanted was a depth/care knob, not a sampling knob. Default `high`; `xhigh` for agentic and critique modes; `low`/`medium` for the cheap helpers.

Also required on Opus 5:
- **Thinking is on by default** (omitting `thinking` runs adaptive), and `max_tokens` caps thinking **plus** response text. ARC's `max_tokens=32768` needs headroom or answers truncate mid-sentence. `thinking: {"type": "disabled"}` is only legal at effort ≤ `high`.
- **Upgrade the web search tool** from `web_search_20250305` to `web_search_20260209` (`query_engine.py:1429`) — dynamic filtering filters results before they enter context. Do **not** also declare `code_execution`; it runs under the hood.
- Stale `claude-3-haiku-20240307` defaults in `hyde.py:64`, `citation_verifier.py:64`, `query_rewriter.py:141`, `entity_extractor.py:226`, `query_engine.py:451`.

Do this **first, as W0.** It gates every later workstream, it is the cheapest quality jump in the plan, and it must precede prompt caching — caches are model-scoped, so configuring them before the model is settled throws the entries away.

### 3b.1 Native Citations API — replaces the custom quote pipeline

This is the single biggest simplification, and it lands requests 1 and 3 directly.

Set `citations: {"enabled": true}` on each `document` content block (all blocks or none; no beta header). The response splits into multiple `text` blocks, and cited blocks carry a `citations` array where each entry has:

- `cited_text` — **the verbatim span, extracted by the API from the document.** Hallucinated quotes are impossible by construction.
- `document_index`, `document_title`
- a location: `char_location` (`start_char_index`/`end_char_index`) for text documents, `page_location` (1-indexed) for PDFs, `content_block_location` for custom-content documents.

**Design for ARC:** pass each retrieved chunk as a plain-text `document` block with citations enabled. Map the returned `char_location` through the chunk's stored offsets (captured in W2) to page + bbox. That yields quote + page + highlight coordinates with no custom validation loop.

**Two constraints this imposes, both of which killed an earlier draft of this plan:**
- Citations are **incompatible with `output_config.format`** (returns 400). The answering call cannot use structured outputs. Keep structured output for side calls — extraction, classification, entailment — and leave the answering call as prose-plus-citations.
- A literal string-match quote validator is **unnecessary and should not be written.** `cited_text` is extracted by the API from the document, so it is verbatim by construction. W3's confidence step therefore has only entailment left to check.

Keep the full-PDF path (`enable_pdf_upload`) as-is but turn citations on there too — it returns `page_location` natively, which is exactly the "p. 7" the user asked for.

### 3b.2 Contextual retrieval, cheaply — ~$50–100 one-time

Prepend a short LLM-written context line to each chunk before embedding. 2026 benchmarks put the gains specifically on **numerical** queries, which is the user's strongest use case.

The naive per-chunk implementation is expensive. The efficient design is **one Haiku call per paper** that emits a context line for every chunk in that paper at once: ~10k input tokens (the paper) + ~1.2k output. At Haiku 4.5 rates that is ≈$0.02/paper, so **≈$96 for all 4,797 papers — ≈$48 via the Batch API (50% off)**. Prompt caching (512-token minimum on Opus 5, reads at 0.1×) makes the per-chunk variant viable too, but one-call-per-paper is cheaper and simpler.

Do this during the single W1+W2 reindex, after the paper record is persisted. Keep the deterministic `[Title — Section > Subsection, p.N]` header as well; the two are complementary.

### 3b.3 Prompt caching — currently zero, and about to matter a lot

ARC caches nothing. Every turn re-bills the full system prompt and all source text. Once W3 starts feeding real conversation history this compounds badly.

- Minimum cacheable prefix is **512 tokens on Opus 5**. Writes cost 1.25× (5-min TTL) or 2× (1-hour); reads 0.1×. Break-even is two requests.
- Render order is `tools` → `system` → `messages`. Put a breakpoint on the last system block, and one on the last content block of the newest turn.
- **Freeze the system prompt.** Do not interpolate memory, profile text, or dates into it — that sits at the front of the prefix and invalidates everything downstream.
- **Inject memory as a mid-conversation system message instead:** `{"role": "system", "content": "..."}` appended to `messages[]`. Supported on Opus 5 with no beta header (not on Sonnet 5). This is both the cache-safe and the prompt-injection-safe channel for operator context, and it is exactly what W3's memory injection needs.
- Verify with `usage.cache_read_input_tokens`; if it stays 0 across turns, something volatile is in the prefix.

### 3b.4 Memory — use the native memory tool plus a structured profile

Request 5 splits cleanly into two layers, and only one of them needs building:

1. **Project notebook — use the client-side memory tool** (`{"type": "memory_20250818", "name": "memory"}`). Claude reads and writes files in a memory directory that you back with storage; Python ships `BetaAbstractMemoryTool` to implement it. This is the right fit for "keep a running tab on my paper ideas" — free-form, Claude-curated, persists across sessions, and far less code than an extraction pipeline. Scope the directory per research project.
2. **Profile paragraph — keep the structured route.** A short generated-and-editable summary of the user, injected per-turn as a mid-conversation system message (3b.3). This is the ChatGPT-style behaviour they described.

For long research threads, use **compaction** (beta `compact-2026-01-12`) rather than hand-rolling summarisation of old turns. Critical detail: append the **full `response.content`** back into messages each turn, not just the text — the compaction blocks must survive or the state is silently lost.

Never write credentials or PII into memory files; they are replayed verbatim into every later session.

### 3b.5 Agentic retrieval — as an explicit mode, not the default

Build the "Develop / Critique / Draft" modes (W3) on the SDK **Tool Runner** (`client.beta.messages.tool_runner` with `@beta_tool` functions) rather than the current fixed pipeline. Tools to expose: `search_library(query, section?, paper_ids?)`, `get_numeric_facts(entity, property)`, `expand_chunk(chunk_id)`, `list_candidate_citations(topic)`.

Scope it deliberately. 2026 results are mixed: reflection loops decreased scores on some benchmarks while raising latency ~6×. So the fixed pipeline stays the default for "Ask", and the loop runs only in the deep modes, at `xhigh` effort. Note the Python runner does **not** auto-resume `pause_turn` — a paused turn ends the loop and returns a silently truncated answer; handle it explicitly.

### 3b.6 1M context changes the single-paper path

Opus 5 has a 1M-token context window at standard pricing. For questions scoped to a handful of papers, retrieval is the wrong tool — send the whole extracted text of those papers as citation-enabled documents, cached. ARC already has the paper-selection UI and an `enable_pdf_upload` flag; the upgrade is to send **extracted text** (cheaper and cleaner than PDF bytes) with citations on, and reserve chunk retrieval for corpus-wide questions.

### 3b.7 SOTA to deliberately skip

- **ColPali / document-as-image late interaction.** Tempting for chemistry figures and schemes, but a 2026 paper is explicitly titled [*Document-as-Image Representations Fall Short for Scientific Retrieval*](https://arxiv.org/pdf/2604.18508). MinerU already extracts the figure captions and table bodies we need. Skip.
- **ColBERT / multivector late interaction.** Qdrant supports it, but it multiplies index size and latency for a gain that hybrid+rerank largely captures. Revisit only if the golden set shows hybrid+rerank plateauing.
- **RAPTOR-style summary trees.** Real technique, but the corpus-level need it serves is cheaper to meet by *fixing abstract extraction* (R16 — only 46% of papers currently have an abstract chunk) and keeping paper-level mean-pooled embeddings. Revisit only after W1 lands and abstracts actually exist.
- **Fine-tuning or a domain embedding model.** `voyage-3-large` is not the bottleneck; extraction and wiring are.
- **Claude Fable 5** ($10/$50). Reserve for nothing here — Opus 5 is the right tier for this workload.

---

## 4. How §1, §2 and §3 merge

The three sections are not three projects. They are three views of the same small set of code surfaces: §1 says what is broken, §2 says what the user feels, §3b says what the platform now provides. **Work is therefore organized by artifact, not by category** — each workstream below rewrites one surface once, and closes defects, lands requests, and adopts platform features in the same pass.

### The keystone: `Chunk.text` is doing four incompatible jobs

`chunker.py` builds chunk text as `context_header + source_text` (lines 233, 247, 257, 272), and prefixes captions with `[Figure/Table Caption]` (412) and tables with `[Table Content]` (439). There is exactly **one** `text` field, and it is used for all of:

| Consumer | What it needs |
|---|---|
| Voyage embedding | source text **plus** context — that is the whole point of contextual retrieval (§3b.2) |
| Citations API | source text **byte-exact**, or `char_location` offsets point into a synthetic string and `cited_text` can surface `[Methods]` as part of a quote (§3b.1) |
| The answer prompt | source text plus context plus a parent window (R5) |
| `SourceCard` in the UI | clean readable prose — today it shows the `[Methods]` prefix to the user |

These cannot be satisfied by one field. So **requests 1 and 3 (verbatim locatable quotes) and §3b.2 (contextual retrieval) are in direct conflict until `text` is split** into `text` (verbatim source span), `embed_text` (context line + source), and `display_text`. That single schema decision is what makes four otherwise-competing goals compatible — and it is invisible if you plan section by section, which is why this restructure matters.

### Convergence matrix

| Workstream | Surface rewritten | Closes | Serves requests | Adopts |
|---|---|---|---|---|
| **W0** Platform baseline | `config.py` + every Anthropic call site | R14, R15 | unblocks all | §3b.0, §3b.3 |
| **W1** Paper record | `preprocessing/pdf_processor.py`, `section_detector.py` | R1, R2, R16, "no persisted extraction" | 1, 2, 3 | §3b.1, §3b.2, §3b.6 (substrate for all three) |
| **W2** Chunk contract | `models.py`, `chunker.py`, `qdrant_store.py`, `index_papers.py`, `query_classifier.py` | R3, R4, R6, R13, R17 | 1, 2, 3 | §3b.2 |
| **W3** Answer pipeline | `query_engine.py`, `citation_verifier.py`, memory, response shape | R5, R7, R8, R9, R10, R11, R12 | 1, 3, 4, 5, 7, 8 | §3b.1, §3b.3, §3b.4, §3b.5, §3b.6 |
| **W4** Bibliographic layer | new `services/bibliography.py`, metadata scripts, library data model | §1 metadata defects | 6, 10 (feeds 9) | CiteRAG framing |
| **W5** Eval harness | `evaluation/` | — | proves 2 | — |
| **W6** Writing surface | prompts + frontend | — | 9 | — |

Each root cause R1–R15 is closed by exactly one workstream. Requests deliberately span several — request 1 (locatable quotes) needs offsets from W1, a split text field from W2, and the Citations call in W3 — which is the point: a user-facing outcome is not a code surface. W3 is the highest-convergence surface in the codebase: seven defects, six requests, and six platform features all land in `_generate_answer` and its callers.

### Where they genuinely conflict, and the resolution

Four places the three sections pull against each other. Each needs a decision, not a merge.

| # | Conflict | Resolution |
|---|---|---|
| C1 | Request 1 (verbatim quotes, via Citations) × request 4 (structured confidence output). `citations` and `output_config.format` are mutually exclusive — 400. | Two calls. Answer call = prose + citations. Entailment = separate side call that *may* use structured output. Never one call. |
| C2 | Request 1/3 (byte-exact quotes) × §3b.2 (context prepended before embedding). | Split `text` / `embed_text` / `display_text` (above). Send **only** `text` as the citation document; use `embed_text` for Voyage and the prompt. |
| C3 | Request 7 (conversational brevity) × request 3 (every number with provenance). | Prompt rule: brevity removes scaffolding and hedging, never data or citations. Enforce with **two** eval metrics — numeric recall *and* answer length — so a regression in either is visible. |
| C4 | Request 2 (recall) × the caption purge, which deletes ~75k chunks. | Gate on the golden set before and after. If figure-related recall drops, the fix is better real-caption extraction, not restoring the regex. |

---

## 5. Workstreams

Total ≈ **4–5 weeks** of engineering, but arranged as three parallel tracks after day 2, so calendar time is shorter. Plus a one-time **60–80 h** unattended reindex and ≈**$60–115** API spend.

### W0 — Platform baseline (~2 days, blocks everything)

R14, R15. Do this first; nothing else can ship on a supported model until it is done.

1. **Models and sampling.** `claude-opus-5` / `claude-sonnet-5` / `claude-haiku-4-5`. Strip every `temperature` from the Opus and Sonnet call sites (they 400: `query_engine.py:1346, 1372, 1448`, `query_classifier.py:232, 329`). Replace the Settings temperature slider and `UserPreferences.temperature` with an effort selector. Raise `max_tokens` for default-on adaptive thinking. Upgrade `web_search_20250305` → `web_search_20260209`. Clear the stale `claude-3-haiku-20240307` defaults in `hyde.py:64`, `citation_verifier.py:64`, `query_rewriter.py:141`, `entity_extractor.py:226`, `query_engine.py:451`. (§3b.0)
2. **Prompt caching, after item 1** — caches are model-scoped, so ordering matters. Breakpoint on the last system block and the newest turn; verify with `usage.cache_read_input_tokens`. (§3b.3)
3. **Quick correctness wins that need no reindex:** full-chunk sources instead of the 1,000-char cap and a parent window centred on the match (R5); `rerank_score` as the displayed relevance (R11); `ConversationMemory` keyed by `conversation_id` as a stopgap, and `selectConversation` resetting server state (R7).

### W1 — The paper record (~3 days + reindex, the keystone artifact)

R1, R2, R16, and the missing persistence. One new artifact, `processed_data/{paper_id}.json`, replaces five separate fixes.

Emit one structured record per paper from MinerU's `content_list`, with every block carrying `type`, `text`, `page_idx`, `bbox`, and a character offset. In the same pass:

1. **Tables** — read `table_body`, attach `table_caption` / `table_footnote` (R1).
2. **Captions** — use MinerU's `image_caption` / `table_caption` lists; delete the body-text regex (R2). Caption count should fall from 93k to ~15–20k real captions.
3. **Equations and page text** — keep `interline_equation` blocks and per-page text, both currently dropped.
4. **Structural section and abstract detection (R16).** Use MinerU's `type: "title"` blocks as section boundaries instead of regex-matching raw lines, and take the abstract from the leading block sequence rather than `extract_abstract`'s pattern list. This is only possible *because* W1 builds the record — and it is what recovers the 14.6% of papers with no sections and the 54% with no abstract. Highest-leverage single item in W1 after the table fix.
5. **The pypdfium2 fallback path (~3.1% of papers)** currently returns `tables=[]`, `captions=[]` and interleaved two-column text. It cannot produce a structured record, so mark those papers as degraded in the record and exclude them from claims about coverage rather than letting them silently look complete.
6. **Persist it.** Today nothing is cached, so every chunking change costs a full 60–80 h re-extraction. After this, re-chunking is minutes and free — which is what makes W2, W5 and every later tuning pass affordable.

The record is simultaneously the substrate for citation offsets (§3b.1), the paper text for contextual retrieval (§3b.2), and the source for the long-context single-paper path (§3b.6). Build it once, correctly.

### W2 — The chunk contract (~4 days, one reindex)

R3, R4, R6, R13. Extend `Chunk` and `to_payload()` once, then reindex once.

1. **Split the text field** — `text` (verbatim source span), `embed_text` (context line + source), `display_text`. This is C2's resolution and it gates W3's citation work.
2. **Positions** — `page_start` / `page_end`, `char_start` / `char_end`, `bbox` (R6).
3. **Stop truncating** — chunk the complete section text; keep truncation only for the coarse section embedding (R3).
4. **Context line** — deterministic `[Title — Section > Subsection, p.N]`, plus the LLM context line from §3b.2 (one Haiku call per paper emitting a line per chunk; ≈$96, ≈$48 on Batch). Both go in `embed_text`, neither in `text`.
5. **Numeric facts** — value + unit + entity triples (concentration, IC50, yield, temperature, equivalents) as a queryable payload field. This is what makes the user's favourite capability addressable rather than incidental.
6. **Hybrid retrieval on** — rebuild the IDF cache so `doc_count` covers all chunks (R13), confirm the collection carries the `bm25` sparse vector, pass `enable_hybrid_search=True` (R4). **A/B on the golden set; do not assume the win.**
7. **Fix the `FACTUAL` strategy (R17).** It serves the majority of real queries and currently excludes `section` and `abstract` while including a `table` type that matches nothing. Add `section`; drop `table` until W1 makes tables exist; reconsider `caption` once W1 has cut the 93k pseudo-captions. **This is a retrieval change, so it is gated on W5 — measure, do not guess.**

### W3 — The answer pipeline (~1.5 weeks, highest convergence)

R5, R7–R12. Requests 1, 3, 4, 5, 7, 8. This is one rewrite of `_generate_answer` and its callers, not six features.

1. **Request builder.** Frozen cached system prompt; conversation history from SQLite with compaction (beta `compact-2026-01-12`, append full `response.content`); memory and profile as **mid-conversation system messages** so the cached prefix survives (R7, R8, R9, §3b.3, §3b.4).
2. **Citations on** — each retrieved chunk as a plain-text `document` with `citations: {"enabled": true}`; map `char_location` through W2's offsets to page + bbox (§3b.1). No quote validator: `cited_text` is verbatim by construction.
3. **Confidence** — delete the self-reported float and keyword-overlap fallback; retire most of `citation_verifier.py`. Entailment is the only remaining signal, computed in a separate side call (C1). Report `Supported / Partially supported / Unsupported` with the span inline; no bare percentage. Remove `ResponseCard`'s min-per-source aggregation (R10).
4. **Voice and modes** — rewrite the prompts to lead with the answer and drop mandated scaffolds (R12); **Ask / Brainstorm / Develop / Refine / Critique / Draft** as stances. Deep modes run on the Tool Runner with `search_library`, `get_numeric_facts`, `expand_chunk`, `list_candidate_citations` at `xhigh` effort; Ask keeps the fixed pipeline (§3b.5). Handle `pause_turn` explicitly — the Python runner does not auto-resume it and returns a silently truncated answer.
5. **Long-context path** — selected-paper questions send whole records as cached citation-enabled documents instead of retrieving (§3b.6).
6. **Memory surfaces** — native memory tool (`memory_20250818`) scoped per research project; `projects` table linking conversations, papers, notes, open questions (§3b.4). The user's own suggestion — ask the model directly what it needs in order to remember better — is worth using as a design step here before fixing the memory-file format.
7. **Frontend** — pdf.js replacing the `<iframe>` viewer, jumping to the cited page and highlighting the span; `SourceCard` showing quote + "p. 7, Methods"; `Source` type gaining `page`, `quote`, `bbox`; mode selector in `QueryInput`; Memory/Profile page and project switcher.

### W4 — Bibliographic layer (~1 week, fully independent)

Requests 6, 10. Depends on nothing in W0–W3, so it can run in parallel throughout.

1. **Metadata backfill** — 2.5% DOI coverage makes every citation feature unusable. Crossref against the W1 record where available; `scripts/batch_extract_doi_metadata.py` already exists.
2. **RIS/BibTeX in-out** — dedupe by DOI → PMID → fuzzy title+author+year; clean against Crossref; return a cleaned file plus a report. Run it against ARC's own library too (166 suffixed duplicates, 71 duplicate-titled papers, 6 untitled).
3. **Metadata-only library entries.** The user's stated flow ends with importing the cleaned library *into ARC* — and they were explicit that this should work "even if it's not uploading all the papers." ARC's data model today assumes PDF → chunks, so a paper with no PDF cannot exist. Add a reference-only record type that appears in the library, participates in citation recommendation and dedupe, and is visibly marked as not full-text-searchable. Without this, request 6's import half has nowhere to land.
4. **Zotero Web API sync** — two-way. EndNote and Mendeley stay file-based; neither has a comparable API.
5. **"What should I cite"** — topic, abstract, or draft paragraph in; ranked library references out, each with a reason and a formatted citation (CSL). Position-specific mode for a given sentence.

### W5 — Eval harness (~2 days up front, then continuous)

Non-negotiable, and it gates W2's decisions. "It consistently misses things" cannot be declared fixed without measurement.

- Golden set from the 46 real conversations and the recovered query log — `Pd(PPh₃)₂Cl₂`, "intracellular concentration of peptides", "Glaser coupling with CuCl and TMEDA". These are the queries that must not miss.
- Reuse `evaluation/evaluator.py` + `evaluation/test_queries.py` (545 lines of scaffolding already).
- Track **recall@k, MRR, citation entailment rate, numeric recall, and answer length**. The last two are C3's guardrail — brevity must not cost data. Do *not* track quote-match rate; once §3b.1 lands it is 100% by definition.

### W6 — Writing surface (~3 days)

Request 9. A thin layer over W3's modes and W4's bibliography: section outlines, "questions this paper must answer", intro citation gathering, target-journal suggestions, grant-aim skeletons.

---

## 6. Sequencing — three parallel tracks

```
W0  platform baseline  ──── hard gate ────┐
    (models, temperature, caching)        │
                                          │
W5  golden set  ──────────────────────────┤
                                          │
     ┌────────────────────────────────────┘
     │
     ├── Track A ──> W1 paper record ──> W2 chunk contract ──> W3 answer pipeline ──> W6 writing
     │                (one reindex spans W1+W2)                      │
     │                                                              │
     ├── Track B ──> W4 bibliographic layer ───────────────────────────┘
     │               (independent; feeds W3's citation formatting and W6)
     │
     └── Track C ──> W5 continuous measurement, gating W2 and W3
```

**W0 is the hard gate.** Until `temperature` is stripped and the models are current, ARC cannot run on a supported model, and caching, compaction, mid-conversation system messages and the Citations API are all unavailable or model-gated.

After the gate, **Track A and Track B are independent** and can proceed simultaneously. Within Track A the ordering is forced: W1's record must exist before W2 can chunk against it, and W2's offsets and split text fields must exist before W3 can map citations. W6 needs both tracks.

The reindex is one operation spanning W1 and W2 — persist extraction first, then chunk, embed, and write contextual context lines in the same pass.

## 7. Open risks

- **Hybrid retrieval is confirmed available, not hypothetical** (verified 2026-09-15 with Qdrant up): the collection declares *and populates* the `bm25` sparse vector across all 212,953 points, and hybrid returns 7-8 new chunks out of 15 on real user queries. The open question is now quality, not availability — and it cannot be judged until R13's re-vectorization lands, because the stored vectors carry 6.6%-coverage IDF.
- **Reindex is 60-80h of wall time.** Persisting the paper record (W1) first converts that from a recurring cost into a one-time one.
- **Caption count will drop ~75%.** That is the intent, but total chunk count and retrieval behaviour will shift noticeably; the golden set is how we confirm it's an improvement.
- **The model migration is not optional and not free.** Every Opus and Sonnet call site currently passes `temperature`, which is a 400 on the target models (§3b.0). Until that is fixed, ARC cannot run on a current model at all — so W0 is load-bearing for the whole plan, and the Settings temperature slider has to be redesigned as part of it.
- **Citations constrain the generation call.** `citations` and `output_config.format` are mutually exclusive. If a later feature wants strict JSON out of the answering call, it has to give up native citations — so keep structured output for side calls (extraction, classification) and leave the answering call in prose-plus-citations form.
- **Adaptive thinking is on by default on Opus 5** and shares the `max_tokens` budget with the answer. Existing limits were tuned for a non-thinking model; under-sizing them truncates answers mid-sentence rather than erroring.
- **Self-reflective retrieval loops** have shown worse results at ~6× latency in 2026 benchmarks. Keep any agentic retrieval narrowly scoped and measured.
- **`pdf_processor.py` has 144 lines of uncommitted work** — the subprocess MinerU isolation that fixes the pdfium deadlock (see `docs/BUG_REPORT_pdfium_deadlock_2026-05-04.md`, also untracked). W1 rewrites exactly that file. **Commit or stash it deliberately before starting W1**, and preserve the subprocess boundary in the rewrite: the paper record must be produced inside the isolated subprocess, not after it returns, or the deadlock fix is lost.
