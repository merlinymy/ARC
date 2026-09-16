# Bug Report: PDF Upload Failure — PDFium State Corruption & Multiprocessing Deadlock

**Date:** 2026-05-04
**Severity:** Critical — all PDF uploads blocked
**Affected component:** `backend/preprocessing/pdf_processor.py`
**Status:** Fixed (pending backend restart)

---

## Summary

Users reported being unable to upload files. Investigation revealed two layered problems:

1. **PDFium native library corruption** — the long-running backend process could no longer open any PDF file
2. **multiprocessing.Queue deadlock** — discovered during the fix; the subprocess approach hung silently due to a well-known Python multiprocessing pitfall

---

## Timeline

| Time | Event |
|------|-------|
| ~May 1 | Backend process started (PID 92385). Successfully indexed papers. |
| May 1 21:47 | Last successful indexing: `Su-2016-Copper Catalysis...pdf` (13 chunks). |
| May 4 06:21 | First failures of the day. Every PDF upload fails with `PDFium: Data format error`. |
| May 4 06:24–08:09 | User retries multiple files. All fail identically. 0 chunks extracted each time. |
| May 4 08:09 | User updates preferences, retries `Steven-2022-Design...pdf`. Still fails. |

---

## Problem 1: PDFium State Corruption

### Symptoms

All PDF uploads failed at the text extraction step:

```
WARNING | MinerU extraction error: Failed to load document (PDFium: Data format error)., using fallback
ERROR   | Fallback extraction also failed: Failed to load document (PDFium: Data format error).
INFO    | Created 0 chunks for <paper_id>
```

Both the primary extractor (MinerU) and the fallback (pypdfium2) failed with the same error because both use PDFium internally.

### Diagnosis

- The backend process had been running continuously for 3+ days (`ps` showed uptime of `02-22:48:07`).
- The failing PDF (`Steven-2022-Design Synthesis and Analytical1_1.pdf`) was verified as valid: `file` reported `PDF document, version 1.6`, and pypdfium2 opened it successfully in a **fresh** Python process (7 pages).
- The same file failed inside the running backend process. Every new PDF failed — not just specific files.
- Disk space was fine (1.6 TB free on `/Volumes/ARC`). Qdrant was healthy (status: green). The API was responsive.

### Root cause

PDFium is a native C library (Google's PDF renderer). When loaded into a long-running Python process, it accumulates corrupted internal state — leaked memory buffers, stale internal caches, exhausted resource pools. After days of loading and closing hundreds of PDF documents, the library enters a state where it cannot open any new document.

This is not a bug in application code. It is a known limitation of using native C libraries in persistent processes. The previous mitigation was to periodically restart the backend, which gave PDFium a fresh start.

### User impact

Users saw a brief error notification ("No chunks extracted from PDF") via SSE, but the paper never appeared in their library. The raw PDF file was saved to `/Volumes/ARC/ARC/papers/` but never indexed — an orphaned file with no database record.

---

## Problem 2: multiprocessing.Queue Deadlock

### Context

The fix for Problem 1 was to run PDF extraction in a **subprocess** instead of a thread, giving each extraction a clean PDFium instance. During testing, this fix appeared to work (MinerU completed all inference stages) but the parent process hung forever waiting for the result.

### Root cause

A classic Python multiprocessing deadlock between `Queue.put()` and `Process.join()`:

```
Parent process                      Child subprocess
──────────────                      ────────────────
proc.start()
proc.join(timeout=600)              # runs MinerU extraction...
  |                                 # all inference stages complete
  |  waiting for child to exit      # result ready (34KB markdown + metadata)
  |                                 result_queue.put(large_result)
  |                                   |
  |                                   |  serialized data exceeds OS pipe
  |                                   |  buffer (~64KB on macOS)
  |                                   |
  |                                   |  put() BLOCKS until parent reads
  |                                   v
  v                                 BLOCKED — cannot exit
BLOCKED — waiting for exit

                    === DEADLOCK ===
```

`multiprocessing.Queue` uses an OS pipe internally. When the serialized result exceeds the pipe buffer capacity, `Queue.put()` blocks until the receiving end reads data to free buffer space. But the parent process called `proc.join()` first — which waits for the child process to exit. The child cannot exit because `put()` is blocked. Neither side can make progress.

From the [Python documentation](https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming):

> *"A process that has put items in a queue will wait before terminating until all the buffered items are fed by the feeder thread to the underlying pipe. A child process can call the cancel_join_thread() method of the queue to avoid this behavior."*

> *"This means that if you try joining that process you may get a deadlock unless you are sure that all items which have been put on the queue have been consumed."*

---

## Fix Applied

**File:** `backend/preprocessing/pdf_processor.py`

### Change 1: Subprocess isolation for PDF extraction

Replaced `ThreadPoolExecutor` (same process, shared PDFium state) with `multiprocessing.Process` (separate OS process, clean PDFium state).

Two top-level worker functions were added (must be top-level for multiprocessing pickling):

- `_subprocess_mineru_extract(pdf_path, lang, use_gpu, result_queue)` — runs the full MinerU pipeline (layout detection, OCR, table extraction, formula recognition)
- `_subprocess_fallback_extract(pdf_path, result_queue)` — runs basic pypdfium2 text extraction

Each extraction now gets a completely fresh Python process. When the subprocess exits, the OS reclaims all native memory. The parent backend process never loads PDFium itself, allowing it to run indefinitely without state corruption.

### Change 2: Queue-before-join to prevent deadlock

```python
# WRONG — deadlocks if result exceeds pipe buffer
proc.join(timeout=self.timeout)
result = result_queue.get_nowait()

# CORRECT — parent reads immediately, unblocking the child's put()
result = result_queue.get(timeout=self.timeout)
proc.join(10)
```

### Change 3: Separate fallback subprocess

MinerU's post-processing (`result_to_middle_json`) can hang after inference completes. If MinerU times out, the subprocess is killed and pypdfium2 runs in a **separate** fresh subprocess. Previously, both were sequential in the same subprocess — if MinerU hung, the fallback never ran.

### Removed

- `ThreadPoolExecutor` and `FuturesTimeoutError` imports (no longer used)
- Thread-based timeout logic with `future.cancel()` and `gc.collect()`

---

## Verification

Tested with the same PDF that was failing in the backend:

| Metric | Before fix | After fix |
|--------|-----------|-----------|
| File | `Steven-2022-Design Synthesis and Analytical1_1.pdf` | Same |
| Status | `PDFium: Data format error` | Success |
| Pages | 0 | 7 |
| Text extracted | 0 chars | 25,842 chars |
| Captions | 0 | 28 |
| Time | Instant failure | 98 seconds |

---

## Deployment

Requires backend restart. The running process (PID 92385) still uses the old in-process code.

After restart, previously failed uploads will need to be re-uploaded by the user. The orphaned PDF files on disk are not automatically re-indexed.

---

## Lessons Learned

1. **Native C libraries in long-running processes are fragile.** PDFium, MuPDF, and similar libraries accumulate state that Python's garbage collector cannot clean up. Subprocess isolation is the robust pattern for production use.

2. **Always read from a multiprocessing Queue before joining the process.** This is documented in the Python stdlib docs but easy to miss. The deadlock is silent — no error, no timeout, just a permanent hang.

3. **MinerU's post-processing can hang independently of inference.** All visible progress bars (Layout Predict, OCR-rec, etc.) complete, but internal steps like `result_to_middle_json` can block indefinitely. The fallback must run in a separate subprocess to avoid being dragged down.
