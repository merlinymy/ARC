# Logging Improvements for Message Persistence Debugging

## Date: 2026-02-03
## Purpose: Enhanced logging to diagnose message content mismatch between streaming and database persistence

---

## Changes Made

### 1. Query Engine Streaming Verification (`backend/retrieval/query_engine.py`)

**Location: Line ~1332-1347** - `_generate_answer()` method

Added logging after RAG answer streaming completes:
- ✅ Total character length
- ✅ Content preview (first 200 chars)
- ✅ Content hash (Python `hash()`)
- ✅ Marks start of streaming

**Location: Line ~913-936** - `query()` method

Added logging after `_generate_answer()` returns:
- ✅ Generation timing (ms)
- ✅ Returned answer length
- ✅ MD5 hash of returned content
- ✅ Content preview

**Location: Line ~1116-1131** - QueryResult creation

Added logging when creating QueryResult object:
- ✅ Answer content length and MD5
- ✅ Web search answer length and MD5 (if present)

### 2. Web Search Result Logging (`backend/retrieval/query_engine.py`)

**Location: Line ~1523-1540** - `_perform_web_search()` method

Enhanced web search result logging:
- ✅ Content length
- ✅ MD5 hash
- ✅ Content preview (first 200 chars)
- ✅ URL count

### 3. API Streaming Endpoint (`backend/api/main.py`)

**Location: Line ~792-810** - `/query/stream` endpoint

Added logging before sending final result to frontend:
- ✅ Result.answer length and MD5
- ✅ Content preview
- ✅ Web search answer length and MD5 (if present)

**Location: Line ~847-922** - Database persistence section

Enhanced database save logging:
- ✅ Start of persistence operation with conversation ID
- ✅ Conversation creation (if new)
- ✅ User message save with length and message ID
- ✅ **RAG answer preview and hash BEFORE saving**
- ✅ RAG answer message ID after save
- ✅ Web search answer length (if present)
- ✅ Web search answer message ID after save
- ✅ Success confirmation with conversation ID
- ✅ Full exception logging with stack trace on error

### 4. Chat Service Database Operations (`backend/services/chat_service.py`)

**Location: Line ~100-142** - `add_message()` method

Added comprehensive logging for all database insertions:
- ✅ Conversation ID
- ✅ Message role (user/assistant)
- ✅ Content length
- ✅ MD5 hash of content
- ✅ Content preview (first 150 chars)
- ✅ Metadata presence check
- ✅ Committed message ID

---

## Log Tags for Easy Filtering

All new logs use prefixed tags for easy filtering:

| Tag | Location | Purpose |
|-----|----------|---------|
| `[STREAMING]` | query_engine.py | Tracks streaming answer generation |
| `[QUERY_ENGINE]` | query_engine.py | Tracks answer generation flow |
| `[WEB_SEARCH]` | query_engine.py | Tracks web search execution |
| `[STREAM]` | api/main.py | Tracks streaming endpoint execution |
| `[DB_SAVE]` | api/main.py | Tracks database persistence operations |
| `[CHAT_SERVICE]` | chat_service.py | Tracks actual database insertions |

---

## How to Use for Debugging

### If the bug occurs again, search logs like this:

```bash
# Get the full journey of a specific query
grep "conversation_id" backend/logs/backend.log | grep "<conversation-id>"

# Track streaming content
grep "\[STREAMING\]" backend/logs/backend.log

# Track what's being saved to DB
grep "\[DB_SAVE\]" backend/logs/backend.log

# Verify content at each stage
grep "MD5\|hash" backend/logs/backend.log

# See all stages in order
grep -E "\[STREAMING\]|\[QUERY_ENGINE\]|\[STREAM\]|\[DB_SAVE\]|\[CHAT_SERVICE\]" backend/logs/backend.log
```

### Content Verification Chain:

For any query, you can now verify:

1. **RAG Answer Generation** (`[STREAMING]`):
   - Length, hash, preview when streaming completes

2. **Return from _generate_answer** (`[QUERY_ENGINE]`):
   - Verify returned value matches streamed value

3. **QueryResult Creation** (`[QUERY_ENGINE]`):
   - Verify QueryResult.answer matches returned value

4. **API Receives Result** (`[STREAM]`):
   - Verify result.answer matches QueryResult

5. **Preparing to Save** (`[DB_SAVE]`):
   - Verify result.answer still matches before DB call

6. **Database Insert** (`[CHAT_SERVICE]`):
   - Verify content parameter matches what was prepared

### Example Log Sequence:

```
[STREAMING] Starting RAG answer generation with streaming
[STREAMING] RAG answer streaming completed - Total length: 10451 chars
[STREAMING] RAG answer preview: # Refined Agenda...
[STREAMING] RAG answer hash: 1234567890

[QUERY_ENGINE] RAG answer generation completed in 63000.00ms
[QUERY_ENGINE] Returned answer length: 10451 chars
[QUERY_ENGINE] Returned answer MD5: abcdef123456
[QUERY_ENGINE] Returned answer preview: # Refined Agenda...

[QUERY_ENGINE] Creating QueryResult object
[QUERY_ENGINE] QueryResult.answer length: 10451 chars, MD5: abcdef123456

[STREAM] Query execution completed, preparing final result
[STREAM] Result.answer length: 10451 chars, MD5: abcdef123456
[STREAM] Result.answer preview: # Refined Agenda...

[DB_SAVE] Preparing to save RAG answer - Length: 10451 chars
[DB_SAVE] RAG answer preview: # Refined Agenda...
[DB_SAVE] RAG answer hash: 1234567890

[CHAT_SERVICE] Adding message to conversation 89d8263a...
[CHAT_SERVICE] Role: assistant, Length: 10451 chars
[CHAT_SERVICE] Content MD5: abcdef123456
[CHAT_SERVICE] Content preview: # Refined Agenda...
[CHAT_SERVICE] Message committed with ID: 161
```

**If hashes don't match between stages = BUG LOCATION IDENTIFIED!**

---

## Hash Comparison

Two hash types are logged:
- **Python `hash()`**: Fast, per-session, good for quick verification within same run
- **MD5**: Consistent across runs, good for comparing logs from different sessions

---

## Next Steps if Bug Recurs

1. Get conversation_id from user
2. Extract all logs for that conversation
3. Compare MD5 hashes at each stage
4. Identify where content changes
5. Review code at that transition point

---

## Deployment

Changes are in:
- `backend/retrieval/query_engine.py`
- `backend/api/main.py`
- `backend/services/chat_service.py`

Restart backend to apply:
```bash
cd /Users/merlin/projects/ARC
./restart_server.sh
```
