# Backend Scripts

This directory contains utility scripts for batch operations and maintenance tasks.

## Batch DOI Extraction and Metadata Update

`batch_extract_doi_metadata.py` - Automatically extracts DOIs from PDFs and updates paper metadata using CrossRef API.

### Features

- Extracts DOIs from PDF files that don't have one stored
- Fetches accurate metadata from CrossRef (title, authors, year, journal)
- Updates papers with poor metadata quality
- Generates proper filenames in "Author-YEAR-Title.pdf" format
- Rate limiting (1 request/second by default) to respect CrossRef API limits
- Progress tracking and detailed logging
- Dry-run mode for testing
- Error handling and recovery

### Requirements

```bash
# Make sure you're in the backend virtual environment
cd /Users/merlin/projects/ARC/backend
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# The script uses only standard library + requests (already installed)
```

### Usage

**Important**: Make sure your backend server is running before executing the script!

```bash
# Start the backend server (in another terminal)
cd /Users/merlin/projects/ARC/backend
source venv/bin/activate
python -m uvicorn api.main:app --reload
```

Then run the script:

```bash
# Dry run first (recommended) - see what would be updated without making changes
python scripts/batch_extract_doi_metadata.py --dry-run

# Actually process all papers
python scripts/batch_extract_doi_metadata.py

# Use custom API URL (if backend is on different port)
python scripts/batch_extract_doi_metadata.py --api-url http://localhost:8080

# Custom rate limit (e.g., 2 seconds between CrossRef requests)
python scripts/batch_extract_doi_metadata.py --rate-limit 2.0
```

### What It Does

The script processes each paper and:

1. **Checks if processing is needed**:
   - Skips papers that already have DOI and good metadata
   - Processes papers with missing DOI, authors, or poor titles

2. **Extracts DOI** (if missing):
   - Scans PDF metadata
   - Searches first page for DOI patterns
   - Validates and normalizes the DOI

3. **Fetches metadata from CrossRef**:
   - Queries CrossRef REST API with the DOI
   - Extracts title, authors, publication year, and journal
   - Respects rate limits (1 second between requests by default)

4. **Updates paper metadata**:
   - Updates title if current is "Unknown" or poor quality
   - Updates authors if current is empty
   - Updates year if missing
   - Generates proper filename: `LastName-YEAR-Title.pdf`

### Output Example

```
======================================================================
DOI Batch Extraction and Metadata Update
======================================================================
API URL: http://localhost:8000
Rate limit: 1.0s between CrossRef requests

[2026-01-08 10:30:15] [INFO] Fetching all papers from API...
[2026-01-08 10:30:16] [INFO] ✓ Successfully fetched 45 papers

======================================================================
Starting processing of 45 papers...
======================================================================

[1/45] [INFO]
[2026-01-08 10:30:16] [INFO] Processing: Machine Learning Applications... (ID: abc123)
[2026-01-08 10:30:16] [INFO]   → Reason: Missing authors
[2026-01-08 10:30:16] [INFO]   → Extracting DOI from PDF...
[2026-01-08 10:30:17] [INFO]   ✓ Found DOI: 10.1234/example.doi
[2026-01-08 10:30:17] [INFO]   → Fetching metadata from CrossRef for DOI: 10.1234/example.doi
[2026-01-08 10:30:19] [INFO]   → Will update authors: John Smith, Jane Doe...
[2026-01-08 10:30:19] [INFO]   → Will update year: 2023
[2026-01-08 10:30:19] [INFO]   → Will update filename: Smith-2023-Machine-Learning-Applications.pdf
[2026-01-08 10:30:19] [INFO]   → Updating paper metadata (3 fields)...
[2026-01-08 10:30:20] [INFO]   ✓ Successfully updated paper metadata

[2/45] [INFO]
[2026-01-08 10:30:20] [INFO] Processing: Deep Neural Networks... (ID: def456)
[2026-01-08 10:30:20] [INFO]   ⊘ Skipping: Already has DOI and good metadata

...

======================================================================
FINAL SUMMARY
======================================================================
Total papers: 45
Processed: 45
DOIs extracted: 12
Metadata updated: 12
Skipped (already good): 28
Errors: 5
======================================================================
```

### Rate Limits

CrossRef API officially allows up to 50 requests per second, but we use a conservative default of **1 request per second** to:
- Be a good API citizen
- Avoid potential rate limiting
- Ensure reliability

You can adjust this with `--rate-limit` if needed, but don't go below 0.5 seconds.

### Error Handling

The script handles errors gracefully:
- **DOI not found**: Logs warning and continues
- **CrossRef 404**: Logs that DOI doesn't exist in CrossRef
- **Network errors**: Logs error and continues to next paper
- **Keyboard interrupt (Ctrl+C)**: Shows partial progress and exits cleanly

### Tips

1. **Always dry-run first**: Use `--dry-run` to see what would be changed
2. **Monitor progress**: The script logs detailed progress every 10 papers
3. **Run during off-hours**: Processing many papers can take time (1-2 seconds per paper)
4. **Check logs**: Review the output for any papers that had errors
5. **Re-run if needed**: The script is idempotent - safe to run multiple times

### Troubleshooting

**"Connection refused" error**:
- Make sure the backend server is running on the specified port
- Check the `--api-url` parameter matches your backend

**"DOI not found" warnings**:
- Some PDFs don't contain DOIs - this is normal
- You can manually add DOIs through the UI

**Rate limit errors from CrossRef**:
- Increase the `--rate-limit` parameter (e.g., `--rate-limit 2.0`)
- Wait a few minutes and try again

**Many "No metadata improvements available"**:
- This means your papers already have good metadata!
- The script only updates when it can improve quality

---

## Staged reindex — extract, then embed (plan §3b.10)

`stage_lib.py`, `extract_stage.py`, `embed_stage.py`, `cohort_cited_papers.py`,
`stage_reuse_check.py`. Driven through `index_papers.py <stage>`, which is now a
dispatcher rather than a 543-line script that did four things.

The reindex is two stages with different economics, and they are kept separable
on purpose:

| stage | cost | repeatable |
|---|---|---|
| `extract` — PDF → `processed_data/{id}.json` + figure crops | ~74 s/paper, $0, GPU-bound | expensive |
| `embed` — record → chunks → vectors → Qdrant | ~1 s/paper + API | **free to redo** |

```bash
# 1. derive the cohort (the papers ARC has actually cited)
python index_papers.py cohort --expect 591

# 2. see what each stage would do; nothing is written
python index_papers.py plan --papers-from data/cohorts/cited_papers.txt

# 3. stage 1 — safe to interrupt and re-run, finished papers are skipped
python index_papers.py extract --papers-from data/cohorts/cited_papers.txt

# 4. stage 2 — builds the collection the active embedding profile names
python index_papers.py embed --papers-from data/cohorts/cited_papers.txt

# later: the rest of the corpus, without re-touching the cohort
python index_papers.py extract --pending
```

Properties worth knowing:

- **Record reuse.** A record is skipped when its `schema_version`,
  `extractor_version`, `file_size` and `file_mtime` still describe the PDF. A
  moved `mtime` with an identical size is reused (copying a volume is not a
  content change) — verified against the PDF's sha256 when the manifest has one.
  `--mtime-policy strict` re-extracts instead. `stage_reuse_check.py` asserts
  both directions.
- **Its own manifests.** `data/stage_manifests/extract.json` and
  `embed__<collection>.json`: append-only `.jsonl` journal per paper plus an
  atomic snapshot, so a `kill -9` cannot truncate them.
  `data/indexing_checkpoint.json` is never read or written by these stages.
- **The live index keeps serving.** The embed stage refuses a collection that
  already holds points unless `--append`, and refuses a collection belonging to
  a different embedding profile at all. Build beside the live one; cut over by
  switching `EMBEDDING_PROFILE`.
- **BM25.** The IDF cache is loaded read-only and never written; document sparse
  vectors carry no corpus statistics. Refresh after the cutover with
  `rebuild_bm25_index.py pass1`.
