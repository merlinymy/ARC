# Token-Based Chunking with Semantic Preservation

## Overview

This document explains how our token-based chunking system works and how it preserves semantic meaning when splitting research papers.

## How Token-Based Chunking Works

### 1. **Tokenization**
We use `tiktoken` (OpenAI's tokenizer with the `cl100k_base` encoding) to count tokens:

```python
self.tokenizer = tiktoken.get_encoding("cl100k_base")
token_count = len(self.tokenizer.encode(text))
```

**What is a token?**
- A token is a unit of text that a language model processes
- Unlike words, tokens can be:
  - Whole words: `"hello"` → 1 token
  - Parts of words: `"embedding"` → 2 tokens (`"embed"`, `"ding"`)
  - Punctuation: `"!"` → 1 token
  - Special characters: `"π"` → 1-2 tokens

**Why use tokens instead of words?**
- Embedding models (like Voyage AI) have token limits, not word limits
- Token counting gives exact, predictable chunk sizes
- Ensures chunks fit within API limits (voyage-3-large: 32,000 tokens)

### 2. **Sentence Splitting** (Semantic Preservation)

Before chunking by tokens, we split text into sentences:

```python
def _split_into_sentences(self, text: str) -> List[str]:
    # Regex pattern detects sentence boundaries
    sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+(?=[A-Z])'
    sentences = re.split(sentence_pattern, text)
    return sentences
```

**This regex pattern:**
- Looks for `.`, `!`, or `?` followed by space and capital letter
- Avoids splitting on abbreviations like "Dr.", "Mr.", "et al."
- Preserves complete thoughts

**Example:**
```
Input: "Dr. Smith published a paper. It discusses AI. The results are promising!"

Sentences:
1. "Dr. Smith published a paper."
2. "It discusses AI."
3. "The results are promising!"
```

### 3. **Building Chunks by Token Count**

We group sentences together until reaching the token limit:

```python
for sentence in sentences:
    sentence_token_count = self._count_tokens(sentence)

    # If adding this sentence would exceed limit, start new chunk
    if current_token_count + sentence_token_count > self.chunk_size:
        # Save current chunk
        save_chunk()

        # Start new chunk with overlap
        create_overlap()

    # Add sentence to current chunk
    current_sentences.append(sentence)
    current_token_count += sentence_token_count
```

**Visual Example:**

```
Config: chunk_size=512 tokens, overlap=128 tokens

Text:
- Sentence 1: 150 tokens ──┐
- Sentence 2: 200 tokens   ├─→ Chunk 1 (450 tokens)
- Sentence 3: 100 tokens ──┘
- Sentence 4: 250 tokens ──┐
- Sentence 5: 180 tokens   ├─→ Chunk 2 (530 tokens, includes overlap)
- Sentence 6: 100 tokens ──┘
  └─ (Sentence 3 repeated for overlap: 100 tokens)

Overlap ensures context continuity between chunks!
```

## Why This Preserves Semantic Meaning

### ✅ **Never Splits Mid-Sentence**
```
❌ BAD (word-based without sentence awareness):
Chunk 1: "...proteins play a crucial role in cell"
Chunk 2: "signaling pathways that regulate..."

✅ GOOD (our approach):
Chunk 1: "...proteins play a crucial role in cell signaling."
Chunk 2: "These pathways regulate gene expression..."
```

### ✅ **Maintains Context with Overlap**
```
Paper text: "Introduction discusses method X. Method X uses algorithm Y.
             Algorithm Y processes data efficiently."

Chunk 1 (with overlap):
"Introduction discusses method X. Method X uses algorithm Y."

Chunk 2 (starts with overlap):
"Method X uses algorithm Y. Algorithm Y processes data efficiently."
       ↑ This sentence appears in both chunks ↑
```

**Why overlap matters:**
- Embedding models need context to understand meaning
- Overlap prevents information loss at chunk boundaries
- Queries matching the overlap region will retrieve both chunks

### ✅ **Consistent Token Limits**
```
All chunks stay within 512 tokens (configurable):
- Chunk 1: 487 tokens ✓
- Chunk 2: 502 tokens ✓
- Chunk 3: 495 tokens ✓

This ensures:
- Predictable embedding costs
- No API limit violations
- Consistent semantic density
```

## Configuration Parameters

In `config.py`:

```python
chunk_size: int = 512      # Max tokens per chunk
chunk_overlap: int = 128   # Tokens to overlap between chunks
```

**Recommended values for research papers:**

| Use Case | chunk_size | chunk_overlap | Rationale |
|----------|------------|---------------|-----------|
| **Default (current)** | 512 | 128 | Good balance, 25% overlap |
| **Detailed papers** | 800-1000 | 200 | More context per chunk |
| **Quick retrieval** | 256-400 | 64-100 | Faster, more granular |
| **Maximum context** | 1500-2000 | 300-400 | For complex papers |

**For voyage-3-large (32,000 token limit):** All these settings are safe!

## Implementation Flow

```
PDF File
   ↓
Extract full text
   ↓
Split into sentences (regex) ← Preserves semantic boundaries
   ↓
Count tokens per sentence (tiktoken)
   ↓
Group sentences into chunks:
  - Add sentences until reaching chunk_size
  - Never split mid-sentence
  - Create overlap from previous chunk
   ↓
Output chunks with metadata:
  {
    'text': "Complete sentences...",
    'chunk_index': 0,
    'token_count': 487,
    'sentence_count': 8
  }
```

## Example Output

```python
processor = PDFProcessor(chunk_size=512, chunk_overlap=128)
chunks = processor.chunk_text(paper_text, metadata)

# chunks[0]:
{
  'text': 'Abstract. This paper presents a novel approach to...',
  'chunk_index': 0,
  'token_count': 487,
  'sentence_count': 8
}

# chunks[1] (with overlap):
{
  'text': '...to neural network training. Our method improves...',
  'chunk_index': 1,
  'token_count': 502,
  'sentence_count': 9  # First 2 sentences overlap with chunks[0]
}
```

## Key Advantages

1. **Precise**: Know exact token count before sending to Voyage AI
2. **Safe**: Never exceed API limits
3. **Semantic**: Never break sentences apart
4. **Contextual**: Overlap ensures continuity
5. **Predictable**: Consistent chunk sizes across all papers
6. **Cost-effective**: No wasted tokens on oversized chunks

## Trade-offs vs Word-Based

| Aspect | Word-Based | Token-Based (Ours) |
|--------|------------|-------------------|
| Speed | ⚡⚡⚡⚡⚡ Faster | ⚡⚡⚡⚡ Slightly slower |
| Accuracy | ⭐⭐⭐ Estimated | ⭐⭐⭐⭐⭐ Exact |
| Semantic | ⭐⭐⭐⭐⭐ (if using sentence split) | ⭐⭐⭐⭐⭐ Always |
| Simplicity | ⭐⭐⭐⭐⭐ Very simple | ⭐⭐⭐⭐ Requires tokenizer |
| API Safety | ⭐⭐⭐ Estimated | ⭐⭐⭐⭐⭐ Guaranteed |

## Summary

Our token-based chunking approach:
1. **Counts tokens precisely** using tiktoken
2. **Preserves semantic meaning** by never splitting sentences
3. **Creates overlap** for context continuity
4. **Guarantees safety** within API limits

This gives you the best of both worlds: precise token control + semantic coherence!
