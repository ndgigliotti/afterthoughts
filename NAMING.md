# Naming Decisions

## Package Name

**Decision:** `afterthoughts`

- "after" = late (as in late chunking)
- "thoughts" = sentences/segments extracted
- Also: chunking is done "as an afterthought" rather than beforehand
- PyPI: available
- SEO: clear (no competing Python libraries)

## Class Names

| Current | New | Notes |
|---------|-----|-------|
| `FinePhrase` | `Encoder` | Main class |
| `FinePhraseLite` | `LiteEncoder` | Memory-optimized variant (PCA, precision reduction, truncation) |

## Method Names

| Current | Keep | Notes |
|---------|------|-------|
| `encode()` | Yes | Main encoding method |
| `encode_queries()` | Yes | Query encoding |

## Parameter Names

### `encode()` parameters

| Current | New | Default | Notes |
|---------|-----|---------|-------|
| `segment_sizes` | `num_sents` | `2` | Sentences per chunk (int or list) |
| `segment_overlap` | `chunk_overlap` | `1` | Sentences to overlap between chunks (int) |
| `chunk_docs` | `prechunk` | `True` | Split docs exceeding context (safe default) |
| `doc_overlap` | `prechunk_overlap` | `0.5` | Fraction of sentences to overlap when prechunking (float) |
| `max_length` | keep | `None` | Model's max length |
| `batch_max_tokens` | `batch_tokens` | `16384` | Approx batch size in tokens |
| `token_batch_size` | (hide) | heuristic | Internal tokenization batch size (auto-scales based on avg doc length) |
| `return_frame` | keep | `"pandas"` | Output format |
| `convert_to_numpy` | `as_numpy` | `True` | Return embeddings as numpy array |
| `debug` | keep | `False` | Include debug columns |
| (new) | `show_progress` | `True` | Show progress bars |

### `__init__` parameters (both classes)

| Current | New | Default | Notes |
|---------|-----|---------|-------|
| `normalize_embeds` | `normalize` | `False` | Normalize output embeddings |
| `num_token_jobs` | (hide) | `-1` | Internal parallelism |

### `__init__` parameters (LiteEncoder only)

| Current | New | Default | Notes |
|---------|-----|---------|-------|
| `reduce_precision` | `half_embeds` | `False` | Output embeddings as fp16 |
| `pca_fit_batch_count` | `pca_early_stop` | `0.5` | Fraction of batches before applying PCA |

## Import Style

```python
from afterthoughts import Encoder

model = Encoder("nomic-ai/nomic-embed-text-v1.5")

# Simple usage - 2 sentences per chunk
df, X = model.encode(docs, num_sents=2)

# Multiple chunk sizes
df, X = model.encode(docs, num_sents=[1, 2, 3])
```

## Module Names

| Current | New | Purpose |
|---------|-----|---------|
| `finephrase.py` | `encoder.py` | Main Encoder/LiteEncoder classes |
| `sentence_utils.py` | `chunking.py` | Sentence detection, chunking logic |
| `tokenize.py` | keep | Tokenization utilities |
| `pca.py` | keep | Incremental PCA |
| `utils.py` | keep | General utilities |

## Terminology

| Term | Meaning |
|------|---------|
| **Chunk** | Group of sentences extracted *after* model forward pass (late chunking) |
| **Prechunk** | Splitting long docs *before* model to fit context window |

## Internal Naming Consistency

### Public API → Internal Parameter Alignment

Internal function parameters should match public API names:

| Public API | Internal (current) | Internal (target) |
|------------|-------------------|-------------------|
| `num_sents` | `chunk_sizes` | `num_sents` |
| `prechunk` | `chunk_docs` | `prechunk` |
| `prechunk_overlap` | `overlap` (in tokenize) | `prechunk_overlap` |
| `chunk_overlap` | `overlap` (in chunk) | `chunk_overlap` |

### Variable/Function Renames

| Current | Target | Location |
|---------|--------|----------|
| `sent_boundary_idx` | `sentence_ids` | chunk.py parameter |
| `segment_*` variables | `chunk_*` | various docstrings |

## Outstanding Questions

- [x] Main class name → `Encoder`
- [x] Lite variant name → `LiteEncoder`
- [x] Method names → keep `encode()` / `encode_queries()`
- [x] Parameter renames:
  - `segment_sizes` → `num_sents`
  - `segment_overlap` → `chunk_overlap`
  - `chunk_docs` → `prechunk`
  - `doc_overlap` → `prechunk_overlap`
