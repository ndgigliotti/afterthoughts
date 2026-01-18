# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Afterthoughts is a Python library implementing late chunking for context-aware sentence-chunk embeddings. Late chunking processes entire documents through a transformer model to capture full context, then extracts embeddings for sentence groups (chunks) by mean-pooling token embeddings within sentence boundaries. This preserves contextual relationships that would be lost with traditional pre-chunking approaches.

Sentence boundaries are detected using configurable tokenizers (BlingFire default, NLTK, pysbd, or syntok).

## Commands

### Install dependencies
```bash
uv pip install -e ".[dev]"
```

### Run tests
```bash
uv run pytest                           # Run all tests
uv run pytest tests/test_encode.py      # Run specific test file
uv run pytest -k "test_name"            # Run test by name pattern
```

### Linting and type checking
```bash
uv run ruff check afterthoughts/        # Lint with ruff
uv run ruff format afterthoughts/       # Format with ruff
uv run pyright                          # Type checking
```

### Pre-commit hooks
```bash
pre-commit run --all                    # Run all hooks (ruff, pyright, codespell)
```

## Architecture

### Module Structure

```
afterthoughts/
├── __init__.py      # Public API: Encoder, configure_logging
├── encode.py        # Encoder class - main entry point
├── tokenize.py      # Tokenization pipeline and dynamic batching
├── chunk.py         # Sentence detection and chunk embedding computation
├── utils.py         # Utilities (logging, normalization, memory)
└── avail.py         # Optional dependency management
```

### Core Components

**`Encoder` class** (`encode.py`):
- Main entry point wrapping a HuggingFace transformer model
- `encode()`: Extracts chunk embeddings from documents (returns DataFrame + NumPy array)
- `encode_queries()`: Encodes query strings for semantic search
- Supports `torch.compile`, AMP, and 16-bit precision
- Memory optimizations: `half_embeds` (float16), `truncate_dims` (MRL dimension truncation)

**Tokenization Pipeline** (`tokenize.py`):
- `tokenize_docs()`: Parallel tokenization with document chunking for long sequences
- `TokenizedDataset`: Sorts sequences by length for efficient batching
- `DynamicTokenSampler`: Batches by total token count (not sequence count)

**Sentence-Aware Processing** (`chunk.py`):
- `get_sentence_offsets()`: Dispatcher for sentence boundary detection
- `tokenize_with_sentence_boundaries()`: Tokenizes while preserving sentence structure
- `get_chunk_idx()`: Extracts chunk indices (groups of consecutive sentences)
- `_compute_chunk_embeds()`: Vectorized mean-pooling of token embeddings

**Optional Dependencies** (`avail.py`):
- Lazy imports for pandas, nltk, pysbd, syntok
- `require_*` functions raise helpful ImportError if missing

### Data Flow

1. Documents tokenized in parallel (joblib) with sentence boundary detection
2. Long sequences chunked with configurable overlap, preserving sentence boundaries
3. Sequences sorted by length and batched by total token count
4. Model inference produces token embeddings
5. Optional dimension truncation applied to token embeddings
6. Chunk embeddings computed by mean-pooling tokens within sentence groups
7. Optional float16 conversion and normalization applied
8. Results returned as Polars DataFrame (chunk metadata) + NumPy array (embeddings)

### Key Parameters

- `num_sents`: Number of sentences per chunk (can be list for multiple sizes)
- `chunk_overlap`: Fraction or count of sentences to overlap between chunks
- `batch_tokens`: Total tokens per batch (enables dynamic batching)
- `sent_tokenizer`: Sentence tokenizer ("blingfire", "nltk", "pysbd", "syntok")
- `half_embeds`: Convert chunk embeddings to float16 for reduced memory
- `truncate_dims`: Truncate embedding dimensions (for MRL models)

## Testing

Tests use session-scoped fixtures in `tests/conftest.py` to avoid repeated model loading. The test model is `sentence-transformers/paraphrase-MiniLM-L3-v2`.

Test files mirror source modules:
- `test_encode.py` - Encoder class integration tests
- `test_chunk.py` - Sentence detection and chunking
- `test_tokenize.py` - Tokenization pipeline
- `test_utils.py` - Utility functions
- `test_alignment.py` - Token/sentence alignment

## Dependencies

**Core**: torch, transformers, blingfire, polars, numpy, joblib, tqdm

**Optional**: pandas (alternative DataFrame output), nltk/pysbd/syntok (alternative sentence tokenizers)

**Dev**: pytest, ruff, pyright, pre-commit, coverage

Python 3.10+ required.
