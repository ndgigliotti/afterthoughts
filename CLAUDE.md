# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Afterthoughts is a Python library implementing late chunking for context-aware sentence-chunk embeddings. It detects sentence boundaries using configurable tokenizers (BlingFire, NLTK, pysbd, or syntok), then extracts overlapping groups of consecutive sentences and computes their embeddings by mean-pooling token embeddings from the model's final hidden state.

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
uv run mypy afterthoughts/              # Type checking (strict mode enabled)
```

## Architecture

### Core Components

**`Encoder` class** (`afterthoughts/encode.py`):
- Main entry point wrapping a HuggingFace transformer model
- `encode()`: Extracts chunk embeddings from documents (returns DataFrame + NumPy array)
- `encode_queries()`: Encodes query strings for semantic search
- Supports model compilation (`torch.compile`), AMP, and 16-bit precision reduction
- Memory optimizations via `half_embeds` (float16 conversion) and `truncate_dims` (dimension truncation)

**Tokenization Pipeline** (`afterthoughts/tokenize.py`):
- `tokenize_docs()`: Parallel tokenization with document chunking for sequences exceeding model max length
- `TokenizedDataset`: Dataset class that sorts sequences by length for efficient dynamic batching
- `DynamicTokenSampler`: Creates batches based on token count (not sequence count) to maximize GPU utilization

**Sentence-Aware Processing** (`afterthoughts/chunk.py`):
- `get_sentence_offsets()`: Dispatcher for sentence boundary detection (blingfire, nltk, pysbd, syntok)
- `tokenize_with_sentence_boundaries()`: Tokenizes while preserving sentence structure
- `get_chunk_idx()`: Extracts chunk indices (groups of consecutive sentences)
- `_compute_chunk_embeds()`: Vectorized computation of chunk embeddings via mean-pooling

**Optional Dependencies** (`afterthoughts/avail.py`):
- Lazy imports for optional packages (pandas, nltk, pysbd, syntok)
- `require_*` functions raise helpful ImportError if package missing

### Data Flow

1. Documents tokenized in parallel (joblib) with sentence boundary detection
2. Long sequences chunked with configurable overlap, preserving sentence boundaries
3. Sequences sorted by length and batched by total token count
4. Model inference produces token embeddings
5. Optional dimension truncation applied to token embeddings
6. Chunk embeddings computed by mean-pooling tokens within sentence groups
7. Optional float16 conversion and normalization applied
8. Results returned as polars/pandas DataFrame (chunk metadata) + NumPy array (embeddings)

### Key Parameters

- `num_sents`: Number of sentences per chunk (can be list for multiple sizes)
- `chunk_overlap`: Fraction or count of sentences to overlap between chunks
- `batch_tokens`: Total tokens per batch (enables dynamic batching)
- `sent_tokenizer`: Sentence tokenizer ("blingfire", "nltk", "pysbd", "syntok")
- `half_embeds`: Convert chunk embeddings to float16 for reduced memory
- `truncate_dims`: Truncate embedding dimensions (for MRL models)

## Dependencies

Core: torch, transformers, blingfire, polars, numpy, joblib, tqdm

Optional: pandas, nltk, pysbd, syntok
