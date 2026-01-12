# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FinePhrase is a Python library for generating fine-grained, context-aware sentence-segment embeddings using transformer models. It detects sentence boundaries using BlingFire, then extracts overlapping groups of consecutive sentences and computes their embeddings by mean-pooling token embeddings from the model's final hidden state.

## Commands

### Install dependencies
```bash
pip install -e ".[dev]"
```

### Run tests
```bash
pytest                           # Run all tests
pytest tests/test_finephrase.py  # Run specific test file
pytest -k "test_name"            # Run test by name pattern
```

### Linting and type checking
```bash
ruff check finephrase/           # Lint with ruff
mypy finephrase/                 # Type checking (strict mode enabled)
```

## Architecture

### Core Components

**`FinePhrase` class** (`finephrase/finephrase.py`):
- Main entry point wrapping a HuggingFace transformer model
- `encode()`: Extracts segment embeddings from documents (returns Polars DataFrame + NumPy array)
- `encode_queries()`: Encodes query strings for semantic search
- Supports model compilation (`torch.compile`), AMP, GPU-based incremental PCA, and 16-bit precision reduction

**Tokenization Pipeline** (`finephrase/tokenize.py`):
- `tokenize_docs()`: Parallel tokenization with document chunking for sequences exceeding model max length
- `TokenizedDataset`: Dataset class that sorts sequences by length for efficient dynamic batching
- `DynamicTokenSampler`: Creates batches based on token count (not sequence count) to maximize GPU utilization
- `dynamic_pad_collate`: Pads batches to power-of-2 lengths for compiled model compatibility

**Sentence-Aware Processing** (`finephrase/sentence_utils.py`):
- `tokenize_with_sentence_boundaries()`: Tokenizes while preserving sentence structure using BlingFire
- `chunk_preserving_sentence_structure()`: Chunks long documents at sentence boundaries
- `get_segment_idx()`: Extracts segment indices (groups of consecutive sentences)
- `_compute_segment_embeds()`: Vectorized computation of segment embeddings
- Sentence IDs track which tokens belong to which sentence (padded with -1)

**PCA Module** (`finephrase/pca.py`):
- `IncrementalPCA`: GPU-accelerated PyTorch implementation adapted from scikit-learn
- Supports incremental fitting across batches and on-the-fly transformation

### Data Flow

1. Documents tokenized in parallel (joblib) with sentence boundary detection via BlingFire
2. Long sequences chunked with configurable overlap, preserving sentence boundaries
3. Sequences sorted by length and batched by total token count (not sequence count)
4. Model inference produces token embeddings
5. Segment embeddings computed by mean-pooling tokens within sentence groups
6. Optional PCA reduction applied incrementally
7. Results returned as Polars DataFrame (segment metadata) + NumPy array (embeddings)

### Key Parameters

- `segment_sizes`: Number of sentences per segment (can be list for multiple sizes)
- `segment_overlap`: Fraction or count of sentences to overlap between segments
- `batch_max_tokens`: Total tokens per batch (enables dynamic batching)
- `pca`: Number of PCA components (GPU-accelerated)
- `pca_fit_batch_count`: Fraction of batches to use for fitting PCA before applying

## Dependencies

Core: PyTorch, transformers, polars, pyarrow, numpy, joblib, blingfire, datasets
