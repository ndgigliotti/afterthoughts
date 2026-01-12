# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FinePhrase is a Python library for generating fine-grained, context-aware phrase embeddings using transformer models. Unlike document-level embeddings, it extracts overlapping sub-sequence embeddings from the model's final hidden state, enabling semantic search and analysis at phrase-level granularity.

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
- `encode()`: Extracts phrase embeddings from documents (returns Polars DataFrame + NumPy array)
- `encode_queries()`: Encodes query strings for semantic search
- `search()`: Performs FAISS-based similarity search against extracted phrases
- Supports model compilation (`torch.compile`), AMP, GPU-based incremental PCA, and 16-bit precision reduction

**Tokenization Pipeline** (`finephrase/tokenize.py`):
- `tokenize_docs()`: Parallel tokenization with document chunking for sequences exceeding model max length
- `TokenizedDataset`: Dataset class that sorts sequences by length for efficient dynamic batching
- `DynamicTokenSampler`: Creates batches based on token count (not sequence count) to maximize GPU utilization
- `dynamic_pad_collate`: Pads batches to power-of-2 lengths for compiled model compatibility

**Sentence-Aware Processing** (`finephrase/sentence_utils.py`):
- `tokenize_with_sentence_boundaries()`: Tokenizes while preserving sentence structure using BlingFire
- `chunk_preserving_sentence_structure()`: Chunks long documents at sentence boundaries
- `_compute_sentence_phrase_embeds()`: Vectorized computation of phrase embeddings grouped by sentences
- Sentence IDs track which tokens belong to which sentence (padded with -1)

**Phrase Extraction** (`finephrase/phrase_utils.py`):
- `get_phrase_idx()`: Computes overlapping phrase indices for token-based extraction
- `_compute_phrase_embeddings()`: Mean-pools token embeddings (excluding special tokens) to create phrase embeddings

**PCA Module** (`finephrase/pca.py`):
- `IncrementalPCA`: GPU-accelerated PyTorch implementation adapted from scikit-learn
- Supports incremental fitting across batches and on-the-fly transformation

### Data Flow

1. Documents tokenized in parallel (joblib) with optional sentence boundary detection
2. Long sequences chunked with configurable overlap, preserving sentence boundaries when enabled
3. Sequences sorted by length and batched by total token count (not sequence count)
4. Model inference produces token embeddings
5. Phrase embeddings computed by mean-pooling tokens within sliding windows
6. Optional PCA reduction applied incrementally
7. Results returned as Polars DataFrame (phrase metadata) + NumPy array (embeddings)

### Key Parameters

- `phrase_sizes`: Token count(s) per phrase (can be list for multiple sizes)
- `phrase_overlap`: Fraction or count of tokens to overlap between phrases
- `sentences=True`: Extract phrases by sentence count instead of token count
- `batch_max_tokens`: Total tokens per batch (enables dynamic batching)
- `pca`: Number of PCA components (GPU-accelerated)
- `pca_fit_batch_count`: Fraction of batches to use for fitting PCA before applying

## Dependencies

Core: PyTorch, transformers, polars, pyarrow, numpy, joblib, blingfire, datasets
Optional: faiss-cpu/faiss-gpu (for search), pandas
