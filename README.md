# Afterthoughts

[![CI](https://github.com/ndgigliotti/afterthoughts/actions/workflows/ci.yml/badge.svg)](https://github.com/ndgigliotti/afterthoughts/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/afterthoughts.svg)](https://pypi.org/project/afterthoughts/)
[![Python versions](https://img.shields.io/pypi/pyversions/afterthoughts.svg)](https://pypi.org/project/afterthoughts/)
[![License](https://img.shields.io/pypi/l/afterthoughts.svg)](https://github.com/ndgigliotti/afterthoughts/blob/main/LICENSE)

**Sentence-aware embeddings with document-level context.** Embeds first, chunks second, extracting any number of chunks in one forward pass while preserving full document context. For related work on late chunking, see [Günther et al., 2024](https://arxiv.org/abs/2409.04701).

## Quick Start

```bash
pip install afterthoughts
```

```python
from afterthoughts import LateEncoder

model = LateEncoder("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

docs = [
    "The Amazon rainforest produces 20% of Earth's oxygen. "
    "Deforestation threatens its biodiversity. "
    "Scientists warn of a tipping point.",  # 1 document, 3 sentences
]
df, X = model.encode(docs, max_chunk_sents=1)  # 1 sentence per chunk
```

```python
>>> df
shape: (3, 6)
┌─────┬──────────────┬───────────┬─────────────────┬───────────┬───────────────────────────────────────────┐
│ idx ┆ document_idx ┆ chunk_idx ┆ max_chunk_sents ┆ num_sents ┆ chunk                                     │
│ --- ┆ ---          ┆ ---       ┆ ---             ┆ ---       ┆ ---                                       │
│ u32 ┆ i64          ┆ i64       ┆ i64             ┆ i64       ┆ str                                       │
╞═════╪══════════════╪═══════════╪═════════════════╪═══════════╪═══════════════════════════════════════════╡
│ 0   ┆ 0            ┆ 0         ┆ 1               ┆ 1         ┆ The Amazon rainforest produces 20% of Ea… │
│ 1   ┆ 0            ┆ 1         ┆ 1               ┆ 1         ┆ Deforestation threatens its biodiversity… │
│ 2   ┆ 0            ┆ 2         ┆ 1               ┆ 1         ┆ Scientists warn of a tipping point.       │
└─────┴──────────────┴───────────┴─────────────────┴───────────┴───────────────────────────────────────────┘

>>> X.shape
(3, 384)  # 3 sentence embeddings, each with full document context
```

## What is Late Chunking?

Traditional RAG pipelines split documents into chunks *before* embedding, which loses contextual information. Consider a technical report that opens with "The new lithium-sulfur battery achieved 400 Wh/kg energy density" and later states "The technology could double EV range" or "Its cycle life remains a challenge." When these sentences are embedded separately, the later chunks lose their connection to lithium-sulfur batteries—a search for "lithium battery limitations" might miss the cycle life sentence entirely.

**Late chunking inverts this process:**

1. **Embed first**: Pass the entire document through the transformer model to get contextually-enriched token embeddings
2. **Chunk second**: Pool token embeddings into chunks *after* the model has established cross-chunk context

This approach ensures that pronouns, references, and contextual cues in each chunk are informed by the full document context.

## Why Late Chunking?

**The problem:** Document-level embeddings are too coarse for long documents. Traditional chunking loses context—pronouns like "it" or "the technology" become meaningless when separated from their referents.

**The solution:** Late chunking embeds the full document first, then pools token embeddings into chunks. Each chunk retains full document context.

**Performance:** One forward pass for the entire document, regardless of chunk count.

## Features

* **Late chunking implementation**: Embed documents first, then pool into chunks for context-aware embeddings
* **Flexible chunk configuration**: Sentence-based, token-based, or combined chunking strategies
* **Multi-configuration support**: Test multiple chunk sizes in a single pass with aligned list parameters
* **Sentence boundary detection**: Choice of BlingFire (default), NLTK, pysbd, or syntok for accurate sentence segmentation
* **Query embedding**: Embed queries in the same space as chunks for semantic search
* **HuggingFace integration**: Works with any transformer model from the HuggingFace Hub
* **Automatic mixed precision (AMP)**: Faster inference with reduced memory footprint
* **Dynamic batching**: Batches by total token count (not sequence count) for optimal GPU utilization
* **Structured output**: Returns chunks and metadata as Polars/pandas DataFrame for easy manipulation
* **Memory optimizations**: Optional float16 embedding conversion and dimension truncation for reduced memory

## Usage Guide

### Basic Usage

1. Install the package using pip:

    ```bash
    pip install afterthoughts
    ```

2. Create a `LateEncoder` object and load a transformer model.

    ```python
    from afterthoughts import LateEncoder

    # Choose a model which works well with mean-tokens pooling
    model = LateEncoder("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    ```

3. Prepare a list of documents `docs` (strings) from which to extract chunk embeddings.

    ```python
    docs = [
        "I am a document. I have multiple sentences.",
        "I am another document. I also have sentences.",
        "I am yet another document. Sentences are great.",
        "I'm not like the others. I'm special.",
    ]
    ```

4. Encode and extract chunk embeddings:

    ```python
    df, X = model.encode(
        docs,
        max_chunk_sents=[1, 2],  # Extract 1-sentence and 2-sentence chunks
    )
    ```
    The `encode` method returns a tuple containing a Polars DataFrame and a NumPy array of chunk embeddings. Pass `return_frame="pandas"` for a pandas DataFrame instead.

    To use a different sentence tokenizer, pass the `sent_tokenizer` parameter:

    ```python
    df, X = model.encode(
        docs,
        max_chunk_sents=2,
        sent_tokenizer="pysbd",  # Options: "blingfire" (default), "nltk", "pysbd", "syntok"
    )
    ```

    The DataFrame contains the following columns:
    * `idx`: Global chunk index (0-based), maps directly to embedding row
    * `document_idx`: The index of the document from which the chunk was extracted
    * `chunk_idx`: The chunk index within each document
    * `max_chunk_sents`: The requested maximum sentences per chunk (only present when specified)
    * `max_chunk_tokens`: The requested maximum tokens per chunk (only present when specified)
    * `num_sents`: The actual number of sentences in the chunk
    * `chunk`: The chunk text

    Additional columns are available when `debug=True`:
    * `embed_idx`: The original embedding index before re-sorting
    * `sequence_idx`: The index of the tokenized sequence (differs from `document_idx` when long documents are split)
    * `batch_idx`: The index of the batch in which the chunk was processed

    To access embeddings for specific chunks, use the `idx` column:

    ```python
    # Get embeddings for all chunks from document 10
    doc_df = df[df["document_idx"] == 10]
    doc_embeds = X[doc_df["idx"]]
    ```

### Advanced Chunking Strategies

#### Token-Based Chunking

In addition to sentence-based chunking, you can specify a maximum token count per chunk using `max_chunk_tokens`. This is useful when you need chunks that fit within a specific token budget:

```python
# Token-based chunking: accumulate sentences until token limit
df, X = model.encode(
    docs,
    max_chunk_tokens=128,  # Maximum 128 tokens per chunk
    max_chunk_sents=None,  # No sentence limit
)
```

You can also combine both constraints—whichever limit is reached first determines the chunk boundary:

```python
# Combined constraints: "at most 3 sentences AND at most 128 tokens"
df, X = model.encode(
    docs,
    max_chunk_sents=3,
    max_chunk_tokens=128,  # Whichever limit is hit first
)
```

**Handling long sentences:** By default, sentences exceeding `max_chunk_tokens` are split into multiple chunks at token boundaries (`split_long_sents=True`). Set `split_long_sents=False` to keep long sentences intact as their own chunks. Note that this parameter only affects chunking at the `max_chunk_tokens` boundary—sentences exceeding the model's `max_length` are handled automatically during tokenization through prechunking.

#### Comparing Multiple Chunk Configurations

To experiment with different chunk sizes simultaneously, pass lists to `max_chunk_sents` and/or `max_chunk_tokens`. When both are lists, they must have the same length and will be processed as **aligned pairs**:

```python
# Multiple sentence sizes
df, X = model.encode(
    docs,
    max_chunk_sents=[1, 2, 3],  # Creates 3 configurations
)

# Multiple token limits
df, X = model.encode(
    docs,
    max_chunk_tokens=[64, 128, 256],  # Creates 3 configurations
)

# Aligned pairs (NOT cartesian product)
df, X = model.encode(
    docs,
    max_chunk_sents=[1, 2, 3],
    max_chunk_tokens=[64, 128, 256],  # Same length required
    # Creates: (1, 64), (2, 128), (3, 256) - only 3 configs!
)
```

The resulting DataFrame includes `max_chunk_sents` and `max_chunk_tokens` columns to identify which configuration produced each chunk:

```python
import polars as pl

# Filter to specific configuration (Polars)
# Cast to int since config columns can contain None
df_small_chunks = df.filter(
    (pl.col("max_chunk_sents").cast(pl.Int64, strict=False) == 1) &
    (pl.col("max_chunk_tokens").cast(pl.Int64, strict=False) == 64)
)

# For pandas DataFrames, direct comparison works:
# df_small_chunks = df[(df["max_chunk_sents"] == 1) & (df["max_chunk_tokens"] == 64)]
```

**Note:** Config columns may contain `None` values when using lists with mixed values (e.g., `[1, None, 2]`). Cast to Int64 for numeric comparisons in Polars.

### Using Pandas Instead of Polars

Afterthoughts uses Polars by default for its speed and memory efficiency, but pandas is fully supported for users who prefer it or need compatibility with existing code. Simply set `return_frame="pandas"`:

```python
df, X = model.encode(
    docs,
    max_chunk_sents=2,
    return_frame="pandas",  # Return a pandas DataFrame
)

# Use familiar pandas operations
df.groupby("document_idx").size()
df[df["max_chunk_sents"] == 2]
```

The pandas integration requires pandas to be installed (`pip install pandas`). The DataFrame schema and all functionality remain identical—only the return type changes.

### Memory Optimizations

The `LateEncoder` class supports two memory optimization parameters:

#### Dimension Truncation (`truncate_dims`)

For models trained with Matryoshka Representation Learning (MRL), you can truncate embeddings to smaller dimensions with minimal quality loss. No retraining required—just slice the first N dimensions.

```python
from afterthoughts import LateEncoder

# This model was trained with MRL at dimensions [768, 512, 256, 128, 64]
model = LateEncoder(
    "tomaarsen/mpnet-base-nli-matryoshka",
    truncate_dims=256,  # Truncate to 256 dimensions
)
```

Truncation is applied to token embeddings *before* pooling, which saves both memory and compute during inference.

Note: Truncation also works on non-MRL models, but may degrade embedding quality since they weren't trained to preserve information in leading dimensions.

#### Float16 Embeddings (`half_embeds`)

Convert chunk embeddings to float16 for 2x memory reduction:

```python
from afterthoughts import LateEncoder

model = LateEncoder(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    half_embeds=True,  # Convert embeddings to float16
)
```

These options can be combined for additional savings.

### Performance Optimizations

#### Using Automatic Mixed Precision (AMP)

To enable automatic mixed precision, set the `amp` parameter to `True` during initialization. This will automatically lower the numerical precision of the most numerically stable layers, reducing the memory footprint of the model and increasing inference speed. Using AMP generally lets you increase the batch size.

```python
import torch
from afterthoughts import LateEncoder

model = LateEncoder(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    amp=True,
    amp_dtype=torch.float16, # Choose the lower-precision data type
)
```

#### Running the Model in 16-Bit Precision

To run the model in 16-bit precision, set the `torch_dtype` parameter to `torch.float16` or `torch.bfloat16` during initialization. This will reduce the memory footprint of the model and increase inference speed. Using 16-bit precision also generally lets you increase the batch size. This is similar to using AMP, but it is a cruder and more aggressive approach.

```python
import torch
from afterthoughts import LateEncoder
model = LateEncoder(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    torch_dtype=torch.float16,  # Run the model in 16-bit precision
)
```

Alternatively, you can convert the model to 16-bit precision after it has been loaded:

```python
from afterthoughts import LateEncoder

model = LateEncoder("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
model.half()  # Convert the model to 16-bit precision
```

### Logging

Afterthoughts uses Python's standard logging module for diagnostic output. By default, logging is silent. To enable logging:

```python
import afterthoughts

# Quick setup with configure_logging
afterthoughts.configure_logging(level="INFO")  # INFO, DEBUG, WARNING, etc.
```

Or use Python's logging module directly for more control:

```python
import logging

# Enable debug output from Afterthoughts
logging.getLogger("afterthoughts").setLevel(logging.DEBUG)
logging.basicConfig()
```

**Log levels:**
- `INFO`: Model loading, compilation, preprocessing time
- `DEBUG`: Batch sizes, token counts, and other diagnostic details

### Instruct-Style Embedding Models

Many modern embedding models require instruction prefixes to achieve optimal performance. Afterthoughts supports these models through `query_prompt` and `document_prompt` parameters.

#### E5-Instruct Models

E5-instruct models (e5-mistral-7b-instruct, multilingual-e5-large-instruct) require a task instruction for queries but not for documents:

```python
from afterthoughts import LateEncoder

model = LateEncoder(
    "intfloat/multilingual-e5-large-instruct",
    query_prompt="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
)

# Queries use the prompt automatically
query_embeds = model.encode_queries(["how much protein should a female eat"])

# Documents are encoded without any prompt
df, X = model.encode(docs, max_chunk_sents=2)
```

#### BGE Models

BGE models use a simpler prefix for queries:

```python
model = LateEncoder(
    "BAAI/bge-large-en-v1.5",
    query_prompt="Represent this sentence for searching relevant passages: ",
)
```

#### Nomic Embed

Nomic requires task prefixes for both queries and documents:

```python
model = LateEncoder(
    "nomic-ai/nomic-embed-text-v1.5",
    query_prompt="search_query: ",
    document_prompt="search_document: ",
)
```

#### Instructor Models

Instructor models use domain-specific instructions for both queries and documents:

```python
model = LateEncoder(
    "hkunlp/instructor-large",
    query_prompt="Represent the Wikipedia question for retrieving supporting documents: ",
    document_prompt="Represent the Wikipedia document for retrieval: ",
)
```

#### Per-Call Prompt Override

You can override the default prompt for specific calls:

```python
# Use a different task for this specific query
query_embeds = model.encode_queries(
    queries,
    prompt="Represent the sentence for clustering: ",
)

# Override document prompt for a specific encoding
df, X = model.encode(
    docs,
    prompt="Represent the scientific abstract: ",
)
```

#### How Prompts Work with Late Chunking

When a document prompt is provided:
1. The prompt is prepended to each document before tokenization
2. Sentence boundaries are detected on the original text (without prompt)
3. Prompt tokens are included in the model input for attention context
4. Prompt tokens are excluded from chunk mean-pooling (they get `sentence_id=-1`)

This ensures that document token embeddings benefit from attending to the prompt during the forward pass, while the final chunk embeddings represent only the actual document content.

## Differences from the Late Chunking Paper

Afterthoughts implements the core late chunking approach from [Günther et al., 2024](https://arxiv.org/abs/2409.04701) with some implementation choices that differ from the paper's recommendations. For details on special token handling, deduplication strategy, and chunk definitions, see [docs/gunther-et-al-2024-differences.md](docs/gunther-et-al-2024-differences.md).

## Known Limitations

#### Memory Requirements

Since each document can contain many chunks, the memory requirements for this approach can be quite high. Use `half_embeds=True` and `truncate_dims` for reduced memory footprint.

#### Sequence Length

Late chunking's contextual benefits are bounded by the model's maximum sequence length. Documents exceeding this limit are split into overlapping sequences at sentence boundaries, which can reduce cross-chunk context at the boundaries. For best results, use long-context embedding models (e.g., models supporting 8K+ tokens) with documents that fit within the context window.

## Future Work

* Add paragraph segmentation
* Support for additional chunking strategies (e.g., semantic chunking)
* Support task-specific LoRA adapters (e.g., jina-embeddings-v3)

## References

Late chunking technique:

> Günther, M., Milliken, I., Geuter, J., Mastrapas, G., Wang, B., & Xiao, H. (2024). *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models*. arXiv:2409.04701. https://arxiv.org/abs/2409.04701

## License

This project is licensed under the Apache License 2.0.

Copyright 2024-2026 Nicholas Gigliotti.

You may use, distribute, and modify this project under the terms of the Apache License 2.0. For detailed information, see the [LICENSE](LICENSE) file included in this repository or visit the official [Apache License website](http://www.apache.org/licenses/LICENSE-2.0).
