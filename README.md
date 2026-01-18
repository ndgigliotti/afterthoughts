# Afterthoughts

A Python library for late chunking, preserving context across chunks for improved RAG retrieval, semantic search, clustering, and exploratory data analysis. Based on the approach described in [Günther et al., 2024](https://arxiv.org/abs/2409.04701).

Independently developed with focus on production robustness, edge case handling, and ease of integration.

## What is Late Chunking?

Traditional RAG pipelines split documents into chunks *before* embedding, which loses contextual information. Consider a Wikipedia article about Berlin where the first sentence mentions "Berlin" and later sentences refer to "its population" or "the city"—when embedded separately, these chunks lose the connection to Berlin.

**Late chunking inverts this process:**

1. **Embed first**: Pass the entire document through the transformer model to get contextually-enriched token embeddings
2. **Chunk second**: Pool token embeddings into chunks *after* the model has established cross-chunk context

This approach ensures that pronouns, references, and contextual cues in each chunk are informed by the full document context.

## How Afterthoughts Implements Late Chunking

Afterthoughts provides a fast, memory-efficient implementation of late chunking optimized for production use:

1. **Sentence boundary detection** using BlingFire, NLTK, pysbd, or syntok for accurate, linguistically-aware chunking
2. **Full document embedding** through transformer models to capture cross-sentence context
3. **Sentence-based pooling** of token embeddings from the model's final hidden state
4. **Overlapping chunk extraction** with configurable sentence counts and overlap ratios

The result is a set of chunk embeddings where each chunk's representation is enriched by the surrounding document context—even if the chunk itself contains ambiguous references or pronouns.

## Why Use Chunk Embeddings?

Document-level embeddings work well for shorter texts but can be too coarse for long or complex documents where multiple topics appear in different sections.

**Example use cases:**

- **Legal document search**: Find specific clauses (e.g., non-compete provisions) buried within lengthy contracts, rather than matching on overall document similarity
- **Review analysis**: Locate specific claims in lengthy reviews (e.g., mentions of "one-dimensional characters") even when they're a minor point rather than the main topic
- **Research paper search**: Find relevant paragraphs discussing specific methods or results within long academic papers

## Key Advantages

### Computational Efficiency

Rather than running each chunk through the model separately, late chunking runs the full document once and derives all chunk embeddings from the resulting token embeddings. This is significantly faster than embedding thousands of short chunks individually.

### Contextual Enrichment

Chunk embeddings capture meaning from surrounding context. For example, "the characters were really something" from a movie review will have different embeddings depending on whether the surrounding text is positive or negative—the model shifts token vectors based on context, producing embeddings that accurately reflect sentiment even without explicit sentiment words in the chunk.


## Features

* **Late chunking implementation**: Embed documents first, then pool into chunks for context-aware embeddings
* **Flexible chunk configuration**: Customize sentences per chunk and overlap between chunks
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

2. Create an `Encoder` object and load a transformer model.

    ```python
    from afterthoughts import Encoder

    # Choose a model which works well with mean-tokens pooling
    model = Encoder("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    ```

3. Prepare a list of documents `docs` (strings) from which to extract chunk embeddings.

    ```python
    docs = [
        "I am a document. It has multiple sentences.",
        "I am another document. This one also has sentences.",
        "I am yet another document. Sentences are great.",
        "I'm not like the others. I'm special.",
    ]
    ```

4. Encode and extract chunk embeddings:

    ```python
    df, X = model.encode(
        docs,
        num_sents=[1, 2],  # Extract 1-sentence and 2-sentence chunks
        chunk_overlap=0.5,  # Overlap between chunks (in sentences)
    )
    ```
    The `encode` method returns a tuple containing the pandas DataFrame and the NumPy array of chunk embeddings. If `return_frame="polars"` is passed, it returns a Polars DataFrame instead.

    To use a different sentence tokenizer, pass the `sent_tokenizer` parameter:

    ```python
    df, X = model.encode(
        docs,
        num_sents=2,
        sent_tokenizer="pysbd",  # Options: "blingfire" (default), "nltk", "pysbd", "syntok"
    )
    ```

    The DataFrame contains the following columns:
    * `sample_idx`: The index of the document from which the chunk was extracted
    * `chunk_idx`: A global index preserving chunk extraction order
    * `chunk_size`: The number of sentences in the chunk
    * `chunk`: The chunk itself, as text

    Additional columns are available when `debug=True`:
    * `embed_idx`: The original embedding index before re-sorting
    * `sequence_idx`: The index of the tokenized sequence (differs from `sample_idx` when long documents are chunked)
    * `batch_idx`: The index of the batch in which the chunk was processed

    To access the chunk embeddings from the `i`-th document, use the following:

    ```python
    i = 10
    doc_mask = df["sample_idx"] == i
    doc_chunks = X[doc_mask]
    ```

    Or in Polars:

    ```python
    i = 10
    doc_chunks = X[df["sample_idx"] == i]
    ```

### Memory Optimizations

The `Encoder` class supports two memory optimization parameters:

#### Dimension Truncation (`truncate_dims`)

For models trained with Matryoshka Representation Learning (MRL) like nomic-embed, jina-v3, or OpenAI v3, you can truncate embeddings to a smaller size without retraining. This is the simplest option—no fitting required, just slice the first N dimensions.

```python
from afterthoughts import Encoder

model = Encoder(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    truncate_dims=256,  # Truncate the embeddings to 256 dimensions
)
```

Truncation is applied to token embeddings *before* pooling, which saves both memory and compute during inference.

#### Float16 Embeddings (`half_embeds`)

Convert chunk embeddings to float16 for 2x memory reduction:

```python
from afterthoughts import Encoder

model = Encoder(
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
from afterthoughts import Encoder

model = Encoder(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    amp=True,
    amp_dtype=torch.float16, # Choose the lower-precision data type
)
```

#### Running the Model in 16-Bit Precision

To run the model in 16-bit precision, set the `torch_dtype` parameter to `torch.float16` or `torch.bfloat16` during initialization. This will reduce the memory footprint of the model and increase inference speed. Using 16-bit precision also generally lets you increase the batch size. This is similar to using AMP, but it is a cruder and more aggressive approach.

```python
import torch
from afterthoughts import Encoder
model = Encoder(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    torch_dtype=torch.float16,  # Run the model in 16-bit precision
)
```

Alternatively, you can convert the model to 16-bit precision after it has been loaded:

```python
from afterthoughts import Encoder

model = Encoder("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
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

## Differences from the Late Chunking Paper

Afterthoughts implements the core late chunking approach from [Günther et al., 2024](https://arxiv.org/abs/2409.04701) with some implementation choices that differ from the paper's recommendations. These can be toggled via parameters.

### Special Token Handling

**Paper recommendation:** Include `[CLS]` in the first chunk's mean pooling and `[SEP]` in the last chunk's mean pooling.

**Afterthoughts default:** Follows the paper's approach (`exclude_special_tokens=False`).

To exclude all special tokens from mean pooling:

```python
df, X = model.encode(docs, exclude_special_tokens=True)
```

### Deduplication of Overlapping Pre-chunks

When documents exceed the model's max sequence length, both approaches split them into overlapping macro chunks. The key difference is how overlapping regions are handled:

**Paper approach (Algorithm 2):** Performs token-level deduplication by keeping only the first occurrence of overlapping token embeddings. After processing each macro chunk, the overlap tokens are dropped before concatenating with subsequent chunks. This creates a single unified token embedding sequence with a bias toward earlier context.

**Afterthoughts approach:** Computes chunk embeddings from each macro chunk separately, then deduplicates at the embedding level by averaging embeddings for chunks covering the exact same sentence IDs. This is more bidirectional, incorporating context from both preceding and following macro chunks. It also enables fast vectorized pooling operations on tensors rather than requiring concatenation of ragged token embedding matrices.

Note that only chunks with identical sentence ID sequences are averaged. Chunks in the overlap region that cover different (even partially overlapping) sentence groups are kept as distinct embeddings. The deduplication uses `np.unique` for grouping and `torch.scatter_add` for vectorized averaging, making it efficient even for large numbers of chunks (e.g., when processing books).

To disable deduplication and keep all duplicate embeddings:

```python
df, X = model.encode(docs, deduplicate=False)
```

### Chunk Definition

**Paper:** Tests multiple chunking strategies - fixed token counts (256 tokens), fixed sentence counts (5 sentences), and semantic boundaries. Late chunking is agnostic to the chunking method.

**Afterthoughts:** Uses sentence-based chunking exclusively (similar to the paper's "Sentence Boundaries" strategy). Chunks are defined as N consecutive sentences, detected via BlingFire, NLTK, or syntok. This means chunk sizes vary based on sentence length rather than being fixed token counts.

## Known Limitations

#### Memory Requirements

Since each document can contain many chunks, the memory requirements for this approach can be quite high. Use `half_embeds=True` and `truncate_dims` for reduced memory footprint.

#### Sequence Length

Late chunking's contextual benefits are bounded by the model's maximum sequence length. Documents exceeding this limit are split into overlapping sequences at sentence boundaries, which can reduce cross-chunk context at the boundaries. For best results, use long-context embedding models (e.g., models supporting 8K+ tokens) with documents that fit within the context window.

## Future Work

* Add paragraph segmentation
* Support for additional chunking strategies (e.g., semantic chunking)
* Support instruct embedding models

## References

Late chunking technique:

> Günther, M., Milliken, I., Geuter, J., Mastrapas, G., Wang, B., & Xiao, H. (2024). *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models*. arXiv:2409.04701. https://arxiv.org/abs/2409.04701

## License

This project is licensed under the Apache License 2.0.

Copyright 2024-2026 Nicholas Gigliotti.

You may use, distribute, and modify this project under the terms of the Apache License 2.0. For detailed information, see the [LICENSE](LICENSE) file included in this repository or visit the official [Apache License website](http://www.apache.org/licenses/LICENSE-2.0).
