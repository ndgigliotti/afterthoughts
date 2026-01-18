# Afterthoughts

A Python implementation of **Late Chunking** for generating fine-grained, context-aware sentence-chunk embeddings with transformer models.

This library implements **late chunking** ([Günther et al., 2024](https://arxiv.org/abs/2409.04701)), a technique that preserves contextual information by embedding documents first and chunking second—ensuring each chunk retains context from the full document.

> **Note:** I began developing Afterthoughts in early 2024, independently exploring ways to derive context-aware chunk embeddings from transformer models. After much iteration, I arrived at essentially the same method that Günther et al. later formalized as "late chunking" in their September 2024 paper. I've since adopted their terminology, as it neatly captures the core idea: chunking happens *late*, after the model has contextualized all tokens.

## What is Late Chunking?

Traditional RAG pipelines split documents into chunks *before* embedding, which loses contextual information. Consider a Wikipedia article about Berlin where the first sentence mentions "Berlin" and later sentences refer to "its population" or "the city"—when embedded separately, these chunks lose the connection to Berlin.

**Late chunking inverts this process:**

1. **Embed first**: Pass the entire document through the transformer model to get contextually-enriched token embeddings
2. **Chunk second**: Pool token embeddings into chunks *after* the model has established cross-chunk context

This approach ensures that pronouns, references, and contextual cues in each chunk are informed by the full document context.

## How Afterthoughts Implements Late Chunking

Afterthoughts provides a fast, memory-efficient implementation of late chunking optimized for production use:

1. **Sentence boundary detection** using BlingFire for accurate, linguistically-aware chunking
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
* **Sentence boundary detection**: BlingFire integration for accurate, fast sentence segmentation
* **Two encoder classes**: `Encoder` for simple usage, `LiteEncoder` for memory-efficient workflows with large datasets
* **GPU-accelerated PCA**: Incremental PCA for dimensionality reduction on massive embedding sets
* **Query embedding**: Embed queries in the same space as chunks for semantic search
* **HuggingFace integration**: Works with any transformer model from the HuggingFace Hub
* **Automatic mixed precision (AMP)**: Faster inference with reduced memory footprint
* **Dynamic batching**: Batches by total token count (not sequence count) for optimal GPU utilization
* **Structured output**: Returns chunks and metadata as Polars/pandas DataFrame for easy manipulation

## Usage Guide

Afterthoughts provides two classes:
- **`Encoder`**: Simple API for most use cases
- **`LiteEncoder`**: Advanced API with memory optimizations (PCA, precision reduction, dimension truncation), enabling in-memory exploration of large datasets that would otherwise exceed available RAM

### Basic Usage

1. Install the package using pip:

    ```bash
    pip install git+https://github.com/ndgigliotti/finephrase.git
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

### Memory Optimizations with LiteEncoder

For advanced users working with large datasets, `LiteEncoder` provides memory-efficient features including PCA, precision reduction, and dimension truncation. These are "lossy" optimizations that trade some embedding quality for significant memory savings.

#### Using PCA

If you are working with an extremely large dataset (hundreds of thousands of documents, extremely long documents, or extremely fine-grained chunk settings), it may be necessary to use the PCA feature. If PCA is enabled, `LiteEncoder` will incrementally learn a PCA transformation and then, once finished, begin applying it to each batch. The transformation is considered fit when it has seen the specified number (or proportion) of batches. This implementation of PCA harnesses the GPU, so it is fast to train and apply. Using PCA can significantly reduce the memory requirements of the pipeline without sacrificing too much quality or speed. Be sure to set the `pca` parameter to a value that balances memory efficiency and accuracy for your use case. Also be sure to set the `pca_early_stop` parameter to a value that is large enough to learn the transformation. Initialize the model like so:

```python
import torch
from afterthoughts import LiteEncoder

model = LiteEncoder(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",  # Lightweight model
    pca=64,  # 64 components should capture a lot of the variance
    pca_early_stop=0.33,  # The first 33% of batches will be used to fit PCA
)
```

By default, `pca_early_stop` is set to `1.0`, meaning that the entire dataset will be used to fit PCA. This is good if you are not worried about memory usage and just want to apply the transformation after all the batches are finished. However, if you are working with a very large dataset and have limited memory, you can set `pca_early_stop` to a value less than `1.0` to fit PCA on a subset of the batches. This will allow you to start applying the transformation sooner, at the cost of potentially lower quality embeddings.

Also keep in mind that using too small a batch size may cause the PCA transformation to be less effective, as each batch will be less representative of the overall dataset. It is recommended to use a batch size that is large enough to capture the overall distribution of the data. You may also want to shuffle your dataset before passing it in, to increase the representativeness of each batch. Furthermore, keep in mind that what PCA is being updated on are the chunk embeddings, of which there are many per sequence. So if the batch size is set to 128 and there are 100 chunks per sequence, then PCA is being updated on batches of 12,800 chunk embeddings.

If you wish to clear the PCA transformation and start over, you can call the `clear_pca` method:

```python
model.clear_pca()
```

This will reset the PCA transformation and let you fit it again.

#### Truncating the Embeddings

If you are working with a very large dataset and have limited memory, you may want to truncate the embeddings to a smaller size. This can be done by setting the `truncate_dims` parameter to a value less than the model's hidden size. For example, if the model's hidden size is 384 and you set `truncate_dims=256`, then the embeddings will be truncated to the first 256 dimensions. This can significantly reduce the memory requirements of the pipeline, but it will also reduce the quality of the embeddings. It is recommended to use this option only if you are working with a very large dataset and have limited memory. Also, PCA generally produces higher quality results and is extremely fast.

```python
from afterthoughts import LiteEncoder

model = LiteEncoder(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    truncate_dims=256,  # Truncate the embeddings to 256 dimensions
)
```

#### Reducing Precision of the Embeddings to 16-bit

To further reduce the memory footprint of the final embeddings, LiteEncoder makes it convenient to reduce their precision to 16-bit floating point. This can be done by setting the `half_embeds` parameter to `True` during initialization. This will reduce the precision of the embeddings to 16-bit floating point after they are extracted from the model and all transformations have been applied. This can be useful when working with large datasets or when memory is a concern, and generally not much quality is lost.

```python
from afterthoughts import LiteEncoder

model = LiteEncoder(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    half_embeds=True,  # Reduce the precision of the final embeds to 16-bit
)
```

> Downcasting the final embeddings to 16-bit may actually lead to slower calculations on CPU, e.g. for semantic search. The main benefit of this option is reducing the memory footprint.

### Performance Optimizations

These optimizations work with both `Encoder` and `LiteEncoder`.

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

#### Example High Efficiency Configuration

An example of a highly memory-efficient configuration is to use `LiteEncoder` with AMP, PCA, and reduced-precision final embeddings. This configuration is ideal for working with large datasets on a machine with limited memory. Here is an example of how to initialize the model with this configuration:

```python
import torch
from afterthoughts import LiteEncoder

model = LiteEncoder(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",  # Lightweight model
    pca=64,  # Enable PCA with 64 components
    pca_early_stop=0.33,  # Use the first 33% of batches to fit PCA
    amp=True,  # Enable automatic mixed precision
    half_embeds=True,  # Reduce the precision of the final embeds to 16-bit
)
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
- `INFO`: Model loading, compilation, preprocessing time, PCA status
- `DEBUG`: Batch sizes, token counts, and other diagnostic details

## Known Limitations

#### Memory Requirements

Since each document can contain many chunks, the memory requirements for this approach can be quite high. Use `LiteEncoder` with PCA and precision reduction for large-scale processing.

#### Sequence Length

Late chunking's contextual benefits are bounded by the model's maximum sequence length. Documents exceeding this limit are split into overlapping sequences at sentence boundaries, which can reduce cross-chunk context at the boundaries. For best results, use long-context embedding models (e.g., models supporting 8K+ tokens) with documents that fit within the context window.

## Future Work

* Add paragraph segmentation
* Support for additional chunking strategies (e.g., semantic chunking)

## References

Late chunking technique:

> Günther, M., Milliken, I., Geuter, J., Mastrapas, G., Wang, B., & Xiao, H. (2024). *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models*. arXiv:2409.04701. https://arxiv.org/abs/2409.04701

## License

This project is licensed under the Apache License 2.0.

Copyright 2024-2026 Nicholas Gigliotti.

You may use, distribute, and modify this project under the terms of the Apache License 2.0. For detailed information, see the [LICENSE](LICENSE) file included in this repository or visit the official [Apache License website](http://www.apache.org/licenses/LICENSE-2.0).
