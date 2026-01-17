<div align="center"><img src="./images/writing_pen.webp" height=150></div>

# Afterthoughts

Generate fine-grained, sentence-based chunk embeddings with state-of-the-art transformers.

This is a new project that is heavily under development. Please check back soon for updates.

## Concept

Afterthoughts provides a fast, memory efficient, and context-aware method of generating embeddings for sentence-based chunks using transformers. It can be used for a variety of tasks, including semantic search, rules-based classification, clustering, and more. Its primary feature is the ability to efficiently combine the transformer's contextually enriched token embeddings to derive chunk embeddings. This is done by detecting sentence boundaries using BlingFire, then extracting overlapping groups of consecutive sentences and averaging the corresponding token embeddings from the model's final hidden state. The result is a set of contextually enriched chunk embeddings.

The purpose of Afterthoughts is to extract embeddings for groups of sentences (chunks) to facilitate fine-grained analysis. Afterthoughts is designed to be highly memory efficient, allowing you to generate chunk embeddings for tens of thousands of documents without running out of memory. That means holding tens of millions of chunk embeddings in memory at once, depending on the configuration.

### Motivation

Typically data scientists opt to use document-level embeddings for tasks like semantic search, clustering, and classification. This works well for a wide range of use cases, especially those which involve shorter documents. However, these embeddings can be too coarse to capture the nuances of the data, representing the overall meaning at the expense of the details. This is particularly true when working with long or complex documents, where multiple topics are discussed in different sections. By using chunk embeddings, you can capture the meaning of the data at a much finer level of granularity.

One example use case would be searching through legal contracts to find certain clauses (e.g. a non-compete clause). If you use document-level embeddings, you may find that you miss contracts where the clause is buried in one small section of the document. However, if you use chunk embeddings, you can find any part of the contract where the clause is mentioned. Furthermore, the chunk embeddings are enriched with meaning from the surrounding context, allowing you to find chunks which semantically match but do not lexically match your query.

Another example use case would be looking for a particular claim of interest in a dataset of lengthy movie reviews. For example, suppose that you are looking for any mention of one-dimensional characters. If you use document-level embeddings, you may find that you miss reviews where character development is a minor concern and not the central topic of the review. However, if you use chunk embeddings, you can find any part of the review where one-dimensional characters are mentioned.

### Advantages

One of the key advantages of this approach is the efficiency of deriving chunk embeddings downstream of the model. Rather than finding chunks first and running each chunk through the model as a separate sequence, the entire document is run through the model at once. Since running sequences through the model is computationally intensive, it is much faster to run a small number of documents through than a massive number of short sequences.

Another key advantage of this approach is that the chunk embeddings are enriched with meaning from the surrounding context. For example, the embedding of "the characters were really something" from a movie review would be enriched with meaning from the surrounding context, allowing it to capture either a positive or negative attitude towards the characters. Even though the chunk does not contain any explicit positive or negative tokens, the model will have shifted the constituent token vectors according to the surrounding context, resulting in a chunk embedding that accurately captures the sentiment.


## Features

* Efficiently derive sentence-based chunk embeddings from state-of-the-art transformer models
* Customize the number of sentences per chunk and overlap between chunks
* Sentence boundary detection using BlingFire for accurate sentence segmentation
* Two classes: `Encoder` for simple usage, `LiteEncoder` for memory-efficient workflows
* Dynamically fit PCA (using GPU) to reduce the dimensionality of the embeddings
* Custom PyTorch implementation of incremental PCA (derived from `sklearn`)
* Easily embed queries or other strings in the same space as the chunks
* Uses the `transformers` library for easy integration with the Hugging Face model hub
* Built in support for automatic mixed precision (AMP)
* Outputs the chunks and indices as a pandas DataFrame for easy, scalable, manipulation

## Usage Guide

Afterthoughts provides two classes:
- **`Encoder`**: Simple API for most use cases
- **`LiteEncoder`**: Advanced API with memory optimizations (PCA, precision reduction, dimension truncation)

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

Since each document can contain many chunks, the memory requirements for this approach can be quite high.

#### Sequence Length

The context-awareness is limited by the maximum sequence length of the model. Currently, documents that exceed the maximum sequence length are handled by chunking the sequence into smaller overlapping sequences while preserving sentence boundaries. This can lead to a loss of context at the boundaries of the chunks and also results in duplicate chunks.

## Future Work

* Add paragraph segmentation
* Decouple tokenization from sentence alignment and chunking

## License

This project is licensed under the Apache License 2.0.

Copyright 2024-2026 Nicholas Gigliotti.

You may use, distribute, and modify this project under the terms of the Apache License 2.0. For detailed information, see the [LICENSE](LICENSE) file included in this repository or visit the official [Apache License website](http://www.apache.org/licenses/LICENSE-2.0).
