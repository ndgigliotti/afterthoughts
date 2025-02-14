<div align="center"><img src="./images/writing_pen.webp" height=150></div>

# FinePhrase

Generate fine-grained phrase embeddings with state-of-the-art transformers.

This is a new project that is heavily under development. Please check back soon for updates.

## Concept

FinePhrase provides a fast, memory efficient, and context-aware method of generating massive numbers of phrase embeddings using transformers. It can be used for a variety of tasks, including semantic search, rules-based classification, clustering, and more. Its primary feature is the ability to efficiently combine the transformer's contextually enriched token embeddings to derive phrase embeddings. This is done by calculating all the possible overlapping sub-sequences and averaging the corresponding token embeddings from the model's final hidden state. The result is a set of contextually enriched phrase embeddings.

Unlike tools like [KeyBERT](https://github.com/MaartenGr/KeyBERT), the purpose of FinePhrase is not to extract the top key-phrases from a document. Rather, the purpose is to extract all of the overlapping sub-sequence embeddings to facilitate fine-grained analysis. FinePhrase is designed to be highly memory efficient, allowing you to generate phrase embeddings for tens of thousands of documents without running out of memory. That means holding tens of millions of phrase embeddings in memory at once, depending on the configuration.

The word "phrase" is used here loosely to mean any sub-sequence of tokens. This can include everything from short sub-sequences (e.g. bigrams or trigrams) to extremely long sub-sequences (e.g. 500-grams). FinePhrase allows you to extract embeddings for sub-sequences of any length, with any degree of overlap. You can even extract embeddings for sub-sequences of multiple different lengths at the same time.

### Motivation

Typically data scientists opt to use document-level embeddings for tasks like semantic search, clustering, and classification. This works well for a wide range of use cases, especially those which involve shorter documents. However, these embeddings can be too coarse to capture the nuances of the data, representing the overall meaning at the expense of the details. This is particularly true when working with long or complex documents, where multiple topics are discussed in different sections. By using phrase embeddings, you can capture the meaning of the data at a much finer level of granularity.

One example use case would be searching through legal contracts to find certain clauses (e.g. a non-compete clause). If you use document-level embeddings, you may find that you miss contracts where the clause is buried in one small section of the document. However, if you use sub-sequence embeddings, you can find any part of the contract where the clause is mentioned. Furthermore, the sub-sequence embeddings are enriched with meaning from the surrounding context, allowing you to find sub-sequences which semantically match but do not lexically match your query.

Another example use case would be looking for a particular claim of interest in a dataset of lengthy movie reviews. For example, suppose that you are looking for any mention of one-dimensional characters. If you use document-level embeddings, you may find that you miss reviews where character development is a minor concern and not the central topic of the review. However, if you use phrase embeddings, you can find any part of the review where one-dimensional characters are mentioned.

### Advantages

One of the key advantages of this approach is the efficiency of deriving phrase embeddings downstream of the model. Rather than finding phrases first and running each phrase through the model as a separate sequence, the entire document is run through the model at once. Since running sequences through the model is computationally intensive, it is much faster to run a small number of documents through than a massive number of short sequences.

Another key advantage of this approach is that the phrase embeddings are enriched with meaning from the surrounding context. For example, the embedding of "the characters were really something" from a movie review would be enriched with meaning from the surrounding context, allowing it to capture either a positive or negative attitude towards the characters. Even though the phrase does not contain any explicit positive or negative tokens, the model will have shifted the constituent token vectors according to the surrounding context, resulting in a phrase embedding that accurately captures the sentiment.


## Features

* Efficiently derive phrase embeddings from state-of-the-art transformer models
* Customize the size and overlap of the phrases to extract
* Dynamically fit PCA (using GPU) to reduce the dimensionality of the embeddings
* Custom PyTorch implementation of incremental PCA (derived from `sklearn`)
* Easily embed queries or other strings in the same space as the phrases
* Uses the `transformers` library for easy integration with the Hugging Face model hub
* Built in support for automatic mixed precision (AMP)
* Outputs the phrases and indices as a Polars DataFrame for easy, scalable, manipulation

## Usage Guide

1. Install the package using pip:

    ```bash
    pip install git+https://github.com/ndgigliotti/finephrase.git
    ```

2. Create a `FinePhrase` object and load a transformer model.

    ```python
    from finephrase import FinePhrase

    # Choose a model which works well with mean-tokens pooling
    model = FinePhrase("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    ```

3. Prepare a list of documents `docs` (strings) from which to extract phrase embeddings.

    ```python
    docs = [
        "I am a document.",
        "I am another document.",
        "I am yet another document.",
        "I'm not like the others.",
    ]
    ```

4. Encode and extract phrase embeddings:

    ```python
    df, X = model.encode(
        docs,
        batch_size=512,  # Model batch size, can set larger if AMP is enabled
        phrase_sizes=[12, 24],  # Range of phrase sizes to extract
        phrase_overlap=0.5,  # Overlap between phrases
    )
    ```
    The `encode` method returns a tuple containing the Polars DataFrame and the NumPy array of phrase embeddings. If `return_frame="pandas"` is passed, it returns a Pandas DataFrame instead.

    The DataFrame contains the following columns:
    * `embed_idx`: The index of the phrase embedding in `X`
    * `sample_idx`: The index of the document from which the phrase was extracted
    * `sequence_idx`: The index of the whole sequence from which the phrase was extracted
        > `sequence_idx` is identical to `sample_idx` if no document chunking was necessary
    * `batch_idx`: The index of the batch in which the phrase was extracted
    * `phrase_size`: The token count of the phrase
    * `phrase`: The the phrase itself, as text

    The most useful columns are `embed_idx`, `sample_idx`, `phrase_size`, and `phrase`. The others are provided for reference and debugging purposes.

    To access the phrase embeddings from the `i`-th document, use the following:

    ```python
    i = 10
    doc_phrases = X[df.filter(pl.col("sample_idx") == i)["embed_idx"]]
    ```

    Or in Pandas:

    ```python
    i = 10
    doc_phrases = X[df.query("sample_idx == @i")["embed_idx"]]
    # or
    doc_phrases = X[df.loc[lambda x: x["sample_idx"] == i, "embed_idx"]]
    ```

5. Optionally, search the phrases (requires FAISS).

    ```python
    queries = [
        "I am a query.",
        "I am another query.",
        "I am yet another query.",
        "I'm a little bit different.",
    ]
    # Perform a one-off search using the model
    search_results = model.search(
        queries, # Encode queries on the fly
        phrase_embeds=X,
        phrase_df=df,
        sim_thresh=0.5,  # Cosine similarity threshold
    )
    ```

    This method is intended as a convenient way to perform one-off searches while encoding the queries on the fly. Note that the following functions and methods can be used
    for greater flexibility:

    * `FinePhrase.encode_queries`
    * `utils.search_phrases`
    * `utils.build_faiss_index`

    For more advanced use cases, it is recommended to use the `faiss` library directly, or other semantic search tools.

### Optimizations

#### Using PCA with FinePhrase

If you are working with an extremely large dataset (hundreds of thousands of documents, extremely long documents, or extremely fine-grained phrase settings), it may be necessary to use the PCA feature. If PCA is enabled, `FinePhrase` will incrementally learn a PCA transformation and then, once finished, begin applying it to each batch. The transformation is considered fit when it has seen the specified number (or proportion) of batches. This implementation of PCA harnesses the GPU, so it is fast to train and apply. Using PCA can significantly reduce the memory requirements of the pipeline without sacrificing too much quality or speed. Be sure to set the `pca` parameter to a value that balances memory efficiency and accuracy for your use case. Also be sure to set the `pca_fit_batch_count` parameter to a value that is large enough to learn the transformation. Initialize the model like so:

```python
import torch
from finephrase import FinePhrase

model = FinePhrase(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",  # Lightweight model
    pca=64,  # 64 components should capture a lot of the variance
    pca_fit_batch_count=0.33,  # The first 33% of batches will be used to fit PCA
)
```

By default, `pca_fit_batch_count` is set to `1.0`, meaning that the entire dataset will be used to fit PCA. This is good if you are not worried about memory usage and just want to apply the transformation after all the batches are finished. However, if you are working with a very large dataset and have limited memory, you can set `pca_fit_batch_count` to a value less than `1.0` to fit PCA on a subset of the batches. This will allow you to start applying the transformation sooner, at the cost of potentially lower quality embeddings.

Also keep in mind that using too small a batch size may cause the PCA transformation to be less effective, as each batch will be less representative of the overall dataset. It is recommended to use a batch size that is large enough to capture the overall distribution of the data. You may also want to shuffle your dataset before passing it in, to increase the representativeness of each batch. Furthermore, keep in mind that what PCA is being updated on are the phrase embeddings, of which there are many per sequence. So if the batch size is set to 128 and there are 100 phrases per sequence, then PCA is being updated on batches of 12,800 phrase embeddings.

If you wish to clear the PCA transformation and start over, you can call the `clear_pca` method:

```python
model.clear_pca()
```

This will reset the PCA transformation and let you fit it again.

#### Truncating the Embeddings

If you are working with a very large dataset and have limited memory, you may want to truncate the embeddings to a smaller size. This can be done by setting the `truncate_dims` parameter to a value less than the model's hidden size. For example, if the model's hidden size is 384 and you set `truncate_dims=256`, then the embeddings will be truncated to the first 256 dimensions. This can significantly reduce the memory requirements of the pipeline, but it will also reduce the quality of the embeddings. It is recommended to use this option only if you are working with a very large dataset and have limited memory. Also, PCA generally produces higher quality results and is extremely fast.

```python
from finephrase import FinePhrase

model = FinePhrase(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    truncate_dims=256,  # Truncate the embeddings to 256 dimensions
)
```

#### Reducing Precision of the Embeddings to 16-bit

To further reduce the memory footprint of the final embeddings, FinePhrase makes it convenient to reduce their precision to 16-bit floating point. This can be done by setting the `reduce_precision` parameter to `True` during initialization. This will reduce the precision of the embeddings to 16-bit floating point after they are extracted from the model and all transformations have been applied. This can be useful when working with large datasets or when memory is a concern, and generally not much quality is lost.

```python
from finephrase import FinePhrase

model = FinePhrase(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    reduce_precision=True,  # Reduce the precision of the final embeds to 16-bit
)
```

> Downcasting the final embeddings to 16-bit may actually lead to slower calculations on CPU, e.g. for semantic search. The main benefit of this option is reducing the memory footprint.

#### Using Automatic Mixed Precision (AMP)

To enable automatic mixed precision, set the `amp` parameter to `True` during initialization. This will automatically lower the numerical precision of the most numerically stable layers, reducing the memory footprint of the model and increasing inference speed. Using AMP generally lets you increase the batch size.

```python
import torch
from finephrase import FinePhrase

model = FinePhrase(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    amp=True,
    amp_dtype=torch.float16, # Choose the lower-precision data type
)
```

#### Running the Model in 16-Bit Precision

To run the model in 16-bit precision, set the `model_dtype` parameter to `torch.float16` or `torch.bfloat16` during initialization. This will reduce the memory footprint of the model and increase inference speed. Using 16-bit precision also generally lets you increase the batch size. This is similar to using AMP, but it is a cruder and more aggressive approach.

```python
import torch
from finephrase import FinePhrase
model = FinePhrase(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    model_dtype=torch.float16,  # Run the model in 16-bit precision
)
```

Alternatively, you can convert the model to 16-bit precision after it has been loaded:

```python
from finephrase import FinePhrase

model = FinePhrase("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
model.half()  # Convert the model to 16-bit precision
```

#### Example High Efficiency Configuration

An example of a highly memory-efficient configuration is to use `FinePhrase` with AMP, PCA, and reduced-precision final embeddings. This configuration is ideal for working with large datasets on a machine with limited memory. Here is an example of how to initialize the model with this configuration:

```python
import torch
from finephrase import FinePhrase

model = FinePhrase(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",  # Lightweight model
    pca=64,  # Enable PCA with 64 components
    pca_fit_batch_count=0.33,  # Use the first 33% of batches to fit PCA
    amp=True,  # Enable automatic mixed precision
    reduce_precision=True,  # Reduce the precision of the final embeds to 16-bit
)
```

## Known Limitations

#### Memory Requirements

Since each document can contain thousands of phrases, the memory requirements for this approach can be quite high.

#### Sequence Length

The context-awareness is limited by the maximum sequence length of the model. Currently, documents that exceed the maximum sequence length are handled by chunking the sequence into smaller overlapping sequences. This can lead to a loss of context at the boundaries of the chunks and also results in duplicate phrases.

## Future Ideas

* ~~Introduce phrase overlap to reduce redundant phrases~~
* ~~Add optional normalization for the phrase embeddings~~
* Add sentence mode using rules-based sentence boundary detection
* Add paragraph mode using rules-based paragraph boundary detection

## License

This project is licensed under the Apache License 2.0.

Copyright 2024 Nicholas Gigliotti.

You may use, distribute, and modify this project under the terms of the Apache License 2.0. For detailed information, see the [LICENSE](LICENSE) file included in this repository or visit the official [Apache License website](http://www.apache.org/licenses/LICENSE-2.0).
