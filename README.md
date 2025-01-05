# FinePhrase

Generate fine-grained phrase embeddings with state-of-the-art transformers.

This is a new project that is heavily under development. Please check back soon for updates.

## Concept

FinePhrase provides a fast, memory efficient, and context-aware method of generating massive numbers of phrase embeddings using transformers. It can be used for a variety of tasks, including semantic search, rules-based classification, clustering, and more. Its primary feature is the ability to efficiently combine the transformer's contextually enriched token embeddings to derive phrase embeddings. This is done by calculating all the possible overlapping sub-sequences and averaging the corresponding token embeddings from the model's final hidden state. The result is a set of contextually enriched phrase embeddings.

Unlike tools like [KeyBERT](https://github.com/MaartenGr/KeyBERT), the purpose of FinePhrase is not to extract the top key-phrases from a document. Rather, the purpose is to extract a large number of overlapping phrase embeddings to facilitate fine-grained analysis. FinePhrase is designed to be highly memory efficient, allowing you to generate phrase embeddings for tens of thousands of documents without running out of memory. That means holding tens of millions of phrase embeddings in memory at once, depending on the configuration.

### Motivation

Typically data scientists opt to use document-level embeddings for tasks like semantic search, clustering, and classification. This is generally much faster, more memory efficient, and more scalable. However, these embeddings can be too coarse to capture the nuances of the data, representing the "overall meaning" at the expense of the details. By using phrase embeddings, you can capture the meaning of the data at a much finer level of granularity. This can be particularly useful when working with long or complex documents, where there are multiple topics discussed in different parts of the document.

For example, suppose that your documents are lengthy movie reviews, and you are looking for one particular claim of interest, such as "the characters are one-dimensional". If you use document-level embeddings, you may find that you miss reviews where character development is a minor concern and not the central topic of the review. However, if you use phrase embeddings, you can find any part of the review where one-dimensional characters are mentioned.

### Advantages and Limitations

One of the key advantages of this approach is the efficiency of deriving phrase embeddings downstream of the model. Rather than finding phrases first and running each phrase through the model as a separate sequence, the entire document is run through the model at once. Since running sequences through the model is computationally intensive, it is much faster to run a small number of documents through than a massive number of short sequences.

The most obvious limitation of this approach is that extracting thousands of embeddings per document (all possible overlapping sub-sequences) is extremely memory intensive. Hence, a considerable amount of engineering has gone into making this process as memory efficient as possible. This includes using the `transformers` library for efficient model loading, using PyTorch for efficient GPU memory management, and dynamically fitting PCA during inference to reduce the dimensionality of the embeddings.

## Features

* Efficiently derive phrase embeddings from state-of-the-art transformer models
* Customize the size and overlap of the phrases to extract
* Dynamically fit PCA (using GPU) to reduce the dimensionality of the embeddings
* Custom PyTorch implementation of incremental PCA (derived from `sklearn`)
* Easily embed queries or other strings in the same space as the phrases
* Uses the `transformers` library for easy integration with the Hugging Face model hub
* Built in support for automatic mixed precision (AMP)

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
    results = model.encode(
        docs,
        batch_size=512,  # Model batch size, can set larger if AMP is enabled
        phrase_sizes=[12, 24],  # Range of phrase sizes to extract
        phrase_overlap=0.5,  # Overlap between phrases
    )
    ```
    The outputs are returned as a dictionary with the following keys:
    * `sequence_idx`: A 1-d array of sample indices corresponding to the input *sequences*
    * `sample_idx`: A 1-d array of phrase indices corresponding to the input documents
    * `phrases`: A list phrase strings extracted from the documents
    * `phrase_embeds`: A 2-d matrix of phrase embeddings

    To access the phrase embeddings from the `i`-th document, use the following:

    ```python
    doc_phrase_embeds = results["phrase_embeds"][results["sample_idx"] == i]
    ```

5. Optionally, encode query strings and find phrases within a nearby radius. First
define a quick search function:

    ```python
    import numpy as np
    from sklearn import neighbors as nb

    def search(
        queries: list[str],
        query_embeds: np.ndarray,
        phrases: np.ndarray,
        phrase_embeds: np.ndarray,
        sample_idx: np.ndarray,
        radius: float = 0.3,
        metric: str = "cosine",
    ):
        search_index = nb.NearestNeighbors(radius=radius, metric=metric)
        search_index.fit(phrase_embeds)
        dists, idx = search_index.radius_neighbors(query_embeds, return_distance=True)
        rankings = [np.argsort(d) for d in dists]
        idx = [i[r] for i, r in zip(idx, rankings)]
        return {
            queries[i]: (phrases[idx[i]], np.unique(sample_idx[idx[i]]))
            for i in range(len(queries))
        }
    ```

    Normally you'd want to keep `search_index` for future use, but this is just an example. Now you can search for phrases near the queries:

    ```python
    queries = [
        "I am a query.",
        "I am another query.",
        "I am yet another query.",
        "I'm a little bit different.",
    ]
    # Encode the queries using the model
    query_embeds = model.encode_queries(queries)
    search_results = search(queries,
                            query_embeds,
                            phrase_embeds=results["phrase_embeds"],
                            phrases=results["phrases"],
                            sample_idx=results["sample_idx"]
    )
    ```
    The `radius` parameter can be adjusted to find phrases within the specified cosine distance of the query.

### Optimizations

#### Using PCA with FinePhrase

If you are working with an extremely large dataset (hundreds of thousands of documents, or extremely fine-grained phrase settings), it is recommended to use the PCA option. This option uses PCA to dynamically reduce the dimensionality of each batch of phrase embeddings. This class fits PCA incrementally until it has seen the specified number of phrases, at which point it stops updating the PCA transformation and begins applying it to each batch. This can significantly reduce the memory requirements of the pipeline without sacrificing too much accuracy. Be sure to set the `pca` parameter to a value that balances memory efficiency and accuracy for your use case. Also be sure to set the `pca_fit_batch_count` parameter to a value that is large enough to learn the transformation. Initialize the model like so:

```python
import torch
from finephrase import FinePhrase

model = FinePhrase(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",  # Lightweight model
    pca=64,  # 64 components should capture a lot of the variance
    pca_fit_batch_count=0.33,  # The first 33% of batches will be used to fit PCA
)
```
#### Using Automatic Mixed Precision (AMP)

To enable automatic mixed precision, set the `amp` parameter to `True` during initialization. This will automatically lower the numerical precision of the most numerically stable layers, reducing the memory footprint of the model and increasing inference speed. You may also want to adjust the `amp_dtype` parameter to select the lower-precision data type.

```python
import torch
from finephrase import FinePhrase

model = FinePhrase(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    amp=True,
    amp_dtype=torch.float16,
)
```
#### Reducing Precision of the Embeddings to 16-bit

To further reduce the memory footprint of the final embeddings, FinePhrase makes it convenient to reduce their precision to 16-bit floating point. This can be done by setting the `reduce_precision` parameter to `True` during initialization. This will reduce the precision of the embeddings to 16-bit floating point after they are extracted from the model and all transformations have been applied. This can be useful when working with large datasets or when memory is a concern, and generally not much accuracy is lost.

#### High Efficiency Configuration

The most memory-efficient configuration is to use `FinePhrase` with automatic mixed precision and quantized embeddings. This configuration is ideal for working with large datasets on a machine with limited memory. Here is an example of how to initialize the model with this configuration:

```python
import torch
from finephrase import FinePhrase

model = FinePhrase(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",  # Lightweight model
    pca=64,  # Enable PCA with 64 components
    pca_fit_batch_count=0.25,  # Use the first 25% of batches to fit PCA
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
* Add features for deduping the phrases
* Add features for filtering phrases
* Features for persistent storage

## License

This project is licensed under the Apache License 2.0.

Copyright 2024 Nicholas Gigliotti.

You may use, distribute, and modify this project under the terms of the Apache License 2.0. For detailed information, see the [LICENSE](LICENSE) file included in this repository or visit the official [Apache License website](http://www.apache.org/licenses/LICENSE-2.0).
