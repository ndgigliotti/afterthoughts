# FinePhrase

Generate fine-grained n-gram embeddings with state-of-the-art transformers.

This is a new project that is heavily under development. Please check back soon for updates.

## Concept

FinePhrase provides a fast, memory efficient, and context-aware method of generating massive numbers of n-gram embeddings using transformers. It can be used for a variety of tasks, including semantic search, rules-based classification, clustering, and more. Its primary feature is the ability to efficiently combine the transformer's contextually enriched token embeddings to derive n-gram embeddings. This is done by calculating all the possible token n-grams and averaging the corresponding token embeddings from the model's final hidden state. The result is a set of contextually enriched n-gram embeddings.

Unlike tools like [KeyBERT](https://github.com/MaartenGr/KeyBERT), the purpose of FinePhrase is not to extract the top key-phrases from a document, nor even to extract their embeddings. Rather, the purpose is to extract *all* the n-gram embeddings to facilitate fine-grained analysis. FinePhrase is designed to be highly memory efficient, allowing you to generate n-gram embeddings for tens of thousands of documents without running out of memory. That means holding tens of millions of n-gram embeddings in memory at once, depending on the configuration.

### Motivation

Typically data scientists opt to use document-level embeddings for tasks like semantic search, clustering, and classification. This is generally much faster, more memory efficient, and more scalable. However, these embeddings can be too coarse to capture the nuances of the data, representing the "overall meaning" at the expense of the details. By using n-gram embeddings, you can capture the meaning of the data at a much finer level of granularity. This can be particularly useful when working with long or complex documents, where there are multiple topics discussed in different parts of the document.

For example, suppose that your documents are lengthy movie reviews, and you are looking for one particular claim of interest, such as "the characters are one-dimensional". If you use document-level embeddings, you may find that you miss reviews where character development is a minor concern and not the central topic of the review. However, if you use n-gram embeddings, you can find any part of the review where one-dimensional characters are mentioned.

### Advantages and Limitations

One of the key advantages of this approach is the efficiency of deriving n-gram embeddings downstream of the model. Rather than finding n-grams first and running each n-gram through the model as a separate sequence, the entire document is run through the model at once. Since running sequences through the model is computationally intensive, it is much faster to run a small number of documents through than a massive number of short sequences.

The most obvious limitation of this approach is that extracting thousands of embeddings per document (all possible n-grams) is extremely memory intensive. Hence, a considerable amount of engineering has gone into making this process as memory efficient as possible. This includes using the `transformers` library for efficient model loading, using PyTorch for efficient GPU memory management, and dynamically fitting PCA during inference to reduce the dimensionality of the embeddings.

## Features

* Efficiently derive n-gram embeddings from state-of-the-art transformer models
* Dynamically fit PCA to reduce the dimensionality of the embeddings
* Easily embed queries or other strings in the same space as the n-grams
* Uses the `transformers` library for easy integration with the Hugging Face model hub
* Built in support for automatic mixed precision (AMP)

## Usage Guide

1. Install the package using pip:

    ```bash
    pip install git+https://github.com/ndgigliotti/finephrase.git
    ```

2. Create a `FinePhrasePCA` object and load a transformer model. I generally recommend using the PCA variant of `FinePhrase`, as it is more memory efficient and scalable to datasets of tens of thousands of documents. The `FinePhrasePCA` class dynamically fits PCA to reduce the dimensionality of the n-gram embeddings, which can significantly reduce the pipeline's memory usage. For a small dataset of only a couple thousand documents, you can use the more precise `FinePhrase` class. Here is an example of how to initialize the model:

    ```python
    from finephrase import FinePhrasePCA

    # Choose a model which works well with mean-tokens pooling
    model = FinePhrasePCA(
        "sentence-transformers/paraphrase-MiniLM-L3-v2"
    )
    ```

3. Prepare a list of documents `docs` (strings) from which to extract n-gram embeddings.

    ```python
    docs = [
        "I am a document.",
        "I am another document.",
        "I am yet another document.",
        "I'm not like the others.",
    ]
    ```

4. Encode and extract n-gram embeddings:

    ```python
    results = model.encode_extract(
        docs,
        batch_size=512, # Model batch size, can set larger if AMP is enabled
        ngram_range=(4, 6), # Range of n-gram sizes to extract
    )
    ```
    The outputs are returned as a dictionary with the following keys:
    * `sequence_idx`: A 1-d array of sample indices corresponding to the input *sequences*
    * `sample_idx`: A 1-d array of n-gram indices corresponding to the input documents
    * `ngrams`: A 1-d array of n-gram strings extracted from the documents
    * `ngram_embeds`: A 2-d matrix of n-gram embeddings

    To access the n-grams from the `i`-th document, use the following:

    ```python
    doc_ngrams = results["ngrams"][results["sample_idx"] == i]
    doc_ngram_embeds = results["ngram_embeds"][results["sample_idx"] == i]
    ```

5. Optionally, encode query strings and find n-grams within a nearby radius. First 
define a quick search function:

    ```python
    import numpy as np
    from sklearn import neighbors as nb

    def search(
        queries: list[str],
        query_embeds: np.ndarray,
        ngrams: np.ndarray,
        ngram_embeds: np.ndarray,
        sample_idx: np.ndarray,
        radius: float = 0.3,
        metric: str = "cosine",
    ):
        search_index = nb.NearestNeighbors(radius=radius, metric=metric)
        search_index.fit(ngram_embeds)
        dists, idx = search_index.radius_neighbors(query_embeds, return_distance=True)
        rankings = [np.argsort(d) for d in dists]
        idx = [i[r] for i, r in zip(idx, rankings)]
        return {
            queries[i]: (ngrams[idx[i]], np.unique(sample_idx[idx[i]]))
            for i in range(len(queries))
        }
    ```

    Normally you'd want to keep `search_index` for future use, but this is just an example. Now you can search for n-grams near the queries:

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
                            ngram_embeds=results["ngram_embeds"],
                            ngrams=results["ngrams"],
                            sample_idx=results["sample_idx"]
    )
    ```
    The `radius` parameter can be adjusted to find n-grams within the specified cosine distance of the query.

### Optimizations

#### Using FinePhrasePCA

If you are working with more than a couple thousand documents, it is recommended to use the `FinePhrasePCA` class. This class uses PCA to dynamically reduce the dimensionality of each batch of n-gram embeddings. This class fits PCA incrementally until it has seen the specified number of n-grams, at which point it stops updating the PCA transformation and begins applying it to each batch (including retroactively). This can significantly reduce the memory requirements of the pipeline without sacrificing too much accuracy. Be sure to set the `n_pca_components` parameter to a value that balances memory efficiency and accuracy for your use case. Also be sure to set the `n_pca_training_samples` parameter to a value that is large enough to learn the transformation but small enough to allow the training to complete early on in the encoding process. Initialize the model like so:

```python
import torch
from FinePhrase import FinePhrasePCA

model = FinePhrasePCA(
    "sentence-transformers/paraphrase-MiniLM-L3-v2", # Lightweight model
    n_pca_components=64, # 64 components will capture a lot of the variance
    n_pca_training_batches=0.33, # The first 33% of batches will be used to fit PCA
)
```
#### Using Automatic Mixed Precision (AMP)

To enable automatic mixed precision, set the `amp` parameter to `True` during initialization. This will automatically lower the numerical precision of the most numerically stable layers, reducing the memory footprint of the model and increasing inference speed. You may also want to adjust the `amp_dtype` parameter to select the lower-precision data type.

```python
import torch
from FinePhrase import FinePhrase

model = FinePhrase(
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
    amp=True,
    amp_dtype=torch.bfloat16,
)
```

The same is true when using `FinePhrasePCA`:

```python
import torch
from FinePhrase import FinePhrasePCA

model = FinePhrasePCA(
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
    amp=True,
    amp_dtype=torch.bfloat16,
)
```

#### Quantizing the Embeddings to 16-bit

To further reduce the memory footprint of the final embeddings, FinePhrase makes it convenient to quantize them to 16-bit floating point. This can be done by setting the `quantize_embeds` parameter to `True` during initialization. This will quantize the embeddings to 16-bit floating point after they are extracted from the model and all transformations have been applied. This can be useful when working with large datasets or when memory is a concern, and generally not much accuracy is lost.

#### High Efficiency Configuration

The most memory-efficient configuration is to use `FinePhrasePCA` with automatic mixed precision and quantized embeddings. This configuration is ideal for working with large datasets on a machine with limited memory. Here is an example of how to initialize the model with this configuration:

```python
import torch
from FinePhrase import FinePhrasePCA

model = FinePhrasePCA(
    "sentence-transformers/paraphrase-MiniLM-L3-v2", # Lightweight model
    n_pca_components=64, # Will capture a lot of the variance
    n_pca_training_batches=0.25, # Use the first 25% of batches for PCA
    amp=True, # Enable automatic mixed precision
    amp_dtype=torch.bfloat16, # Use bfloat16 for better numerical stability
    quantize_embeds=True, # Quantize the final embeddings to 16-bit floating point
)
```

## Known Limitations

#### Memory Requirements

Since each document can contain thousands of n-grams, the memory requirements for this approach can be quite high.

#### Sequence Length

The context-awareness is limited by the maximum sequence length of the model. Currently, documents that exceed the maximum sequence length are handled by chunking the sequence into smaller overlapping sequences. This can lead to a loss of context at the boundaries of the chunks and also results in duplicate n-grams.

## Future Ideas

* Add optional normalization for the n-gram embeddings
* Add features for deduping the n-grams
* Add features for filtering n-grams
* Features for large datasets and persistent storage

## License

This project is licensed under the Apache License 2.0.

Copyright 2024 Nicholas Gigliotti.

You may use, distribute, and modify this project under the terms of the Apache License 2.0. For detailed information, see the [LICENSE](LICENSE) file included in this repository or visit the official [Apache License website](http://www.apache.org/licenses/LICENSE-2.0).
