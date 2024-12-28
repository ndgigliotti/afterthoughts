# Phrase Foundry

Forge n-gram embeddings using state-of-the-art transformers.

This is a new project that is heavily under development. Please check back soon for updates.

## Concept

Phrase Foundry provides a fast and context-aware method of embedding n-grams using transformers. It can be used for a variety of tasks, including semantic search, search-based classification, clustering, cluster interpretation, and more. Its primary feature is the ability to "melt down" the transformer's contextually enriched token embeddings and "cast" them into n-gram embeddings. This is done by calculating all the possible token n-grams and averaging the corresponding token embeddings from the model's final hidden state. The result is a set of contextually enriched n-gram embeddings.

One of the key advantages of this approach is the efficiency of deriving n-gram embeddings downstream of the model. Rather than finding n-grams using e.g. `nltk` and running each n-gram through the model as a separate sequence, the entire document is run through the model at once. Since running sequences through the model is the bottleneck of the process, it is much faster to run through a small number of documents than a large number of short sequences.

## Features

* Efficiently derive n-gram embeddings from state-of-the-art transformer models
* Easily embed queries or other strings in the same space as the n-grams
* Uses the `transformers` library for easy integration with the Hugging Face model hub
* Built in support for automatic mixed precision (AMP)

## Usage Guide

1. Install the package using pip:

    ```bash
    pip install pip@git+https://github.com/ndgigliotti/phrase-foundry.git
    ```

2. Create a `PhraseFoundry` object and load a transformer model:

    ```python
    from phrase_foundry import PhraseFoundry

    model = PhraseFoundry(
        # Choose a model which works well with mean-tokens pooling
        "sentence-transformers/all-mpnet-base-v2",
        device="cuda",
        # Disallow n-grams that start with a subword token
        invalid_start_token_pattern=r"^##",
    )
    ```

3. Prepare a list of documents `docs` (strings) from which to extract n-gram embeddings.

4. Encode and extract n-gram embeddings:

    ```python
    samp_idx, ngrams, ngram_vecs = model.encode_extract(
        docs,
        batch_size=256, # Model batch size, can set larger if AMP is enabled
        ngram_range=(4, 6), # Range of n-gram sizes to extract
    )
    ```
    Three outputs are returned:
    * `samp_idx`: A 1-d array of sample indices corresponding to the input documents
    * `ngrams`: A 1-d array of n-gram strings extracted from the documents
    * `ngram_vecs`: A 2-d matrix of n-gram embeddings

    To access the n-grams from the `i`-th document, use the following:

    ```python
    doc_ngrams = ngrams[samp_idx == i]
    doc_ngram_vecs = ngram_vecs[samp_idx == i]
    ```

5. Optionally, encode a query string and find ngrams within a nearby radius:

    ```python
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(radius=0.45, metric="cosine")
    nn.fit(ngram_vecs)
    queries = model.encode_queries(["interest rate is too high"])
    dists, idx = nn.radius_neighbors(queries)
    ngrams[idx[0]] # N-grams
    samp_idx[idx[0]] # Source document indices
    ```
    The `radius` parameter can be adjusted to find n-grams within a certain cosine distance of the query.

### Using Automatic Mixed Precision (AMP)

To enable automatic mixed precision, set the `amp` parameter to `True` when calling any of the encoding methods. This will automatically lower the numerical precision of the most numerically stable layers, reducing the memory footprint of the model and increasing inference speed. You may also want to adjust the `amp_dtype` parameter to select the lower-precision data type.

```python
import torch

samp_idx, ngrams, ngram_vecs = model.encode_extract(
    docs,
    batch_size=512, # Model batch size set larger since AMP is enabled
    ngram_range=(4, 6),
    amp=True,
    amp_dtype=torch.bfloat16,
)
```
## Known Limitations

#### Memory Requirements

Since each document can contain thousands of n-grams, the memory requirements for this approach can be quite high. Currently, it is limited to usage on small datasets, or processing small batches of large datasets incrementally. In the future, I may add features to make it more convenient to work with large datasets.

#### Sequence Length

The context-awareness is limited by the maximum sequence length of the model. Currently, documents that exceed the maximum sequence length are handled by chunking the sequence into smaller overlapping sequences. This can lead to a loss of context at the boundaries of the chunks and also results in duplicate n-grams.

## License

This project is licensed under the Apache License 2.0.

Copyright 2024 Nicholas Gigliotti.

You may use, distribute, and modify this project under the terms of the Apache License 2.0. For detailed information, see the [LICENSE](LICENSE) file included in this repository or visit the official [Apache License website](http://www.apache.org/licenses/LICENSE-2.0).
