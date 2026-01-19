# Afterthoughts

A Python library for late chunking, preserving context across chunks for improved RAG retrieval, semantic search, clustering, and exploratory data analysis. Similar to the approach described in [Günther et al., 2024](https://arxiv.org/abs/2409.04701).

Independently developed with focus on production robustness, edge case handling, and ease of integration. It also emphasizes sentence-boundary awareness, since sentences are units of thought.

## What is Late Chunking?

Traditional RAG pipelines split documents into chunks *before* embedding, which loses contextual information. Consider a Wikipedia article about Berlin where the first sentence mentions "Berlin" and later sentences refer to "its population" or "the city"—when embedded separately, these chunks lose the connection to Berlin.

**Late chunking inverts this process:**

1. **Embed first**: Pass the entire document through the transformer model to get contextually-enriched token embeddings
2. **Chunk second**: Pool token embeddings into chunks *after* the model has established cross-chunk context

This approach ensures that pronouns, references, and contextual cues in each chunk are informed by the full document context.

## How Afterthoughts Implements Late Chunking

Afterthoughts provides a fast, memory-efficient implementation of late chunking optimized for production use:

1. **Sentence boundary awareness** using BlingFire, NLTK, pysbd, or syntok for accurate, linguistically-aware chunking
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
    The `encode` method returns a tuple containing a Polars DataFrame and a NumPy array of chunk embeddings. Pass `return_frame="pandas"` for a pandas DataFrame instead.

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

    To access the chunk embeddings from the `i`-th document:

    ```python
    i = 10
    doc_chunks = X[df["sample_idx"] == i]
    ```

    This works identically for both Polars and pandas DataFrames.

### Using pandas Instead of Polars

Afterthoughts uses Polars by default for its speed and memory efficiency, but pandas is fully supported for users who prefer it or need compatibility with existing code. Simply set `return_frame="pandas"`:

```python
df, X = model.encode(
    docs,
    num_sents=2,
    return_frame="pandas",  # Return a pandas DataFrame
)

# Use familiar pandas operations
df.groupby("sample_idx").size()
df[df["chunk_size"] == 2]
```

The pandas integration requires pandas to be installed (`pip install pandas`). The DataFrame schema and all functionality remain identical—only the return type changes.

### Memory Optimizations

The `Encoder` class supports two memory optimization parameters:

#### Dimension Truncation (`truncate_dims`)

For models trained with Matryoshka Representation Learning (MRL), you can truncate embeddings to smaller dimensions with minimal quality loss. No retraining required—just slice the first N dimensions.

```python
from afterthoughts import Encoder

# This model was trained with MRL at dimensions [768, 512, 256, 128, 64]
model = Encoder(
    "tomaarsen/mpnet-base-nli-matryoshka",
    truncate_dims=256,  # Truncate to 256 dimensions
)
```

Truncation is applied to token embeddings *before* pooling, which saves both memory and compute during inference.

Note: Truncation also works on non-MRL models, but may degrade embedding quality since they weren't trained to preserve information in leading dimensions.

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

### Instruct-Style Embedding Models

Many modern embedding models require instruction prefixes to achieve optimal performance. Afterthoughts supports these models through `query_prompt` and `document_prompt` parameters.

#### E5-Instruct Models

E5-instruct models (e5-mistral-7b-instruct, multilingual-e5-large-instruct) require a task instruction for queries but not for documents:

```python
from afterthoughts import Encoder

model = Encoder(
    "intfloat/multilingual-e5-large-instruct",
    query_prompt="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
)

# Queries use the prompt automatically
query_embeds = model.encode_queries(["how much protein should a female eat"])

# Documents are encoded without any prompt
df, X = model.encode(docs, num_sents=2)
```

#### BGE Models

BGE models use a simpler prefix for queries:

```python
model = Encoder(
    "BAAI/bge-large-en-v1.5",
    query_prompt="Represent this sentence for searching relevant passages: ",
)
```

#### Nomic Embed

Nomic requires task prefixes for both queries and documents:

```python
model = Encoder(
    "nomic-ai/nomic-embed-text-v1.5",
    query_prompt="search_query: ",
    document_prompt="search_document: ",
)
```

#### Instructor Models

Instructor models use domain-specific instructions for both queries and documents:

```python
model = Encoder(
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

Afterthoughts implements the core late chunking approach from [Günther et al., 2024](https://arxiv.org/abs/2409.04701) with some implementation choices that differ from the paper's recommendations. These can be toggled via parameters.

### Special Token and Prompt Handling

**Paper recommendation:** Include `[CLS]` in the first chunk's mean pooling and `[SEP]` in the last chunk's mean pooling. For instruct models, include instruction prefix tokens in the first chunk.

The paper's discussion (Section 4.1, "Dealing with Non-Context Tokens"):

> Not all tokens correspond to characters in the original string. For instance, the tokenizers of all models add a [CLS] token at the beginning and append a [SEP] token at the end of the text. Additionally, jina-embeddings-v3 and nomic-embed-text-v1 prepend an instruction to the string for distinguishing queries and documents. During late chunking, we include all embeddings of prepended tokens in the mean pooling of the first chunk and all embeddings of appended tokens to the last chunk.

No empirical comparison or theoretical justification is provided—it's presented as a practical convention. Note that they include not just `[CLS]` but also instruction prefix tokens in the first chunk's mean pool. Presumably this is done in the spirit of not throwing away information.

**Afterthoughts default:** Excludes all special tokens (`exclude_special_tokens=True`) and all prompt prefix tokens.

**Rationale:** Including `[CLS]` only in the first micro-chunk doesn't make semantic sense. In BERT-style models, `[CLS]` is trained to aggregate information from the *entire sequence* via self-attention, or at least the entire macro-chunk. Its embedding represents a summary of everything the model saw—not the first few sentences. It's called `[CLS]` as short for "classification," since in the original BERT it was trained on a binary next-sentence classification task where pairs of sentences were encoded together in the same sequence. There's no principled reason why averaging `[CLS]` into the first micro-chunk's embedding is better than including it in all micro-chunks, or excluding it entirely. The same applies to `[SEP]`, which is simply an end-of-sequence marker.

This becomes more problematic for long documents split into multiple macro chunks. Each macro chunk has its own `[CLS]` and `[SEP]` tokens, but only the first macro chunk's `[CLS]` represents anything close to "document start." The paper's reference implementation sidesteps this by concatenating all token embeddings before pooling (Algorithm 2), but this approach doesn't scale to book-length documents due to memory constraints.

**Prompt tokens:** When using instruct-style models with a `document_prompt`, Afterthoughts excludes prompt tokens from chunk mean-pooling entirely. The prompt tokens are included in the model input so document tokens can attend to them, but only actual document content contributes to the final chunk embeddings. This differs from the paper's approach of including instruction tokens in the first chunk's mean pool. Our rationale: instruction tokens describe the task, not the document content—averaging them into chunk embeddings conflates metadata with content.

Excluding all special tokens and prompt tokens is simpler, consistent across all chunks, and more semantically justifiable.

To include special tokens in boundary chunks (similar to the paper):

```python
df, X = model.encode(docs, exclude_special_tokens=False)
```

### Deduplication of Overlapping Pre-chunks

When documents exceed the model's max sequence length, both approaches split them into overlapping macro chunks. The key difference is how overlapping regions are handled:

**Paper approach (Algorithm 2):** Performs token-level deduplication by keeping only the first occurrence of overlapping token embeddings. After processing each macro chunk, the overlap tokens are dropped before concatenating with subsequent chunks. This creates a single unified token embedding sequence with a bias toward earlier context.

**Afterthoughts approach:** Computes chunk embeddings from each macro chunk separately, then deduplicates at the embedding level by averaging embeddings for chunks covering the exact same sentence IDs. This is more bidirectional, incorporating context from both preceding and following macro chunks. It also enables fast vectorized pooling operations on tensors rather than requiring concatenation of ragged token embedding matrices.

Note that only chunks with identical sentence ID sequences are averaged. Chunks in the overlap region that cover different (even partially overlapping) sentence groups are kept as distinct embeddings. The deduplication uses `np.unique` for grouping and `torch.scatter_add` for vectorized averaging, making it efficient even for large numbers of chunks (e.g., when processing books).

**Memory scalability:** The paper's Algorithm 2 constructs a single token embedding matrix for the entire document before pooling. For book-length documents (100k+ tokens), this requires holding an enormous `(tokens × hidden_dim)` matrix in memory for each book. Afterthoughts instead extracts chunk embeddings immediately after each macro chunk, discarding token embeddings before processing the next. This keeps memory at `O(max_length × hidden_dim)` regardless of document length, making it practical for arbitrarily long documents.

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
* Support task-specific LoRA adapters (e.g., jina-embeddings-v3)

## References

Late chunking technique:

> Günther, M., Milliken, I., Geuter, J., Mastrapas, G., Wang, B., & Xiao, H. (2024). *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models*. arXiv:2409.04701. https://arxiv.org/abs/2409.04701

## License

This project is licensed under the Apache License 2.0.

Copyright 2024-2026 Nicholas Gigliotti.

You may use, distribute, and modify this project under the terms of the Apache License 2.0. For detailed information, see the [LICENSE](LICENSE) file included in this repository or visit the official [Apache License website](http://www.apache.org/licenses/LICENSE-2.0).
