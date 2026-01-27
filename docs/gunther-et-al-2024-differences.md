# Differences from the Late Chunking Paper

Afterthoughts implements the core late chunking approach from [Günther et al., 2024](https://arxiv.org/abs/2409.04701) with some implementation choices that differ from the paper's recommendations.

## Special Token and Prompt Handling

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

## Deduplication of Overlapping Pre-chunks

When documents exceed the model's max sequence length, both approaches split them into overlapping macro chunks. The key difference is how overlapping regions are handled:

**Paper approach (Algorithm 2):** Performs token-level deduplication by keeping only the first occurrence of overlapping token embeddings. After processing each macro chunk, the overlap tokens are dropped before concatenating with subsequent chunks. This creates a single unified token embedding sequence with a bias toward earlier context.

**Afterthoughts approach:** Computes chunk embeddings from each macro chunk separately, then deduplicates at the embedding level by averaging embeddings for chunks that cover identical content. Deduplication uses a compound key of `(document_idx, max_chunk_sents, max_chunk_tokens, first_sent_id, last_sent_id)` to identify duplicates. This is more bidirectional than the paper's approach, incorporating context from both preceding and following macro chunks. It also enables fast vectorized pooling operations on tensors rather than requiring concatenation of ragged token embedding matrices.

**Configuration-aware deduplication:** Only chunks with identical sentence ID sequences *and* configuration parameters are averaged. This allows running multiple chunking configurations simultaneously (e.g., different `max_chunk_tokens` limits) without incorrectly merging chunks from different experiments. Split chunks (portions of sentences exceeding token limits) are kept separate even when covering the same sentence boundaries.

**Memory scalability:** The paper's Algorithm 2 constructs a single token embedding matrix for the entire document before pooling. For book-length documents (100k+ tokens), this requires holding an enormous `(tokens × hidden_dim)` matrix in memory for each book. Afterthoughts instead extracts chunk embeddings immediately after each macro chunk, discarding token embeddings before processing the next. This keeps memory at `O(max_length × hidden_dim)` regardless of document length, making it practical for arbitrarily long documents.

To disable deduplication and keep all duplicate embeddings:

```python
df, X = model.encode(docs, deduplicate=False)
```

## Chunk Definition

**Paper:** Tests multiple chunking strategies - fixed token counts (256 tokens), fixed sentence counts (5 sentences), and semantic boundaries. Late chunking is agnostic to the chunking method.

**Afterthoughts:** Supports both sentence-based and token-based chunking strategies:

**Sentence-based chunking** (similar to the paper's "Sentence Boundaries" strategy): Chunks are defined as N consecutive sentences, detected via BlingFire, NLTK, pysbd, or syntok. Chunk sizes vary based on sentence length.

```python
df, X = model.encode(docs, max_chunk_sents=5)  # 5 sentences per chunk
```

**Token-based chunking** (similar to the paper's fixed token count strategy): Chunks are built by greedily accumulating consecutive sentences until a token limit is reached. This provides more consistent chunk sizes in terms of tokens while still respecting sentence boundaries.

```python
df, X = model.encode(docs, max_chunk_tokens=256)  # ~256 tokens per chunk
```

**Combined constraints:** Both limits can be specified simultaneously. Chunks are built by accumulating sentences until either the sentence count or token count limit is hit, whichever comes first.

```python
df, X = model.encode(docs, max_chunk_sents=5, max_chunk_tokens=256)
```

**Handling long sentences:** When individual sentences exceed `max_chunk_tokens`, the `split_long_sents` parameter controls behavior:
- `split_long_sents=True` (default): Splits the sentence into multiple chunks at token boundaries, with a warning.
- `split_long_sents=False`: Keeps the sentence intact as its own chunk, exceeding the limit, with a warning.

Note that sentences exceeding the model's `max_length` are automatically split during prechunking regardless of this setting.
