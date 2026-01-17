# Afterthoughts

**Late chunking for transformer embeddings.**

Generate fine-grained, context-aware sentence embeddings by chunking *after* the model forward pass rather than before. Each chunk retains full document context from the transformer's attention mechanism.

## Coming Soon

This package is under active development. Features will include:

- Sentence-level embeddings with full document context
- Overlapping chunk extraction for dense retrieval
- Memory-efficient processing with incremental PCA
- Dynamic batching for optimal GPU utilization

## Why "Afterthoughts"?

- **"After"** = late (as in late chunking)
- **"Thoughts"** = the sentences/segments extracted
- Chunking is done "as an afterthought" rather than beforehand

## License

Apache 2.0
