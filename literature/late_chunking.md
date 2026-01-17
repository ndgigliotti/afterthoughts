# Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models

**Authors:** Michael Günther, Isabelle Mohr, Daniel James Williams, Bo Wang, Han Xiao (Jina AI)

**arXiv:** [2409.04701](https://arxiv.org/abs/2409.04701) | **Latest Version:** v3 (July 7, 2025)

**Abstract:** Proposes "late chunking" which applies text segmentation *after* the transformer model processes complete documents, rather than before. This preserves contextual richness in chunk embeddings. The key insight is that chunking happens just before mean pooling, allowing all tokens to attend to the full document context first.

## Algorithm

1. Tokenize the entire text (or maximum supported length)
2. Apply the transformer model to generate token embeddings with full document context
3. Apply mean pooling *after* chunking—hence "late" in its naming

**Long Late Chunking (for documents exceeding context):**
- Split text into overlapping macro-chunks of maximum tokens
- Process each macro-chunk separately with contextual overlap (ω tokens)
- Concatenate token embeddings before final pooling

## Sentence Boundary Experiments (Table 2)

| Chunking Strategy | Description |
|-------------------|-------------|
| Fixed-Size | 256 tokens per chunk |
| Sentence-Based | 5 sentences per chunk |
| Semantic | Similarity-based sentence grouping |

**Results across 3 models (jina-v2-small, jina-v3, nomic-v1) and 4 datasets:**
- Late chunking with sentence boundaries: **3.63% relative improvement** (1.9% absolute)
- Late chunking with fixed-size: **3.46% improvement** (1.8% absolute)
- Late chunking with semantic boundaries: **2.70% improvement** (1.5% absolute)

## Key Findings

**Overlap Finding (Table 6):** Adding 16-token overlap showed **no clear advantage**—nDCG@10 remained similar regardless of overlap presence.

**Chunk Size Finding:** "Late chunking performs better than naive chunking, specifically for small chunk sizes." Diminishing returns with very large chunks.

**Limitation:** On synthetic datasets (Needle-8192, Passkey-8192) with irrelevant surrounding context, naive chunking sometimes performed better—late chunking works optimally when document context remains semantically relevant.

**Training Enhancement (Span Pooling):** Optional fine-tuning applies mean pooling only to annotated answer spans rather than full documents, teaching models to encode relevant information into token embeddings.

## Resources

- [PDF](https://arxiv.org/pdf/2409.04701)
- [GitHub](https://github.com/jina-ai/late-chunking)
- [Jina AI Blog Part I](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)
- [Jina AI Blog Part II](https://jina.ai/news/what-late-chunking-really-is-and-what-its-not-part-ii/)

---

## Detailed Insights

### Algorithm Details (for implementation verification)

**Standard Late Chunking (Algorithm 1):**
1. Chunk text using any chunking strategy (sentence, fixed-size, semantic)
2. Tokenize the ENTIRE text (not chunks individually)
3. Apply transformer model to get token embeddings with full document context
4. Map chunk boundaries to token positions (character offsets → token indices)
5. Mean pool token embeddings within each chunk's token range

**Long Late Chunking (Algorithm 2) - for documents exceeding context length:**
1. Split into macro-chunks of `l_max` tokens with overlap `ω`
2. Process each macro-chunk through the model
3. For first macro-chunk: keep all token embeddings
4. For subsequent macro-chunks: discard first `ω` embeddings (overlap region), keep rest
5. Concatenate all kept embeddings, then apply standard late chunking pooling

**Key implementation detail:** The overlap tokens provide context during transformer attention but their embeddings are discarded to avoid duplicate representations.

### Special Token Handling

The paper explicitly addresses how to handle non-content tokens:
- **[CLS] token embeddings**: Include in mean pooling of the **first** chunk
- **[SEP] token embeddings**: Include in mean pooling of the **last** chunk
- **Instruction prefix tokens** (for models like jina-v3, nomic): Include in mean pooling of the **first** chunk

**FinePhrase action:** Verify current implementation handles special tokens correctly. Check if sentence_ids properly excludes or includes special tokens in segment pooling.

### Quantitative Results Deep Dive

**Table 2 - Chunking Strategy Comparison (nDCG@10 averaged across 3 models, 4 datasets):**

| Strategy | Naive → Late | Improvement |
|----------|--------------|-------------|
| Sentence (5 sent/chunk) | 52.4 → 54.3 | +3.63% relative |
| Fixed-size (256 tok/chunk) | 52.2 → 54.0 | +3.46% relative |
| Semantic boundaries | 52.4 → 53.8 | +2.70% relative |

**Insight:** Sentence boundaries provide the largest improvement with late chunking. FinePhrase's sentence-aware approach is optimal.

**Table 6 - Overlap Analysis (jina-v2-small, 256 tokens, 16 token overlap):**

| Dataset | Late w/ Overlap | Late w/o Overlap |
|---------|-----------------|------------------|
| SciFact | 66.1 | 65.9 |
| NFCorpus | 30.0 | 30.5 |
| FiQA | 33.8 | 34.0 |
| TRECCOVID | 64.7 | 64.9 |

**Insight:** Overlap shows no consistent benefit—sometimes slightly worse. With late chunking, context is already captured through full-document attention. FinePhrase could potentially simplify by defaulting `segment_overlap=0`.

### Chunk Size Sweet Spot (Figure 3)

Performance vs chunk size (tokens) on long document tasks:
- **4-64 tokens**: Late chunking dramatically outperforms naive
- **64-256 tokens**: Late chunking maintains advantage, gap narrows
- **256-512 tokens**: Advantages diminish, sometimes converge

**FinePhrase insight:** With sentence-level segments (~15-30 tokens typical), FinePhrase operates in the optimal zone where late chunking provides maximum benefit.

### Failure Modes (Section 4.2)

Late chunking underperforms or shows no advantage when:
1. **Needle-in-haystack tasks**: Target text surrounded by irrelevant content
2. **Passkey-style tasks**: Single relevant phrase in random text
3. **Very large chunks**: Context benefit diminishes as chunk approaches document size

**FinePhrase guidance:** Late chunking excels for semantically coherent documents (articles, papers, reports) but may not help for documents with isolated relevant snippets in irrelevant filler.

### Span Pooling Training (Section 3.2)

Optional training enhancement:
- Training data: `(query, document, <start, end>)` tuples with answer span annotations
- During training: pool only the span tokens, not full document
- Result: Model learns to encode relevant information into span token embeddings

**Datasets used:** FEVER (sentence-level annotations), TriviaQA (phrase-level)

**Performance gain (Table 3):** +0.5-1% nDCG@10 improvement over standard mean pooling training

**FinePhrase relevance:** If training custom models, span pooling could improve segment retrieval quality. Requires span-annotated training data.

### Comparison to LLM Contextual Embedding (Section 4.5)

Direct comparison using financial document example:

| Method | Correct chunk similarity | Advantage |
|--------|-------------------------|-----------|
| Late Chunking | 0.8516 | No LLM required |
| Contextual Embedding (Claude) | 0.8590 | Slightly higher |
| Naive Chunking | 0.6343 | - |

**Insight:** Late chunking achieves ~99% of contextual embedding quality without LLM inference costs.

### Benchmarks Used

For replication or comparison:
- **Short docs:** BeIR (SciFact, NFCorpus, FiQA, TRECCOVID)
- **Long docs:** LongEmbed (NarrativeQA, 2WikiMultiHopQA, SummScreenFD, QMSum)
- **Synthetic:** Needle-8192, Passkey-8192 (where late chunking struggles)

### Implementation Checklist for FinePhrase

Based on paper findings, verify FinePhrase:

- [ ] **Tokenizes full documents before segmentation** (core late chunking principle) ✓
- [ ] **Handles special tokens correctly** in segment pooling
- [ ] **Chunk overlap during macro-chunk processing** uses contextual overlap but discards duplicate embeddings
- [ ] **Sentence boundary detection** is accurate (BlingFire validated)
- [ ] **Default overlap setting** could be 0 given paper findings
- [ ] **Documentation notes** limitations with needle-in-haystack content
