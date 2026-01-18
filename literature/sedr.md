# SeDR: Segment Representation Learning for Long Documents Dense Retrieval

**Authors:** Junying Chen, Qingcai Chen, Dongfang Li, Yutao Huang (Harbin Institute of Technology, Shenzhen)

**arXiv:** [2211.10841](https://arxiv.org/abs/2211.10841) (November 2022)

**Abstract:** Addresses dense retrieval for long documents where only 29.5% of MS MARCO documents fit in 512 tokens. Proposes Segment-Interaction Transformer that encodes documents into document-aware and segment-sensitive representations. Introduces Late-Cache Negative for training optimization with long documents.

## Core Problem

Current DR approaches use suboptimal strategies:
1. **Truncation**: Loses information from unused segments
2. **Splitting-and-pooling (MaxP)**: Encodes segments independently, losing document context
3. **Fixed multiple representations**: Redundant for short docs, insufficient for long docs

## Key Innovations

1. **Segment-Interaction Transformer**: [CLS] tokens from different segments attend to each other at every layer, enabling document-aware segment representations while keeping O(n_s²) complexity
2. **Segment Embedding**: Positional embedding for segment order (analogous to position embedding for tokens)
3. **Late-Cache Negative**: Stores recent embeddings to provide additional training negatives when batch size is constrained by long documents

## Segment Interaction vs Alternatives

| Method | Approach | Performance | Issue |
|--------|----------|-------------|-------|
| MaxP | Independent segments | Baseline | No cross-segment context |
| Global Attention | [CLS] attends to all tokens | Lower | Embeddings **collapse** to same point |
| Longformer | Sparse attention | Similar | 5.5x slower training |
| **SeDR** | [CLS]-to-[CLS] interaction | **Best** | None |

**Critical Finding:** Global attention causes segment embeddings to collapse into identical representations (shown via t-SNE). SeDR's approach keeps segments distinct but document-aware.

## Results on MS MARCO Document

- SeDR: MRR@100 = **0.409** (vs STAR 0.390, STAR-MaxP 0.394)
- SeDR especially outperforms on documents >512 tokens
- With ADORE: MRR@100 = **0.421**

## Relevance to FinePhrase

- Validates segment-level representations are effective for retrieval
- FinePhrase uses sentence-level segments (finer-grained than SeDR's 512-token segments)
- The collapse finding warns against certain cross-segment attention patterns
- FinePhrase's late chunking approach (full document attention → segment pooling) avoids collapse while maintaining document awareness

## Resources

- [PDF](https://arxiv.org/pdf/2211.10841)
- [GitHub](https://github.com/jymChen/SeDR)

---

## Detailed Insights

### The Long Document Problem

**MS MARCO Document Statistics:**
- Only **29.5%** of documents are <512 tokens
- Documents range from 0 to 2560+ tokens
- Current DR models truncate or use suboptimal splitting strategies

**Three core problems identified:**
1. **Length Limitation**: Transformer O(n²) complexity prevents direct processing
2. **Finite Representation Capacity**: Single embedding insufficient for diverse document content
3. **Memory Bottleneck**: Long documents → smaller batches → fewer in-batch negatives

### Segment-Interaction Transformer Architecture

**Document Processing Pipeline:**
1. Split document `d = [t₁, t₂, ..., tₙ]` into segments of `m` tokens
2. Segment count `k = ⌈n/m⌉`, last segment padded if needed
3. Each segment: `sᵢ = [CLS, tᵢₘ₊₁, ..., tᵢₘ₊ₘ, SEP]`
4. Add **segment embedding** to indicate segment order (like position embedding but for segments)
5. Apply Segment-Interaction mechanism at each transformer layer

**Segment-Interaction Mechanism (Equation 7):**
- Each token attends to: (a) tokens in same segment, (b) [CLS] tokens from ALL other segments
- [CLS] tokens form a "communication backbone" between segments
- Maintains O(n_s²) complexity (n_s = segment length) vs O(n²) for full attention

**Key insight for FinePhrase:** The [CLS]-to-[CLS] interaction pattern enables document-awareness without full attention. FinePhrase achieves similar document-awareness through late chunking (full attention within macro-chunks) without needing architectural modifications.

### Segment Embedding Details

**Purpose:** Signal segment order within document, analogous to position embedding for tokens.

**Implementation:**
- Input representation `H⁰_sᵢ` = token embedding + position embedding + i-th segment embedding
- Segment embeddings are learned during training

**FinePhrase consideration:** FinePhrase's sentence_ids already track sentence positions. Could consider adding segment-order information if multi-segment groupings need relative position context.

### Why Global Attention Fails

**t-SNE Visualization Finding (Figure 5):**

| Method | Segment Distribution | Effect on Retrieval |
|--------|---------------------|---------------------|
| MaxP (independent) | Diffuse arbitrarily | No document awareness |
| Global Attention | **Collapse to single point** | Reduced capacity |
| Longformer | Collapse to single point | Reduced capacity |
| SeDR | Scatter to document area | Best—distinct but related |
| Transformer-Head | Scatter to document area | Good—similar to SeDR |

**Critical insight:** Global attention (where [CLS] attends to ALL document tokens) causes segment embeddings to converge to identical representations. This eliminates the benefit of multiple representations.

**FinePhrase implication:** FinePhrase's approach of:
1. Full attention within macro-chunks (provides document context)
2. Mean pooling to segment embeddings (maintains segment distinctiveness)

...naturally avoids the collapse problem while achieving document-awareness.

### Late-Cache Negative for Training

**Problem:** Long document encoding → smaller batch size → fewer in-batch negatives.

**Solution:** Store recent embeddings in a queue Q (detached from gradient computation):
- Cache positive docs `d⁺` and hard negatives `d⁻` from recent batches
- Use cached embeddings as additional negatives without memory cost
- Also use cached queries to further constrain document representations

**Negative Set with Late-Cache:**
```
D⁻_q = d⁻ ∪ {d⁺, d⁻}_{q'∈B, q'≠q} ∪ {d̂⁺, d̂⁻}_{q̂∈Q}
```

**Hyperparameter findings:**
- Cache size C < 50: Significantly degrades performance
- Cache size C ≥ 50: Near-optimal performance
- Recommend: C = 50, hard negative top-K = 100

**FinePhrase relevance:** If training custom models for segment retrieval, Late-Cache Negative could improve training efficiency with long documents.

### Quantitative Results Deep Dive

**MS MARCO Document Retrieval (Table 1):**

| Model | Long Input | MRR@100 | Recall@100 | Index Size | Latency |
|-------|------------|---------|------------|------------|---------|
| BM25 | ✓ | 0.277 | 0.807 | - | 87.2ms |
| STAR | ✗ | 0.390 | 0.913 | 9.2G | 1.3ms |
| STAR-Multi (4 vectors) | ✗ | 0.404 | 0.913 | 36.8G | 4.8ms |
| ANCE(MaxP) | ✓ | 0.384 | 0.906 | 21.5G | 2.7ms |
| STAR(MaxP) | ✓ | 0.394 | 0.909 | 21.5G | 2.7ms |
| **SeDR** | ✓ | **0.409** | **0.921** | 21.5G | 2.7ms |
| SeDR + ADORE | ✓ | **0.421** | **0.933** | 21.5G | 2.7ms |

**Ablation (Table 1):**
- Without Segment-Interaction: MRR drops to 0.403
- Without Late-Cache Negative: MRR drops to 0.400
- Both components contribute, but Segment-Interaction has larger impact

**Performance by Document Length (Figure 4):**
- Documents <512 tokens: SeDR slightly underperforms STAR
- Documents 512-1024: SeDR matches others
- Documents 1024-2048: SeDR significantly outperforms
- Documents >2048: SeDR dramatically outperforms

**FinePhrase insight:** SeDR's advantage is specific to long documents. FinePhrase should similarly show greatest benefit on longer documents where more sentences provide richer segment-level retrieval.

### Segment-Interaction Pattern Comparison (Table 2)

| Method | MRR | NDCG@10 | Training Time | Indexing Time | Params | Latency |
|--------|-----|---------|---------------|---------------|--------|---------|
| SeDR-MaxP | 0.403 | 0.611 | 15.8h | 20.6h | 125M | 2.7ms |
| SeDR-Transformer-Head | 0.405 | 0.622 | 15.8h | 20.7h | 132M | 2.7ms |
| SeDR-Global-Attention | 0.406 | 0.600 | 21.2h | 26.9h | 149M | 3.4ms |
| SeDR-Longformer | 0.408 | 0.625 | 88.2h | 65.2h | 149M | 21.1ms |
| **SeDR** | **0.409** | **0.632** | 15.8h | 20.7h | 125M | 2.7ms |

**Key insights:**
1. Segment-Interaction Transformer achieves best performance
2. Longformer is 5.5x slower to train with minimal benefit
3. Global attention hurts NDCG despite similar MRR
4. SeDR adds negligible parameters vs MaxP baseline

### MaxSim Pooling for Retrieval

**Scoring function:**
```
f(q, d) = max_i {sim(E_Q(q), s̃_i)}
```

Where `s̃_i` is the i-th segment embedding. This enables:
1. Pre-compute and index all segment embeddings offline
2. Single ANN search at query time
3. Same retrieval latency as single-vector approaches

**FinePhrase parallel:** FinePhrase returns all segment embeddings, enabling similar MaxSim retrieval or more sophisticated scoring.

### Implementation Checklist for FinePhrase

Based on SeDR findings, consider:

- [ ] **Document late chunking advantage**: FinePhrase's approach achieves document-awareness like SeDR's Segment-Interaction, but through late chunking rather than architectural changes
- [ ] **Avoid collapse patterns**: Verify that FinePhrase's segment embeddings remain distinct (could add t-SNE visualization to tests)
- [ ] **Variable segment count**: Document that FinePhrase naturally produces variable segment counts based on document length (like SeDR, unlike fixed multi-vector approaches)
- [ ] **Benchmark on long documents**: Test specifically on documents >512 tokens where segment-level retrieval provides most benefit
- [ ] **MaxSim retrieval example**: Provide example code showing MaxSim scoring over FinePhrase segment embeddings
- [ ] **Sentence segments are finer-grained**: Note that FinePhrase's sentence-level segments (~15-30 tokens) provide finer granularity than SeDR's 512-token segments, combined with Late Chunking paper's finding that smaller chunks benefit most from contextual embeddings
