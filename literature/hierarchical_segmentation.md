# Enhancing RAG with Hierarchical Text Segmentation Chunking

**Authors:** Hai-Toan Nguyen, Tien-Dat Nguyen, Viet-Ha Nguyen (VNU University of Engineering and Technology)

**arXiv:** [2507.09935](https://arxiv.org/abs/2507.09935) (July 2025)

**Abstract:** Proposes a bottom-up hierarchical framework that combines supervised text segmentation with unsupervised clustering for RAG. Each chunk is represented by multiple vectors (segment embeddings + cluster embedding), enabling more precise retrieval through multi-vector matching.

## Key Framework Components

1. **Text Segmentation**: Supervised BiLSTM model predicts sentence-level section boundaries
2. **Chunk Clustering**: Graph-based clustering groups related segments based on semantic similarity and sequential order
3. **Multi-Vector Retrieval**: `cos(q, Ci) = max(cos(q, Es1), ..., cos(q, Esm), cos(q, Ec))` - takes max over segment and cluster embeddings

## Clustering Algorithm

1. Build relatedness graph: segments as nodes, edges if similarity > τ = μ + k·σ
2. Find maximal cliques in graph
3. Merge adjacent segments sharing cliques → initial clusters
4. Merge adjacent clusters connected via cliques
5. Final merge: orphan segments to nearest cluster
6. Cluster embedding = mean pooling of segment embeddings

## k-Parameter for Cluster Size Control

| Target chunk size | k value |
|-------------------|---------|
| 512 tokens | 1.2 |
| 1024 tokens | 0.7 |
| 2048 tokens | 0.4 |

## Results Summary

| Dataset | Best Config | Method | Score | Baseline |
|---------|-------------|--------|-------|----------|
| NarrativeQA | 1024 tokens | Seg+Cluster | **26.54** ROUGE-L | 23.86 |
| QuALITY | 512 tokens | Seg+Cluster | **63.77%** Acc | 60.23% |
| QASPER | 1024 tokens | Seg+Cluster | **24.67** F1 | 22.07 |

## Critical Findings

- **Segment+Cluster > Cluster Only > Base**: Both segment and cluster embeddings contribute
- **Diminishing returns at 2048 tokens**: "Larger chunks become too broad, causing the reader model to lose focus on query-relevant details"
- **Sweet spot around 1024 tokens** for most tasks

## Relevance to FinePhrase

- Validates multi-vector segment retrieval approach FinePhrase uses
- Clustering could be optional post-processing on FinePhrase segments
- Similar MaxSim retrieval pattern as SeDR paper
- Confirms sentence-boundary segmentation outperforms fixed-size

## Resources

- [PDF](https://arxiv.org/pdf/2507.09935)

---

## Detailed Insights

### The Problem with Traditional Chunking

**Core issues identified:**
1. Fixed-size chunking fails to capture semantic meaning
2. No awareness of underlying textual structure
3. Complex queries require understanding multiple document parts

**Why bottom-up (not top-down):**
- Top-down would naturally suit hierarchical structures
- But current models lack multi-level training data
- Models struggle with processing very long documents
- Bottom-up works well with RAG's non-sequential retrieval (chunks retrieved by relevance, not position)

### Multi-Vector Retrieval Strategy

**Indexing phase:**
```
Indexing(D) = {(Es, Ec) | Es = f_segment(Si), Ec = f_cluster(Cj)}
```

Each document produces:
- Multiple segment embeddings (Es1, Es2, ..., Esm)
- Cluster embeddings (Ec) for grouped segments

**Retrieval phase:**
```
cos(q, Ci) = max(cos(q, Es1), ..., cos(q, Esm), cos(q, Ec))
```

The max-pooling over multiple embeddings provides:
1. More matching options during retrieval
2. Can match on specific segment details OR broader cluster context
3. Increases likelihood of precise matches

**FinePhrase parallel:** FinePhrase's overlapping segments (`segment_overlap > 0`) and multiple segment sizes (`segment_sizes=[1,3,5]`) naturally create multiple embeddings per document region, enabling similar multi-vector retrieval.

### Text Segmentation Model Details

**Architecture (Koshorek et al. [14]):**
- **Sentence embedding layer**: 2-layer bidirectional LSTM (input=300, hidden=256)
- Max-pooling over LSTM outputs → fixed-length sentence representations
- **Classifier layer**: 2-layer bidirectional LSTM (input=512, hidden=256)
- Binary classification: 1 = end of segment, 0 = continuation

**Training:**
- SGD, batch size 32, 20 epochs
- 100,000 documents from Wiki727k dataset
- Cross-entropy loss
- Early stopping at epoch 14

**Evaluation metric:** pk score (probability of error at random boundary points)
- Paper achieved pk=35 on WIKI-50 test set
- Original Koshorek paper: pk=20 (with more training data/epochs)

**FinePhrase comparison:** FinePhrase uses BlingFire for sentence boundary detection (rule-based, faster) vs this paper's neural approach. The neural approach could detect topical segments (sections), not just sentences.

### Graph-Based Clustering Algorithm

**Step 1 - Graph Construction:**
```
G = (V, E) where V = {segments}
E = {(Si, Sj) | similarity(Si, Sj) > τ}
τ = μ + k·σ
```
- μ = mean similarity between all segment pairs
- σ = standard deviation of similarities
- k = sensitivity parameter (controls number of clusters)

**Step 2 - Maximal Clique Detection:**
Find all maximal cliques Q in graph G

**Step 3 - Initial Clustering:**
Merge adjacent segments that share at least one clique

**Step 4 - Cluster Merging:**
Merge adjacent clusters ci and ci+1 if any clique Q contains segments from both

**Step 5 - Final Merging:**
Orphan single-segment clusters → merge with nearest neighbor (cosine similarity)

**Step 6 - Cluster Embedding:**
Mean pooling over segment embeddings within each cluster

**Example (Table 1):**
```
Cliques Q:       {1,2,6}, {2,4,7}, {3,4,5}, {1,6,7}
Initial clusters: {1,2}, {3,4,5}, {6,7}
After merging:   {1,2,3,4,5}, {6,7}
```

**FinePhrase implementation consideration:** Could implement as optional clustering layer:
```python
def cluster_segments(segment_embeddings, k=0.7):
    # Compute pairwise similarities
    # Threshold at μ + k·σ
    # Find maximal cliques (networkx.find_cliques)
    # Merge adjacent segments/clusters
    # Return cluster assignments + cluster embeddings
```

### Quantitative Results Deep Dive

**Table 2 - QASPER and QuALITY:**

| Chunk size | Method | F1 (QASPER) | Accuracy (QuALITY) |
|------------|--------|-------------|---------------------|
| 256 | Base | 19.28 | 58.16 |
| 256 | Semantic | 18.07 | 57.23 |
| 512 | Base | 20.33 | 60.23 |
| 512 | Cluster Only | 21.64 | 62.36 |
| 512 | **Seg + Cluster** | **21.95** | **63.77** |
| 1024 | Base | 22.07 | 58.23 |
| 1024 | Cluster Only | 23.31 | 58.84 |
| 1024 | **Seg + Cluster** | **24.67** | 59.08 |
| 2048 | Base | 22.05 | 57.54 |
| 2048 | Cluster Only | 22.76 | 57.71 |
| 2048 | Seg + Cluster | 23.89 | 58.85 |

**Key observations:**
1. **512 best for QuALITY** (shorter context passages)
2. **1024 best for QASPER** (scientific papers, longer context)
3. **Seg + Cluster consistently beats Cluster Only** by 0.3-1.3 F1 points
4. **2048 shows diminishing returns**

**Table 3 - NarrativeQA (long narratives):**

| Chunk size | Method | ROUGE-L | BLEU-1 | BLEU-4 | METEOR |
|------------|--------|---------|--------|--------|--------|
| 256 | Base | 22.21 | 16.99 | 5.06 | 27.11 |
| 512 | Seg + Cluster | 24.67 | 18.97 | 6.83 | 28.64 |
| 1024 | Base | 23.86 | 18.05 | 6.59 | 27.12 |
| 1024 | Cluster Only | 25.15 | 19.28 | 6.97 | 29.05 |
| 1024 | **Seg + Cluster** | **26.54** | **20.03** | **7.58** | **30.26** |
| 2048 | Seg + Cluster | 26.39 | 19.62 | 7.38 | 30.07 |

**Key finding:** 1024-token Seg+Cluster achieves best across ALL metrics on NarrativeQA. Even 2048 with Seg+Cluster slightly underperforms 1024.

### Why Multi-Vector Helps (Figure 4 Analysis)

**Example: "The Olympic Gene Pool" question about athletic ability changes**

**512 Segment + Cluster retrieval:**
- Retrieved chunks about Ethiopian/Kenyan runners, environmental factors, healthcare
- These factors together provide context for "environment" as the answer
- **Correct answer: 3 (Environment)**

**Base 512 retrieval:**
- Retrieved chunks about genetic inheritance, saber-toothed tigers, natural selection
- Fragmented ideas about genetics without environmental context
- **Incorrect answer: 2 (Innate factors)**

**Insight:** Multi-vector retrieval captures broader themes by allowing matches on different segment aspects. Even segments that seem tangential individually provide relevant context together.

### Comparison to Other RAG Approaches

| Approach | Chunking | Retrieval | Trade-off |
|----------|----------|-----------|-----------|
| Fixed-size | Token count | Single vector | Simple but breaks semantic units |
| Recursive | Markers (newlines, spaces) | Single vector | Depends on formatting |
| Semantic | Cosine similarity | Single vector | Inconsistent boundaries |
| LongRAG | Entire documents | Single vector | Huge retrieval units |
| RAPTOR | Hierarchical summaries | Multi-level | Requires LLM summarization |
| GraphRAG | Entity extraction | Knowledge graph | Disrupts text flow |
| **This paper** | Neural segmentation + clustering | Multi-vector | Preserves structure + multiple match options |

**FinePhrase positioning:** Similar to this paper's approach but:
- Uses rule-based sentence detection (BlingFire) instead of neural
- Creates overlapping segments directly instead of clustering
- Late chunking provides document context without explicit clustering

### Implementation Checklist for FinePhrase

Based on Hierarchical Segmentation findings, consider:

- [ ] **Optional clustering post-processing**: Implement graph-based clustering over FinePhrase segments as optional feature
- [ ] **Multi-segment-size retrieval**: Document that `segment_sizes=[1,3,5]` provides similar multi-vector benefits
- [ ] **Target ~1024 token segments**: For users wanting larger chunks, recommend sentence group sizes that average ~1024 tokens
- [ ] **MaxSim retrieval example**: Extend existing MaxSim example to show segment + aggregate retrieval
- [ ] **Benchmark on NarrativeQA, QuALITY, QASPER**: These datasets test long-document retrieval where FinePhrase should excel
- [ ] **Avoid 2048+ token segments**: Document diminishing returns for very large segments
- [ ] **Cluster embedding option**: Consider adding `cluster_segments=True` parameter that groups semantically similar segments and returns cluster embeddings alongside segment embeddings
