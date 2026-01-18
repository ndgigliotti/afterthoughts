# Literature Review

Research papers on late chunking, multi-vector retrieval, and embedding compression relevant to FinePhrase.

## Papers

| Paper | Topic | Key Finding |
|-------|-------|-------------|
| [Late Chunking](late_chunking.md) | Contextual chunk embeddings | 3.63% improvement with sentence boundaries vs fixed-size |
| [Contextual Retrieval](contextual_retrieval.md) | LLM-generated context | 35-67% retrieval improvement; late chunking achieves 99% of quality at fraction of cost |
| [SeDR](sedr.md) | Segment representations | Validates segment-level retrieval; warns against global attention collapse |
| [LongEmbed](longembed.md) | Long context extension | RoPE models extend better; BM25 still beats dense on long docs |
| [Hierarchical Segmentation](hierarchical_segmentation.md) | Clustering for RAG | Multi-vector retrieval with MaxSim; 1024-token segments optimal |
| [ColBERT](colbert.md) | Late interaction | MaxSim over token embeddings; 170x faster than BERT |
| [ColBERTv2](colbertv2.md) | Residual compression | 6-10x storage reduction with 0% quality loss |

## Summary: Sentence Boundary Findings

| Paper | Sentence Boundary Treatment | Finding |
|-------|---------------------------|---------|
| Late Chunking | Explicit ablation study | 3.63% improvement with sentence boundaries vs. fixed-size |
| Contextual Retrieval | 800-token chunks, no specific boundary guidance | LLM-generated context achieves 35-67% retrieval improvement; ~$1/M tokens cost; Late chunking achieves 99% of quality at fraction of cost |
| Hierarchical Segmentation | Neural boundary detection + clustering | Multi-vector retrieval (segment + cluster embeddings) with MaxSim; 1024-token segments optimal; Segment+Cluster beats Cluster Only beats Base |
| SeDR | Fixed 512-token segments | Validates segment-level retrieval; FinePhrase's sentence-level segments are finer-grained |
| LongEmbed | Not addressed directly | Focus on context length; PCW uses fixed-size chunks. FinePhrase's sentence-boundary chunking is an improvement over their approach |
| ColBERT | Token-level (no chunking) | MaxSim over token embeddings; 170x faster than BERT; Query augmentation via [MASK] tokens essential; FinePhrase is middle ground between single-vector and token-level |
| ColBERTv2 | Token-level with compression | Residual compression achieves 6-10x storage reduction with 0% quality loss; Embeddings cluster by word sense enabling efficient compression; LoTTE benchmark for long-tail topics |

---

## Recommendations for FinePhrase

### From Late Chunking Paper

1. **Late chunking is directly relevant** - FinePhrase already implements a similar approach by embedding full documents before extracting segment embeddings via mean pooling.

2. **Sentence boundaries matter** - The Late Chunking paper explicitly shows sentence-boundary chunking outperforms fixed-size and semantic boundaries (3.63% vs 3.46% improvement).

3. **BlingFire for boundary detection** - FinePhrase's use of BlingFire aligns with best practices. Consider comparing against neural boundary detectors if quality issues arise.

4. **Overlap may not help** - The Late Chunking paper found 16-token overlap showed "no clear advantage" (Table 6). FinePhrase's `segment_overlap` parameter may be less beneficial than expected when using late chunking—the full-context embeddings already capture cross-boundary information. Worth benchmarking overlap vs no-overlap on target tasks.

5. **Small chunks benefit most** - The paper notes late chunking "performs better than naive chunking, specifically for small chunk sizes." FinePhrase's fine-grained sentence-level segments should benefit substantially.

6. **Multi-scale segments** - FinePhrase's support for multiple `segment_sizes` (groups of consecutive sentences) is a differentiator not explored in the late chunking paper.

7. **Context relevance matters** - Late chunking underperforms when surrounding context is irrelevant (e.g., needle-in-haystack tasks). For documents with high information density throughout, late chunking excels.

### From LongEmbed Paper

8. **Prefer RoPE-based models** - For documents exceeding 512 tokens, recommend jina-embeddings-v2, nomic-embed-text, or E5-Mistral over BERT-based models (E5, GTE, BGE). RoPE models extend context more effectively without fine-tuning.

9. **FinePhrase improves on PCW** - LongEmbed's Parallel Context Windows uses fixed-size chunks and averages embeddings for single document representation. FinePhrase's approach is superior: sentence-boundary chunking + segment-level embeddings.

10. **LongEmbed as evaluation benchmark** - Use LongEmbed benchmark (NarrativeQA, QMSum, SummScreenFD, 2WikiMultihopQA) to evaluate FinePhrase on long documents. Compare against BM25 baseline (90.4 avg) to measure effectiveness.

11. **Dense retrieval gap remains** - Even the best extended models (75.3 avg) significantly underperform BM25 (90.4 avg) on long documents. FinePhrase's late chunking + sentence boundaries may help close this gap.

12. **Model context length matters** - For documents exceeding model context, FinePhrase's macro-chunk processing is necessary. The LongEmbed paper confirms that longer native context (4k, 8k, 32k) yields better results than extension methods on short-context models.

### From SeDR Paper

13. **Segment-level retrieval is validated** - SeDR proves that segment representations (rather than single document vectors) are effective for retrieval. FinePhrase's approach of returning segment-level embeddings is well-supported.

14. **Avoid global attention patterns** - SeDR shows that global attention causes segment embeddings to collapse into identical representations. FinePhrase's late chunking approach (full attention within chunks → mean pooling) naturally avoids this.

15. **Variable segment count is beneficial** - SeDR uses variable numbers of segments based on document length rather than fixed counts. FinePhrase does this naturally with sentence-level segments.

16. **MaxP is suboptimal** - Independent segment encoding (MaxP) loses document context. FinePhrase's late chunking provides better document-awareness than SeDR's MaxP baseline while being simpler than Segment-Interaction Transformer.

17. **FinePhrase is finer-grained than SeDR** - SeDR uses 512-token segments; FinePhrase uses sentence-level segments (~15-30 tokens). The Late Chunking paper shows smaller chunks benefit more from contextual embeddings, suggesting FinePhrase's granularity is optimal.

### From Hierarchical Segmentation Paper

18. **Multi-vector retrieval validated** - The paper confirms that using multiple vectors per chunk (segment embeddings + cluster embedding) with MaxSim improves retrieval. FinePhrase's overlapping multi-sentence segments serve a similar role, providing multiple "views" of each document region.

19. **Clustering as optional enhancement** - The graph-based clustering algorithm (threshold τ = μ + k·σ, maximal clique detection, sequential merging) could be implemented as optional post-processing to group related FinePhrase segments into clusters.

20. **Sweet spot around 1024 tokens** - Best results achieved with ~1024-token average chunk sizes. FinePhrase's `segment_sizes` parameter can target similar token counts by choosing appropriate sentence group sizes.

21. **Avoid very large chunks** - Paper shows diminishing returns at 2048 tokens—chunks become "too broad" and readers lose focus. Validates FinePhrase's fine-grained sentence-level approach.

22. **Segment + Cluster > Cluster Only** - Both granularities contribute to retrieval quality. FinePhrase could expose both fine-grained segments and coarser aggregations (via `segment_sizes=[1,3,5]` for example).

### From Contextual Retrieval (Anthropic)

23. **FinePhrase is the efficient alternative** - Late chunking achieves ~99% of Contextual Retrieval quality (0.8516 vs 0.8590 similarity) without LLM costs. Position FinePhrase as the cost-effective choice for users who need contextual chunk embeddings at scale.

24. **Hybrid search matters** - Contextual Retrieval gets biggest gains from BM25 fusion (+14% on top of contextual embeddings). FinePhrase users should consider combining with BM25/lexical search for production systems.

25. **Reranking as post-processing** - The 67% total improvement includes reranking (150 candidates → top 20). Document that FinePhrase embeddings work well with rerankers (Cohere, Voyage, cross-encoders).

26. **Small knowledge bases don't need RAG** - For <200k tokens (~500 pages), just include full content in LLM context. FinePhrase is for larger document collections where chunking is necessary.

27. **Generic summaries don't work** - Anthropic found "very limited gains" from prepending generic document summaries. FinePhrase's implicit context (via full-document attention) is more effective than naive summary approaches.

### From ColBERT Paper

28. **MaxSim is the right interaction** - ColBERT's ablation shows max-pooling over similarities outperforms average. FinePhrase's segment-level MaxSim retrieval follows this validated pattern.

29. **FinePhrase is the practical middle ground** - ColBERT stores ~100+ embeddings per document (every token); single-vector stores 1. FinePhrase's ~10-50 segment embeddings balance granularity vs storage.

30. **Dimension reduction works** - ColBERT shows 24-dim embeddings lose only 1% MRR vs 128-dim (27GB vs 286GB). FinePhrase's PCA option for dimension reduction is well-supported.

31. **Query augmentation insight** - ColBERT pads queries with [MASK] tokens for learned expansion. FinePhrase could explore similar query-side techniques if needed.

32. **End-to-end retrieval validated** - ColBERT's FAISS-based retrieval (issue Nq searches, map to docs, re-rank) achieves Recall@50 > BM25's Recall@1000. FinePhrase segments can use identical retrieval pattern.

### From ColBERTv2 Paper

33. **Residual compression opportunity** - ColBERTv2 shows token embeddings can be compressed 6-10x with centroid + quantized residual encoding. FinePhrase segment embeddings likely cluster similarly by semantic meaning, making this compression technique directly applicable.

34. **Semantic clustering validation** - ColBERTv2 demonstrates that BERT embeddings naturally cluster by word sense (~90% of clusters have ≤16 distinct tokens). FinePhrase segment embeddings should exhibit similar clustering by segment topic/meaning.

35. **Aggressive quantization works** - 1-2 bits per dimension for residuals achieves near-lossless compression (0% quality loss at 2-bit, only 0.7% loss at 1-bit). FinePhrase's current float16 + PCA could be augmented with residual quantization for further compression.

36. **Cross-encoder distillation for training** - If training custom embedding models for segment retrieval, ColBERTv2's supervision strategy (retrieve with model, score with cross-encoder, train on KL-divergence with hard negatives) is the proven approach.

37. **LoTTE for out-of-domain evaluation** - ColBERTv2 introduces LoTTE benchmark (StackExchange communities + GooAQ queries) for testing on long-tail topics. FinePhrase should be evaluated on LoTTE to measure generalization beyond MS MARCO/Wikipedia.

38. **Centroid selection at scale** - Use k-means on √N samples to select centroids efficiently. ColBERTv2 uses |C| ∝ √n_embeddings, which could apply to FinePhrase's segment embedding index.

39. **Inverted index for candidate generation** - ColBERTv2's retrieval groups embedding IDs by centroid, enabling fast candidate generation before full MaxSim scoring. FinePhrase could adopt similar structure for large-scale deployment.

40. **Denoised supervision improves multi-vector models** - ColBERTv2 proves that hard negative mining + distillation works for multi-vector representations, not just single-vector. Multi-vector architectures like FinePhrase's segment approach benefit from the same training improvements.

---

## Fetching Papers

Run `scripts/fetch_papers.sh` to download all referenced papers from arXiv.
