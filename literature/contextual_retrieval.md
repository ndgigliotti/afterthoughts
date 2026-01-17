# Contextual Retrieval (Anthropic)

**Authors:** Anthropic Research Team

**Publication:** [Anthropic Blog](https://www.anthropic.com/news/contextual-retrieval) (September 2024)

**Problem Statement:** Traditional RAG loses critical context when chunking. Example: *"The company's revenue grew by 3% over the previous quarter"* becomes ambiguous without knowing which company or time period.

## Core Approach

Use Claude to generate 50-100 token context for each chunk, prepended before embedding:

```
<document>{{WHOLE_DOCUMENT}}</document>
Here is the chunk we want to situate within the whole document
<chunk>{{CHUNK_CONTENT}}</chunk>
Please give a short succinct context to situate this chunk within the overall
document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
```

## Performance Results

Retrieval failure rate = 1 - recall@20:

| Configuration | Failure Rate | Reduction |
|--------------|--------------|-----------|
| Baseline (no context) | 5.7% | - |
| Contextual Embeddings | 3.7% | 35% |
| Contextual + BM25 | 2.9% | 49% |
| Contextual + BM25 + Reranking | 1.9% | **67%** |

## Implementation Details

- **Chunk size:** 800 tokens used in cost calculations
- **Retrieval:** Top-20 chunks returned
- **Reranking:** Fetch 150 candidates, rerank (Cohere/Voyage), return top 20
- **Embedding models:** Gemini and Voyage performed best
- **Hybrid search:** BM25 for exact matches + semantic embeddings, combined via rank fusion

**Cost:** $1.02 per million document tokens (one-time contextualization with prompt caching)

**When NOT to use:** For knowledge bases <200,000 tokens (~500 pages), include entire content in prompt with caching instead.

## Alternative Methods Tested (underperformed)

- Generic document summaries: "very limited gains"
- Hypothetical document embedding: less effective
- Summary-based indexing: low performance

## Trade-off vs Late Chunking

| Aspect | Contextual Retrieval | Late Chunking (FinePhrase) |
|--------|---------------------|---------------------------|
| Context method | LLM-generated explicit | Implicit via full-doc attention |
| Cost | $1.02/M tokens + LLM calls | Embedding model only |
| Model requirement | Any embedding model | Long-context embedding model |
| Quality | Slightly higher (~0.8590) | Very close (~0.8516)* |
| Latency | Higher (LLM generation) | Lower (no LLM) |

*From Late Chunking paper's direct comparison (Table in Section 4.5)

---

## Detailed Insights

### The Core Problem

Traditional chunking loses context. The canonical example:

> *"The company's revenue grew by 3% over the previous quarter"*

This chunk is ambiguous: Which company? Which quarter? Without context, retrieval may fail even when the chunk contains the answer.

**Two solutions:**
1. **Contextual Retrieval**: LLM generates explicit context, prepended to chunk
2. **Late Chunking (FinePhrase)**: Implicit context via full-document transformer attention

### The Contextual Retrieval Pipeline

**Step 1 - Contextualization (one-time, at indexing):**
```
For each chunk in document:
    context = Claude(prompt=CONTEXT_PROMPT, document=full_doc, chunk=chunk)
    contextualized_chunk = context + " " + chunk
```

**Step 2 - Embedding:**
```
embedding = embed(contextualized_chunk)  # Any embedding model works
```

**Step 3 - Retrieval (at query time):**
```
candidates = semantic_search(query, top_k=150)
candidates += bm25_search(query, top_k=150)
candidates = deduplicate(candidates)
reranked = reranker(query, candidates, top_k=20)
return reranked
```

### Cost Analysis

**Assumptions (from blog):**
- 800-token chunks
- 8k-token documents (10 chunks per doc)
- 50-token instruction
- 100-token generated context per chunk

**Calculation:**
- With prompt caching: read cached doc once per chunk
- Claude 3 Haiku pricing with caching
- Result: **$1.02 per million document tokens**

**FinePhrase comparison:**
- No LLM calls needed
- Cost = embedding model inference only
- For most embedding models: <$0.10 per million tokens
- **~10x cheaper** than Contextual Retrieval

### Performance Breakdown

**Baseline:** 5.7% retrieval failure rate (1 - recall@20)

| Enhancement | Failure Rate | Δ from Baseline | Δ Incremental |
|-------------|--------------|-----------------|---------------|
| + Contextual Embeddings | 3.7% | -35% | -35% |
| + BM25 Hybrid | 2.9% | -49% | -14% |
| + Reranking | 1.9% | -67% | -18% |

**Key insight:** Each component contributes meaningfully:
- Contextual embeddings: largest single improvement
- BM25 hybrid: catches exact matches embeddings miss
- Reranking: final precision boost

### Direct Comparison: Late Chunking vs Contextual Retrieval

From Late Chunking paper (Section 4.5), financial document example:

| Method | Similarity to Correct Chunk | LLM Required |
|--------|----------------------------|--------------|
| Naive Chunking | 0.6343 | No |
| Late Chunking | 0.8516 | No |
| Contextual Retrieval | 0.8590 | Yes |

**Late chunking achieves 99.1% of Contextual Retrieval quality** without any LLM inference.

**When to use each:**

| Use Case | Recommendation |
|----------|----------------|
| Cost-sensitive, high volume | **FinePhrase** (late chunking) |
| Maximum quality, cost not critical | Contextual Retrieval |
| Any embedding model (no long-context) | Contextual Retrieval |
| Long-context embedding model available | **FinePhrase** |
| Real-time indexing needed | **FinePhrase** (no LLM latency) |
| One-time index, query-heavy workload | Either (amortize LLM cost) |

### Hybrid Search Implementation

**Why BM25 helps:**
- Embeddings can miss exact keyword matches
- BM25 excels at precise phrase matching
- Combination catches both semantic similarity and lexical overlap

**Rank fusion approach (reciprocal rank fusion):**
```python
def reciprocal_rank_fusion(results_lists, k=60):
    scores = defaultdict(float)
    for results in results_lists:
        for rank, doc in enumerate(results):
            scores[doc] += 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
```

**FinePhrase integration:** FinePhrase returns segment text alongside embeddings, enabling easy BM25 integration:
```python
df, embeddings = fp.encode(docs)
# df contains segment text for BM25 indexing
# embeddings for semantic search
# Combine results with rank fusion
```

### Reranking Strategy

**Two-stage retrieval:**
1. **Stage 1 (recall):** Fetch 150 candidates via semantic + BM25
2. **Stage 2 (precision):** Rerank to top 20 via cross-encoder

**Reranker options:**
- Cohere Rerank API
- Voyage Rerank API
- Open-source: ms-marco-MiniLM-L-6-v2, bge-reranker-base

**Why 150 → 20:**
- Larger candidate set increases recall
- Reranker is expensive but only runs on candidates
- Final 20 chunks fit comfortably in LLM context

### What Didn't Work

**Generic document summaries:**
- Prepending document summary to each chunk
- "Very limited gains" - summary too generic to help specific chunks

**Hypothetical document embedding:**
- Generate hypothetical document that would answer query
- Less effective than direct chunk contextualization

**Summary-based indexing:**
- Index summaries instead of full chunks
- "Low performance" - loses detail needed for retrieval

**FinePhrase advantage:** Late chunking avoids these pitfalls by preserving full chunk content while adding implicit context through transformer attention.

### When RAG Isn't Needed

**Threshold: ~200,000 tokens (~500 pages)**

Below this threshold:
- Include full knowledge base in LLM context
- Use prompt caching for efficiency
- No chunking/retrieval complexity needed

**FinePhrase target use case:** Document collections significantly exceeding 200k tokens where chunking is necessary.

### Implementation Checklist for FinePhrase

Based on Contextual Retrieval analysis, consider:

- [ ] **Position as efficient alternative**: Document the 99% quality at 10x lower cost vs Contextual Retrieval
- [ ] **BM25 hybrid example**: Provide code showing FinePhrase + BM25 rank fusion
- [ ] **Reranking integration**: Document compatibility with Cohere/Voyage rerankers
- [ ] **Benchmark comparison**: Run FinePhrase on same datasets as Anthropic (codebases, fiction, ArXiv, science) to get direct comparison numbers
- [ ] **Cost comparison table**: Add to docs showing FinePhrase vs Contextual Retrieval costs at various scales
- [ ] **Decision tree**: Help users decide between FinePhrase (late chunking) vs Contextual Retrieval based on their constraints
