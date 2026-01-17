# LongEmbed: Extending Embedding Models for Long Context Retrieval

**Authors:** Dawei Zhu, Liang Wang, Nan Yang, Yifan Song, Wenhao Wu, Furu Wei, Sujian Li (Peking University, Microsoft)

**arXiv:** [2404.12096](https://arxiv.org/abs/2404.12096) | **Venue:** EMNLP 2024 | **Version:** v3 (November 7, 2024)

**Abstract:** Explores extending embedding model context windows from 512 to 32,768 tokens. Introduces the LongEmbed benchmark with synthetic and real-world tasks featuring dispersed target information. Demonstrates that training-free strategies like position interpolation can effectively extend context windows by several folds.

## Key Findings

- RoPE-based models superior to APE for context extension
- NTK-Aware Interpolation and SelfExtend work best for RoPE models
- Training-free methods can extend context by 8x (512→4k or 4k→32k)
- Further fine-tuning yields additional +5 point gains for APE models
- Released E5-Base-4k and E5-RoPE-Base models
- LongEmbed benchmark now integrated into MTEB

## Context Extension Methods Explored

1. **Parallel Context Windows (PCW)**: Divide long document into chunks, process separately, average embeddings
2. **Grouped/Recurrent Positions (GP/RP)**: Reuse position IDs to accommodate longer inputs
3. **Linear Position Interpolation (PI)**: Interpolate new position embeddings from existing ones
4. **NTK-Aware Interpolation**: Scales RoPE frequencies non-uniformly to preserve high-frequency features
5. **SelfExtend**: Re-introduces normal relative positions within a neighbor window for RoPE models

## Best Results by Model Type

- **APE models (E5, GTE)**: Further tuning on PI yields best results (+15.6 points)
- **RoPE models (E5-RoPE, E5-Mistral)**: NTK and SelfExtend without tuning (+10.9 to +20.3 points)

## Relevance to FinePhrase

Long-context models are prerequisites for effective late chunking. FinePhrase's macro-chunk processing aligns with PCW approach. This paper provides guidance on model selection and potential context extension strategies.

## Resources

- [PDF](https://arxiv.org/pdf/2404.12096)
- [GitHub](https://github.com/dwzhu-pku/LongEmbed)
- [ACL Anthology](https://aclanthology.org/2024.emnlp-main.47/)

---

## Detailed Insights

### The LongEmbed Benchmark

**Design Principles:**
1. Documents must be long enough (thousands to tens of thousands of words)
2. Target information should be dispersed throughout the document (not biased to beginning)

**Why Existing Benchmarks Fall Short:**
- **BEIR**: Documents average <300 words—too short for long-context evaluation
- **LoCo**: Biased target distribution—E5-Base achieves >85% on 3/8 tasks with only 512 context

**Benchmark Tasks:**

| Dataset | Domain | # Queries | # Docs | Avg Query Words | Avg Doc Words |
|---------|--------|-----------|--------|-----------------|---------------|
| NarrativeQA | Literature, Film | 10,449 | 355 | 9 | **50,474** |
| QMSum | Meeting | 1,527 | 197 | 71 | 10,058 |
| 2WikiMultihopQA | Wikipedia | 300 | 300 | 12 | 6,132 |
| SummScreenFD | ScreenWriting | 336 | 336 | 102 | 5,582 |
| Passkey | Synthetic | 400 | 800 | 11 | Controllable |
| Needle | Synthetic | 400 | 800 | 7 | Controllable |

**FinePhrase evaluation guidance:** Use LongEmbed for evaluating long-document retrieval. NarrativeQA (50k words) tests extreme long-context; QMSum/SummScreenFD test meeting/screenplay retrieval with dispersed information.

### Position Encoding: APE vs RoPE

**Absolute Position Embedding (APE):**
- Used by BERT-based models (E5, GTE, BGE)
- Embeds absolute position IDs into vectors added to token embeddings
- Limited to pre-trained context length (typically 512)

**Rotary Position Embedding (RoPE):**
- Used by LLaMA-based models (E5-Mistral, Jina-V2, Nomic-V1)
- Encodes relative positions via rotation matrices in attention
- Naturally supports length extrapolation

**Key Finding:** RoPE-based models consistently outperform APE models for context extension, even without training. At 4k context, E5-RoPE (no tuning) surpasses E5 (tuned) by a large margin.

**FinePhrase model selection:** Prefer RoPE-based embedding models (jina-embeddings-v2, nomic-embed, E5-Mistral) when processing documents exceeding model context length.

### Context Extension Strategies Deep Dive

**For APE-based Models (E5, GTE, BGE):**

| Method | Approach | Performance |
|--------|----------|-------------|
| PCW | Split → process → average | Baseline |
| GP | Group position IDs: `pid → ⌊pid/s⌋` | Similar to PCW |
| RP | Recurrent positions: `pid → pid mod L_o` | Similar to PCW |
| PI | Interpolate positions: `pid → pid/s` | Similar to PCW |
| **Tuning on PI** | Fine-tune only new position embeddings | **Best (+5 pts)** |

**Key insight:** For APE models, plug-and-play methods yield comparable results. Further tuning is required for significant improvement.

**For RoPE-based Models (E5-Mistral, Jina-V2):**

| Method | Approach | E5-RoPE (512→4k) | E5-Mistral (4k→32k) |
|--------|----------|------------------|---------------------|
| PCW | Split → average | 52.9 | 68.7 |
| GP | Group positions | 52.5 | 64.7 |
| PI | Interpolate | 51.9 | 59.4 |
| **SE** | SelfExtend | **60.8** | 72.4 |
| **NTK** | NTK-Aware | 56.1 | **75.3** |

**Key insight:** NTK and SelfExtend dramatically outperform other methods for RoPE models without any training.

### Quantitative Results (Table 2)

**Existing Models on LongEmbed:**

| Model | Params | Context | Passkey | Needle | NQA | QMS | SFD | WQA | **Avg** |
|-------|--------|---------|---------|--------|-----|-----|-----|-----|---------|
| E5-Base | 110M | 512 | 38.0 | 28.5 | 25.3 | 23.8 | 74.7 | 55.8 | 41.0 |
| Contriever | 110M | 512 | 38.5 | 29.0 | 26.7 | 25.5 | 73.5 | 47.3 | 40.1 |
| Jina-V2 | 137M | 8,192 | 50.3 | 54.5 | 37.9 | 38.9 | 93.5 | 74.0 | 58.2 |
| Nomic-V1 | 137M | 8,192 | 60.7 | 39.5 | 41.2 | 36.7 | 93.0 | 73.8 | 57.5 |
| BGE-M3 | 568M | 8,192 | 59.3 | 40.5 | 45.8 | 35.5 | 94.0 | 78.0 | 58.9 |
| E5-Mistral | 7B | 4,096 | 71.0 | 48.3 | 44.6 | 43.6 | 96.8 | 82.0 | 64.4 |

**Extended Models:**

| Model | Extension | Passkey | Needle | NQA | QMS | SFD | WQA | **Avg** | Δ |
|-------|-----------|---------|--------|-----|-----|-----|-----|---------|---|
| E5-Base + Tuning | 512→4k | 67.3 | 41.5 | 30.4 | 35.7 | 95.2 | 69.2 | 56.6 | **+15.6** |
| E5-RoPE + SE | 512→4k | 73.5 | 53.5 | 32.3 | 39.1 | 91.9 | 74.6 | 60.8 | **+20.3** |
| E5-Mistral + NTK | 4k→32k | 93.8 | 66.8 | 49.8 | 49.2 | 97.1 | 95.2 | 75.3 | **+10.9** |

**FinePhrase insight:** Even the best extended model (75.3 avg) has huge room for improvement compared to BM25 (90.4 avg on same benchmark). Late chunking + sentence boundaries may help close this gap.

### NTK-Aware Interpolation Details

**Problem with Linear PI:** Uniformly scaling position indices prevents learning high-frequency features (per Neural Tangent Kernel theory).

**NTK Solution:** Scale high frequencies less, low frequencies more:
- Original: `θ_j = 10000^(-2j/d)`
- NTK: `θ'_j = (10000 × λ)^(-2j/d)` where `λ` is slightly greater than scaling factor

**Recommended λ values:**
| Extension | λ (base multiplier) |
|-----------|---------------------|
| 2x (512→1k or 4k→8k) | 3 (10,000 → 30,000) |
| 4x (512→2k or 4k→16k) | 5 (10,000 → 50,000) |
| 8x (512→4k or 4k→32k) | 10 (10,000 → 100,000) |

**FinePhrase action:** If using RoPE-based models, consider exposing NTK scaling parameter for users processing very long documents.

### SelfExtend Details

**Problem with Grouped Positions:** Loses fine-grained relative position information within grouped tokens.

**SelfExtend Solution:** Maintain normal relative positions within a neighbor window `w`, use grouped positions only for distant tokens.

**Hyperparameters:**
| Extension | Group size (g) | Window size (w) |
|-----------|----------------|-----------------|
| 2x | 3 | 256 (small model) / 2,048 (large) |
| 4x | 5 | 128 / 1,024 |
| 8x | 9 | 64 / 512 |

### Parallel Context Windows (PCW) - Most Relevant to FinePhrase

**Algorithm:**
1. Segment long document D into chunks of `L_o` tokens (original context length)
2. Process each chunk independently through the model
3. Average chunk embeddings to represent the full document

**Implementation notes:**
- Set overlap between adjacent chunks to 0 for simplicity
- Exception: Last chunk can overlap to ensure it contains `L_o` tokens

**Comparison to FinePhrase's macro-chunking:**
- FinePhrase chunks at **sentence boundaries** (better than fixed-size)
- FinePhrase uses **overlap** (paper suggests this may not help with late chunking)
- FinePhrase computes **segment-level** embeddings, not document-level average

**Key difference:** PCW averages chunk embeddings for single document representation; FinePhrase extracts fine-grained segment embeddings. FinePhrase's approach is more aligned with late chunking for retrieval.

### Further Tuning Strategy (APE Models)

**Key insight:** For APE models, freeze original weights and only train new position embeddings. This:
1. Strictly preserves original behavior within 512 context
2. Gains long-context capability as "free lunch"

**Training approach (PoSE - Positional Skip-wise):**
1. Given input of `L_o` tokens, introduce skipping bias `u` at beginning
2. Transform position IDs from `{0, 1, ..., L_o-1}` to `{u, u+1, ..., u+L_o-1}`
3. Sample `u` uniformly from `{0, 1, ..., L_t - L_o}` for each training sample

**Tuning on PI vs RP:**
- PI with interleaved frozen/learnable vectors yields better results
- Frozen vectors act as anchors preventing convergence to suboptimal values

### BM25 vs Dense Retrieval on LongEmbed

| Method | Passkey | Needle | NQA | QMS | SFD | WQA | Avg |
|--------|---------|--------|-----|-----|-----|-----|-----|
| BM25 | 100 | 95.3 | 71.5 | 81.3 | 97.6 | 96.5 | **90.4** |
| E5-Mistral | 71.0 | 48.3 | 44.6 | 43.6 | 96.8 | 82.0 | 64.4 |
| E5-Mistral + NTK | 93.8 | 66.8 | 49.8 | 49.2 | 97.1 | 95.2 | 75.3 |

**Key insight:** BM25 still significantly outperforms dense retrieval on long-document tasks. The gap is largest on needle/passkey (synthetic) and QA tasks. Dense models approach BM25 only on summarization tasks (SFD, QMS with dispersed but related content).

**FinePhrase opportunity:** Late chunking with sentence boundaries may help dense retrieval close the gap with BM25 on these challenging long-document tasks.

### Implementation Checklist for FinePhrase

Based on LongEmbed findings, consider:

- [ ] **Prefer RoPE-based models** in documentation for long-document use cases
- [ ] **LongEmbed benchmark** can be used for evaluating FinePhrase on long documents
- [ ] **PCW approach comparison**: FinePhrase's macro-chunking differs by using sentence boundaries and segment-level (not document-average) embeddings—document this advantage
- [ ] **NTK/SelfExtend support**: Consider exposing context extension parameters for RoPE models
- [ ] **Benchmark against BM25**: Use LongEmbed to measure how close FinePhrase gets to BM25 on long documents
- [ ] **Document model recommendations**: Suggest jina-v2, nomic-v1, or E5-Mistral for documents exceeding 512 tokens
