# Plan: Align Late Chunking Implementation with Jina AI Paper

## Summary

Two changes to align with the late chunking paper's recommendations:

1. **Handle duplicate chunks from overlapping pre-chunks**: Average embeddings for the same sentence group instead of keeping duplicates
2. **Configurable special token handling**: Add `exclude_special_tokens: bool` parameter

---

## Change 1: Average Duplicate Chunk Embeddings

### Problem
When documents exceed `max_length`, overlapping pre-chunks create duplicate chunk embeddings for the same sentence groups (with different attention contexts). Currently all duplicates are kept.

### Solution
Post-process to average embeddings for identical sentence groups within each document.

### Files to Modify

**`afterthoughts/encode.py`**

1. Add new function `_deduplicate_chunk_embeds()` after `_build_results_df()` (~line 368):

```python
def _deduplicate_chunk_embeds(
    results: dict,
    method: Literal["average", "first"] = "average",
) -> dict:
    """
    Deduplicate chunk embeddings from overlapping pre-chunks.

    Groups by (document_idx, chunk_size, sentence_ids) and either
    averages embeddings or keeps first occurrence.
    """
```

2. Modify `encode()` method (~line 840) to call deduplication before `_build_results_df()`:
   - Add parameter `deduplicate: bool = True`
   - Call `_deduplicate_chunk_embeds()` when True

### Implementation Details

- Group key: `(document_idx, chunk_size, tuple(sorted(sentence_ids)))`
- For "average": compute mean of embeddings in each group
- Keep first occurrence's metadata (chunk_idx, sequence_idx, etc.)
- Update chunk_idx to be sequential after deduplication

---

## Change 2: Configurable Special Token Handling

### Problem
Current implementation excludes ALL special tokens from mean pooling. Paper recommends including [CLS] in first chunk and [SEP] in last chunk.

### Solution
Add `exclude_special_tokens: bool = True` parameter (default preserves current behavior).

### Files to Modify

**`afterthoughts/chunk.py`**

1. Modify `_compute_chunk_embeds()` (line 641) signature:
```python
def _compute_chunk_embeds(
    ...
    exclude_special_tokens: bool = True,  # NEW
) -> dict[str, torch.Tensor]:
```

2. Modify mask computation (lines 690-695):
```python
if exclude_special_tokens:
    # Current behavior: exclude all special tokens
    valid_token_mask = torch.isin(
        chunk_data["chunk_token_ids"],
        torch.tensor(tokenizer.all_special_ids, device=input_ids.device),
        invert=True,
    ).float()
else:
    # Paper's approach: include [CLS] in first chunk, [SEP] in last chunk
    valid_token_mask = _compute_boundary_special_token_mask(
        chunk_data, tokenizer, input_ids.device
    )
```

3. Add helper function `_compute_boundary_special_token_mask()` (~line 640):
```python
def _compute_boundary_special_token_mask(
    chunk_data: dict,
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """
    Create mask that includes [CLS] in first chunk and [SEP] in last chunk
    of each sequence, excludes other special tokens.
    """
```

4. Similarly update `_compute_chunk_embeds_slow()` (line 566) for consistency.

**`afterthoughts/encode.py`**

5. Modify `_generate_chunk_embeds()` (line 482) to accept and pass through `exclude_special_tokens`

6. Modify `Encoder.encode()` (line 686) and `LiteEncoder.encode()` to accept `exclude_special_tokens: bool = True`

7. Modify `encode_queries()` (line 558) similarly - for single-sequence queries, when `exclude_special_tokens=False`, include both [CLS] and [SEP]

---

## API Changes Summary

```python
def encode(
    self,
    docs: list[str],
    ...
    # Existing parameters unchanged
    ...
    # NEW parameters:
    exclude_special_tokens: bool = True,  # False = include [CLS] in first, [SEP] in last
    deduplicate: bool = True,             # Average duplicate chunks from overlapping pre-chunks
) -> tuple[DataFrame, np.ndarray]:
```

---

## Implementation Order

1. **Change 2 first** (lower risk, isolated changes):
   - Add `_compute_boundary_special_token_mask()`
   - Modify `_compute_chunk_embeds()` and `_compute_chunk_embeds_slow()`
   - Thread parameter through `_generate_chunk_embeds()` → `encode()`
   - Update `encode_queries()`
   - Add tests

2. **Change 1 second**:
   - Add `_deduplicate_chunk_embeds()`
   - Integrate into `encode()` flow
   - Add tests

---

## Testing

### New Test File: `tests/test_late_chunking_paper.py`

```python
# Change 2: Special token handling
def test_exclude_special_tokens_default():
    """Default behavior excludes all special tokens (backward compat)."""

def test_include_boundary_special_tokens():
    """exclude_special_tokens=False includes [CLS] in first, [SEP] in last."""

def test_single_chunk_includes_both_special_tokens():
    """Single-chunk doc with exclude_special_tokens=False includes both."""

# Change 1: Deduplication
def test_deduplicate_averages_overlapping_chunks():
    """Overlapping pre-chunks produce averaged embeddings."""

def test_no_duplicates_after_dedup():
    """Same (doc, chunk_size, sentences) appears only once."""

def test_short_docs_unaffected_by_dedup():
    """Documents not requiring pre-chunking are unchanged."""
```

### Verification Steps

1. Run existing tests to ensure backward compatibility:
   ```bash
   pytest tests/test_encode.py
   ```

2. Run new tests:
   ```bash
   pytest tests/test_late_chunking_paper.py
   ```

3. Manual verification with long document:
   ```python
   encoder = Encoder("sentence-transformers/all-MiniLM-L6-v2")
   long_doc = "..." * 1000  # Exceeds max_length

   # Compare before/after deduplication
   df_dup, emb_dup = encoder.encode([long_doc], deduplicate=False)
   df_dedup, emb_dedup = encoder.encode([long_doc], deduplicate=True)

   assert len(df_dedup) < len(df_dup)  # Fewer rows after dedup
   ```

---

## Files Modified

| File | Changes |
|------|---------|
| `afterthoughts/chunk.py` | Add `_compute_boundary_special_token_mask()`, modify `_compute_chunk_embeds()` and `_compute_chunk_embeds_slow()` |
| `afterthoughts/encode.py` | Add `_deduplicate_chunk_embeds()`, modify `encode()`, `_generate_chunk_embeds()`, `encode_queries()` |
| `tests/test_late_chunking_paper.py` | New test file |
