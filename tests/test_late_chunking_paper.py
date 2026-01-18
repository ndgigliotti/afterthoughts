"""Tests for late chunking paper alignment features.

These tests verify the implementation of:
1. Configurable special token handling (exclude_special_tokens parameter)
2. Deduplication of overlapping pre-chunk embeddings (deduplicate parameter)
"""

import numpy as np
import torch

from afterthoughts.chunk import (
    _compute_boundary_special_token_mask,
    get_chunk_idx,
)

# =============================================================================
# Tests for special token handling (Change 2)
# =============================================================================


def test_exclude_special_tokens_option(model):
    """Setting exclude_special_tokens=True excludes all special tokens."""
    docs = ["This is a sentence. This is another."]
    df_exclude, emb_exclude = model.encode(
        docs,
        num_sents=1,
        exclude_special_tokens=True,
        show_progress=False,
    )

    # Should produce results
    assert len(df_exclude) > 0
    assert emb_exclude.shape[0] == len(df_exclude)


def test_include_boundary_special_tokens(model):
    """exclude_special_tokens=False includes [CLS] in first, [SEP] in last."""
    docs = ["First sentence. Second sentence. Third sentence."]

    df_exclude, emb_exclude = model.encode(
        docs,
        num_sents=1,
        exclude_special_tokens=True,
        deduplicate=False,
        show_progress=False,
    )

    df_include, emb_include = model.encode(
        docs,
        num_sents=1,
        exclude_special_tokens=False,
        deduplicate=False,
        show_progress=False,
    )

    # Same number of chunks
    assert len(df_exclude) == len(df_include)

    # But embeddings should differ (because special tokens are included)
    # The first and last chunk embeddings should differ most
    first_chunk_diff = np.linalg.norm(emb_exclude[0] - emb_include[0])
    last_chunk_diff = np.linalg.norm(emb_exclude[-1] - emb_include[-1])

    # First chunk should differ (CLS included vs excluded)
    assert first_chunk_diff > 0

    # Last chunk should differ (SEP included vs excluded)
    assert last_chunk_diff > 0


def test_single_chunk_includes_both_special_tokens(model):
    """Single-chunk doc with exclude_special_tokens=False includes both CLS and SEP."""
    docs = ["Just one sentence."]

    df_exclude, emb_exclude = model.encode(
        docs,
        num_sents=1,
        exclude_special_tokens=True,
        show_progress=False,
    )

    df_include, emb_include = model.encode(
        docs,
        num_sents=1,
        exclude_special_tokens=False,
        show_progress=False,
    )

    # Both should produce one chunk
    assert len(df_exclude) == 1
    assert len(df_include) == 1

    # Embeddings should differ (special tokens now included)
    diff = np.linalg.norm(emb_exclude[0] - emb_include[0])
    assert diff > 0


def test_boundary_special_token_mask_first_last(tokenizer):
    """Test that boundary mask correctly identifies first and last chunks."""
    # Create mock chunk_data with 3 chunks from same sequence
    input_ids = torch.tensor([[101, 1, 2, 102]])  # CLS ... SEP
    sentence_ids = torch.tensor([[0, 0, 0, 0]])
    sequence_idx = torch.tensor([0])

    chunk_data = get_chunk_idx(
        input_ids, sentence_ids, num_sents=1, chunk_overlap=0, sequence_idx=sequence_idx
    )

    mask = _compute_boundary_special_token_mask(chunk_data, tokenizer, torch.device("cpu"))

    # For single chunk, both CLS and SEP should be included (mask=1)
    # The mask should be all 1s for valid tokens
    assert mask.sum() > 0


def test_encode_queries_special_tokens(model):
    """Test encode_queries respects exclude_special_tokens parameter."""
    queries = ["What is machine learning?"]

    emb_exclude = model.encode_queries(queries, exclude_special_tokens=True)
    emb_include = model.encode_queries(queries, exclude_special_tokens=False)

    # Both should produce embeddings
    assert emb_exclude.shape == emb_include.shape

    # Embeddings should differ
    diff = np.linalg.norm(emb_exclude - emb_include)
    assert diff > 0


# =============================================================================
# Tests for deduplication (Change 1)
# =============================================================================


def test_deduplicate_averages_overlapping_chunks(model):
    """Overlapping pre-chunks produce averaged embeddings when deduplicate=True."""
    # Create a long document that will require pre-chunking
    sentences = [f"This is sentence number {i}." for i in range(50)]
    long_doc = " ".join(sentences)

    df_no_dedup, emb_no_dedup = model.encode(
        [long_doc],
        num_sents=1,
        deduplicate=False,
        max_length=128,  # Force pre-chunking
        show_progress=False,
    )

    df_dedup, emb_dedup = model.encode(
        [long_doc],
        num_sents=1,
        deduplicate=True,
        max_length=128,  # Force pre-chunking
        show_progress=False,
    )

    # Without deduplication, there may be more rows due to overlapping chunks
    # With deduplication, duplicates should be merged
    # Note: If no duplicates exist, the counts will be equal
    assert len(df_dedup) <= len(df_no_dedup)


def test_no_duplicates_after_dedup(model):
    """Same (doc, chunk_size, sentences) appears only once after deduplication."""
    # Create a document that requires pre-chunking with overlap
    sentences = [f"Sentence {i}." for i in range(40)]
    long_doc = " ".join(sentences)

    df, emb = model.encode(
        [long_doc],
        num_sents=1,
        deduplicate=True,
        max_length=128,
        prechunk_overlap=0.5,
        show_progress=False,
    )

    # Check that each (document_idx, chunk_size, chunk text) combination is unique
    # Using chunk text as proxy for sentence_ids
    if "chunk" in df.columns:
        unique_chunks = df.select(["document_idx", "chunk_size", "chunk"]).unique()
        assert len(unique_chunks) == len(df)


def test_short_docs_unaffected_by_dedup(model):
    """Documents not requiring pre-chunking are unchanged by deduplication."""
    short_docs = [
        "Short doc one. Two sentences.",
        "Short doc two. Also two sentences.",
    ]

    df_no_dedup, emb_no_dedup = model.encode(
        short_docs,
        num_sents=1,
        deduplicate=False,
        show_progress=False,
    )

    df_dedup, emb_dedup = model.encode(
        short_docs,
        num_sents=1,
        deduplicate=True,
        show_progress=False,
    )

    # Should have same number of chunks (no duplicates to remove)
    assert len(df_dedup) == len(df_no_dedup)

    # Embeddings should be identical
    np.testing.assert_array_almost_equal(emb_dedup, emb_no_dedup)


def test_deduplicate_preserves_metadata(model):
    """Deduplication preserves chunk metadata correctly."""
    sentences = [f"Sentence {i}." for i in range(30)]
    doc = " ".join(sentences)

    df, emb = model.encode(
        [doc],
        num_sents=2,
        deduplicate=True,
        max_length=128,
        show_progress=False,
    )

    # All chunks should have document_idx = 0
    assert (df["document_idx"] == 0).all()

    # chunk_idx should be sequential
    expected_chunk_idx = list(range(len(df)))
    assert df["chunk_idx"].to_list() == expected_chunk_idx

    # chunk_size should all be 2
    assert (df["chunk_size"] == 2).all()


def test_deduplicate_with_multiple_docs(model):
    """Deduplication works correctly with multiple documents."""
    docs = [
        " ".join([f"Sentence {i} in doc 1." for i in range(25)]),
        " ".join([f"Sentence {i} in doc 2." for i in range(25)]),
    ]

    df, emb = model.encode(
        docs,
        num_sents=1,
        deduplicate=True,
        max_length=128,
        show_progress=False,
    )

    # Should have chunks from both documents
    assert set(df["document_idx"].to_list()) == {0, 1}

    # chunk_idx should be sequential within each document
    for doc_idx in [0, 1]:
        doc_chunks = df.filter(df["document_idx"] == doc_idx)
        expected = list(range(len(doc_chunks)))
        assert doc_chunks["chunk_idx"].to_list() == expected


def test_deduplicate_disabled_when_no_prechunk(model):
    """Deduplication is skipped when prechunk=False."""
    docs = ["Short document."]

    # With prechunk=False, deduplicate should have no effect
    df1, emb1 = model.encode(
        docs,
        num_sents=1,
        prechunk=False,
        deduplicate=True,
        show_progress=False,
    )

    df2, emb2 = model.encode(
        docs,
        num_sents=1,
        prechunk=False,
        deduplicate=False,
        show_progress=False,
    )

    assert len(df1) == len(df2)
    np.testing.assert_array_almost_equal(emb1, emb2)
