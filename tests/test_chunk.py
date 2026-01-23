import pytest
import torch

from afterthoughts.chunk import (
    _add_special_tokens,
    _compute_boundary_special_token_mask,
    _split_long_sentences,
    get_chunk_idx,
    get_chunk_idx_by_tokens,
    get_sentence_offsets,
    get_sentence_offsets_blingfire,
    get_sentence_offsets_nltk,
    get_sentence_offsets_pysbd,
    get_sentence_offsets_syntok,
)


def test_get_sentence_offsets_syntok_returns_tensor():
    pytest.importorskip("syntok")
    text = "Hello world. This is a test."
    offsets = get_sentence_offsets_syntok(text)
    # Check that the result is a tensor with 2 columns
    assert isinstance(offsets, torch.Tensor)
    assert offsets.ndim == 2
    assert offsets.size(1) == 2


def test_get_sentence_offsets_syntok_non_empty():
    pytest.importorskip("syntok")
    text = "Hello world. This is a test."
    offsets = get_sentence_offsets_syntok(text)
    # Ensure that at least one sentence is detected
    assert offsets.size(0) >= 1
    # Check that each offset pair has a start less than end
    for pair in offsets:
        start, end = pair.tolist()
        assert start < end


def test_get_sentence_offsets_syntok_empty_text():
    pytest.importorskip("syntok")
    text = ""
    offsets = get_sentence_offsets_syntok(text)
    # For empty text, expect no sentence offsets
    assert isinstance(offsets, torch.Tensor)
    assert offsets.size() == (0, 2)


def test_get_sentence_offsets_blingfire_returns_tensor():
    text = "Hello world. This is a test."
    offsets = get_sentence_offsets_blingfire(text)
    # Check that the result is a tensor with 2 columns
    assert isinstance(offsets, torch.Tensor)
    assert offsets.ndim == 2
    assert offsets.size(1) == 2


def test_get_sentence_offsets_blingfire_non_empty():
    text = "Hello world. This is a test."
    offsets = get_sentence_offsets_blingfire(text)
    # Ensure that at least one sentence is detected
    assert offsets.size(0) >= 1
    # Check that each offset pair has a start less than end
    for pair in offsets:
        start, end = pair.tolist()
        assert start < end


def test_get_sentence_offsets_blingfire_empty_text():
    text = ""
    offsets = get_sentence_offsets_blingfire(text)
    # For empty text, expect no sentence offsets
    assert isinstance(offsets, torch.Tensor)
    assert offsets.size() == (0, 2)


def test_get_sentence_offsets_nltk_returns_tensor():
    pytest.importorskip("nltk")
    text = "Hello world. This is a test."
    offsets = get_sentence_offsets_nltk(text)
    # Check that the result is a tensor with 2 columns
    assert isinstance(offsets, torch.Tensor)
    assert offsets.ndim == 2
    assert offsets.size(1) == 2


def test_get_sentence_offsets_nltk_non_empty():
    pytest.importorskip("nltk")
    text = "Hello world. This is a test."
    offsets = get_sentence_offsets_nltk(text)
    # Ensure that at least one sentence is detected
    assert offsets.size(0) >= 1
    # Check that each offset pair has a start less than end
    for pair in offsets:
        start, end = pair.tolist()
        assert start < end


def test_get_sentence_offsets_nltk_empty_text():
    pytest.importorskip("nltk")
    text = ""
    offsets = get_sentence_offsets_nltk(text)
    # For empty text, expect no sentence offsets
    assert isinstance(offsets, torch.Tensor)
    assert offsets.size() == (0, 2)


def test_get_sentence_offsets_pysbd_returns_tensor():
    pytest.importorskip("pysbd")
    text = "Hello world. This is a test."
    offsets = get_sentence_offsets_pysbd(text)
    # Check that the result is a tensor with 2 columns
    assert isinstance(offsets, torch.Tensor)
    assert offsets.ndim == 2
    assert offsets.size(1) == 2


def test_get_sentence_offsets_pysbd_non_empty():
    pytest.importorskip("pysbd")
    text = "Hello world. This is a test."
    offsets = get_sentence_offsets_pysbd(text)
    # Ensure that at least one sentence is detected
    assert offsets.size(0) >= 1
    # Check that each offset pair has a start less than end
    for pair in offsets:
        start, end = pair.tolist()
        assert start < end


def test_get_sentence_offsets_pysbd_empty_text():
    pytest.importorskip("pysbd")
    text = ""
    offsets = get_sentence_offsets_pysbd(text)
    # For empty text, expect no sentence offsets
    assert isinstance(offsets, torch.Tensor)
    assert offsets.size() == (0, 2)


def test_get_sentence_offsets_pysbd_abbreviations():
    pytest.importorskip("pysbd")
    # pysbd should handle abbreviations correctly
    text = "Dr. Smith went to Washington. He met with U.S. officials."
    offsets = get_sentence_offsets_pysbd(text)
    # Should detect exactly 2 sentences (not split on Dr. or U.S.)
    assert offsets.size(0) == 2


def test_get_sentence_offsets_blingfire_method():
    text = "Hello world. This is a test."
    offsets = get_sentence_offsets(text, method="blingfire")
    # Check that the result is a tensor with 2 columns
    assert isinstance(offsets, torch.Tensor)
    assert offsets.ndim == 2
    assert offsets.size(1) == 2


def test_get_sentence_offsets_nltk_method():
    pytest.importorskip("nltk")
    text = "Hello world. This is a test."
    offsets = get_sentence_offsets(text, method="nltk")
    # Check that the result is a tensor with 2 columns
    assert isinstance(offsets, torch.Tensor)
    assert offsets.ndim == 2
    assert offsets.size(1) == 2


def test_get_sentence_offsets_syntok_method():
    pytest.importorskip("syntok")
    text = "Hello world. This is a test."
    offsets = get_sentence_offsets(text, method="syntok")
    # Check that the result is a tensor with 2 columns
    assert isinstance(offsets, torch.Tensor)
    assert offsets.ndim == 2
    assert offsets.size(1) == 2


def test_get_sentence_offsets_pysbd_method():
    pytest.importorskip("pysbd")
    text = "Hello world. This is a test."
    offsets = get_sentence_offsets(text, method="pysbd")
    # Check that the result is a tensor with 2 columns
    assert isinstance(offsets, torch.Tensor)
    assert offsets.ndim == 2
    assert offsets.size(1) == 2


def test_get_sentence_offsets_invalid_method():
    text = "Hello world. This is a test."
    with pytest.raises(ValueError, match="Invalid method: 'invalid'"):
        get_sentence_offsets(text, method="invalid")


def test_get_sentence_offsets_parallel():
    texts = ["Hello world. This is a test.", "Another sentence. And another one."]
    offsets = get_sentence_offsets(texts, method="blingfire", n_jobs=2)
    # Check that the result is a list of tensors
    assert isinstance(offsets, list)
    assert all(isinstance(o, torch.Tensor) for o in offsets)
    assert all(o.ndim == 2 and o.size(1) == 2 for o in offsets)


def test_add_special_tokens_no_special_tokens():
    input_ids = torch.tensor([1, 2, 3, 4])
    result = _add_special_tokens(input_ids)
    expected = torch.tensor([1, 2, 3, 4])
    assert torch.equal(result, expected)


def test_add_special_tokens_with_cls_token():
    input_ids = torch.tensor([1, 2, 3, 4])
    cls_token_id = 101
    result = _add_special_tokens(input_ids, cls_token_id=cls_token_id)
    expected = torch.tensor([101, 1, 2, 3, 4])
    assert torch.equal(result, expected)


def test_add_special_tokens_with_sep_token():
    input_ids = torch.tensor([1, 2, 3, 4])
    sep_token_id = 102
    result = _add_special_tokens(input_ids, sep_token_id=sep_token_id)
    expected = torch.tensor([1, 2, 3, 4, 102])
    assert torch.equal(result, expected)


def test_add_special_tokens_with_cls_and_sep_tokens():
    input_ids = torch.tensor([1, 2, 3, 4])
    cls_token_id = 101
    sep_token_id = 102
    result = _add_special_tokens(input_ids, cls_token_id=cls_token_id, sep_token_id=sep_token_id)
    expected = torch.tensor([101, 1, 2, 3, 4, 102])
    assert torch.equal(result, expected)


def test_split_long_sentences_no_split_needed():
    sentence_ids = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32)
    max_length = 3
    result = _split_long_sentences(sentence_ids, max_length)
    expected = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32)
    assert torch.equal(result, expected)


def test_split_long_sentences_split_needed():
    sentence_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32)
    max_length = 2
    result = _split_long_sentences(sentence_ids, max_length)
    expected = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int32)
    assert torch.equal(result, expected)


def test_split_long_sentences_multiple_splits():
    sentence_ids = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.int32)
    max_length = 2
    result = _split_long_sentences(sentence_ids, max_length)
    expected = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=torch.int32)
    assert torch.equal(result, expected)


def test_split_long_sentences_with_padding():
    sentence_ids = torch.tensor([0, 0, 0, 1, 1, 1, -1, -1], dtype=torch.int32)
    max_length = 2
    result = _split_long_sentences(sentence_ids, max_length)
    expected = torch.tensor([0, 0, 1, 2, 2, 3, -1, -1], dtype=torch.int32)
    assert torch.equal(result, expected)


def test_split_long_sentences_empty_input():
    sentence_ids = torch.tensor([], dtype=torch.int32)
    max_length = 2
    result = _split_long_sentences(sentence_ids, max_length)
    expected = torch.tensor([], dtype=torch.int32)
    assert torch.equal(result, expected)


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


# =============================================================================
# Tests for get_chunk_idx_by_tokens
# =============================================================================


def test_get_chunk_idx_by_tokens_basic():
    """Test basic token-based chunking."""
    # 3 sentences: sent0 has 3 tokens, sent1 has 3 tokens, sent2 has 3 tokens
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]])  # 0 is padding
    sentence_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, -1]])

    # Max 6 tokens per chunk should group sent0+sent1, then sent2
    result = get_chunk_idx_by_tokens(input_ids, sentence_ids, max_chunk_tokens=6, chunk_overlap=0)

    assert "chunk_token_idx" in result
    assert "chunk_token_ids" in result
    assert "sentence_ids" in result
    assert "num_sents" in result
    assert "sequence_idx" in result
    assert "chunk_idx" in result

    # Should have 2 chunks
    assert result["num_sents"].size(0) == 2
    # First chunk has 2 sentences, second has 1
    assert result["num_sents"][0].item() == 2
    assert result["num_sents"][1].item() == 1


def test_get_chunk_idx_by_tokens_single_chunk():
    """Test when entire document fits in one chunk."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    sentence_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])

    result = get_chunk_idx_by_tokens(input_ids, sentence_ids, max_chunk_tokens=100, chunk_overlap=0)

    # All sentences should fit in one chunk
    assert result["num_sents"].size(0) == 1
    assert result["num_sents"][0].item() == 2


def test_get_chunk_idx_by_tokens_with_overlap():
    """Test token-based chunking with sentence overlap."""
    # 4 sentences: 2 tokens each
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    sentence_ids = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]])

    # Max 4 tokens per chunk with 50% overlap
    result = get_chunk_idx_by_tokens(input_ids, sentence_ids, max_chunk_tokens=4, chunk_overlap=0.5)

    # Should have overlapping chunks
    assert result["num_sents"].size(0) >= 2
    # With overlap, later chunks should share sentences with earlier ones


def test_get_chunk_idx_by_tokens_with_num_sents_limit():
    """Test combined num_sents and max_chunk_tokens limits."""
    # 4 sentences: 2 tokens each
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    sentence_ids = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]])

    # Max 100 tokens (won't be hit) but max 2 sentences
    result = get_chunk_idx_by_tokens(
        input_ids, sentence_ids, max_chunk_tokens=100, num_sents=2, chunk_overlap=0
    )

    # Should have 2 chunks of 2 sentences each (limited by num_sents)
    assert result["num_sents"].size(0) == 2
    assert all(result["num_sents"] == 2)


def test_get_chunk_idx_by_tokens_num_sents_hit_first():
    """Test that num_sents limit is respected even when token limit allows more."""
    # 3 sentences: 2 tokens each
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    sentence_ids = torch.tensor([[0, 0, 1, 1, 2, 2]])

    # Max 100 tokens but only 1 sentence per chunk
    result = get_chunk_idx_by_tokens(
        input_ids, sentence_ids, max_chunk_tokens=100, num_sents=1, chunk_overlap=0
    )

    # Should have 3 chunks of 1 sentence each
    assert result["num_sents"].size(0) == 3
    assert all(result["num_sents"] == 1)


def test_get_chunk_idx_by_tokens_token_limit_hit_first():
    """Test that token limit is respected even when num_sents allows more."""
    # 3 sentences: 3 tokens each
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    sentence_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2]])

    # Max 5 tokens but up to 10 sentences per chunk
    result = get_chunk_idx_by_tokens(
        input_ids, sentence_ids, max_chunk_tokens=5, num_sents=10, chunk_overlap=0
    )

    # Token limit should be hit first - can only fit 1 sentence (3 tokens) per chunk
    assert result["num_sents"].size(0) == 3
    assert all(result["num_sents"] == 1)


def test_get_chunk_idx_by_tokens_empty_document():
    """Test handling of empty document."""
    input_ids = torch.tensor([[0]])  # Just padding
    sentence_ids = torch.tensor([[-1]])  # No valid sentences

    result = get_chunk_idx_by_tokens(input_ids, sentence_ids, max_chunk_tokens=100, chunk_overlap=0)

    # Should return empty results
    assert result["num_sents"].size(0) == 0


def test_get_chunk_idx_by_tokens_large_sentence_warning():
    """Test that warning is issued when single sentence exceeds max_chunk_tokens."""
    # One sentence with 10 tokens, limit is 5
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    sentence_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    with pytest.warns(UserWarning, match="exceeding max_chunk_tokens"):
        result = get_chunk_idx_by_tokens(
            input_ids, sentence_ids, max_chunk_tokens=5, chunk_overlap=0
        )

    # The large sentence should still be included as its own chunk
    assert result["num_sents"].size(0) == 1


def test_get_chunk_idx_by_tokens_multiple_sequences():
    """Test token-based chunking across multiple sequences."""
    # 2 sequences
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 0, 0],  # 6 tokens
            [7, 8, 9, 10, 0, 0, 0, 0],  # 4 tokens
        ]
    )
    sentence_ids = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, -1, -1],  # 2 sentences
            [0, 0, 1, 1, -1, -1, -1, -1],  # 2 sentences
        ]
    )

    result = get_chunk_idx_by_tokens(input_ids, sentence_ids, max_chunk_tokens=4, chunk_overlap=0)

    # Should have chunks from both sequences
    assert result["num_sents"].size(0) >= 2
    # sequence_idx should track which sequence each chunk came from
    assert 0 in result["sequence_idx"].tolist()
    assert 1 in result["sequence_idx"].tolist()


def test_get_chunk_idx_by_tokens_chunk_idx_per_document():
    """Test that chunk_idx resets for each document."""
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
        ]
    )
    sentence_ids = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ]
    )

    result = get_chunk_idx_by_tokens(input_ids, sentence_ids, max_chunk_tokens=4, chunk_overlap=0)

    # chunk_idx should start at 0 for each sequence
    seq0_chunks = result["chunk_idx"][result["sequence_idx"] == 0]
    seq1_chunks = result["chunk_idx"][result["sequence_idx"] == 1]

    # Both should start with 0
    assert seq0_chunks[0].item() == 0
    assert seq1_chunks[0].item() == 0
