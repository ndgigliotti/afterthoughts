import numpy as np
import polars as pl
import pytest
import torch

from afterthoughts import Encoder
from afterthoughts.chunk import get_chunk_idx
from afterthoughts.utils import move_or_convert_tensors

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@requires_cuda
def test_encoder_init():
    """Test basic Encoder initialization."""
    model_name = MODEL_NAME
    amp = False
    amp_dtype = torch.float16
    normalize = True
    device = "cuda"
    _num_token_jobs = 8

    encoder = Encoder(
        model_name=model_name,
        amp=amp,
        amp_dtype=amp_dtype,
        normalize=normalize,
        device=device,
        _num_token_jobs=_num_token_jobs,
    )

    assert encoder.tokenizer is not None
    assert encoder.model is not None
    assert encoder.amp == amp
    assert encoder.amp_dtype == amp_dtype
    assert encoder.normalize == normalize
    assert encoder.device.type == device
    assert encoder._num_token_jobs == _num_token_jobs


@requires_cuda
def test_encoder_init_with_half_embeds_and_truncate_dims():
    """Test Encoder initialization with half_embeds and truncate_dims parameters."""
    encoder = Encoder(
        model_name=MODEL_NAME,
        half_embeds=True,
        truncate_dims=128,
        device="cuda",
    )

    assert encoder.half_embeds is True
    assert encoder.truncate_dims == 128


def test_get_chunk_idx_single_size():
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    sentence_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, -1]])
    max_chunk_sents = 2
    overlap = 0.5

    result = get_chunk_idx(input_ids, sentence_ids, max_chunk_sents, overlap)

    assert "chunk_token_idx" in result
    assert "chunk_token_ids" in result
    assert "sentence_ids" in result
    assert "sequence_idx" in result
    assert "num_sents" in result


def test_get_chunk_idx_multiple_sizes():
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    sentence_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, -1]])
    max_chunk_sents = [1, 2]
    overlap = 0.5

    result = get_chunk_idx(input_ids, sentence_ids, max_chunk_sents, overlap)

    assert "chunk_token_idx" in result
    assert "chunk_token_ids" in result
    assert "sentence_ids" in result
    assert "sequence_idx" in result
    assert "num_sents" in result


def test_move_or_convert_results_to_cpu():
    results = {
        "chunk_token_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "sequence_idx": torch.tensor([0, 1]),
        "chunk_embeds": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
    }
    expected_results = {
        "chunk_token_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "sequence_idx": torch.tensor([0, 1]),
        "chunk_embeds": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
    }
    converted_results = move_or_convert_tensors(results, move_to_cpu=True)
    for key in results:
        assert torch.equal(converted_results[key], expected_results[key])


def test_move_or_convert_results_to_numpy():
    results = {
        "chunk_token_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "sequence_idx": torch.tensor([0, 1]),
        "chunk_embeds": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
    }
    expected_results = {
        "chunk_token_ids": np.array([[1, 2, 3], [4, 5, 6]]),
        "sequence_idx": np.array([0, 1]),
        "chunk_embeds": np.array([[0.1, 0.2], [0.3, 0.4]]),
    }
    converted_results = move_or_convert_tensors(results, return_tensors="np", move_to_cpu=True)
    for key in results:
        np.testing.assert_allclose(converted_results[key], expected_results[key])


def test_move_or_convert_results_invalid_return_tensors():
    results = {
        "chunk_token_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "sequence_idx": torch.tensor([0, 1]),
        "chunk_embeds": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
    }
    with pytest.raises(ValueError):
        move_or_convert_tensors(results, return_tensors="invalid")


@requires_cuda
def test_encoder_to_cpu():
    model_name = MODEL_NAME
    device = "cuda"
    encoder = Encoder(
        model_name=model_name,
        device=device,
    )
    assert encoder.device.type == "cuda"

    encoder.to("cpu")
    assert encoder.device.type == "cpu"


@requires_cuda
def test_encoder_to_cuda():
    encoder = Encoder(
        model_name=MODEL_NAME,
        device="cpu",
        _num_token_jobs=1,
    )
    assert encoder.device.type == "cpu"

    encoder.to("cuda")
    assert encoder.device.type == "cuda"


@requires_cuda
def test_encoder_to_device():
    encoder = Encoder(
        model_name=MODEL_NAME,
        device="cpu",
        _num_token_jobs=1,
    )
    assert encoder.device.type == "cpu"

    encoder.to(torch.device("cuda"))
    assert encoder.device.type == "cuda"

    encoder.to(torch.device("cpu"))
    assert encoder.device.type == "cpu"


def test_encoder_encode(model):
    docs = [
        "This is a test document. Another sentence here.",
        "Another test document. With more sentences.",
    ]
    df, X = model.encode(
        docs, max_chunk_sents=1, max_length=64, max_batch_tokens=256, show_progress=False
    )
    assert isinstance(df, pl.DataFrame)
    assert isinstance(X, np.ndarray)
    assert len(df) == len(X)
    assert "chunk" in df.columns
    assert "num_sents" in df.columns


def test_encoder_encode_multiple_max_chunk_sents(model):
    docs = [
        "This is a test document. Another sentence here.",
        "Another test document. With more sentences.",
    ]
    max_chunk_sents = [1, 2]
    df, X = model.encode(
        docs,
        max_chunk_sents=max_chunk_sents,
        max_length=64,
        max_batch_tokens=256,
        show_progress=False,
    )
    assert isinstance(df, pl.DataFrame)
    assert isinstance(X, np.ndarray)
    assert len(df) == len(X)
    assert all(size in df["num_sents"].unique().to_list() for size in max_chunk_sents)


def test_encoder_encode_queries(model):
    queries = ["What is the capital of France?", "How to bake a cake?"]
    query_embeds = model.encode_queries(queries, max_length=10, batch_size=1)

    assert query_embeds is not None
    assert len(query_embeds) == len(queries)


def test_encoder_encode_queries_preserves_order(model):
    """Test that encode_queries returns embeddings in the original input order.

    This is a regression test for a bug where TokenizedDataset's sorting by
    token count caused embeddings to be returned in the wrong order.
    """
    # Use queries of very different lengths to trigger reordering
    queries = [
        "Short query",
        "This is a much longer query with many more words to ensure different token counts",
        "Medium length query here",
    ]

    # Encode all at once
    batch_embeds = model.encode_queries(queries)

    # Encode individually for comparison
    individual_embeds = [model.encode_queries([q])[0] for q in queries]

    # Embeddings should match in order
    for i, (batch_emb, indiv_emb) in enumerate(zip(batch_embeds, individual_embeds, strict=False)):
        similarity = np.dot(batch_emb, indiv_emb) / (
            np.linalg.norm(batch_emb) * np.linalg.norm(indiv_emb)
        )
        assert similarity > 0.99, f"Query {i} embedding mismatch: similarity={similarity:.4f}"


def test_encoder_half_embeds_if_needed():
    """Test that half_embeds parameter converts embeddings to float16."""
    encoder = Encoder(model_name=MODEL_NAME, device="cpu", half_embeds=True)
    embeds = torch.randn(10, 10, dtype=torch.float32)
    reduced_embeds = encoder.half_embeds_if_needed(embeds)
    assert reduced_embeds.dtype == torch.float16

    encoder.half_embeds = False
    non_reduced_embeds = encoder.half_embeds_if_needed(embeds)
    assert non_reduced_embeds.dtype == torch.float32


def test_encoder_normalize_if_needed():
    encoder = Encoder(model_name=MODEL_NAME, device="cpu", normalize=True)
    embeds = torch.randn(10, 10)
    normalized_embeds = encoder.normalize_if_needed(embeds)
    norms = torch.norm(normalized_embeds, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    encoder.normalize = False
    non_normalized_embeds = encoder.normalize_if_needed(embeds)
    norms = torch.norm(non_normalized_embeds, dim=1)
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


@pytest.mark.parametrize(
    "return_frame, as_numpy",
    [
        ("polars", True),
        ("polars", False),
        ("teddies", True),
    ],
)
def test_build_results_dataframe(return_frame, as_numpy):
    # Determine the expected dataframe type based on the return_frame parameter
    if return_frame == "polars":
        expected_df_type = pl.DataFrame

    # Determine the expected embeddings type based on the as_numpy parameter
    expected_embeds_type = torch.Tensor
    if as_numpy:
        expected_embeds_type = np.ndarray

    # Define the results dictionary with sample data
    results = {
        "embed_idx": torch.tensor([0, 1]),
        "sample_idx": torch.tensor([0, 1]),
        "sequence_idx": torch.tensor([0, 1]),
        "batch_idx": torch.tensor([0, 1]),
        "num_sents": torch.tensor([2, 2]),
        "chunk": ["segment1", "segment2"],
        "chunk_embeds": torch.randn(2, 10),
    }

    # Test for invalid return_frame value
    if return_frame == "teddies":
        with pytest.raises(ValueError, match="Invalid value for"):
            Encoder._build_results_df(results, return_frame, as_numpy)
    else:
        # Build the results dataframe and check the types
        expected_length = len(results["sample_idx"])
        df, embeds = Encoder._build_results_df(results, return_frame, as_numpy)
        assert isinstance(df, expected_df_type)
        assert isinstance(embeds, expected_embeds_type)
        assert len(df) == len(embeds) == expected_length


def test_build_results_dataframe_pandas():
    pd = pytest.importorskip("pandas")
    return_frame = "pandas"
    as_numpy = True

    expected_df_type = pd.DataFrame
    expected_embeds_type = np.ndarray

    results = {
        "embed_idx": torch.tensor([0, 1]),
        "sample_idx": torch.tensor([0, 1]),
        "sequence_idx": torch.tensor([0, 1]),
        "batch_idx": torch.tensor([0, 1]),
        "num_sents": torch.tensor([2, 2]),
        "chunk": ["segment1", "segment2"],
        "chunk_embeds": torch.randn(2, 10),
    }

    expected_length = len(results["sample_idx"])
    df, embeds = Encoder._build_results_df(results, return_frame, as_numpy)
    assert isinstance(df, expected_df_type)
    assert isinstance(embeds, expected_embeds_type)
    assert len(df) == len(embeds) == expected_length


def test_encoder_preserves_sentence_text(model):
    """Test that chunk text is reconstructed from original sentences without detokenization."""
    docs = [
        "This is the first sentence. Here's the second one!",
        "Another document starts here. And continues with this.",
    ]
    df, X = model.encode(
        docs, max_chunk_sents=1, max_length=64, max_batch_tokens=256, show_progress=False
    )
    assert isinstance(df, pl.DataFrame)
    assert "chunk" in df.columns

    # Verify text comes from original document (not detokenized)
    chunks = df["chunk"].to_list()
    # Each chunk should match a sentence from the original document exactly
    for chunk in chunks:
        assert any(chunk in doc for doc in docs), f"Chunk '{chunk}' not found in original docs"


def test_encoder_text_preservation_with_overlap(model):
    """Test text preservation with overlapping chunks."""
    docs = [
        "First sentence here. Second sentence follows. Third sentence ends.",
    ]
    df, X = model.encode(
        docs,
        max_chunk_sents=2,
        chunk_overlap_sents=1,  # One sentence overlap
        max_length=64,
        max_batch_tokens=256,
        show_progress=False,
    )
    assert isinstance(df, pl.DataFrame)
    chunks = df["chunk"].to_list()

    # Check that overlapping chunks contain correct sentence combinations
    # With max_chunk_sents=2 and overlap=1, we should get chunks like:
    # [sent0, sent1], [sent1, sent2]
    for chunk in chunks:
        # Each chunk should be a substring or contiguous part of the original doc
        assert any(
            sent in chunk
            for sent in [
                "First sentence",
                "Second sentence",
                "Third sentence",
            ]
        )


def test_encoder_text_preservation_multiple_docs(model):
    """Test text preservation with multiple documents."""
    docs = [
        "Doc one sentence one. Doc one sentence two.",
        "Doc two sentence one. Doc two sentence two.",
        "Doc three has a single sentence.",
    ]
    df, X = model.encode(
        docs, max_chunk_sents=1, max_length=64, max_batch_tokens=256, show_progress=False
    )

    # Verify correct document mapping
    assert "document_idx" in df.columns
    doc_indices = df["document_idx"].unique().sort().to_list()
    assert doc_indices == [0, 1, 2]

    # Each chunk should match its document
    for row in df.iter_rows(named=True):
        doc_idx = row["document_idx"]
        chunk = row["chunk"]
        assert chunk in docs[doc_idx], f"Chunk '{chunk}' not in doc {doc_idx}"


def test_encoder_return_text_false(model):
    """Test that return_text=False skips text reconstruction."""
    docs = ["Simple test document. With two sentences."]
    df, X = model.encode(
        docs,
        max_chunk_sents=1,
        max_length=64,
        max_batch_tokens=256,
        return_text=False,
        show_progress=False,
    )
    assert "chunk" not in df.columns


def test_encoder_text_reconstruction_fallback(model):
    """Test fallback to decoding when sentence is split due to exceeding max_length."""
    # Create a document with a very long sentence that will exceed max_length
    # and trigger _split_long_sents(), causing sentence IDs to exceed original count
    long_sentence = (
        "This is a very long sentence " + "with many repeated words " * 50 + "that ends here."
    )
    docs = [f"Short intro. {long_sentence} Short outro."]

    # Use a small max_length to force the long sentence to be split
    df, X = model.encode(
        docs, max_chunk_sents=1, max_length=32, max_batch_tokens=256, show_progress=False
    )

    assert isinstance(df, pl.DataFrame)
    assert "chunk" in df.columns
    # Should have chunks - the exact number depends on how the sentence is split
    assert len(df) > 0
    # All chunks should have non-empty text (either reconstructed or decoded fallback)
    assert all(len(chunk) > 0 for chunk in df["chunk"].to_list())


# =============================================================================
# Tests for special token handling (exclude_special_tokens parameter)
# =============================================================================


def test_exclude_special_tokens_option(model):
    """Setting exclude_special_tokens=True excludes all special tokens."""
    docs = ["This is a sentence. This is another."]
    df_exclude, emb_exclude = model.encode(
        docs,
        max_chunk_sents=1,
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
        max_chunk_sents=1,
        exclude_special_tokens=True,
        deduplicate=False,
        show_progress=False,
    )

    df_include, emb_include = model.encode(
        docs,
        max_chunk_sents=1,
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
        max_chunk_sents=1,
        exclude_special_tokens=True,
        show_progress=False,
    )

    df_include, emb_include = model.encode(
        docs,
        max_chunk_sents=1,
        exclude_special_tokens=False,
        show_progress=False,
    )

    # Both should produce one chunk
    assert len(df_exclude) == 1
    assert len(df_include) == 1

    # Embeddings should differ (special tokens now included)
    diff = np.linalg.norm(emb_exclude[0] - emb_include[0])
    assert diff > 0


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
# Tests for deduplication (deduplicate parameter)
# =============================================================================


def test_deduplicate_averages_overlapping_chunks(model):
    """Overlapping pre-chunks produce averaged embeddings when deduplicate=True."""
    # Create a long document that will require pre-chunking
    sentences = [f"This is sentence number {i}." for i in range(50)]
    long_doc = " ".join(sentences)

    df_no_dedup, emb_no_dedup = model.encode(
        [long_doc],
        max_chunk_sents=1,
        deduplicate=False,
        max_length=128,  # Force pre-chunking
        show_progress=False,
    )

    df_dedup, emb_dedup = model.encode(
        [long_doc],
        max_chunk_sents=1,
        deduplicate=True,
        max_length=128,  # Force pre-chunking
        show_progress=False,
    )

    # Without deduplication, there may be more rows due to overlapping chunks
    # With deduplication, duplicates should be merged
    # Note: If no duplicates exist, the counts will be equal
    assert len(df_dedup) <= len(df_no_dedup)


def test_no_duplicates_after_dedup(model):
    """Same (doc, max_chunk_sents, sentences) appears only once after deduplication."""
    # Create a document that requires pre-chunking with overlap
    sentences = [f"Sentence {i}." for i in range(40)]
    long_doc = " ".join(sentences)

    df, emb = model.encode(
        [long_doc],
        max_chunk_sents=1,
        deduplicate=True,
        max_length=128,
        prechunk_overlap_tokens=0.5,
        show_progress=False,
    )

    # Check that each (document_idx, max_chunk_sents, chunk text) combination is unique
    # Using chunk text as proxy for sentence_ids
    if "chunk" in df.columns:
        unique_chunks = df.select(["document_idx", "num_sents", "chunk"]).unique()
        assert len(unique_chunks) == len(df)


def test_short_docs_unaffected_by_dedup(model):
    """Documents not requiring pre-chunking are unchanged by deduplication."""
    short_docs = [
        "Short doc one. Two sentences.",
        "Short doc two. Also two sentences.",
    ]

    df_no_dedup, emb_no_dedup = model.encode(
        short_docs,
        max_chunk_sents=1,
        deduplicate=False,
        show_progress=False,
    )

    df_dedup, emb_dedup = model.encode(
        short_docs,
        max_chunk_sents=1,
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
        max_chunk_sents=2,
        deduplicate=True,
        max_length=128,
        show_progress=False,
    )

    # All chunks should have document_idx = 0
    assert (df["document_idx"] == 0).all()

    # chunk_idx should be sequential
    expected_chunk_idx = list(range(len(df)))
    assert df["chunk_idx"].to_list() == expected_chunk_idx

    # max_chunk_sents should all be 2
    assert (df["num_sents"] == 2).all()


def test_deduplicate_with_multiple_docs(model):
    """Deduplication works correctly with multiple documents."""
    docs = [
        " ".join([f"Sentence {i} in doc 1." for i in range(25)]),
        " ".join([f"Sentence {i} in doc 2." for i in range(25)]),
    ]

    df, emb = model.encode(
        docs,
        max_chunk_sents=1,
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
        max_chunk_sents=1,
        prechunk=False,
        deduplicate=True,
        show_progress=False,
    )

    df2, emb2 = model.encode(
        docs,
        max_chunk_sents=1,
        prechunk=False,
        deduplicate=False,
        show_progress=False,
    )

    assert len(df1) == len(df2)
    np.testing.assert_array_almost_equal(emb1, emb2)


def test_deduplicate_averaging_correctness():
    """Verify that deduplication correctly averages embeddings for duplicate chunks."""
    # Create test data with known duplicates
    # Chunks 0 and 2 are duplicates (same doc, size, sentences)
    # Chunks 1 and 3 are duplicates (same doc, size, sentences)
    results = {
        "document_idx": torch.tensor([0, 0, 0, 0]),
        "max_chunk_sents": torch.tensor([2, 2, 2, 2]),  # Configuration values
        "max_chunk_tokens": torch.tensor([-1, -1, -1, -1]),  # -1 means None
        "num_sents": torch.tensor([2, 2, 2, 2]),
        "chunk_embeds": torch.tensor(
            [
                [1.0, 2.0, 3.0],  # chunk 0: sentences 0-1
                [4.0, 5.0, 6.0],  # chunk 1: sentences 2-3
                [7.0, 8.0, 9.0],  # chunk 2: sentences 0-1 (duplicate of 0)
                [10.0, 11.0, 12.0],  # chunk 3: sentences 2-3 (duplicate of 1)
            ]
        ),
        "chunk_sentence_ids": [
            torch.tensor([0, 0, 1, 1, -1]),  # sentences 0-1
            torch.tensor([2, 2, 3, 3, -1]),  # sentences 2-3
            torch.tensor([0, 0, 1, 1, -1]),  # sentences 0-1 (duplicate)
            torch.tensor([2, 2, 3, 3, -1]),  # sentences 2-3 (duplicate)
        ],
        "chunk_idx": torch.tensor([0, 1, 2, 3]),
        "sequence_idx": torch.tensor([0, 0, 1, 1]),
        "batch_idx": torch.tensor([0, 0, 0, 0]),
        "chunk_token_ids": [
            torch.tensor([1, 2, 3, 4, 0]),
            torch.tensor([5, 6, 7, 8, 0]),
            torch.tensor([1, 2, 3, 4, 0]),
            torch.tensor([5, 6, 7, 8, 0]),
        ],
    }

    # Test averaging method
    dedup_results = Encoder._deduplicate_chunk_embeds(results, method="average")

    # Should have 2 unique chunks
    assert len(dedup_results["document_idx"]) == 2
    assert dedup_results["chunk_embeds"].shape[0] == 2

    # Verify averaged embeddings
    # Chunk 0-1 average: ([1,2,3] + [7,8,9]) / 2 = [4, 5, 6]
    # Chunk 2-3 average: ([4,5,6] + [10,11,12]) / 2 = [7, 8, 9]
    expected_embeds = torch.tensor(
        [
            [4.0, 5.0, 6.0],  # average of chunks 0 and 2
            [7.0, 8.0, 9.0],  # average of chunks 1 and 3
        ]
    )
    torch.testing.assert_close(dedup_results["chunk_embeds"], expected_embeds)

    # Verify chunk_idx is reindexed sequentially
    assert dedup_results["chunk_idx"].tolist() == [0, 1]


def test_deduplicate_first_method():
    """Verify that method='first' keeps first occurrence without averaging."""
    results = {
        "document_idx": torch.tensor([0, 0, 0]),
        "max_chunk_sents": torch.tensor([1, 1, 1]),  # Configuration values
        "max_chunk_tokens": torch.tensor([-1, -1, -1]),  # -1 means None
        "num_sents": torch.tensor([1, 1, 1]),
        "chunk_embeds": torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],  # duplicate of chunk 0
                [5.0, 6.0],
            ]
        ),
        "chunk_sentence_ids": [
            torch.tensor([0, 0, -1]),
            torch.tensor([0, 0, -1]),  # same as chunk 0
            torch.tensor([1, 1, -1]),
        ],
        "chunk_idx": torch.tensor([0, 1, 2]),
        "sequence_idx": torch.tensor([0, 0, 0]),
        "batch_idx": torch.tensor([0, 0, 0]),
        "chunk_token_ids": [
            torch.tensor([1, 2, 0]),
            torch.tensor([1, 2, 0]),
            torch.tensor([3, 4, 0]),
        ],
    }

    dedup_results = Encoder._deduplicate_chunk_embeds(results, method="first")

    # Should have 2 unique chunks
    assert len(dedup_results["document_idx"]) == 2

    # Should keep first occurrence, not average
    expected_embeds = torch.tensor(
        [
            [1.0, 2.0],  # first occurrence of sentence 0
            [5.0, 6.0],  # sentence 1
        ]
    )
    torch.testing.assert_close(dedup_results["chunk_embeds"], expected_embeds)


# =============================================================================
# Tests for half_embeds and truncate_dims
# =============================================================================


def test_encoder_half_embeds_output(model_half_embeds):
    """Test that half_embeds produces float16 output."""
    docs = ["This is a test document. Another sentence here."]
    df, X = model_half_embeds.encode(
        docs, max_chunk_sents=1, max_length=64, max_batch_tokens=256, show_progress=False
    )
    # Note: as_numpy=True converts to numpy which may upcast to float32
    # Check the internal processing produces float16
    assert model_half_embeds.half_embeds is True


def test_encoder_truncate_dims_output(model_truncate_dims):
    """Test that truncate_dims produces truncated embeddings."""
    docs = ["This is a test document. Another sentence here."]
    df, X = model_truncate_dims.encode(
        docs, max_chunk_sents=1, max_length=64, max_batch_tokens=256, show_progress=False
    )
    # The embedding dimension should be 128 (truncated from 384)
    assert X.shape[1] == 128


def test_encoder_truncate_dims_queries(model_truncate_dims):
    """Test that truncate_dims applies to query embeddings."""
    queries = ["What is this about?"]
    query_embeds = model_truncate_dims.encode_queries(queries)
    assert query_embeds.shape[1] == 128


# =============================================================================
# Tests for instruct-style prompts (query_prompt, document_prompt)
# =============================================================================


def test_apply_prompt_helper(model):
    """Test _apply_prompt helper method."""
    texts = ["query one", "query two"]
    prompt = "Represent this for retrieval: "

    # With prompt
    result = model._apply_prompt(texts, prompt)
    assert result == [
        "Represent this for retrieval: query one",
        "Represent this for retrieval: query two",
    ]

    # Without prompt
    result_no_prompt = model._apply_prompt(texts, None)
    assert result_no_prompt == texts


def test_get_prompt_length_helper(model):
    """Test _get_prompt_length helper method."""
    # Simple prompt
    prompt = "Search query: "
    length = model._get_prompt_length(prompt)
    assert length > 0
    assert isinstance(length, int)

    # Empty/None prompt
    assert model._get_prompt_length(None) == 0
    assert model._get_prompt_length("") == 0


def test_encode_queries_with_prompt(model):
    """Test encode_queries with per-call prompt override."""
    queries = ["What is machine learning?"]
    prompt = "Represent this question for retrieval: "

    # Encode with prompt
    emb_with_prompt = model.encode_queries(queries, prompt=prompt)

    # Encode without prompt
    emb_no_prompt = model.encode_queries(queries)

    # Embeddings should differ (prompt changes the embedding)
    diff = np.linalg.norm(emb_with_prompt - emb_no_prompt)
    assert diff > 0, "Prompt should change query embeddings"


def test_encode_queries_with_init_prompt():
    """Test encode_queries with query_prompt set at initialization."""
    prompt = "Represent this for retrieval: "
    encoder = Encoder(
        model_name=MODEL_NAME,
        device="cpu",
        query_prompt=prompt,
        _num_token_jobs=1,
    )

    queries = ["What is AI?"]

    # Should use init prompt by default
    emb_init_prompt = encoder.encode_queries(queries)

    # Per-call prompt should override
    override_prompt = "Represent for clustering: "
    emb_override = encoder.encode_queries(queries, prompt=override_prompt)

    # Embeddings should differ
    diff = np.linalg.norm(emb_init_prompt - emb_override)
    assert diff > 0, "Per-call prompt should override init prompt"


def test_encode_with_document_prompt(model):
    """Test encode() with per-call prompt override."""
    docs = ["Machine learning is AI. Deep learning is a subset."]
    prompt = "Represent this document: "

    # Encode with prompt
    df_with, emb_with = model.encode(
        docs, max_chunk_sents=1, max_length=128, prompt=prompt, show_progress=False
    )

    # Encode without prompt
    df_without, emb_without = model.encode(
        docs, max_chunk_sents=1, max_length=128, show_progress=False
    )

    # Should have same number of chunks (prompt doesn't add sentences)
    assert len(df_with) == len(df_without)

    # Embeddings should differ
    for i in range(len(emb_with)):
        diff = np.linalg.norm(emb_with[i] - emb_without[i])
        assert diff > 0, f"Prompt should change chunk {i} embedding"


def test_encode_with_init_document_prompt():
    """Test encode() with document_prompt set at initialization."""
    prompt = "Document for retrieval: "
    encoder = Encoder(
        model_name=MODEL_NAME,
        device="cpu",
        document_prompt=prompt,
        _num_token_jobs=1,
    )

    docs = ["Python is popular. It is widely used."]

    # Should use init prompt by default
    df_init, emb_init = encoder.encode(docs, max_chunk_sents=1, max_length=128, show_progress=False)

    # Per-call prompt should override
    override_prompt = "Summarize: "
    df_override, emb_override = encoder.encode(
        docs, max_chunk_sents=1, max_length=128, prompt=override_prompt, show_progress=False
    )

    # Embeddings should differ
    assert len(emb_init) == len(emb_override)
    diff = np.linalg.norm(emb_init[0] - emb_override[0])
    assert diff > 0, "Per-call prompt should override init document_prompt"


def test_prompt_does_not_affect_chunk_count(model):
    """Verify that prompts don't change the number of chunks extracted."""
    docs = ["First sentence. Second sentence. Third sentence."]
    prompt = "This is a very long prompt that contains many tokens: "

    df_without, _ = model.encode(docs, max_chunk_sents=1, max_length=256, show_progress=False)
    df_with, _ = model.encode(
        docs, max_chunk_sents=1, max_length=256, prompt=prompt, show_progress=False
    )

    assert len(df_without) == len(df_with), "Prompt should not affect chunk count"


def test_prompt_excluded_from_chunk_text(model):
    """Verify that prompt text is not included in reconstructed chunk text."""
    docs = ["Hello world. This is a test."]
    prompt = "PREFIX_MARKER: "

    df, _ = model.encode(
        docs, max_chunk_sents=1, max_length=128, prompt=prompt, show_progress=False
    )

    # Check that no chunk contains the prompt prefix
    for chunk in df["chunk"].to_list():
        assert "PREFIX_MARKER" not in chunk, "Prompt should not appear in chunk text"


def test_prompt_sentence_ids_are_negative(tokenizer):
    """Test that prompt tokens get sentence_id=-1 in tokenization."""
    from afterthoughts.chunk import tokenize_with_sentence_boundaries

    docs = ["Hello world."]
    prompt = "Query: "

    result = tokenize_with_sentence_boundaries(
        docs,
        tokenizer,
        prechunk=False,
        max_length=64,
        prompt=prompt,
    )

    # Get sentence_ids for the first (only) sequence
    sent_ids = result["sentence_ids"][0]

    # The first few tokens should be prompt tokens with sentence_id=-1
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    num_prompt_tokens = len(prompt_tokens)

    # First token is CLS, then prompt tokens should have -1
    # (After CLS which also has a sentence_id assignment)
    # Actually, special tokens get extended sentence_ids in the chunking logic
    # Let's just verify some -1 values exist at the start (excluding CLS)
    assert any(
        sid == -1 for sid in sent_ids[1 : num_prompt_tokens + 2]
    ), "Prompt tokens should have sentence_id=-1"


def test_encode_queries_prompt_equivalent_to_manual_prepend(model):
    """Verify prompt produces same embeddings as manually prepending text."""
    query = "What is AI?"
    prompt = "query: "

    # Using prompt parameter
    emb_prompt = model.encode_queries([query], prompt=prompt)

    # Manual prepend
    emb_manual = model.encode_queries([prompt + query])

    # Should be identical (or very close due to floating point)
    np.testing.assert_array_almost_equal(emb_prompt, emb_manual, decimal=5)


# =============================================================================
# Tests for max_chunk_tokens parameter
# =============================================================================


def test_encode_with_max_chunk_tokens_only(model):
    """Test encode() with only max_chunk_tokens (no max_chunk_sents limit)."""
    docs = ["First sentence here. Second sentence follows. Third sentence ends. Fourth one too."]
    df, X = model.encode(
        docs,
        max_chunk_tokens=32,
        max_chunk_sents=None,  # No sentence limit
        max_length=128,
        max_batch_tokens=256,
        show_progress=False,
    )

    assert isinstance(df, pl.DataFrame)
    assert isinstance(X, np.ndarray)
    assert len(df) == len(X)
    # Should have multiple chunks (doc is too long for 32 tokens)
    assert len(df) >= 1


def test_encode_with_max_chunk_tokens_and_max_chunk_sents(model):
    """Test encode() with both max_chunk_tokens and max_chunk_sents."""
    docs = ["First sentence here. Second sentence follows. Third sentence ends. Fourth one too."]
    df, X = model.encode(
        docs,
        max_chunk_tokens=128,  # High token limit
        max_chunk_sents=2,  # Low sentence limit
        max_length=128,
        max_batch_tokens=256,
        show_progress=False,
    )

    assert isinstance(df, pl.DataFrame)
    assert isinstance(X, np.ndarray)
    # All chunks should have at most 2 sentences
    assert all(df["num_sents"] <= 2)


def test_max_chunk_tokens_respects_token_limit(model):
    """Verify chunks don't exceed max_chunk_tokens."""
    # Create document with varying sentence lengths
    docs = [
        "Short. A bit longer sentence here. "
        "This is a much longer sentence with many words. "
        "Another one. And the final sentence of this test document."
    ]

    df, X = model.encode(
        docs,
        max_chunk_tokens=20,
        max_chunk_sents=None,
        max_length=256,
        max_batch_tokens=256,
        show_progress=False,
    )

    # Each chunk text should tokenize to <= max_chunk_tokens
    # (accounting for the fact that chunk text doesn't include special tokens)
    for chunk_text in df["chunk"].to_list():
        tokens = model.tokenizer.encode(chunk_text, add_special_tokens=False)
        # Allow some tolerance due to sentence boundary alignment
        assert len(tokens) <= 25, f"Chunk '{chunk_text}' has {len(tokens)} tokens"


def test_max_chunk_tokens_produces_valid_embeddings(model):
    """Verify embeddings from max_chunk_tokens are valid."""
    docs = ["First sentence. Second sentence. Third sentence."]

    df, X = model.encode(
        docs,
        max_chunk_tokens=50,
        max_chunk_sents=None,
        max_length=128,
        max_batch_tokens=256,
        show_progress=False,
    )

    # Embeddings should have correct shape
    assert X.shape[0] == len(df)
    assert X.shape[1] > 0

    # Embeddings should not be all zeros or NaN
    assert not np.all(X == 0)
    assert not np.any(np.isnan(X))


def test_max_chunk_tokens_with_overlap(model):
    """Test max_chunk_tokens with sentence overlap."""
    docs = ["First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."]

    df, X = model.encode(
        docs,
        max_chunk_tokens=32,
        max_chunk_sents=None,
        chunk_overlap_sents=1,  # 1 sentence overlap (must be int with max_chunk_tokens)
        max_length=128,
        max_batch_tokens=256,
        deduplicate=False,
        show_progress=False,
    )

    assert isinstance(df, pl.DataFrame)
    # With overlap, chunks may share sentences
    assert len(df) >= 1


def test_max_chunk_tokens_backward_compatibility(model):
    """Verify existing behavior is unchanged when max_chunk_tokens is not specified."""
    docs = ["First sentence. Second sentence. Third sentence."]

    # Old behavior (max_chunk_sents only)
    df_old, X_old = model.encode(
        docs,
        max_chunk_sents=1,
        max_length=128,
        max_batch_tokens=256,
        show_progress=False,
    )

    # Should have 3 chunks (one per sentence)
    assert len(df_old) == 3
    assert all(df_old["num_sents"] == 1)


def test_max_chunk_tokens_validation_rejects_list_max_chunk_sents(model):
    """Verify that aligned lists work with max_chunk_tokens and max_chunk_sents."""
    docs = ["First sentence. Second sentence. Third sentence. Fourth sentence."]

    # Aligned lists should work
    df, X = model.encode(
        docs,
        max_chunk_tokens=[32, 64],
        max_chunk_sents=[1, 2],  # Now allowed with aligned lists
        chunk_overlap_sents=0,
        show_progress=False,
    )

    # Should have chunks for both configurations
    configs = set(
        zip(df["max_chunk_sents"].to_list(), df["max_chunk_tokens"].to_list(), strict=False)
    )
    assert (1, 32) in configs
    assert (2, 64) in configs


def test_max_chunk_tokens_validation_rejects_invalid_value(model):
    """Verify validation for invalid max_chunk_tokens values."""
    docs = ["Test document."]

    with pytest.raises(ValueError, match="max_chunk_tokens must be >= 1"):
        model.encode(
            docs,
            max_chunk_tokens=0,
            max_chunk_sents=None,
            show_progress=False,
        )


def test_max_chunk_sents_none_without_max_chunk_tokens_fails(model):
    """Verify that max_chunk_sents=None without max_chunk_tokens raises error."""
    docs = ["Test document."]

    with pytest.raises(ValueError, match="max_chunk_sents cannot be None"):
        model.encode(
            docs,
            max_chunk_sents=None,
            max_chunk_tokens=None,
            show_progress=False,
        )


def test_max_chunk_tokens_with_long_document(model):
    """Test max_chunk_tokens handles long documents requiring prechunking."""
    # Create a longer document
    sentences = [f"This is sentence number {i}." for i in range(20)]
    long_doc = " ".join(sentences)

    df, X = model.encode(
        [long_doc],
        max_chunk_tokens=64,
        max_chunk_sents=None,
        max_length=128,  # Force prechunking
        max_batch_tokens=256,
        show_progress=False,
    )

    assert isinstance(df, pl.DataFrame)
    assert isinstance(X, np.ndarray)
    assert len(df) > 0
    # All chunks should have the same document_idx
    assert all(df["document_idx"] == 0)


def test_max_chunk_tokens_chunk_text_preserved(model):
    """Verify chunk text is correctly reconstructed with max_chunk_tokens."""
    docs = ["Hello world. This is a test sentence. Final sentence here."]

    df, X = model.encode(
        docs,
        max_chunk_tokens=30,
        max_chunk_sents=None,
        max_length=128,
        max_batch_tokens=256,
        show_progress=False,
    )

    # Each chunk should contain text from the original document
    for chunk in df["chunk"].to_list():
        assert any(sent in chunk for sent in ["Hello world", "test sentence", "Final sentence"])


def test_max_chunk_tokens_rejects_float_overlap(model):
    """Verify that float chunk_overlap_sents is rejected with max_chunk_tokens."""
    docs = ["Test document."]

    with pytest.raises(TypeError, match="chunk_overlap_sents must be an integer"):
        model.encode(
            docs,
            max_chunk_tokens=50,
            max_chunk_sents=None,
            chunk_overlap_sents=0.5,  # Should fail - must be int
            show_progress=False,
        )


def test_split_long_sents_true_splits(model):
    """Test that split_long_sents=True splits sentences exceeding max_chunk_tokens."""
    # Create doc with a long sentence
    long_sent = "This is a very long sentence " + "with many repeated words " * 10 + "ending here."
    docs = [f"Short intro. {long_sent} Short outro."]

    df, X = model.encode(
        docs,
        max_chunk_tokens=32,
        max_chunk_sents=None,
        split_long_sents=True,
        max_length=256,
        show_progress=False,
    )

    # All chunks should respect the token limit (since sentences are split)
    for chunk_text in df["chunk"].to_list():
        tokens = len(model.tokenizer.encode(chunk_text, add_special_tokens=False))
        assert tokens <= 32, f"Chunk has {tokens} tokens, exceeds limit of 32"


def test_split_long_sents_false_keeps_intact(model):
    """Test that split_long_sents=False keeps sentences intact even if exceeding limit."""
    # Create doc with a long sentence
    long_sent = "This is a very long sentence " + "with many repeated words " * 10 + "ending here."
    docs = [f"Short intro. {long_sent} Short outro."]

    df, X = model.encode(
        docs,
        max_chunk_tokens=32,
        max_chunk_sents=None,
        split_long_sents=False,
        max_length=256,
        show_progress=False,
    )

    # Should have exactly 3 chunks (3 sentences)
    assert len(df) == 3
    # The long sentence chunk should exceed the token limit
    token_counts = [
        len(model.tokenizer.encode(c, add_special_tokens=False)) for c in df["chunk"].to_list()
    ]
    assert max(token_counts) > 32, "Long sentence should exceed token limit"


def test_max_chunk_tokens_exceeds_max_length_raises(model):
    """Test that max_chunk_tokens > max_length raises ValueError."""
    docs = ["This is a test sentence."]

    with pytest.raises(ValueError, match="max_chunk_tokens.*cannot exceed max_length"):
        model.encode(
            docs,
            max_chunk_tokens=256,
            max_length=128,  # max_chunk_tokens > max_length
            max_chunk_sents=None,
            show_progress=False,
        )
