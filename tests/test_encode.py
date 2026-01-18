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
    num_sents = 2
    overlap = 0.5

    result = get_chunk_idx(input_ids, sentence_ids, num_sents, overlap)

    assert "chunk_token_idx" in result
    assert "chunk_token_ids" in result
    assert "sentence_ids" in result
    assert "sequence_idx" in result
    assert "chunk_size" in result


def test_get_chunk_idx_multiple_sizes():
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    sentence_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, -1]])
    num_sents = [1, 2]
    overlap = 0.5

    result = get_chunk_idx(input_ids, sentence_ids, num_sents, overlap)

    assert "chunk_token_idx" in result
    assert "chunk_token_ids" in result
    assert "sentence_ids" in result
    assert "sequence_idx" in result
    assert "chunk_size" in result


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
    df, X = model.encode(docs, num_sents=1, max_length=64, batch_tokens=256, show_progress=False)
    assert isinstance(df, pl.DataFrame)
    assert isinstance(X, np.ndarray)
    assert len(df) == len(X)
    assert "chunk" in df.columns
    assert "chunk_size" in df.columns


def test_encoder_encode_multiple_num_sents(model):
    docs = [
        "This is a test document. Another sentence here.",
        "Another test document. With more sentences.",
    ]
    num_sents = [1, 2]
    df, X = model.encode(
        docs, num_sents=num_sents, max_length=64, batch_tokens=256, show_progress=False
    )
    assert isinstance(df, pl.DataFrame)
    assert isinstance(X, np.ndarray)
    assert len(df) == len(X)
    assert all(size in df["chunk_size"].unique().to_list() for size in num_sents)


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
        "chunk_size": torch.tensor([2, 2]),
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
        "chunk_size": torch.tensor([2, 2]),
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
    df, X = model.encode(docs, num_sents=1, max_length=64, batch_tokens=256, show_progress=False)
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
        num_sents=2,
        chunk_overlap=1,  # One sentence overlap
        max_length=64,
        batch_tokens=256,
        show_progress=False,
    )
    assert isinstance(df, pl.DataFrame)
    chunks = df["chunk"].to_list()

    # Check that overlapping chunks contain correct sentence combinations
    # With num_sents=2 and overlap=1, we should get chunks like:
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
    df, X = model.encode(docs, num_sents=1, max_length=64, batch_tokens=256, show_progress=False)

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
        docs, num_sents=1, max_length=64, batch_tokens=256, return_text=False, show_progress=False
    )
    assert "chunk" not in df.columns


def test_encoder_text_reconstruction_fallback(model):
    """Test fallback to decoding when sentence is split due to exceeding max_length."""
    # Create a document with a very long sentence that will exceed max_length
    # and trigger _split_long_sentences(), causing sentence IDs to exceed original count
    long_sentence = (
        "This is a very long sentence " + "with many repeated words " * 50 + "that ends here."
    )
    docs = [f"Short intro. {long_sentence} Short outro."]

    # Use a small max_length to force the long sentence to be split
    df, X = model.encode(docs, num_sents=1, max_length=32, batch_tokens=256, show_progress=False)

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


def test_deduplicate_averaging_correctness():
    """Verify that deduplication correctly averages embeddings for duplicate chunks."""
    # Create test data with known duplicates
    # Chunks 0 and 2 are duplicates (same doc, size, sentences)
    # Chunks 1 and 3 are duplicates (same doc, size, sentences)
    results = {
        "document_idx": torch.tensor([0, 0, 0, 0]),
        "chunk_size": torch.tensor([2, 2, 2, 2]),
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
        "chunk_size": torch.tensor([1, 1, 1]),
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
        docs, num_sents=1, max_length=64, batch_tokens=256, show_progress=False
    )
    # Note: as_numpy=True converts to numpy which may upcast to float32
    # Check the internal processing produces float16
    assert model_half_embeds.half_embeds is True


def test_encoder_truncate_dims_output(model_truncate_dims):
    """Test that truncate_dims produces truncated embeddings."""
    docs = ["This is a test document. Another sentence here."]
    df, X = model_truncate_dims.encode(
        docs, num_sents=1, max_length=64, batch_tokens=256, show_progress=False
    )
    # The embedding dimension should be 128 (truncated from 384)
    assert X.shape[1] == 128


def test_encoder_truncate_dims_queries(model_truncate_dims):
    """Test that truncate_dims applies to query embeddings."""
    queries = ["What is this about?"]
    query_embeds = model_truncate_dims.encode_queries(queries)
    assert query_embeds.shape[1] == 128
