import numpy as np
import polars as pl
import pytest
import torch

from afterthoughts import Encoder, LiteEncoder
from afterthoughts.chunk import get_chunk_idx
from afterthoughts.encode import _EncoderBase
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
def test_encoder_lite_init():
    """Test LiteEncoder initialization with lossy params."""
    model_name = MODEL_NAME
    amp = False
    amp_dtype = torch.float16
    quantize = "float16"
    truncate_dims = 128
    normalize = True
    pca = 50
    pca_early_stop = 1.0
    device = "cuda"
    _num_token_jobs = 8

    encoder = LiteEncoder(
        model_name=model_name,
        amp=amp,
        amp_dtype=amp_dtype,
        quantize=quantize,
        truncate_dims=truncate_dims,
        normalize=normalize,
        pca=pca,
        pca_early_stop=pca_early_stop,
        device=device,
        _num_token_jobs=_num_token_jobs,
    )

    assert encoder.tokenizer is not None
    assert encoder.model is not None
    assert encoder.amp == amp
    assert encoder.amp_dtype == amp_dtype
    assert encoder.quantize == quantize
    assert encoder.truncate_dims == truncate_dims
    assert encoder.normalize == normalize
    assert encoder.pca == pca
    assert encoder.pca_early_stop == pca_early_stop
    assert encoder.device.type == device
    assert encoder._num_token_jobs == _num_token_jobs

    # Test invalid truncate_dims and pca combination
    with pytest.raises(ValueError):
        LiteEncoder(
            model_name=model_name,
            truncate_dims=10,
            pca=50,
        )


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


def test_encoder_lite_quantize_if_needed():
    model = LiteEncoder(model_name=MODEL_NAME, device="cpu", quantize="float16")
    embeds = torch.randn(10, 10, dtype=torch.float32)
    reduced_embeds = model.quantize_if_needed(embeds)
    assert reduced_embeds.dtype == torch.float16

    model.quantize = None
    non_reduced_embeds = model.quantize_if_needed(embeds)
    assert non_reduced_embeds.dtype == torch.float32


def test_encoder_lite_truncate_dims_if_needed():
    encoder = LiteEncoder(model_name=MODEL_NAME, device="cpu", truncate_dims=5)
    embeds = torch.randn(10, 10)
    truncated_embeds = encoder.truncate_dims_if_needed(embeds)
    assert truncated_embeds.shape[1] == 5

    encoder.truncate_dims = None
    non_truncated_embeds = encoder.truncate_dims_if_needed(embeds)
    assert non_truncated_embeds.shape[1] == 10


def test_encoder_lite_quantize_options():
    """Test that LiteEncoder accepts valid quantize options."""
    for opt in LiteEncoder.QUANTIZE_OPTIONS:
        if opt == "binary":
            # binary is incompatible with normalize=True (default is False, so this works)
            encoder = LiteEncoder(model_name=MODEL_NAME, device="cpu", quantize=opt)
        else:
            encoder = LiteEncoder(model_name=MODEL_NAME, device="cpu", quantize=opt)
        assert encoder.quantize == opt


def test_encoder_lite_quantize_invalid():
    """Test that invalid quantize value raises error."""
    with pytest.raises(ValueError, match="must be one of"):
        LiteEncoder(model_name=MODEL_NAME, device="cpu", quantize="invalid")


@pytest.mark.parametrize("quantize", ["int8", "binary"])
def test_encoder_lite_quantize_normalize_incompatible(quantize):
    """Test that quantize='int8' and 'binary' are incompatible with normalize=True."""
    with pytest.raises(ValueError, match="incompatible"):
        LiteEncoder(model_name=MODEL_NAME, device="cpu", quantize=quantize, normalize=True)


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
            _EncoderBase._build_results_df(results, return_frame, as_numpy)
    else:
        # Build the results dataframe and check the types
        expected_length = len(results["sample_idx"])
        df, embeds = _EncoderBase._build_results_df(results, return_frame, as_numpy)
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
    df, embeds = _EncoderBase._build_results_df(results, return_frame, as_numpy)
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
