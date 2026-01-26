import pytest

from afterthoughts.validation import (
    validate_chunk_overlap_sents,
    validate_docs,
    validate_encode_params,
    validate_encode_queries_params,
    validate_max_chunk_sents,
    validate_positive_int,
    validate_prechunk_overlap_tokens,
    validate_return_frame,
    validate_sent_tokenizer,
)


class TestValidateDocs:
    def test_valid_docs(self):
        validate_docs(["doc1", "doc2"])  # Should not raise

    def test_empty_docs(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_docs([])

    def test_non_string_docs(self):
        with pytest.raises(TypeError, match="must be a list of strings"):
            validate_docs(["valid", 123])  # type: ignore[list-item]


class TestValidateNumSents:
    def test_valid_int(self):
        validate_max_chunk_sents(1)
        validate_max_chunk_sents(5)

    def test_valid_list(self):
        validate_max_chunk_sents([1, 2, 3])

    def test_valid_tuple(self):
        validate_max_chunk_sents((1, 2))

    def test_zero_int(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            validate_max_chunk_sents(0)

    def test_negative_int(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            validate_max_chunk_sents(-1)

    def test_empty_list(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_max_chunk_sents([])

    def test_list_with_zero(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            validate_max_chunk_sents([1, 0, 2])

    def test_list_with_non_int(self):
        with pytest.raises(TypeError, match="must be int or None"):
            validate_max_chunk_sents([1, 2.5])  # type: ignore[list-item]


class TestValidateChunkOverlapSents:
    """Test validation for chunk_overlap_sents parameter (now int only)."""

    def test_valid_int(self):
        validate_chunk_overlap_sents(0)
        validate_chunk_overlap_sents(5)

    def test_negative_int(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            validate_chunk_overlap_sents(-1)

    def test_invalid_type(self):
        with pytest.raises(TypeError, match="must be an integer"):
            validate_chunk_overlap_sents(0.5)  # type: ignore[arg-type]


class TestValidatePrechunkOverlapTokens:
    """Test validation for prechunk_overlap_tokens parameter."""

    def test_valid_float(self):
        validate_prechunk_overlap_tokens(0.5)

    def test_valid_int(self):
        validate_prechunk_overlap_tokens(10)

    def test_float_out_of_range(self):
        with pytest.raises(ValueError, match=r"must be in \[0, 1\)"):
            validate_prechunk_overlap_tokens(1.5)

    def test_negative_int(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            validate_prechunk_overlap_tokens(-1)


class TestValidateSentTokenizer:
    def test_valid_tokenizers(self):
        for tokenizer in ["blingfire", "nltk", "pysbd", "syntok"]:
            validate_sent_tokenizer(tokenizer)

    def test_invalid_tokenizer(self):
        with pytest.raises(ValueError, match="Invalid sent_tokenizer"):
            validate_sent_tokenizer("invalid")


class TestValidateReturnFrame:
    def test_valid_frames(self):
        validate_return_frame("polars")
        validate_return_frame("pandas")

    def test_invalid_frame(self):
        with pytest.raises(ValueError, match="Invalid return_frame"):
            validate_return_frame("invalid")


class TestValidatePositiveInt:
    def test_valid_positive(self):
        validate_positive_int(1, "test")
        validate_positive_int(100, "test")

    def test_none_allowed(self):
        validate_positive_int(None, "test")

    def test_zero_invalid(self):
        with pytest.raises(ValueError, match="must be > 0"):
            validate_positive_int(0, "test")

    def test_negative_invalid(self):
        with pytest.raises(ValueError, match="must be > 0"):
            validate_positive_int(-1, "test")


class TestValidateEncodeParams:
    def test_valid_params(self):
        validate_encode_params(
            docs=["doc1"],
            max_chunk_sents=1,
            chunk_overlap_sents=0,
            prechunk_overlap_tokens=0.5,
            sent_tokenizer="blingfire",
            return_frame="polars",
            max_batch_tokens=8192,
            max_length=512,
        )

    def test_invalid_docs_caught(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_encode_params(
                docs=[],
                max_chunk_sents=1,
                chunk_overlap_sents=0,
                prechunk_overlap_tokens=0.5,
                sent_tokenizer="blingfire",
                return_frame="polars",
                max_batch_tokens=8192,
                max_length=None,
            )


class TestValidateEncodeQueriesParams:
    def test_valid_params(self):
        validate_encode_queries_params(
            queries=["query1"],
            batch_size=32,
            max_length=512,
        )

    def test_invalid_queries_caught(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_encode_queries_params(
                queries=[],
                batch_size=32,
                max_length=None,
            )

    def test_invalid_batch_size_caught(self):
        with pytest.raises(ValueError, match="must be > 0"):
            validate_encode_queries_params(
                queries=["query"],
                batch_size=0,
                max_length=None,
            )
