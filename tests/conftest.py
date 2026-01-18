"""Shared pytest fixtures for test performance optimization."""

import pytest
from transformers import AutoTokenizer

from afterthoughts import Encoder

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"


@pytest.fixture(scope="session")
def model():
    """Load model once per test session to avoid repeated initialization."""
    return Encoder(
        model_name=MODEL_NAME,
        device="cpu",
        _num_token_jobs=1,
    )


@pytest.fixture(scope="session")
def tokenizer():
    """Load tokenizer once per test session."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="session")
def encoded_2docs(model):
    """Pre-encoded ML/Python docs for reuse across tests."""
    docs = [
        "Machine learning is AI. It enables learning.",
        "Python is a programming language. It is popular.",
    ]
    return model.encode(docs, num_sents=[1, 2], chunk_overlap=0, show_progress=False)


@pytest.fixture(scope="session")
def model_half_embeds():
    """Load model with half_embeds=True for testing float16 conversion."""
    return Encoder(
        model_name=MODEL_NAME,
        device="cpu",
        half_embeds=True,
        _num_token_jobs=1,
    )


@pytest.fixture(scope="session")
def model_truncate_dims():
    """Load model with truncate_dims=128 for testing dimension truncation."""
    return Encoder(
        model_name=MODEL_NAME,
        device="cpu",
        truncate_dims=128,
        _num_token_jobs=1,
    )
