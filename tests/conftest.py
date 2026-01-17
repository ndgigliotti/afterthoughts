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
