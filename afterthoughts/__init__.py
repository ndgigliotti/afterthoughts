"""Afterthoughts: Context-aware sentence-chunk embeddings via late chunking.

This library implements late chunking for extracting sentence-chunk embeddings
using transformer models. Late chunking processes entire documents through the
model to capture full context, then extracts embeddings for sentence groups
(chunks) by mean-pooling token embeddings within sentence boundaries.

Key Features
------------
- Context-aware embeddings: Full document context is preserved during encoding
- Flexible chunking: Configure chunk size (number of sentences) and overlap
- Sentence boundary detection: Multiple backends (BlingFire, NLTK, pysbd, syntok)
- Memory efficient: Dynamic batching by token count, optional float16 conversion
- Matryoshka support: Dimension truncation for MRL-trained models
- Multiple output formats: Polars or pandas DataFrames with NumPy/PyTorch arrays

Basic Usage
-----------
    >>> from afterthoughts import LateEncoder
    >>>
    >>> # Initialize encoder
    >>> encoder = LateEncoder("sentence-transformers/all-MiniLM-L6-v2")
    >>>
    >>> # Encode documents into sentence chunks
    >>> docs = ["First sentence. Second sentence.", "Another document."]
    >>> df, embeddings = encoder.encode(docs, max_chunk_sents=1)
    >>>
    >>> # Encode queries for semantic search
    >>> query_embeds = encoder.encode_queries(["search query"])

Main Classes
------------
LateEncoder : Main API for encoding documents and queries into embeddings

Utility Functions
-----------------
configure_logging : Configure logging output for the library
get_device : Auto-detect the best available device (CUDA > MPS > CPU)

See Also
--------
The full documentation provides details on chunking strategies, sentence
tokenizers, and advanced configuration options.
"""

__author__ = """Nicholas Gigliotti"""
__email__ = "ndgigliotti@gmail.com"
__version__ = "0.1.1"

# Copyright 2024-2026 Nicholas Gigliotti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

from afterthoughts.encode import LateEncoder
from afterthoughts.utils import configure_logging, get_device

__all__ = ["Encoder", "LateEncoder", "configure_logging", "get_device"]


def __getattr__(name: str):
    """Provide deprecated Encoder alias for backwards compatibility."""
    if name == "Encoder":
        warnings.warn(
            "Encoder is deprecated and will be removed in a future version. "
            "Use LateEncoder instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return LateEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
