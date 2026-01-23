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

"""Input validation for encoding parameters.

This module provides validation functions that fail fast with clear error
messages before expensive computation begins.
"""

VALID_SENT_TOKENIZERS = frozenset({"blingfire", "nltk", "pysbd", "syntok"})
VALID_RETURN_FRAMES = frozenset({"polars", "pandas"})


def validate_docs(docs: list[str]) -> None:
    """Validate document input."""
    match docs:
        case []:
            raise ValueError("docs cannot be empty")
        case list(items) if not all(isinstance(d, str) for d in items):
            raise TypeError("docs must be a list of strings")


def validate_num_sents(
    num_sents: int | list[int] | tuple[int, ...] | None,
    max_chunk_tokens: int | None = None,
) -> None:
    """Validate num_sents parameter."""
    # Allow None only when max_chunk_tokens is specified
    if num_sents is None:
        if max_chunk_tokens is None:
            raise ValueError("num_sents cannot be None unless max_chunk_tokens is specified")
        return
    match num_sents:
        case int(n) if n < 1:
            raise ValueError(f"num_sents must be >= 1, got {n}")
        case list() | tuple() as sizes if len(sizes) == 0:
            raise ValueError("num_sents cannot be empty")
        case list() | tuple() as sizes if any(not isinstance(s, int) for s in sizes):
            raise TypeError("num_sents values must be integers")
        case list() | tuple() as sizes if any(s < 1 for s in sizes):
            raise ValueError(f"num_sents values must be >= 1, got {list(sizes)}")


def validate_chunk_overlap(chunk_overlap: int | float | list[int] | dict[int, int]) -> None:
    """Validate chunk_overlap parameter."""
    match chunk_overlap:
        case float(f) if not (0 <= f < 1):
            raise ValueError(f"chunk_overlap as float must be in [0, 1), got {f}")
        case int(i) if i < 0:
            raise ValueError(f"chunk_overlap as int must be >= 0, got {i}")
        case list() as overlaps if any(not isinstance(o, int) or o < 0 for o in overlaps):
            raise ValueError(
                f"chunk_overlap list values must be non-negative integers, got {overlaps}"
            )
        case dict() as mapping if any(not isinstance(v, int) or v < 0 for v in mapping.values()):
            raise ValueError("chunk_overlap dict values must be non-negative integers")


def validate_prechunk_overlap(prechunk_overlap: float | int) -> None:
    """Validate prechunk_overlap parameter."""
    match prechunk_overlap:
        case float(f) if not (0 <= f < 1):
            raise ValueError(f"prechunk_overlap as float must be in [0, 1), got {f}")
        case int(i) if i < 0:
            raise ValueError(f"prechunk_overlap as int must be >= 0, got {i}")


def validate_sent_tokenizer(sent_tokenizer: str) -> None:
    """Validate sent_tokenizer parameter."""
    match sent_tokenizer:
        case str(s) if s in VALID_SENT_TOKENIZERS:
            pass
        case _:
            raise ValueError(
                f"Invalid sent_tokenizer: {sent_tokenizer!r}. "
                f"Valid options: {sorted(VALID_SENT_TOKENIZERS)}"
            )


def validate_return_frame(return_frame: str) -> None:
    """Validate return_frame parameter."""
    match return_frame:
        case str(s) if s in VALID_RETURN_FRAMES:
            pass
        case _:
            raise ValueError(
                f"Invalid return_frame: {return_frame!r}. "
                f"Valid options: {sorted(VALID_RETURN_FRAMES)}"
            )


def validate_positive_int(value: int | None, name: str) -> None:
    """Validate that a value is a positive integer (if provided)."""
    match value:
        case None:
            pass
        case int(n) if n > 0:
            pass
        case int(n):
            raise ValueError(f"{name} must be > 0, got {n}")
        case _:
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}")


def validate_max_chunk_tokens(
    max_chunk_tokens: int | None,
    num_sents: int | list[int] | tuple[int, ...] | None,
) -> None:
    """Validate max_chunk_tokens parameter."""
    if max_chunk_tokens is not None:
        if not isinstance(max_chunk_tokens, int):
            raise TypeError(
                f"max_chunk_tokens must be an integer, got {type(max_chunk_tokens).__name__}"
            )
        if max_chunk_tokens < 1:
            raise ValueError(f"max_chunk_tokens must be >= 1, got {max_chunk_tokens}")
        # When max_chunk_tokens is used, num_sents can only be int or None (not list/tuple)
        if isinstance(num_sents, (list, tuple)):
            raise ValueError(
                "num_sents cannot be a list/tuple when max_chunk_tokens is specified. "
                "Use a single int value or None."
            )


def validate_encode_params(
    docs: list[str],
    num_sents: int | list[int] | tuple[int, ...] | None,
    chunk_overlap: int | float | list[int] | dict[int, int],
    prechunk_overlap: float | int,
    sent_tokenizer: str,
    return_frame: str,
    batch_tokens: int,
    max_length: int | None,
    max_chunk_tokens: int | None = None,
) -> None:
    """Validate all parameters for Encoder.encode()."""
    validate_docs(docs)
    validate_max_chunk_tokens(max_chunk_tokens, num_sents)
    validate_num_sents(num_sents, max_chunk_tokens)
    validate_chunk_overlap(chunk_overlap)
    validate_prechunk_overlap(prechunk_overlap)
    validate_sent_tokenizer(sent_tokenizer)
    validate_return_frame(return_frame)
    validate_positive_int(batch_tokens, "batch_tokens")
    validate_positive_int(max_length, "max_length")


def validate_encode_queries_params(
    queries: list[str],
    batch_size: int,
    max_length: int | None,
) -> None:
    """Validate all parameters for Encoder.encode_queries()."""
    validate_docs(queries)  # Same validation as docs
    validate_positive_int(batch_size, "batch_size")
    validate_positive_int(max_length, "max_length")
