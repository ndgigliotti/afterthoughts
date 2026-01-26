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


def validate_max_chunk_sents(
    max_chunk_sents: int | list[int | None] | tuple[int | None, ...] | None,
    max_chunk_tokens: int | list[int] | tuple[int, ...] | None = None,
) -> None:
    """Validate max_chunk_sents parameter.

    Supports None as a value in lists when max_chunk_tokens is specified.
    """
    # Allow None scalar only when max_chunk_tokens is specified
    if max_chunk_sents is None:
        if max_chunk_tokens is None:
            raise ValueError("max_chunk_sents cannot be None unless max_chunk_tokens is specified")
        return

    # Validate single int value
    if isinstance(max_chunk_sents, int):
        if max_chunk_sents < 1:
            raise ValueError(f"max_chunk_sents must be >= 1, got {max_chunk_sents}")
        return

    # Validate list/tuple
    if isinstance(max_chunk_sents, (list, tuple)):
        if len(max_chunk_sents) == 0:
            raise ValueError("max_chunk_sents cannot be empty")

        # Check types (allow int or None)
        for i, s in enumerate(max_chunk_sents):
            if s is None:
                # None is allowed only if max_chunk_tokens is specified
                if max_chunk_tokens is None:
                    raise ValueError(
                        f"max_chunk_sents[{i}] is None, but max_chunk_tokens is not specified. "
                        "None values require max_chunk_tokens to be set."
                    )
            elif isinstance(s, int):
                if s < 1:
                    raise ValueError(f"max_chunk_sents[{i}] must be >= 1, got {s}")
            else:
                raise TypeError(f"max_chunk_sents[{i}] must be int or None, got {type(s).__name__}")
        return

    raise TypeError(
        f"max_chunk_sents must be int, list, tuple, or None, got {type(max_chunk_sents).__name__}"
    )


def validate_chunk_overlap_sents(chunk_overlap_sents: int) -> None:
    """Validate chunk_overlap_sents parameter."""
    match chunk_overlap_sents:
        case int(i) if i < 0:
            raise ValueError(f"chunk_overlap_sents must be >= 0, got {i}")
        case int():
            pass
        case _:
            raise TypeError(
                f"chunk_overlap_sents must be an integer, got {type(chunk_overlap_sents).__name__}"
            )


def validate_prechunk_overlap_tokens(prechunk_overlap_tokens: float | int) -> None:
    """Validate prechunk_overlap_tokens parameter."""
    match prechunk_overlap_tokens:
        case float(f) if not (0 <= f < 1):
            raise ValueError(f"prechunk_overlap_tokens as float must be in [0, 1), got {f}")
        case int(i) if i < 0:
            raise ValueError(f"prechunk_overlap_tokens as int must be >= 0, got {i}")


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
    max_chunk_tokens: int | list[int] | tuple[int, ...] | None,
    max_chunk_sents: int | list[int] | tuple[int, ...] | None,
) -> None:
    """Validate max_chunk_tokens parameter.

    Supports single values or lists for cartesian product generation.
    """
    if max_chunk_tokens is None:
        return

    # Validate single value
    if isinstance(max_chunk_tokens, int):
        if max_chunk_tokens < 1:
            raise ValueError(f"max_chunk_tokens must be >= 1, got {max_chunk_tokens}")
        return

    # Validate list/tuple
    if isinstance(max_chunk_tokens, (list, tuple)):
        if len(max_chunk_tokens) == 0:
            raise ValueError("max_chunk_tokens cannot be empty")
        if any(not isinstance(t, int) for t in max_chunk_tokens):
            raise TypeError("max_chunk_tokens values must be integers")
        if any(t < 1 for t in max_chunk_tokens):
            raise ValueError(f"max_chunk_tokens values must be >= 1, got {list(max_chunk_tokens)}")
        return

    raise TypeError(
        f"max_chunk_tokens must be an int, list, or tuple, got {type(max_chunk_tokens).__name__}"
    )


def validate_chunk_config_pair(
    max_chunk_sents: int | None,
    max_chunk_tokens: int | None,
) -> None:
    """Validate a single (max_chunk_sents, max_chunk_tokens) configuration pair.

    Parameters
    ----------
    max_chunk_sents : int or None
        Maximum sentences per chunk for this config.
    max_chunk_tokens : int or None
        Maximum tokens per chunk for this config.

    Raises
    ------
    ValueError
        If the pair is invalid (e.g., both None).
    """
    if max_chunk_sents is None and max_chunk_tokens is None:
        raise ValueError(
            "At least one of max_chunk_sents or max_chunk_tokens must be specified. "
            "Got both as None."
        )


def validate_encode_params(
    docs: list[str],
    max_chunk_sents: int | list[int] | tuple[int, ...] | None,
    chunk_overlap_sents: int,
    prechunk_overlap_tokens: float | int,
    sent_tokenizer: str,
    return_frame: str,
    max_batch_tokens: int,
    max_length: int | None,
    max_chunk_tokens: int | list[int] | tuple[int, ...] | None = None,
) -> None:
    """Validate all parameters for Encoder.encode()."""
    validate_docs(docs)
    validate_max_chunk_tokens(max_chunk_tokens, max_chunk_sents)
    validate_max_chunk_sents(max_chunk_sents, max_chunk_tokens)
    validate_chunk_overlap_sents(chunk_overlap_sents)
    validate_prechunk_overlap_tokens(prechunk_overlap_tokens)
    validate_sent_tokenizer(sent_tokenizer)
    validate_return_frame(return_frame)
    validate_positive_int(max_batch_tokens, "max_batch_tokens")
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
