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

"""Sentence-aware chunking and chunk embedding computation.

This module provides functions for:
1. Sentence boundary detection using multiple backends (BlingFire, NLTK, pysbd, syntok)
2. Sentence-aware document chunking that preserves sentence boundaries
3. Chunk embedding computation via mean-pooling of token embeddings
4. Tokenization with sentence boundary preservation

The late chunking approach processes entire documents through the model to
capture full context, then extracts embeddings for sentence groups (chunks)
by mean-pooling token embeddings within sentence boundaries.

Key Functions
-------------
get_sentence_offsets : Detect sentence boundaries in text
tokenize_with_sentence_boundaries : Tokenize while preserving sentence structure
get_chunk_idx : Extract chunk indices from tokenized sequences
_compute_chunk_embeds : Compute chunk embeddings via vectorized mean pooling
chunk_preserving_sentence_structure : Split long sequences preserving sentences

Sentence Tokenizers
-------------------
- BlingFire: Fast C++ implementation (default, recommended)
- NLTK: Punkt tokenizer with abbreviation handling
- pysbd: Rule-based with extensive punctuation handling
- syntok: Sophisticated segmentation with token-level analysis

Notes
-----
All chunking functions preserve sentence boundaries to maintain semantic coherence.
Long sentences that exceed max_length are automatically split into sub-segments.
"""

import logging
import warnings
from typing import Any

import blingfire as bf
import numpy as np
import torch
from joblib import Parallel, delayed
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

from afterthoughts.avail import require_nltk, require_pysbd, require_syntok
from afterthoughts.tokenize import TokenizedDataset, get_max_length, pad, tokenize_docs
from afterthoughts.utils import get_overlap_count

logger = logging.getLogger(__name__)


def get_sentence_offsets_syntok(text: str) -> torch.Tensor:
    """
    Extracts sentence offsets from the given text using the syntok library.

    Parameters
    ----------
    text : str
        The input text from which to extract sentence offsets.

    Returns
    -------
    torch.Tensor
        A tensor containing tuples of start and end offsets for each sentence in the text.

    Raises
    ------
    ImportError
        If syntok is not installed.
    """
    analyze = require_syntok()
    if len(text) == 0:
        offsets = torch.tensor([]).reshape(0, 2)
    else:
        offsets = torch.tensor([(s[0].offset, s[-1].offset + 1) for p in analyze(text) for s in p])
    return offsets


def get_sentence_offsets_nltk(text: str) -> torch.Tensor:
    """
    Tokenizes the input text into sentences and returns their offsets using NLTK's PunktSentenceTokenizer.

    Parameters
    ----------
    text : str
        The input text to be tokenized into sentences.

    Returns
    -------
    torch.Tensor
        A tensor containing the start and end offsets of each sentence in the input text.

    Raises
    ------
    ImportError
        If NLTK is not installed.
    """
    nltk = require_nltk()
    if len(text) == 0:
        offsets = torch.tensor([]).reshape(0, 2)
    else:
        sent_tokenizer = nltk.PunktSentenceTokenizer()
        offsets = torch.tensor(list(sent_tokenizer.span_tokenize(text)))
    return offsets


def get_sentence_offsets_blingfire(text: str) -> torch.Tensor:
    """
    Get the sentence offsets from the given text using BlingFire.

    Parameters
    ----------
    text : str
        The input text to be processed.

    Returns
    -------
    torch.Tensor
        A tensor containing the offsets of each sentence in the input text.
    """
    if len(text) == 0:
        offsets = torch.tensor([]).reshape(0, 2)
    else:
        offsets = torch.tensor(bf.text_to_sentences_and_offsets(text)[1])
    return offsets


def get_sentence_offsets_pysbd(text: str) -> torch.Tensor:
    """
    Get the sentence offsets from the given text using pysbd.

    pysbd (Python Sentence Boundary Disambiguation) is designed to handle
    edge cases like abbreviations (Dr., U.S., etc.) more accurately than
    simpler tokenizers.

    Parameters
    ----------
    text : str
        The input text to be processed.

    Returns
    -------
    torch.Tensor
        A tensor containing the offsets of each sentence in the input text.

    Raises
    ------
    ImportError
        If pysbd is not installed.
    """
    Segmenter = require_pysbd()
    if len(text) == 0:
        offsets = torch.tensor([]).reshape(0, 2)
    else:
        segmenter = Segmenter(language="en", clean=False)
        sentences = segmenter.segment(text)
        # Build offsets by finding each sentence in the original text
        offset_list = []
        pos = 0
        for sent in sentences:
            start = text.find(sent, pos)
            if start == -1:
                # Fallback: use current position if exact match fails
                start = pos
            end = start + len(sent)
            offset_list.append((start, end))
            pos = end
        offsets = torch.tensor(offset_list)
    return offsets


def get_sentence_offsets(
    text: str | list[str], method: str = "blingfire", n_jobs: int | None = None
) -> torch.Tensor | list[torch.Tensor]:
    """
    Get sentence offsets for a given text using a specified method.

    Parameters
    ----------
    text : str or list of str
        The input text to be processed. If a list of strings is provided,
        the function will process each string in parallel.
    method : str, optional
        The method to use for sentence boundary detection.
        Options are 'blingfire', 'nltk', 'pysbd', and 'syntok'.
        Defaults to 'blingfire'.
    n_jobs : int, optional
        The number of jobs to use for parallel processing when the input
        is a list of strings. If None, defaults to using all available cores.
        Only relevant when `text` is a list.

    Returns
    -------
    torch.Tensor or list of torch.Tensor
        A tensor containing the start and end offsets of each sentence in the input text.
        If the input is a list of strings, the function returns a list of tensors.

    Raises
    ------
    ValueError
        If an invalid method is specified.

    See Also
    --------
    get_sentence_offsets_blingfire : Sentence offset detection using BlingFire.
    get_sentence_offsets_nltk : Sentence offset detection using NLTK.
    get_sentence_offsets_pysbd : Sentence offset detection using pysbd.
    get_sentence_offsets_syntok : Sentence offset detection using syntok.
    """
    methods = {
        "blingfire": get_sentence_offsets_blingfire,
        "nltk": get_sentence_offsets_nltk,
        "pysbd": get_sentence_offsets_pysbd,
        "syntok": get_sentence_offsets_syntok,
    }

    if method in methods:
        get_offsets = methods[method]
        if isinstance(text, str):
            offsets = get_offsets(text)
        else:
            offsets = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(get_offsets)(t) for t in text
            )
    else:
        raise ValueError(f"Invalid method: '{method}'")
    return offsets


def _add_special_tokens(
    input_ids: torch.Tensor, cls_token_id: int | None = None, sep_token_id: int | None = None
) -> torch.Tensor:
    """Adds special tokens (CLS, SEP) to the input token IDs.

    Parameters
    ----------
    input_ids : torch.Tensor
        The input token IDs to which special tokens will be added.
    cls_token_id : int, optional
        The token ID of the CLS token. If provided, the CLS token is added to the beginning of the input IDs.
    sep_token_id : int, optional
        The token ID of the SEP token. If provided, the SEP token is added to the end of the input IDs.

    Returns
    -------
    torch.Tensor
        The input IDs with the special tokens added.
    """

    to_cat = [input_ids]
    if cls_token_id is not None:
        to_cat.insert(0, torch.tensor([cls_token_id]))
    if sep_token_id is not None:
        to_cat.append(torch.tensor([sep_token_id]))
    result = torch.cat(to_cat)

    return result


@torch.inference_mode()
def _split_long_sentences(sentence_ids: torch.Tensor, max_length: int) -> torch.Tensor:
    """Splits long sentences into smaller segments.

    Parameters
    ----------
    sentence_ids : torch.Tensor
        A tensor containing sentence IDs.
    max_length : int
        The maximum length for each segment.

    Returns
    -------
    torch.Tensor
        A tensor of sentence IDs with long sentences split into segments.
    """
    sent_lengths = torch.bincount(sentence_ids[sentence_ids != -1])
    if torch.any(sent_lengths > max_length):
        masks = [sentence_ids == i for i in range(len(sent_lengths))]
        for i in reversed(range(len(masks))):
            mask = masks[i]
            if mask.sum() > max_length:
                del masks[i]
                # Divide the sentence into smaller segments
                subsegment_id = torch.arange(0, int(mask.sum().item())) // max_length
                # Create a new mask for each subsegment
                for sub_id in reversed(torch.unique(subsegment_id)):
                    new_mask = torch.zeros_like(mask, dtype=torch.bool)
                    new_mask[mask] = subsegment_id == sub_id
                    # Insert the new mask into the list of masks
                    masks.insert(i, new_mask)
        # Combine the masks into a single tensor of sentence IDs
        sentence_ids = torch.full_like(sentence_ids, -1)
        for i, mask in enumerate(masks):
            sentence_ids[mask] = i
    return sentence_ids


@torch.inference_mode()
def chunk_preserving_sentence_structure(
    input_ids: torch.Tensor,
    sentence_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    sample_idx: int,
    max_length: int = 512,
    overlap: float = 0.5,
    add_special_tokens: bool = True,
    padding: str | None = "max_length",
    reset_sentence_ids_on_overflow: bool = False,
) -> dict[str, Any]:
    """Chunk a sequence of input IDs while preserving sentence structure.
    This function splits a long sequence of input IDs into smaller chunks,
    ensuring that sentence boundaries are respected as much as possible.
    It also supports overlapping chunks and padding to a specified length.

    Parameters
    ----------
    input_ids : list of int
        List of input IDs representing the tokenized text.
    sentence_ids : list of int
        List of sentence IDs indicating to which sentence each token belongs.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer used to encode the input text.  It is used to retrieve
        special tokens (e.g., CLS, SEP, PAD).
    sample_idx : int
        The index of the sample being processed.  This is used to populate
        the `overflow_to_sample_mapping` in the output.
    max_length : int, optional
        The maximum length of each chunk (including special tokens), by default 512.
    overlap : float, optional
        The fraction of sentences to overlap between consecutive chunks, by default 0.5.
        Must be in the range [0, 1).
    add_special_tokens : bool, optional
        Whether to add special tokens (e.g., CLS, SEP) to the beginning and end
        of each chunk, by default True.
    padding : str, optional
        The padding strategy to use. Must be one of 'max_length', 'longest', or None.
        If 'max_length', all chunks are padded to `max_length`. If 'longest', chunks
        are padded to the length of the longest chunk in the batch. If None, no padding is applied, by default "max_length".
    reset_sentence_ids_on_overflow : bool, optional
        Whether to reset sentence IDs to start from 0 for each chunk. If False, sentence IDs
        are adjusted to be relative to the start of each chunk. Default is False.

    Returns
    -------
    dict
        A dictionary containing the following keys:
            - "input_ids": list of list of int
                List of padded chunked input IDs.
            - "attention_mask": list of list of bool
                List of attention masks corresponding to the padded input IDs.
            - "overflow_to_sample_mapping": torch.Tensor
                Tensor mapping each chunk to the original sample index.
            - "sentence_ids": list of list of int
                List of sentence IDs for each chunk, adjusted to the chunk's local indexing.

    Raises
    ------
    ValueError
        If `padding` is not one of 'max_length', 'longest', or None.
    ValueError
        If `overlap` is not in the range [0, 1).

    Notes
    -----
    - The function prioritizes preserving sentence boundaries when creating chunks.
    - Overlapping chunks can be created by specifying a value for the `overlap` parameter.
    - The `sentence_ids` in the output is adjusted to be relative to the start of each chunk.
    """
    max_length = get_max_length(max_length, tokenizer)
    # Basic validation
    if padding not in ["max_length", "longest", None]:
        raise ValueError("Padding must be either 'max_length', 'longest', or None.")
    if not (0 <= overlap < 1):
        raise ValueError("Overlap must be in the range [0, 1).")

    # Setup special tokens
    cls_token_id = tokenizer.cls_token_id or tokenizer.bos_token_id
    sep_token_id = tokenizer.sep_token_id or tokenizer.eos_token_id
    num_specials = sum(x is not None for x in [cls_token_id, sep_token_id])

    # Identify leading prompt tokens (sentence_id=-1 before first valid sentence)
    # These need to be preserved in each chunk for attention context
    prompt_mask = torch.zeros(len(sentence_ids), dtype=torch.bool)
    for i, sid in enumerate(sentence_ids):
        if sid == -1:
            prompt_mask[i] = True
        else:
            break  # Stop at first valid sentence token

    num_prompt_tokens = prompt_mask.sum().item()
    prompt_input_ids = (
        torch.tensor(input_ids)[:num_prompt_tokens] if num_prompt_tokens > 0 else None
    )
    prompt_sentence_ids = sentence_ids[:num_prompt_tokens] if num_prompt_tokens > 0 else None

    # Effective max length must account for prompt tokens
    eff_max_length = max_length - num_specials if add_special_tokens else max_length
    eff_max_length_for_sentences = eff_max_length - num_prompt_tokens

    sentence_ids = _split_long_sentences(sentence_ids, eff_max_length_for_sentences)
    sent_lengths = torch.bincount(sentence_ids[sentence_ids != -1])
    unique_sent_ids = torch.unique(sentence_ids[sentence_ids != -1])

    chunks: list[list[int]] = []
    current_chunk: list[int] = []

    def current_length() -> torch.Tensor:
        return sent_lengths[current_chunk].sum()

    next_sent_id = 0
    while next_sent_id in unique_sent_ids:
        if current_length() + sent_lengths[next_sent_id] <= eff_max_length:
            current_chunk.append(next_sent_id)
        elif current_chunk:
            chunks.append(current_chunk)
            overlap_start = get_overlap_count(overlap, len(current_chunk))
            current_chunk = current_chunk[overlap_start:] + [next_sent_id]
        else:
            raise RuntimeError(f"Sentence ID {next_sent_id} exceeds max_length {max_length}.")
        next_sent_id += 1
    # Process the last chunk
    if current_chunk:
        chunks.append(current_chunk)
        del current_chunk
    # Create token IDs and sentence IDs for each chunk
    chunked_input_ids_tensors: list[torch.Tensor] = []
    chunked_sentence_ids_tensors: list[torch.Tensor] = []
    for chunk in chunks:
        mask = torch.isin(sentence_ids, torch.tensor(chunk))
        chunk_input_ids = torch.tensor(input_ids)[mask]
        chunk_sentence_ids_tensor = sentence_ids[mask]

        # Prepend prompt tokens (if any) to each chunk
        if prompt_input_ids is not None and prompt_sentence_ids is not None:
            chunk_input_ids = torch.cat([prompt_input_ids, chunk_input_ids])
            chunk_sentence_ids_tensor = torch.cat([prompt_sentence_ids, chunk_sentence_ids_tensor])

        chunk_length = len(chunk_input_ids)
        if chunk_length > eff_max_length:
            # Truncate the chunk to fit within the max_length
            # and raise warning
            chunk_input_ids = chunk_input_ids[:eff_max_length]
            chunk_sentence_ids_tensor = chunk_sentence_ids_tensor[:eff_max_length]
            warnings.warn(
                f"Chunk length {chunk_length} exceeds effective max_length {eff_max_length}. "
                "Truncating to fit.",
                stacklevel=2,
            )
        if add_special_tokens:
            chunk_input_ids = _add_special_tokens(chunk_input_ids, cls_token_id, sep_token_id)
            # Extend first sentence ID to include CLS token
            # and last sentence ID to include SEP token
            # For CLS: use -1 if we have prompt tokens, otherwise use first sentence's ID
            cls_sentence_id = -1 if prompt_input_ids is not None else chunk_sentence_ids_tensor[0]
            chunk_sentence_ids_tensor = torch.cat(
                [
                    torch.tensor([cls_sentence_id]),
                    chunk_sentence_ids_tensor,
                    torch.tensor([chunk_sentence_ids_tensor[-1]]),
                ]
            )
        chunked_input_ids_tensors.append(chunk_input_ids)
        if reset_sentence_ids_on_overflow:
            # Reset sentence IDs to start from 0 for each chunk
            chunk_sentence_ids_tensor = chunk_sentence_ids_tensor - chunk_sentence_ids_tensor[0]
        chunked_sentence_ids_tensors.append(chunk_sentence_ids_tensor)

    results: dict[str, Any]
    if padding is None:
        chunked_input_ids_lists = [x.tolist() for x in chunked_input_ids_tensors]
        chunked_sentence_ids_lists = [x.tolist() for x in chunked_sentence_ids_tensors]
        attention_mask_lists = [
            [tok != tokenizer.pad_token_id for tok in chunk] for chunk in chunked_input_ids_lists
        ]
        overflow_to_sample_mapping_list = [sample_idx] * len(chunked_input_ids_lists)
        results = {
            "input_ids": chunked_input_ids_lists,
            "attention_mask": attention_mask_lists,
            "overflow_to_sample_mapping": overflow_to_sample_mapping_list,
            "sentence_ids": chunked_sentence_ids_lists,
        }
    else:
        # Pad the chunks
        padded_chunks = pad(
            chunked_input_ids_tensors,
            tokenizer.pad_token_id,
            strategy=padding,
            max_length=max_length,
        )
        padded_sentence_ids = pad(
            chunked_sentence_ids_tensors, -1, strategy=padding, max_length=max_length
        )
        # Create attention mask
        if isinstance(padded_chunks, torch.Tensor):
            attention_mask_tensor = padded_chunks.ne(tokenizer.pad_token_id)
            # Create overflow_to_sample_mapping
            overflow_to_sample_mapping_tensor = torch.full((len(padded_chunks),), sample_idx)
            results = {
                "input_ids": padded_chunks,
                "attention_mask": attention_mask_tensor,
                "overflow_to_sample_mapping": overflow_to_sample_mapping_tensor,
                "sentence_ids": padded_sentence_ids,
            }
        else:
            raise TypeError("Expected padded_chunks to be a Tensor when padding is not None")
    return results


def check_contiguous(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Check if all tensors in a dictionary are contiguous.

    Any non-contiguous tensors are converted to contiguous tensors.

    Parameters
    ----------
    tensors : dict[str, torch.Tensor]
        Dictionary of tensors to check.

    Raises
    ------
    ValueError
        If any tensor in the dictionary is not contiguous.
    """
    for key, tensor in tensors.items():
        if not tensor.is_contiguous():
            print(f"Tensor '{key}' is not contiguous. Converting to contiguous tensor.")
            tensors[key] = tensor.contiguous()
    return tensors


def check_tensors(
    tensors: dict[str, torch.Tensor], tokenizer: PreTrainedTokenizerBase
) -> dict[str, torch.Tensor]:
    """Check the shapes, dtypes, contiguity, and token IDs of tensors in a dictionary."""
    tensors = check_contiguous(tensors)
    if "input_ids" in tensors:
        # Check that IDs are within vocabulary range
        vocab_size = tokenizer.vocab_size
        if not torch.all((tensors["input_ids"] >= 0) & (tensors["input_ids"] < vocab_size)):
            raise ValueError("Input IDs contain values outside the vocabulary range.")
    return tensors


def get_chunk_idx(
    input_ids: torch.Tensor,
    sentence_ids: torch.Tensor,
    num_sents: list[int] | tuple[int, ...] | int,
    chunk_overlap: int | float | list[int] | dict[int, int] = 0.5,
    sequence_idx: torch.Tensor | None = None,
    pad_token_id: int = 0,
) -> dict[str, Any]:
    """Get chunk indices while preserving sentence structure.

    Parameters
    ----------
    input_ids : torch.Tensor
        Tensor containing input token IDs.
    sentence_ids : torch.Tensor
        Tensor containing sentence IDs, padded with -1.
    num_sents : list, tuple, or int
        Number of sentences per chunk.
    chunk_overlap : int, float, list, or dict, optional
        Overlap between chunks (number or fraction of sentences).
        Default is 0.5.
    sequence_idx : torch.Tensor or None, optional
        Tensor containing sequence indices. If None, a default sequence index is generated.
        Default is None.
    pad_token_id : int, optional
        Token ID used for padding chunk token IDs. Default is 0.

    Returns
    -------
    dict
        Dictionary containing the following keys:
        - "chunk_token_idx": 0-padded matrix of token indices within chunks.
        - "chunk_token_ids": Padded matrix of chunk token IDs.
        - "sentence_ids": Padded matrix of sentence IDs.
        - "attention_mask": Attention mask for the chunk indices.
        - "num_sents": Number of sentences in each chunk.
        - "sequence_idx": Sequence indices.

    Raises
    ------
    ValueError
        If `sequence_idx` length does not match the number of rows in `input_ids`.
        If `sentence_ids` length does not match the size of `input_ids`.

    Notes
    -----
    This function processes each sentence boundary index and extracts chunks of specified sizes,
    ensuring that the sentence structure is preserved. The overlap between chunks can be controlled
    using the `chunk_overlap` parameter.
    """
    if sequence_idx is None:
        sequence_idx = torch.arange(input_ids.size(0), device=input_ids.device)
    else:
        if len(sequence_idx) != input_ids.size(0):
            raise ValueError(
                f"`sequence_idx` must have the same number of rows as `input_ids` "
                f"({len(sequence_idx)} != {input_ids.size(0)})."
            )
        if sequence_idx.device != input_ids.device:
            sequence_idx = sequence_idx.to(input_ids.device)
    if len(sentence_ids) != input_ids.size(0):
        raise ValueError(
            f"Length of `sentence_ids` ({len(sentence_ids)}) "
            f"must match size of `input_ids` ({input_ids.size(0)})."
        )
    results: dict[str, Any] = {
        "chunk_token_idx": [],
        "chunk_token_ids": [],
        "sentence_ids": [],
        "num_sents": [],
        "sequence_idx": [],
        "chunk_idx": [],
    }
    if isinstance(num_sents, int):
        num_sents = [num_sents]
    # Process sequences first, then chunk sizes within each sequence
    # chunk_idx resets for each sequence (per-document indexing)
    for seq_input_ids, seq_sentence_ids, seq_idx in zip(
        input_ids, sentence_ids, sequence_idx, strict=False
    ):
        unique_sent_ids = torch.unique(seq_sentence_ids[seq_sentence_ids != -1])
        chunk_counter = 0  # Reset for each sequence
        for i, size in enumerate(num_sents):
            overlap_sents = get_overlap_count(chunk_overlap, size, i)
            step = size - overlap_sents
            eff_size = min(size, len(unique_sent_ids))
            chunk_sent_ids = unique_sent_ids.unfold(0, eff_size, step)
            chunk_masks = [torch.isin(seq_sentence_ids, chunk) for chunk in chunk_sent_ids]
            chunk_token_ids = [seq_input_ids[mask] for mask in chunk_masks]
            chunk_token_idx = [
                torch.nonzero(mask, as_tuple=False).squeeze() for mask in chunk_masks
            ]
            n_chunks = len(chunk_token_ids)
            results["chunk_token_ids"].extend(chunk_token_ids)
            results["sentence_ids"].extend([seq_sentence_ids[mask] for mask in chunk_masks])
            results["chunk_token_idx"].extend(chunk_token_idx)
            results["num_sents"].append(torch.full((n_chunks,), size))
            results["sequence_idx"].append(seq_idx.repeat_interleave(n_chunks))
            results["chunk_idx"].append(torch.arange(chunk_counter, chunk_counter + n_chunks))
            chunk_counter += n_chunks
    results["num_sents"] = torch.cat(results["num_sents"])
    results["sequence_idx"] = torch.cat(results["sequence_idx"])
    results["chunk_idx"] = torch.cat(results["chunk_idx"])
    results["attention_mask"] = pad_sequence(
        [torch.ones_like(p) for p in results["chunk_token_idx"]],
        batch_first=True,
        padding_value=0,
    ).to(input_ids.device)
    results["chunk_token_idx"] = pad_sequence(
        results["chunk_token_idx"], batch_first=True, padding_value=0
    ).to(input_ids.device)
    results["chunk_token_ids"] = pad_sequence(
        results["chunk_token_ids"], batch_first=True, padding_value=pad_token_id
    ).to(input_ids.device)
    results["sentence_ids"] = pad_sequence(
        results["sentence_ids"], batch_first=True, padding_value=-1
    ).to(input_ids.device)
    assert (
        results["chunk_token_idx"].size(0)
        == results["chunk_token_ids"].size(0)
        == results["sentence_ids"].size(0)
        == results["num_sents"].size(0)
        == results["sequence_idx"].size(0)
        == results["chunk_idx"].size(0)
    )
    return results


def _compute_chunk_embeds_slow(
    input_ids: torch.Tensor,
    token_embeds: torch.Tensor,
    sentence_ids: torch.Tensor,
    sequence_idx: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    num_sents: int | list[int] | tuple[int, ...] = 2,
    chunk_overlap: int | float | list[int] | dict[int, int] = 0.5,
    exclude_special_tokens: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Compute chunk embeddings (slow version).

    Parameters
    ----------
    input_ids : torch.Tensor
        Tensor containing input token IDs.
    token_embeds : torch.Tensor
        Tensor containing token embeddings.
    sentence_ids : list[torch.Tensor] or torch.Tensor
        Tensor containing sentence IDs, padded with -1.
    sequence_idx : torch.Tensor
        Tensor containing sequence indices.
    tokenizer : object
        Tokenizer object used to identify special tokens.
    num_sents : int or list or tuple, optional
        Number of sentences per chunk. Default is 2.
    chunk_overlap : int or float or list or dict, optional
        Overlap between chunks. Default is 0.5.
    exclude_special_tokens : bool, optional
        If True (default), exclude all special tokens from mean pooling.
        If False, include [CLS] in first chunk and [SEP] in last chunk
        of each sequence.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing the following keys:
        - "sequence_idx": Tensor of sequence indices.
        - "chunk_token_ids": Tensor of chunk token IDs.
        - "num_sents": Tensor of chunk sizes.
        - "chunk_embeds": Tensor of chunk embeddings.
    """
    chunk_data = get_chunk_idx(
        input_ids,
        sentence_ids,
        num_sents=num_sents,
        chunk_overlap=chunk_overlap,
        sequence_idx=sequence_idx,
    )

    if exclude_special_tokens:
        # Default behavior: mask all special tokens at input level
        special_tokens_mask = torch.isin(
            input_ids,
            torch.tensor(tokenizer.all_special_ids, device=input_ids.device),
            invert=True,
        ).to(torch.uint8)
    else:
        # Paper's approach: use boundary mask computed at chunk level
        valid_token_mask = _compute_boundary_special_token_mask(
            chunk_data, tokenizer, input_ids.device
        )

    results = {
        "sequence_idx": chunk_data["sequence_idx"],
        "chunk_token_ids": chunk_data["chunk_token_ids"],
        "num_sents": chunk_data["num_sents"],
        "chunk_embeds": [],
    }

    if exclude_special_tokens:
        masked_token_embeds = token_embeds * special_tokens_mask.unsqueeze(-1)
        for seq_idx, chunk_token_idx in zip(
            chunk_data["sequence_idx"], chunk_data["chunk_token_idx"], strict=False
        ):
            divisor = special_tokens_mask[seq_idx, chunk_token_idx].sum().clamp(min=1)
            embed = masked_token_embeds[seq_idx, chunk_token_idx].sum(dim=0) / divisor
            results["chunk_embeds"].append(embed)
    else:
        # Use pre-computed boundary mask for each chunk
        for i, (seq_idx, chunk_token_idx) in enumerate(
            zip(chunk_data["sequence_idx"], chunk_data["chunk_token_idx"], strict=False)
        ):
            chunk_embeds = token_embeds[seq_idx, chunk_token_idx]
            chunk_mask = valid_token_mask[i, : len(chunk_token_idx)]
            divisor = chunk_mask.sum().clamp(min=1)
            embed = (chunk_embeds * chunk_mask.unsqueeze(-1)).sum(dim=0) / divisor
            results["chunk_embeds"].append(embed)

    results["chunk_embeds"] = torch.vstack(results["chunk_embeds"])
    assert (
        len(results["chunk_token_ids"])
        == results["chunk_embeds"].size(0)
        == results["sequence_idx"].numel()
        == results["num_sents"].numel()
    )
    return results


def _compute_boundary_special_token_mask(
    chunk_data: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> torch.Tensor:
    """
    Create mask that includes [CLS] in first chunk and [SEP] in last chunk
    of each sequence, excludes other special tokens.

    This implements the paper's recommendation to include boundary special tokens
    in the mean pooling for the first and last chunks of each sequence.

    Parameters
    ----------
    chunk_data : dict
        Dictionary from get_chunk_idx() containing chunk_token_ids, sequence_idx, etc.
    tokenizer : PreTrainedTokenizer
        Tokenizer used to identify special tokens.
    device : torch.device
        Device to place tensors on.

    Returns
    -------
    torch.Tensor
        Float mask of shape (num_chunks, max_chunk_len) where 1.0 means include token.
    """
    chunk_token_ids = chunk_data["chunk_token_ids"]
    sequence_idx = chunk_data["sequence_idx"]
    num_chunks, max_chunk_len = chunk_token_ids.shape

    # Get special token IDs
    cls_token_id = tokenizer.cls_token_id or tokenizer.bos_token_id
    sep_token_id = tokenizer.sep_token_id or tokenizer.eos_token_id
    all_special_ids = set(tokenizer.all_special_ids)

    # Start with mask that excludes all special tokens
    is_special = torch.zeros(num_chunks, max_chunk_len, dtype=torch.bool, device=device)
    for special_id in all_special_ids:
        is_special |= chunk_token_ids == special_id

    # Find first and last chunk indices for each sequence
    # We need to identify which chunks are "first" or "last" within their sequence
    unique_seqs = torch.unique(sequence_idx)
    is_first_chunk = torch.zeros(num_chunks, dtype=torch.bool, device=device)
    is_last_chunk = torch.zeros(num_chunks, dtype=torch.bool, device=device)

    for seq in unique_seqs:
        seq_mask = sequence_idx == seq
        seq_indices = torch.where(seq_mask)[0]
        if len(seq_indices) > 0:
            is_first_chunk[seq_indices[0]] = True
            is_last_chunk[seq_indices[-1]] = True

    # For first chunks: allow [CLS] token (typically first position)
    if cls_token_id is not None:
        is_cls = chunk_token_ids == cls_token_id
        # Only allow CLS in first chunks
        allow_cls = is_first_chunk.unsqueeze(1) & is_cls
        is_special = is_special & ~allow_cls

    # For last chunks: allow [SEP] token (typically last non-pad position)
    if sep_token_id is not None:
        is_sep = chunk_token_ids == sep_token_id
        # Only allow SEP in last chunks
        allow_sep = is_last_chunk.unsqueeze(1) & is_sep
        is_special = is_special & ~allow_sep

    # Return mask where 1.0 = include, 0.0 = exclude
    valid_token_mask = (~is_special).float()
    return valid_token_mask


@torch.inference_mode()
def _compute_chunk_embeds(
    input_ids: torch.Tensor,
    token_embeds: torch.Tensor,
    sentence_ids: torch.Tensor,
    sequence_idx: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    num_sents: int | list[int] | tuple[int, ...] = 2,
    chunk_overlap: int | float | list[int] | dict[int, int] = 0.5,
    exclude_special_tokens: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute chunk embeddings via vectorized mean-pooling of token embeddings.

    This is the core function implementing late chunking. It extracts embeddings
    for sentence groups (chunks) by mean-pooling token embeddings within sentence
    boundaries. Uses advanced indexing for efficient vectorized computation.

    The function groups consecutive sentences into chunks, then averages the token
    embeddings for all tokens within each chunk.

    Parameters
    ----------
    input_ids : torch.Tensor
        Input token IDs of shape (batch_size, seq_len). Contains tokenized text
        including special tokens.
    token_embeds : torch.Tensor
        Token embeddings of shape (batch_size, seq_len, hidden_size). Output from
        the transformer model's last hidden state.
    sentence_ids : torch.Tensor
        Sentence IDs of shape (batch_size, seq_len). Each token is labeled with its
        sentence ID (0, 1, 2, ...). Padding positions are marked with -1.
    sequence_idx : torch.Tensor
        Sequence indices of shape (batch_size,). Maps each sequence to its position
        in the batch (used for tracking across chunks).
    tokenizer : PreTrainedTokenizerBase
        Tokenizer for identifying special tokens (CLS, SEP, PAD).
    num_sents : int, list[int], or tuple[int, ...], optional
        Number of sentences per chunk. Can be a single int or a list/tuple to
        extract multiple chunk sizes simultaneously, by default 2.
        Example: [1, 2, 3] extracts 1-sentence, 2-sentence, and 3-sentence chunks.
    chunk_overlap : int, float, list[int], or dict[int, int], optional
        Overlap between consecutive chunks in number of sentences, by default 0.5.
        - float: Fraction of chunk size (e.g., 0.5 means 50% overlap)
        - int: Absolute number of sentences to overlap
        - list: Overlap values corresponding to each value in num_sents
        - dict: Maps chunk size to overlap count
    exclude_special_tokens : bool, optional
        How to handle special tokens during mean pooling, by default True.
        - True (default): Exclude all special tokens from mean pooling.
        - False: Include [CLS] in first chunk and [SEP] in last chunk
          of each sequence.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing chunk data and embeddings with keys:
        - "sequence_idx" (torch.Tensor): Sequence index for each chunk, shape (num_chunks,)
        - "chunk_idx" (torch.Tensor): Chunk index within document, shape (num_chunks,)
        - "chunk_token_ids" (torch.Tensor): Token IDs for each chunk (padded), shape (num_chunks, max_chunk_len)
        - "sentence_ids" (torch.Tensor): Sentence IDs for each chunk (padded), shape (num_chunks, max_chunk_len)
        - "num_sents" (torch.Tensor): Number of sentences in each chunk, shape (num_chunks,)
        - "chunk_embeds" (torch.Tensor): Mean-pooled chunk embeddings, shape (num_chunks, hidden_size)

    Notes
    -----
    - Uses vectorized advanced indexing for efficient batch processing
    - Handles variable-length chunks through padding
    - Preserves sentence boundaries in all chunks
    - Mean pooling weights all non-padding, non-excluded tokens equally

    Examples
    --------
    >>> # Assume we have token embeddings from a model
    >>> results = _compute_chunk_embeds(
    ...     input_ids=batch["input_ids"],
    ...     token_embeds=model_output.last_hidden_state,
    ...     sentence_ids=batch["sentence_ids"],
    ...     sequence_idx=batch["sequence_idx"],
    ...     tokenizer=tokenizer,
    ...     num_sents=2,
    ...     chunk_overlap=0.5
    ... )
    >>> chunk_embeddings = results["chunk_embeds"]  # Shape: (num_chunks, hidden_size)
    """
    # Get the chunk grouping information
    chunk_data = get_chunk_idx(
        input_ids,
        sentence_ids,
        num_sents=num_sents,
        chunk_overlap=chunk_overlap,
        sequence_idx=sequence_idx,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Compute the mask for valid tokens
    if exclude_special_tokens:
        # Default behavior: exclude all special tokens
        valid_token_mask = torch.isin(
            chunk_data["chunk_token_ids"],
            torch.tensor(tokenizer.all_special_ids, device=input_ids.device),
            invert=True,
        ).float()
    else:
        # Paper's approach: include [CLS] in first chunk, [SEP] in last chunk
        valid_token_mask = _compute_boundary_special_token_mask(
            chunk_data, tokenizer, input_ids.device
        )

    # -----------------------------------------------------------
    # Use advanced, vectorized indexing to select tokens
    # corresponding to each chunk.
    #
    # token_embeds:         [batch, tokens, embed_dim]
    # batch_sequence_idx:  [num_chunks, 1]
    # chunk_token_idx:    [num_chunks, max_chunk_len]
    #
    # The resulting tensor will have shape:
    # [num_chunks, max_chunk_len, embed_dim]
    # -----------------------------------------------------------
    # Create a mapping from original sequence indices to their batch positions.
    # IMPORTANT: We must use the order that sequences appear in the input batch,
    # NOT sorted order. token_embeds[i] contains embeddings for sequence_idx[i].
    sequence_to_batch_pos = {seq.item(): idx for idx, seq in enumerate(sequence_idx)}
    batch_sequence_idx = torch.tensor(
        [sequence_to_batch_pos[seq.item()] for seq in chunk_data["sequence_idx"]],
        device=chunk_data["sequence_idx"].device,
    ).unsqueeze(1)

    chunk_token_embeds = token_embeds[batch_sequence_idx, chunk_data["chunk_token_idx"]]

    # Sum the embeddings over the token dimension
    # (taking into account only valid positions)
    summed_embeds = torch.sum(chunk_token_embeds * valid_token_mask.unsqueeze(-1), dim=1)

    # Compute the divisor: sum of mask values, clamp to at least one.
    valid_token_count = valid_token_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)

    chunk_embeds = summed_embeds / valid_token_count

    results = {
        "sequence_idx": chunk_data["sequence_idx"],
        "chunk_idx": chunk_data["chunk_idx"],
        "chunk_token_ids": chunk_data["chunk_token_ids"],
        "sentence_ids": chunk_data["sentence_ids"],
        "num_sents": chunk_data["num_sents"],
        "chunk_embeds": chunk_embeds,
    }
    assert (
        results["chunk_token_ids"].size(0)
        == results["sentence_ids"].size(0)
        == results["chunk_embeds"].size(0)
        == results["sequence_idx"].numel()
        == results["num_sents"].numel()
        == results["chunk_idx"].numel()
    )
    return results


def _as_sentence_ids(
    sent_boundary_idx: torch.Tensor,
    max_length: int | None = None,
    pad_value: int = -1,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """
    Converts sentence boundary indices to token sentence IDs.

    Parameters
    ----------
    sent_boundary_idx : torch.Tensor
        A tensor of shape (N, 2) where N is the number of sentences. Each row
        contains the start and end indices of a sentence.
    max_length : int, optional
        The maximum length of the token sequence. If None, it will be set to
        the end index of the last sentence. Default is None.
    pad_value : int, optional
        The value to use for padding tokens that do not belong to any sentence.
        Default is -1.
    dtype : torch.dtype, optional
        The data type of the output tensor. Default is torch.int32.

    Returns
    -------
    torch.Tensor
        A tensor of shape (max_length,) where each element is the sentence ID
        of the corresponding token. Tokens that do not belong to any sentence
        are filled with `pad_value`.

    Examples
    --------
    >>> import torch
    >>> sent_boundary_idx = torch.tensor([[0, 5], [5, 10]])
    >>> _as_sentence_ids(sent_boundary_idx)
    tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.int32)
    """
    eff_max_length: int
    if max_length is None:
        eff_max_length = int(sent_boundary_idx[-1, 1].item())
    else:
        eff_max_length = max_length
    token_type_ids = torch.full((eff_max_length,), pad_value, dtype=dtype)
    for i, (start, end) in enumerate(sent_boundary_idx):
        token_type_ids[start:end] = i
    return token_type_ids.contiguous()


@torch.inference_mode()
def tokenize_with_sentence_boundaries(
    docs: list[str],
    tokenizer: PreTrainedTokenizerBase,
    method: str = "blingfire",
    max_length: int | None = 512,
    prechunk: bool = True,
    prechunk_overlap: float | int = 0.5,
    return_tokenized_dataset: bool = False,
    batch_size: int = 10,
    n_jobs: int | None = None,
    show_progress: bool = True,
    prompt: str | None = None,
) -> dict[str, Any] | tuple[TokenizedDataset, list[list[str]]]:
    """Tokenize documents while preserving sentence boundaries for late chunking.

    This function performs sentence-aware tokenization, which is critical for late
    chunking. It detects sentence boundaries in the original text, tokenizes the
    documents, then maps token positions back to sentence IDs. Long documents are
    optionally split into overlapping sequences while preserving complete sentences.

    The sentence boundary information enables extracting chunk embeddings from
    groups of consecutive sentences after encoding the full document context.

    Parameters
    ----------
    docs : list[str]
        Documents to tokenize. Can be of any length.
    tokenizer : transformers.PreTrainedTokenizerBase
        HuggingFace tokenizer (fast tokenizer recommended for offset mapping).
    method : str, optional
        Sentence boundary detection method, by default "blingfire".
        Options: "blingfire" (fast, recommended), "nltk" (accurate),
        "pysbd" (handles abbreviations), "syntok" (sophisticated).
    max_length : int or None, optional
        Maximum sequence length in tokens, by default 512.
        If None, uses tokenizer.model_max_length.
    prechunk : bool, optional
        Whether to split documents exceeding max_length into overlapping sequences,
        by default True. If False, documents are truncated at max_length.
    prechunk_overlap : float or int, optional
        Overlap for splitting long documents into sequences, by default 0.5.
        - float in [0, 1): Fraction of max_length to overlap
        - int: Absolute number of sentences to overlap
        Only relevant when prechunk=True.
    return_tokenized_dataset : bool, optional
        Return format, by default False.
        - True: Returns (TokenizedDataset, list[list[str]]) where second element
          contains original sentence texts for text reconstruction.
        - False: Returns dict with tokenization results.
    batch_size : int, optional
        Number of documents to process per batch during tokenization, by default 10.
        Smaller batches for longer documents reduce memory usage.
    n_jobs : int or None, optional
        Number of parallel jobs for tokenization, by default None.
        - None or 1: Sequential processing
        - -1: Use all CPU cores
        - n > 1: Use n parallel workers
    show_progress : bool, optional
        Whether to display progress bar during chunking, by default True.
    prompt : str | None, optional
        Prompt to prepend to each document, by default None. Used for instruct-style
        embedding models. Sentence detection is performed on the original document
        (without prompt), then prompt tokens are assigned sentence_id=-1 so they
        are excluded from chunk mean-pooling.

    Returns
    -------
    dict or tuple
        If return_tokenized_dataset=False (default):
            Dictionary with keys:
            - "input_ids" (list[list[int]]): Token IDs for each sequence
            - "overflow_to_sample_mapping" (list[int]): Maps sequences to original docs
            - "sentence_ids" (list[list[int]]): Sentence ID for each token (-1 for padding)
            - "sequence_idx" (list[int]): Unique index for each sequence

        If return_tokenized_dataset=True:
            Tuple of (TokenizedDataset, list[list[str]]) where:
            - TokenizedDataset: Sorted dataset ready for batching
            - list[list[str]]: Per-document sentence texts for reconstruction

    Notes
    -----
    - Sentence IDs are consecutive integers (0, 1, 2, ...) within each document
    - Padding tokens have sentence_id = -1
    - Prompt tokens (if prompt is provided) have sentence_id = -1
    - When prechunk=True, sentences are never split across sequence boundaries
    - Original sentence texts are preserved for later text reconstruction
    - Token offsets are used to map tokens back to sentences in the original text

    Examples
    --------
    Basic usage with sentence boundary preservation:

    >>> docs = ["First sentence. Second sentence.", "Another document."]
    >>> result = tokenize_with_sentence_boundaries(docs, tokenizer)
    >>> result.keys()
    dict_keys(['input_ids', 'overflow_to_sample_mapping', 'sentence_ids', 'sequence_idx'])

    Get dataset and sentence texts for late chunking:

    >>> dataset, sentence_texts = tokenize_with_sentence_boundaries(
    ...     docs,
    ...     tokenizer,
    ...     return_tokenized_dataset=True
    ... )
    >>> len(sentence_texts)  # One list per document
    2
    >>> sentence_texts[0]  # Sentences from first document
    ['First sentence.', 'Second sentence.']

    Handle long documents with overlapping sequences:

    >>> long_docs = ["Very long document..." * 1000]
    >>> result = tokenize_with_sentence_boundaries(
    ...     long_docs,
    ...     tokenizer,
    ...     max_length=512,
    ...     prechunk=True,
    ...     prechunk_overlap=0.5
    ... )

    With instruct-style prompt:

    >>> docs = ["What is machine learning?"]
    >>> result = tokenize_with_sentence_boundaries(
    ...     docs,
    ...     tokenizer,
    ...     prompt="Represent this question for retrieval: "
    ... )
    """
    # Determine prompt offset for adjusting character positions
    prompt_char_offset = len(prompt) if prompt is not None else 0

    # Prepend prompt to documents for tokenization (if provided)
    docs_for_tokenization = [prompt + doc for doc in docs] if prompt else docs

    # Tokenize the documents using the provided tokenizer.
    # We disable truncation and padding at this stage to retain full document context.
    # Special tokens are also disabled, but will be added later.
    tokenize_max_length: int | None = max_length if not prechunk else None
    inputs_raw = tokenize_docs(
        docs_for_tokenization,
        tokenizer,  # type: ignore[arg-type]
        max_length=tokenize_max_length,
        truncation=not prechunk,
        add_special_tokens=not prechunk,
        return_attention_mask=False,
        return_offsets_mapping=True,
        prechunk=False,
        batch_size=batch_size,
        n_jobs=n_jobs,
    )
    if not isinstance(inputs_raw, dict):
        raise TypeError("Expected tokenize_docs to return a dict")
    inputs: dict[str, Any] = inputs_raw
    # Get sentence offsets for each document using the specified method.
    # Sentence detection is performed on original documents (without prompt).
    sent_offsets = get_sentence_offsets(docs, method=method)

    # Adjust sentence offsets by prompt length so they align with tokenized text.
    # This way, prompt tokens will have character offsets before any sentence
    # and will naturally get sentence_id=-1.
    if prompt_char_offset > 0:
        sent_offsets = [offsets + prompt_char_offset for offsets in sent_offsets]

    token_offsets = [torch.as_tensor(x) for x in inputs["offset_mapping"]]

    # Extract original sentence texts using character offsets (per-document, not per-sequence).
    # This allows reconstructing chunk text without detokenization later.
    # Note: We use the original docs (without prompt) for sentence text extraction.
    original_sent_offsets = get_sentence_offsets(docs, method=method)
    sentence_texts = [
        [doc[start:end] for start, end in offsets.tolist()]
        for doc, offsets in zip(docs, original_sent_offsets, strict=False)
    ]

    # Initialize a dictionary to store the results.
    results: dict[str, Any] = {
        "input_ids": [],
        "overflow_to_sample_mapping": [],
        "sentence_ids": [],
    }

    # Iterate through each document and its corresponding tokenization and sentence offsets.
    for i, (
        input_ids,
        current_token_offsets,
        current_sent_offsets,
    ) in enumerate(
        zip(
            tqdm(inputs["input_ids"], desc="Chunking", disable=not show_progress),
            token_offsets,
            sent_offsets,
            strict=False,
        )
    ):
        # Remove offsets for pad tokens
        current_token_offsets = current_token_offsets
        # Use searchsorted to find the indices in token offsets that correspond to the start of each sentence.
        start_idx = torch.searchsorted(
            current_token_offsets[:, 0].contiguous(),
            current_sent_offsets[:, 0].contiguous(),
            side="left",
        )
        # Use searchsorted to find the indices in token offsets that correspond to the end of each sentence.
        stop_idx = torch.searchsorted(
            current_token_offsets[:, 1].contiguous(),
            current_sent_offsets[:, 1].contiguous(),
            side="right",
        )
        # Stack the start and end indices to create a tensor representing sentence boundaries.
        sent_boundary_idx = torch.stack([start_idx, stop_idx], dim=1)
        # Check for gaps between sentences
        if (sent_boundary_idx[1:, 0] - sent_boundary_idx[:-1, 1]).any():
            warnings.warn(
                "Gaps between sentences detected. "
                "Please ensure sentence boundaries are correctly identified.",
                stacklevel=2,
            )
        sentence_ids = _as_sentence_ids(sent_boundary_idx)

        # Split long docs before model if needed.
        if prechunk:
            if max_length is None:
                resolved_max_length = get_max_length(None, tokenizer, required=True)
            else:
                resolved_max_length = max_length
            chunked_inputs = chunk_preserving_sentence_structure(
                input_ids,
                sentence_ids,
                tokenizer,
                sample_idx=i,
                max_length=resolved_max_length,
                padding=None,
                overlap=prechunk_overlap,
                add_special_tokens=True,
                reset_sentence_ids_on_overflow=False,
            )
            # Append the chunked inputs to the results dictionary.
            results["input_ids"].extend(chunked_inputs["input_ids"])
            results["overflow_to_sample_mapping"].extend(
                chunked_inputs["overflow_to_sample_mapping"]
            )
            results["sentence_ids"].extend(chunked_inputs["sentence_ids"])
        else:
            # Add the input IDs, attention mask, and sentence boundary indices to the results dictionary.
            # Convert tensors to lists for TokenizedDataset compatibility
            input_ids_list = (
                input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
            )
            sentence_ids_list = (
                sentence_ids.tolist() if isinstance(sentence_ids, torch.Tensor) else sentence_ids
            )
            results["input_ids"].append(input_ids_list)
            results["sentence_ids"].append(sentence_ids_list)
            results["overflow_to_sample_mapping"].append(i)

    results["sequence_idx"] = list(range(len(results["input_ids"])))
    # Add sentence_texts for text reconstruction without detokenization.
    # This is per-document, not per-sequence, so must be extracted before creating TokenizedDataset.
    results["sentence_texts"] = sentence_texts
    # Return the results dictionary.
    if return_tokenized_dataset:
        # Extract sentence_texts before creating TokenizedDataset (different length)
        sentence_texts = results.pop("sentence_texts")
        return TokenizedDataset(results), sentence_texts
    return results


@delayed  # type: ignore[untyped-decorator]
def _tokenize_batch_with_sentence_boundaries(
    docs: list[str],
    sample_idx: list[int],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int | None = None,
    prechunk: bool = True,
    prechunk_overlap: float | int = 0.5,
) -> dict[str, Any]:
    """Tokenize a list of documents into input sequences for the model."""
    inputs_result = tokenize_with_sentence_boundaries(
        docs,
        tokenizer,
        method="blingfire",
        max_length=max_length,
        prechunk=prechunk,
        prechunk_overlap=prechunk_overlap,
    )
    if not isinstance(inputs_result, dict):
        raise TypeError("Expected tokenize_with_sentence_boundaries to return a dict")
    inputs: dict[str, Any] = inputs_result
    # Globalize overflow_to_sample_mapping
    if "overflow_to_sample_mapping" in inputs:
        inputs["overflow_to_sample_mapping"] = np.asarray(sample_idx)[
            inputs["overflow_to_sample_mapping"]
        ]
    return inputs
