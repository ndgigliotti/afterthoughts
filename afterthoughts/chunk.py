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
import logging
import warnings

import blingfire as bf
import numpy as np
import torch
import transformers
from joblib import Parallel, delayed
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

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
    """
    from syntok.segmenter import analyze

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
    LookupError
        If the NLTK 'punkt' tokenizer models are not found and cannot be downloaded.
    """
    import nltk

    # Check if punkt is installed
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
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
        Options are 'syntok', 'nltk', and 'blingfire'.
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
    get_sentence_offsets_syntok : Sentence offset detection using Syntok.
    get_sentence_offsets_nltk : Sentence offset detection using NLTK.
    get_sentence_offsets_blingfire : Sentence offset detection using Blingfire.
    """
    methods = {
        "syntok": get_sentence_offsets_syntok,
        "nltk": get_sentence_offsets_nltk,
        "blingfire": get_sentence_offsets_blingfire,
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


@torch.no_grad()
def _split_long_sentences(sentence_ids: torch.Tensor, max_length: int) -> list[torch.Tensor]:
    """Splits long sentences into smaller segments.

    Parameters
    ----------
    sentence_ids : torch.Tensor
        A tensor containing sentence IDs.
    max_length : int
        The maximum length for each segment.

    Returns
    -------
    list of torch.Tensor
        A list of tensors, each containing a segment of the original sentence IDs.
    """
    sent_lengths = torch.bincount(sentence_ids[sentence_ids != -1])
    if torch.any(sent_lengths > max_length):
        masks = [sentence_ids == i for i in range(len(sent_lengths))]
        for i in reversed(range(len(masks))):
            mask = masks[i]
            if mask.sum() > max_length:
                del masks[i]
                # Divide the sentence into smaller segments
                subsegment_id = torch.arange(0, mask.sum()) // max_length
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


@torch.no_grad()
def chunk_preserving_sentence_structure(
    input_ids: torch.Tensor,
    sentence_ids: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizer,
    sample_idx: int,
    max_length: int = 512,
    overlap: float = 0.5,
    add_special_tokens: bool = True,
    padding: str = "max_length",
    reset_sentence_ids_on_overflow: bool = False,
) -> dict:
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
    eff_max_length = max_length - num_specials if add_special_tokens else max_length
    sentence_ids = _split_long_sentences(sentence_ids, eff_max_length)
    sent_lengths = torch.bincount(sentence_ids[sentence_ids != -1])
    unique_sent_ids = torch.unique(sentence_ids[sentence_ids != -1])

    chunks = []
    current_chunk = []
    current_length = lambda: sent_lengths[current_chunk].sum()
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
    chunked_input_ids = []
    chunked_sentence_ids = []
    for chunk in chunks:
        mask = torch.isin(sentence_ids, torch.tensor(chunk))
        chunk_input_ids = torch.tensor(input_ids)[mask]
        chunk_sentence_ids = sentence_ids[mask]
        chunk_length = len(chunk_input_ids)
        if chunk_length > eff_max_length:
            # Truncate the chunk to fit within the max_length
            # and raise warning
            chunk_input_ids = chunk_input_ids[:eff_max_length]
            chunk_sentence_ids = chunk_sentence_ids[:eff_max_length]
            warnings.warn(
                f"Chunk length {chunk_length} exceeds effective max_length {eff_max_length}. "
                "Truncating to fit.",
                stacklevel=2,
            )
        if add_special_tokens:
            chunk_input_ids = _add_special_tokens(chunk_input_ids, cls_token_id, sep_token_id)
            # Extend first sentence ID to include CLS token
            # and last sentence ID to include SEP token
            chunk_sentence_ids = torch.cat(
                [
                    torch.tensor([chunk_sentence_ids[0]]),
                    chunk_sentence_ids,
                    torch.tensor([chunk_sentence_ids[-1]]),
                ]
            )
        chunked_input_ids.append(chunk_input_ids)
        if reset_sentence_ids_on_overflow:
            # Reset sentence IDs to start from 0 for each chunk
            chunk_sentence_ids = chunk_sentence_ids - chunk_sentence_ids[0]
        chunked_sentence_ids.append(chunk_sentence_ids)

    if padding is None:
        chunked_input_ids = [x.tolist() for x in chunked_input_ids]
        chunked_sentence_ids = [x.tolist() for x in chunked_sentence_ids]
        attention_mask = [chunk != tokenizer.pad_token_id for chunk in chunked_input_ids]
        overflow_to_sample_mapping = [sample_idx] * len(chunked_input_ids)
        results = {
            "input_ids": chunked_input_ids,
            "attention_mask": attention_mask,
            "overflow_to_sample_mapping": overflow_to_sample_mapping,
            "sentence_ids": chunked_sentence_ids,
        }
    else:
        # Pad the chunks
        padded_chunks = pad(
            chunked_input_ids,
            tokenizer.pad_token_id,
            strategy=padding,
            max_length=max_length,
        )
        padded_sentence_ids = pad(chunked_sentence_ids, -1, strategy=padding, max_length=max_length)
        # Create attention mask
        attention_mask = padded_chunks != tokenizer.pad_token_id
        # Create overflow_to_sample_mapping
        overflow_to_sample_mapping = torch.full((len(padded_chunks),), sample_idx)
        results = {
            "input_ids": padded_chunks,
            "attention_mask": attention_mask,
            "overflow_to_sample_mapping": overflow_to_sample_mapping,
            "sentence_ids": padded_sentence_ids,
        }
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


def check_tensors(tensors: dict[str, torch.Tensor], tokenizer) -> dict[str, torch.Tensor]:
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
    num_sents: list | tuple | int,
    chunk_overlap: int | float | list | dict = 0.5,
    sequence_idx: torch.Tensor | None = None,
    pad_token_id: int = 0,
) -> dict:
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
        - "chunk_size": Number of sentences in each chunk.
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
    results = {
        "chunk_token_idx": [],
        "chunk_token_ids": [],
        "sentence_ids": [],
        "chunk_size": [],
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
            results["chunk_size"].append(torch.full((n_chunks,), size))
            results["sequence_idx"].append(seq_idx.repeat_interleave(n_chunks))
            results["chunk_idx"].append(torch.arange(chunk_counter, chunk_counter + n_chunks))
            chunk_counter += n_chunks
    results["chunk_size"] = torch.cat(results["chunk_size"])
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
        == results["chunk_size"].size(0)
        == results["sequence_idx"].size(0)
        == results["chunk_idx"].size(0)
    )
    return results


def _compute_chunk_embeds_slow(
    input_ids: torch.Tensor,
    token_embeds: torch.Tensor,
    sentence_ids: list[torch.Tensor] | torch.Tensor,
    sequence_idx: torch.Tensor,
    tokenizer,
    num_sents: int | list | tuple = 2,
    chunk_overlap: int | float | list | dict = 0.5,
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

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing the following keys:
        - "sequence_idx": Tensor of sequence indices.
        - "chunk_token_ids": Tensor of chunk token IDs.
        - "chunk_size": Tensor of chunk sizes.
        - "chunk_embeds": Tensor of chunk embeddings.
    """
    chunk_data = get_chunk_idx(
        input_ids,
        sentence_ids,
        num_sents=num_sents,
        chunk_overlap=chunk_overlap,
        sequence_idx=sequence_idx,
    )
    # Mask all special tokens
    special_tokens_mask = torch.isin(
        input_ids,
        torch.tensor(tokenizer.all_special_ids, device=input_ids.device),
        invert=True,
    ).to(torch.uint8)
    results = {
        "sequence_idx": chunk_data["sequence_idx"],
        "chunk_token_ids": chunk_data["chunk_token_ids"],
        "chunk_size": chunk_data["chunk_size"],
        "chunk_embeds": [],
    }
    masked_token_embeds = token_embeds * special_tokens_mask.unsqueeze(-1)
    for seq_idx, segment_token_idx in zip(
        chunk_data["sequence_idx"], chunk_data["chunk_token_idx"], strict=False
    ):
        divisor = special_tokens_mask[seq_idx, segment_token_idx].sum().clamp(min=1)
        embed = masked_token_embeds[seq_idx, segment_token_idx].sum(dim=0) / divisor
        results["chunk_embeds"].append(embed)

    results["chunk_embeds"] = torch.vstack(results["chunk_embeds"])
    assert (
        len(results["chunk_token_ids"])
        == results["chunk_embeds"].size(0)
        == results["sequence_idx"].numel()
        == results["chunk_size"].numel()
    )
    return results


@torch.no_grad()
def _compute_chunk_embeds(
    input_ids: torch.Tensor,
    token_embeds: torch.Tensor,
    sentence_ids: torch.Tensor,
    sequence_idx: torch.Tensor,
    tokenizer,
    num_sents: int | list | tuple = 2,
    chunk_overlap: int | float | list | dict = 0.5,
) -> dict[str, torch.Tensor]:
    """
    Compute embeddings for chunks (sentence groups) using token embeddings.

    Parameters
    ----------
    input_ids : torch.Tensor
        Tensor containing input token IDs.
    token_embeds : torch.Tensor
        Tensor containing token embeddings.
    sentence_ids : torch.Tensor
        Tensor containing sentence IDs, padded with -1.
    sequence_idx : torch.Tensor
        Tensor containing sequence indices.
    tokenizer : PreTrainedTokenizer
        Tokenizer used to process the input text.
    num_sents : int or list or tuple, optional
        Number of sentences per chunk, by default 2.
    chunk_overlap : int or float or list or dict, optional
        Overlap between chunks (number or fraction of sentences),
        by default 0.5.
    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing the following keys:
        - "sequence_idx": Tensor of sequence indices for each chunk.
        - "chunk_token_ids": Tensor of chunk token IDs.
        - "chunk_size": Tensor of chunk sizes.
        - "chunk_embeds": Tensor of computed chunk embeddings.
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

    # Compute the mask for non-special tokens
    valid_token_mask = torch.isin(
        chunk_data["chunk_token_ids"],
        torch.tensor(tokenizer.all_special_ids, device=input_ids.device),
        invert=True,
    ).float()

    # -----------------------------------------------------------
    # Use advanced, vectorized indexing to select tokens
    # corresponding to each segment.
    #
    # token_embeds:         [batch, tokens, embed_dim]
    # batch_sequence_idx:  [num_segments, 1]
    # segment_token_idx:    [num_segments, max_segment_len]
    #
    # The resulting tensor will have shape:
    # [num_segments, max_segment_len, embed_dim]
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

    segment_embeds = summed_embeds / valid_token_count

    results = {
        "sequence_idx": chunk_data["sequence_idx"],
        "chunk_idx": chunk_data["chunk_idx"],
        "chunk_token_ids": chunk_data["chunk_token_ids"],
        "sentence_ids": chunk_data["sentence_ids"],
        "chunk_size": chunk_data["chunk_size"],
        "chunk_embeds": segment_embeds,
    }
    assert (
        results["chunk_token_ids"].size(0)
        == results["sentence_ids"].size(0)
        == results["chunk_embeds"].size(0)
        == results["sequence_idx"].numel()
        == results["chunk_size"].numel()
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
    if max_length is None:
        max_length = sent_boundary_idx[-1, 1].item()
    token_type_ids = torch.full((max_length,), pad_value, dtype=dtype)
    for i, (start, end) in enumerate(sent_boundary_idx):
        token_type_ids[start:end] = i
    return token_type_ids.contiguous()


@torch.no_grad()
def tokenize_with_sentence_boundaries(
    docs: list[str],
    tokenizer: transformers.PreTrainedTokenizer,
    method: str = "blingfire",
    max_length: int = 512,
    prechunk: bool = True,
    prechunk_overlap: float = 0.5,
    return_tokenized_dataset: bool = False,
    batch_size: int = 10,
    n_jobs: int | None = None,
    show_progress: bool = True,
) -> dict | TokenizedDataset:
    """Tokenizes documents while preserving sentence boundaries.
    This function takes a list of documents, a tokenizer, and optional parameters
    to tokenize the documents into chunks, ensuring that sentence boundaries are
    respected. It leverages sentence boundary detection to split the documents
    into sentences and then chunks the tokens while keeping sentences intact.

    Parameters
    ----------
    docs : list of str
        A list of documents to tokenize.
    tokenizer : transformers.PreTrainedTokenizer
        A pre-trained tokenizer from the transformers library.
    method : str, optional
        The method used for sentence boundary detection, by default "blingfire".
    max_length : int, optional
        The maximum length of each chunk, by default 512.
    prechunk : bool, optional
        Whether to split documents exceeding max_length before model, by default True.
    prechunk_overlap : float, optional
        The fraction of overlap between prechunked sequences, by default 0.5.
        Must be in the range [0, 1).
    return_tokenized_dataset : bool, optional
        Whether to return a TokenizedDataset instead of a dictionary, by default False.
    batch_size : int, optional
        The batch size for processing documents, by default 10.
    n_jobs : int, optional
        The number of parallel jobs to run for tokenization. If None, it uses
        sequential processing. If -1, it uses all available cores. Default is None.
    show_progress : bool, optional
        Show progress bar during chunking, by default True.


    Returns
    -------
    dict
        A dictionary containing the tokenized inputs, attention masks,
        overflow mappings, and sentence boundary indices. The dictionary has the
        following keys:
        *   `input_ids`: torch.Tensor
            A tensor containing the input token IDs.
        *   `attention_mask`: torch.Tensor
            A tensor containing the attention masks.
        *   `overflow_to_sample_mapping`: torch.Tensor
            A tensor mapping overflowing tokens to their original sample index.
        *   `sentence_ids`: list of torch.Tensor
            A list of tensors containing sentence IDs for each token.
    """
    # Tokenize the documents using the provided tokenizer.
    # We disable truncation and padding at this stage to retain full document context.
    # Special tokens are also disabled, but will be added later.
    inputs = tokenize_docs(
        docs,
        tokenizer,
        max_length=max_length if not prechunk else torch.inf,
        truncation=not prechunk,
        add_special_tokens=not prechunk,
        return_attention_mask=False,
        return_offsets_mapping=True,
        prechunk=False,
        batch_size=batch_size,
        n_jobs=n_jobs,
    )
    # Get sentence offsets for each document using the specified method.
    sent_offsets = get_sentence_offsets(docs, method=method)
    token_offsets = [torch.as_tensor(x) for x in inputs["offset_mapping"]]

    # Initialize a dictionary to store the results.
    results = {
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
        sent_boundary_idx = torch.stack([start_idx, stop_idx], axis=1)
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
            chunked_inputs = chunk_preserving_sentence_structure(
                input_ids,
                sentence_ids,
                tokenizer,
                sample_idx=i,
                max_length=max_length,
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
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()
            if isinstance(sentence_ids, torch.Tensor):
                sentence_ids = sentence_ids.tolist()
            results["input_ids"].append(input_ids)
            results["sentence_ids"].append(sentence_ids)
            results["overflow_to_sample_mapping"].append(i)

    results["sequence_idx"] = list(range(len(results["input_ids"])))
    # Return the results dictionary.
    return TokenizedDataset(results) if return_tokenized_dataset else results


@delayed
def _tokenize_batch_with_sentence_boundaries(
    docs: list[str],
    sample_idx: list[int],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int | None = None,
    prechunk: bool = True,
    prechunk_overlap: float = 0.5,
) -> dict:
    """Tokenize a list of documents into input sequences for the model."""
    inputs = tokenize_with_sentence_boundaries(
        docs,
        tokenizer,
        method="blingfire",
        max_length=max_length,
        prechunk=prechunk,
        prechunk_overlap=prechunk_overlap,
    )
    # Globalize overflow_to_sample_mapping
    if "overflow_to_sample_mapping" in inputs:
        inputs["overflow_to_sample_mapping"] = np.asarray(sample_idx)[
            inputs["overflow_to_sample_mapping"]
        ]
    return inputs
