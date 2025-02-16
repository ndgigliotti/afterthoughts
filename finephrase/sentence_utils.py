# Copyright 2024 Nicholas Gigliotti
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
import math

import blingfire as bf
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from torch.nn.utils.rnn import pad_sequence


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
        offsets = torch.tensor(
            [(s[0].offset, s[-1].offset + 1) for p in analyze(text) for s in p]
        )
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
    text: str | list[str], method: str = "blingfire", n_jobs: int = None
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
    input_ids: torch.Tensor, cls_token_id: int = None, sep_token_id: int = None
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
    print(f"Adding special tokens to input_ids: {input_ids}")
    to_cat = [input_ids]
    if cls_token_id is not None:
        print(f"Adding CLS token: {cls_token_id}")
        to_cat.insert(0, torch.tensor([cls_token_id]))
    if sep_token_id is not None:
        print(f"Adding SEP token: {sep_token_id}")
        to_cat.append(torch.tensor([sep_token_id]))
    result = torch.cat(to_cat)
    print(f"Result after adding special tokens: {result}")
    return result


def _pad(
    input_ids: list[torch.Tensor],
    pad_token_id: int,
    how: str = "longest",
    max_length: int = None,
) -> torch.Tensor | list[torch.Tensor]:
    """Pads a list of sequences to a uniform length.

    Parameters
    ----------
    input_ids : list of torch.Tensor
        A list of sequences to pad. Each sequence should be a list or a PyTorch tensor of numerical IDs.
    pad_token_id : int
        The ID to use for padding.
    how : str, optional
        The padding strategy to use. Can be one of the following:
        - 'longest': Pad all sequences to the length of the longest sequence in the list.
        - 'max_length': Pad all sequences to a specified maximum length. `max_length` must be provided.
        - None: No padding is applied. The input sequences are returned as is.
        (default: 'longest')
    max_length : int, optional
        The maximum length to pad sequences to when `how='max_length'`. Must be provided if `how='max_length'`. (default: None)

    Returns
    -------
    torch.Tensor or list of torch.Tensor
        A PyTorch tensor containing the padded sequences if padding is applied, or the original list of sequences if `how=None`.
        If padding is applied, the tensor has shape (len(input_ids), max_len), where max_len is the length of the longest sequence
        in `input_ids` or `max_length` if specified.

    Raises
    ------
    ValueError
        If `input_ids` is empty.
    ValueError
        If `how='max_length'` and `max_length` is not specified.
    ValueError
        If `how='max_length'` and any sequence in `input_ids` exceeds `max_length`.
    ValueError
        If `how` is not one of 'longest', 'max_length', or None.
    """
    print(
        f"Padding input_ids: {input_ids} with pad_token_id: {pad_token_id}, how: {how}, max_length: {max_length}"
    )
    if not len(input_ids):
        raise ValueError("Input list must not be empty.")
    if how == "max_length":
        if max_length is None:
            raise ValueError("max_length must be specified when how='max_length'.")
        if any([len(ids) > max_length for ids in input_ids]):
            raise ValueError("Input sequence length exceeds `max_length`.")
        # Pad the first sequence to `max_length`
        input_ids[0] = F.pad(
            input_ids[0], (0, max_length - len(input_ids[0])), value=pad_token_id
        )
        # Pad the remaining sequences to the length of the first sequence
        padded_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id
        )
    elif how == "longest":
        padded_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id
        )
    elif how is None:
        padded_ids = input_ids
    else:
        raise ValueError(f"Invalid value '{how}' for `how`.")
    print(f"Padded input_ids: {padded_ids}")
    return padded_ids


def _adjust_sent_boundary_idx(
    sent_boundary_idx: torch.Tensor,
    start_idx: int,
    chunk: torch.Tensor,
    current_chunk_sent_idx: list[int],
    cls_token_id: int = None,
    sep_token_id: int = None,
    add_special_tokens: bool = True,
) -> torch.Tensor:
    """Adjusts sentence boundary indices relative to a chunk of text.

    This function takes sentence boundary indices from a larger context and adjusts them to be relative to a smaller chunk of text.
    It also handles the addition of special tokens (CLS and SEP) if specified.

    Parameters
    ----------
    sent_boundary_idx : torch.Tensor
        A 2D tensor of shape (num_sentences, 2) containing the start and end indices of sentences in the original text.
    start_idx : int
        The starting index of the current chunk within the original text.
    chunk : torch.Tensor
        A sequence of token IDs representing the current chunk of text.
    current_chunk_sent_idx : list[int]
        A list of indices indicating which sentences from the original text are present in the current chunk.
    cls_token_id : int, optional
        The ID of the CLS token. If not None, it's assumed a CLS token is added at the beginning of the chunk.
    sep_token_id : int, optional
        The ID of the SEP token. If not None, it's assumed a SEP token is added at the end of the chunk.
    add_special_tokens : bool
        A boolean indicating whether special tokens (CLS and SEP) are added to the chunk.

    Returns
    -------
    torch.Tensor
        A 2D tensor of shape (num_sentences_in_chunk, 2) containing the adjusted start and end indices of sentences within the chunk.

    """
    print(f"Original sentence boundaries: {sent_boundary_idx}")
    print(f"Start index: {start_idx}")
    print(f"Chunk: {chunk}")
    print(f"Current chunk sentence indices: {current_chunk_sent_idx}")
    current_chunk_sent_idx = torch.as_tensor(current_chunk_sent_idx)
    adjusted_sent_boundary_idx = sent_boundary_idx[current_chunk_sent_idx] - start_idx
    print(
        f"Adjusted sentence boundaries after subtraction: {adjusted_sent_boundary_idx}"
    )
    if add_special_tokens:
        if cls_token_id is not None:
            adjusted_sent_boundary_idx += 1
        adjusted_sent_boundary_idx[0, 0] = 0
        if sep_token_id is not None:
            adjusted_sent_boundary_idx[-1, 1] = len(chunk)
        adjusted_sent_boundary_idx = adjusted_sent_boundary_idx.clamp(0, len(chunk))
    print(f"Final adjusted sentence boundaries: {adjusted_sent_boundary_idx}")
    return adjusted_sent_boundary_idx


def chunk_preserving_sentence_structure(
    input_ids: torch.Tensor,
    sent_boundary_idx: torch.Tensor,
    tokenizer: "transformers.PreTrainedTokenizer",
    sample_idx: int,
    max_length: int = 512,
    overlap: float = 0.5,
    add_special_tokens: bool = True,
    padding: str = "max_length",
) -> dict:
    """Chunk a sequence of input IDs while preserving sentence structure.
    This function splits a long sequence of input IDs into smaller chunks,
    ensuring that sentence boundaries are respected as much as possible.
    It also supports overlapping chunks and padding to a specified length.

    Parameters
    ----------
    input_ids : list of int
        List of input IDs representing the tokenized text.
    sent_boundary_idx : list of tuple of int
        List of tuples, where each tuple contains the start and end index
        of a sentence within the `input_ids`.
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
            - "sent_boundary_idx": list of list of tuple of int
                List of sentence boundary indices for each chunk, adjusted to the
                chunk's local indexing.

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
    - The `sent_boundary_idx` in the output is adjusted to be relative to the start of each chunk.
    """
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

    # Compute each sentence length (in tokens) from sent_boundary_idx.
    sent_lengths = [end - start for start, end in sent_boundary_idx]

    chunks = []
    chunk_boundaries = []
    current_chunk_sent_idx = []  # List of sentence indices for current chunk
    current_length = 0
    i = 0
    n_sent = len(sent_lengths)

    while i < n_sent:
        this_length = sent_lengths[i]
        # Always include at least one sentence per chunk.
        if current_length + this_length <= eff_max_length or not current_chunk_sent_idx:
            current_chunk_sent_idx.append(i)
            current_length += this_length
            i += 1
        else:
            # Build chunk from the current sentences.
            start_idx = sent_boundary_idx[current_chunk_sent_idx[0]][0]
            end_idx = sent_boundary_idx[current_chunk_sent_idx[-1]][1]
            chunk = input_ids[start_idx:end_idx]
            if add_special_tokens:
                chunk = _add_special_tokens(chunk, cls_token_id, sep_token_id)
            # Ensure chunk length does not exceed max_length.
            if len(chunk) > max_length:
                chunk = chunk[:max_length]
            chunks.append(chunk)
            adjusted_sent_boundary_idx = _adjust_sent_boundary_idx(
                sent_boundary_idx,
                start_idx,
                chunk,
                current_chunk_sent_idx,
                cls_token_id,
                sep_token_id,
                add_special_tokens,
            )
            chunk_boundaries.append(adjusted_sent_boundary_idx)
            print(f"Chunk created: {chunk}")
            print(f"Adjusted sentence boundaries: {adjusted_sent_boundary_idx}")
            # Apply overlap: carry over a fraction of sentences to the next chunk.
            overlap_count = math.ceil(len(current_chunk_sent_idx) * overlap)
            # Make sure at least one sentence is carried over if possible.
            if overlap_count < 1 and len(current_chunk_sent_idx) > 1:
                overlap_count = 1
            # Reset current chunk with the overlapping sentences.
            if overlap_count:
                overlap_start = current_chunk_sent_idx[-overlap_count]
                current_chunk_sent_idx = list(
                    range(overlap_start, current_chunk_sent_idx[-1] + 1)
                )
                # Recompute the current_length based on the carried sentences.
                current_length = sum(sent_lengths[j] for j in current_chunk_sent_idx)
            else:
                current_chunk_sent_idx = []
                current_length = 0

    # Process any remaining sentences.
    if current_chunk_sent_idx:
        start_idx = sent_boundary_idx[current_chunk_sent_idx[0]][0]
        end_idx = sent_boundary_idx[current_chunk_sent_idx[-1]][1]
        chunk = input_ids[start_idx:end_idx]
        if add_special_tokens:
            chunk = _add_special_tokens(chunk, cls_token_id, sep_token_id)
        if len(chunk) > max_length:
            chunk = chunk[:max_length]
        chunks.append(chunk)
        adjusted_sent_boundary_idx = _adjust_sent_boundary_idx(
            sent_boundary_idx,
            start_idx,
            chunk,
            current_chunk_sent_idx,
            cls_token_id,
            sep_token_id,
            add_special_tokens,
        )
        chunk_boundaries.append(adjusted_sent_boundary_idx)
        print(f"Final chunk created: {chunk}")
        print(f"Final adjusted sentence boundaries: {adjusted_sent_boundary_idx}")

    padded_chunks = _pad(
        chunks, tokenizer.pad_token_id, how=padding, max_length=max_length
    )
    print(f"Padded chunks: {padded_chunks}")
    return {
        "input_ids": padded_chunks,
        "attention_mask": padded_chunks != tokenizer.pad_token_id,
        "overflow_to_sample_mapping": torch.full((len(padded_chunks),), sample_idx),
        "sent_boundary_idx": chunk_boundaries,
    }


def tokenize_with_sentence_boundaries(
    docs: list[str],
    tokenizer: "transformers.PreTrainedTokenizer",
    method: str = "blingfire",
    max_length: int = 512,
    overlap: float = 0.5,
) -> dict:
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
    overlap : float, optional
        The amount of overlap between adjacent chunks, by default 0.5.

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
        *   `sent_boundary_idx`: list of torch.Tensor
            A list of tensors, where each tensor contains the start and end
            indices of sentences within the chunks.
    """
    # Tokenize the documents using the provided tokenizer.
    # We disable truncation and padding at this stage to retain full document context.
    # Special tokens are also disabled, but will be added later.
    inputs = tokenizer(
        docs,
        return_tensors="pt",
        return_overflowing_tokens=False,
        truncation=False,
        padding="longest",
        max_length=torch.inf,
        stride=None,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    # Get sentence offsets for each document using the specified method.
    sent_offsets = get_sentence_offsets(docs, method=method)

    # Initialize a dictionary to store the results.
    results = {
        "input_ids": [],
        "attention_mask": [],
        "overflow_to_sample_mapping": [],
        "sent_boundary_idx": [],
    }

    # Iterate through each document and its corresponding tokenization and sentence offsets.
    for i, (
        input_ids,
        attention_mask,
        current_token_offsets,
        current_sent_offsets,
    ) in enumerate(
        zip(
            inputs["input_ids"],
            inputs["attention_mask"].bool(),
            inputs["offset_mapping"],
            sent_offsets,
        )
    ):
        print(f"Processing document {i}")
        # Remove offsets for pad tokens
        current_token_offsets = current_token_offsets[attention_mask]
        # Use searchsorted to find the indices in token offsets that correspond to the start of each sentence.
        start_idx = torch.searchsorted(
            current_token_offsets[:, 0], current_sent_offsets[:, 0], side="left"
        )
        # Use searchsorted to find the indices in token offsets that correspond to the end of each sentence.
        stop_idx = torch.searchsorted(
            current_token_offsets[:, 1], current_sent_offsets[:, 1], side="right"
        )
        # Stack the start and end indices to create a tensor representing sentence boundaries.
        sent_boundary_idx = torch.stack([start_idx, stop_idx], axis=1)
        print(f"Sentence boundaries: {sent_boundary_idx}")
        # Chunk the input IDs while preserving sentence structure.
        chunked_inputs = chunk_preserving_sentence_structure(
            input_ids[attention_mask],
            sent_boundary_idx,
            tokenizer,
            sample_idx=i,
            max_length=max_length,
            padding="max_length",
            overlap=overlap,
            add_special_tokens=True,
        )
        # Append the chunked inputs to the results dictionary.
        results["input_ids"].append(chunked_inputs["input_ids"])
        results["attention_mask"].append(chunked_inputs["attention_mask"])
        results["overflow_to_sample_mapping"].append(
            chunked_inputs["overflow_to_sample_mapping"]
        )
        results["sent_boundary_idx"].append(chunked_inputs["sent_boundary_idx"])

    # Concatenate the lists of tensors in the results dictionary to create single tensors.
    results["input_ids"] = torch.cat(results["input_ids"])
    results["attention_mask"] = torch.cat(results["attention_mask"])
    results["overflow_to_sample_mapping"] = torch.cat(
        results["overflow_to_sample_mapping"]
    )
    # Flatten the list of lists of sentence boundary indices into a single list.
    results["sent_boundary_idx"] = [y for x in results["sent_boundary_idx"] for y in x]

    # Return the results dictionary.
    return results
