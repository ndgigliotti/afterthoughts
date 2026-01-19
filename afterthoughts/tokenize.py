import logging
import math
import random
import warnings
from collections.abc import Iterator
from functools import cached_property
from types import MappingProxyType
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from afterthoughts.utils import (
    disable_tokenizer_parallelism,
    get_overlap_count,
    order_by_indices,
    round_up_to_power_of_2,
)

logger = logging.getLogger(__name__)


DEFAULT_PAD_VALUES = MappingProxyType(
    {
        "attention_mask": 0,
        "token_type_ids": 0,
        "special_tokens_mask": 0,
        "sentence_ids": -1,
    }
)


def _get_tokenization_batch_size(docs: list[str], sample_size: int = 100) -> int:
    """Determine tokenization batch size based on average document length.

    Smaller batches are used for longer documents to manage memory usage
    and improve work distribution across parallel workers.

    Parameters
    ----------
    docs : list[str]
        List of documents to be tokenized.
    sample_size : int, optional
        Maximum number of documents to sample for length estimation.
        Uses deterministic sampling based on dataset size. Default is 100.

    Returns
    -------
    int
        Recommended batch size for tokenization.
    """
    if not docs:
        return 10
    # Limit sample size to available documents
    sample_size = min(sample_size, len(docs))
    # Sample for large datasets to avoid iterating through all docs
    if len(docs) > sample_size:
        rng = random.Random(len(docs))  # Deterministic based on dataset size
        sample = rng.sample(docs, sample_size)
    else:
        sample = docs
    avg_doc_chars = sum(len(d) for d in sample) / len(sample)
    # Estimate: ~4 chars per token on average
    if avg_doc_chars > 50_000:  # ~12k tokens, book chapters / long articles
        return 1
    elif avg_doc_chars > 10_000:  # ~2.5k tokens, medium articles
        return 5
    else:
        return 10


class TokenizedDataset(Dataset[dict[str, Any]]):
    """Dataset for tokenized sequences with automatic sorting and batching support.

    This dataset stores tokenized sequences and automatically sorts them by token
    count (longest first) to enable efficient dynamic batching. This minimizes
    padding when creating batches and improves GPU utilization.

    Attributes
    ----------
    data : dict[str, list]
        Dictionary containing tokenized data with keys like 'input_ids',
        'sequence_idx', 'sentence_ids', and 'overflow_to_sample_mapping'.
        Each value is a list where each element corresponds to one sequence.
    sort_by_token_count : bool
        Whether the dataset has been sorted by token count (longest first).
    token_counts : list[int]
        Number of tokens in each sequence (cached property).
    sort_idx : list[int]
        Indices that sort sequences by token count descending (cached property).
    unsort_idx : list[int]
        Inverse mapping to restore original order after sorting (cached property).

    Examples
    --------
    Create a dataset from tokenized data:

    >>> data = {
    ...     "input_ids": [[1, 2, 3], [4, 5], [6, 7, 8, 9]],
    ...     "sequence_idx": [0, 1, 2]
    ... }
    >>> dataset = TokenizedDataset(data, sort_by_token_count=True)
    >>> len(dataset)
    3

    Access sequences in sorted order:

    >>> first_item = dataset[0]  # Longest sequence
    >>> len(first_item["input_ids"])
    4

    Notes
    -----
    - Sequences are sorted longest-first to minimize padding in batches
    - Use the unsort_idx property to restore results to original document order
    - All lists in data must have the same length (one entry per sequence)
    """

    def __init__(self, data: dict[str, list[Any]], sort_by_token_count: bool = True) -> None:
        """Initialize the TokenizedDataset.

        Parameters
        ----------
        data : dict[str, list]
            Dictionary containing tokenized data. Must include 'input_ids' key
            with a list of token ID lists. All values must be lists of the same length.
        sort_by_token_count : bool, optional
            Whether to sort sequences by token count (longest first), by default True.
            Sorting improves batching efficiency but requires using unsort_idx to
            restore original order.

        Raises
        ------
        ValueError
            If data validation fails (see validate_data for details).
        """
        self.data = data
        self.validate_data()
        self.sort_by_token_count = sort_by_token_count
        self.sort_data()

    def validate_data(self) -> None:
        """Validate the data in the dataset.

        Raises
        ------
        ValueError
            If the data is not a dictionary.
        ValueError
            If the data does not contain any tensors.
        ValueError
            If the data contains empty lists.
        ValueError
            If the data does not contain 'input_ids'.
        ValueError
            If the data contains lists of different lengths.
        """
        if not isinstance(self.data, dict):
            raise ValueError("Data must be a dictionary.")
        if not all(isinstance(val, list) for val in self.data.values()):
            raise ValueError("Data must contain only lists.")
        # Check non-empty
        if not all(len(val) > 0 for val in self.data.values()):
            raise ValueError("Data must contain only non-empty lists.")
        # Check consistent length
        lengths = [len(val) for val in self.data.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All lists in the data must have the same length.")
        if "input_ids" not in self.data:
            raise ValueError("Data must contain 'input_ids'.")
        if not isinstance(self.data["input_ids"], list) or not all(
            isinstance(x, list) for x in self.data["input_ids"]
        ):
            raise ValueError("'input_ids' must be a list of ragged lists.")
        if "sentence_ids" in self.data and (
            not isinstance(self.data["sentence_ids"], list)
            or not all(isinstance(x, list) for x in self.data["sentence_ids"])
        ):
            raise ValueError("'sentence_ids' must be a list of ragged lists.")

    def sort_data(self) -> None:
        """Sort the data by token count."""
        for key in self.data:
            if len(self.data[key]):
                self.data[key] = order_by_indices(self.data[key], self.sort_idx)
        self.token_counts = order_by_indices(self.token_counts, self.sort_idx)

    def keys(self) -> list[str]:
        """Return the keys of the dataset."""
        return list(self.data.keys())

    @cached_property
    def token_counts(self) -> list[int]:
        """Return the token counts of the dataset."""
        return [len(x) for x in self.data["input_ids"]]

    @cached_property
    def sort_idx(self) -> list[int]:
        """Return the indices for sorting the dataset by token count."""
        return sorted(
            range(len(self.token_counts)),
            key=lambda i: self.token_counts[i],
            reverse=True,
        )

    @cached_property
    def unsort_idx(self) -> list[int]:
        """Return the indices for unsorting the dataset.

        This creates an inverse mapping of sort_idx to restore the original order.
        For example, if sort_idx is [2, 0, 1] (meaning item 2 is longest, then 0, then 1),
        then unsort_idx should be [1, 2, 0] to restore the original order.
        """
        # Create a list of the same length as sort_idx
        result = [0] * len(self.sort_idx)
        # For each position in sort_idx, place the original index
        for new_pos, orig_idx in enumerate(self.sort_idx):
            result[orig_idx] = new_pos
        return result

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data["input_ids"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a single item from the sorted dataset."""
        return {key: self.data[key][idx] for key in self.keys()}

    def unsort_results(
        self, results: dict[str, torch.Tensor | np.ndarray[Any, Any] | list[Any]]
    ) -> dict[str, torch.Tensor | np.ndarray[Any, Any] | list[Any]]:
        """Unsort the results of the dataset."""
        unsorted: dict[str, torch.Tensor | np.ndarray[Any, Any] | list[Any]] = {}
        for key in results:
            val = results[key]
            if isinstance(val, list):
                if len(val):
                    unsorted[key] = order_by_indices(val, self.unsort_idx)
                else:
                    unsorted[key] = []
            elif isinstance(val, torch.Tensor):
                if len(val):
                    unsorted[key] = val[torch.tensor(self.unsort_idx)]
                else:
                    unsorted[key] = torch.empty(0)
            elif isinstance(val, np.ndarray):
                if len(val):
                    unsorted[key] = val[np.array(self.unsort_idx)]
                else:
                    unsorted[key] = torch.empty(0)
        return unsorted


class DynamicTokenSampler(Sampler[list[int]]):
    """Batch sampler that groups sequences by total token count for efficient GPU usage.

    This sampler creates variable-size batches where each batch contains up to
    max_tokens total tokens (accounting for padding). It assumes sequences are
    pre-sorted longest-first, which minimizes padding and maximizes GPU utilization.

    Traditional fixed-size batching wastes computation on padding tokens. This
    sampler ensures consistent computational load per batch by counting tokens
    rather than sequences.

    Attributes
    ----------
    data_source : TokenizedDataset
        Dataset containing tokenized sequences, sorted by token count.
    max_tokens : int
        Maximum total tokens per batch (including padding).
    token_counts : list[int]
        Token count for each sequence (alias for data_source.token_counts).
    batches : list[list[int]]
        Pre-computed list of batches, where each batch is a list of sequence indices.

    Examples
    --------
    Create a dataloader with dynamic batching:

    >>> dataset = TokenizedDataset(data, sort_by_token_count=True)
    >>> sampler = DynamicTokenSampler(dataset, max_tokens=8192)
    >>> loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)

    Batch sizes adapt to sequence lengths:

    >>> # Long sequences -> small batches (e.g., 4 sequences x 2048 tokens)
    >>> # Short sequences -> large batches (e.g., 64 sequences x 128 tokens)

    Notes
    -----
    - Sequences are padded to the next power of 2 for efficiency
    - Longest sequence in batch determines padding length
    - Pre-sorting is critical: random order would create inefficient batches
    - Total tokens = batch_size * next_power_of_2(max_seq_len_in_batch)
    """

    def __init__(
        self,
        data_source: TokenizedDataset,
        max_tokens: int,
    ) -> None:
        self.data_source = data_source
        if not isinstance(data_source, TokenizedDataset):
            raise ValueError("`data_source` must be a `TokenizedDataset`.")
        self.max_tokens = max_tokens

    @property
    def token_counts(self) -> list[int]:
        """Alias for `data_source.token_counts`."""
        return self.data_source.token_counts

    @cached_property
    def batches(self) -> list[list[int]]:
        """Return the list of batches of indices."""
        batch_list: list[list[int]] = []
        current_batch: list[int] = []
        current_longest = 0
        batch_token_counts = []  # Keep track of token counts for debugging

        # Use true indices for both token counting and batching
        for idx in range(len(self.data_source)):
            seq_tokens = self.token_counts[idx]
            current_longest = round_up_to_power_of_2(max(current_longest, seq_tokens))
            current_tokens = len(current_batch) * current_longest
            # If adding this sequence would exceed target, add current batch to batches
            if current_tokens + current_longest > self.max_tokens and current_batch:
                batch_list.append(current_batch)
                batch_token_counts.append(current_tokens)  # Store token count for this batch
                current_batch = []
                current_longest = 0

            current_batch.append(idx)  # Use true index

        # Add final batch if not empty
        if current_batch:
            batch_list.append(current_batch)
            batch_token_counts.append(current_tokens)  # Store token count for final batch

        logger.debug("Batch sizes: %s", [len(x) for x in batch_list])
        logger.debug("Batch token counts: %s", batch_token_counts)
        return batch_list

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.batches)

    def __len__(self) -> int:
        """Return the number of batches in the dataset."""
        return len(self.batches)


def get_max_length(
    max_length: int | None, tokenizer: PreTrainedTokenizerBase, required: bool = False
) -> int:
    """
    Get the maximum length for tokenization.

    Parameters
    ----------
    max_length : int or None
        The maximum length for tokenization. If None, the
        tokenizer's model_max_length will be used.
    tokenizer : PreTrainedTokenizer
        The tokenizer to use for determining the maximum length.
    required : bool, optional
        Whether the max_length is required. If True, raises an
        error if max_length is None and tokenizer does not have
        a model_max_length.

    Returns
    -------
    int
        The maximum length for tokenization.

    Raises
    ------
    ValueError
        If `max_length` is None and the tokenizer does not have
        a `model_max_length`.
    """
    if max_length is None:
        if tokenizer.model_max_length is None and required:
            raise ValueError(
                "The `max_length` parameter must be specified if the tokenizer does not have a `model_max_length`."
            )
        else:
            max_length = tokenizer.model_max_length
    return max_length


def _tokenize_batch(
    docs: list[str],
    sample_idx: list[int],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int | None = None,
    padding: bool | str = False,
    truncation: bool = True,
    return_overflowing_tokens: bool = True,
    stride: int = 0,
    return_tensors: str | None = None,
    add_special_tokens: bool = True,
    return_attention_mask: bool = True,
    return_offsets_mapping: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Tokenize a list of documents into input sequences for the model.

    Parameters
    ----------
    docs : list of str
        List of documents to be tokenized.
    sample_idx : list of int
        List of sample indices corresponding to the documents.
    tokenizer : PreTrainedTokenizer
        Tokenizer to be used for tokenizing the documents.
    max_length : int or None, optional
        Maximum length of the tokenized sequences. If None, the tokenizer's default max length is used.
    padding : str, optional
        Padding strategy to use. Default is False.
    truncation : bool, optional
        Whether to truncate sequences to the maximum length. Default is True.
    return_overflowing_tokens : bool, optional
        Whether to return overflowing tokens. Default is True.
    stride : int, optional
        Stride to use when handling overflowing tokens. Default is 0.
    return_tensors : str, optional
        Format of the returned tensors. Default is None.
    add_special_tokens : bool, optional
        Whether to add special tokens to the sequences. Default is True.
    return_attention_mask : bool, optional
        Whether to return the attention mask. Default is True.
    return_offsets_mapping : bool, optional
        Whether to return the offsets mapping. Default is False.
    **kwargs
        Additional keyword arguments to be passed to the tokenizer.

    Returns
    -------
    dict of str to torch.Tensor
        Dictionary containing the tokenized inputs and other relevant information.
    """
    max_length = get_max_length(max_length, tokenizer, required=padding == "max_length")
    idx_arr: list[int] | np.ndarray[Any, np.dtype[np.int64]] | torch.Tensor
    if return_tensors == "np":
        idx_arr = np.array(sample_idx)
    elif return_tensors == "pt":
        idx_arr = torch.tensor(sample_idx)
    else:
        idx_arr = list(sample_idx)
    inputs = tokenizer(
        docs,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_overflowing_tokens=return_overflowing_tokens,
        stride=stride,
        return_tensors=return_tensors,
        add_special_tokens=add_special_tokens,
        return_attention_mask=return_attention_mask,
        return_offsets_mapping=return_offsets_mapping,
        **kwargs,
    )
    # Globalize overflow_to_sample_mapping
    if "overflow_to_sample_mapping" in inputs:
        if isinstance(idx_arr, list):
            inputs["overflow_to_sample_mapping"] = [
                idx_arr[i] for i in inputs["overflow_to_sample_mapping"]
            ]
        else:
            inputs["overflow_to_sample_mapping"] = idx_arr[inputs["overflow_to_sample_mapping"]]
    # Flatten offset mapping
    if "offset_mapping" in inputs:
        inputs["offset_mapping"] = [
            [tuple(offset) for offset in offsets] for offsets in inputs["offset_mapping"]
        ]
    return dict(inputs)


@torch.inference_mode()
def pad(
    sequences: list[torch.Tensor] | list[np.ndarray[Any, Any]] | list[list[int]],
    pad_value: int,
    strategy: str | None = "longest",
    max_length: int | None = None,
) -> torch.Tensor | np.ndarray[Any, Any] | list[torch.Tensor]:
    """Pads a list of sequences to a uniform length.

    Parameters
    ----------
    sequences : list of torch.Tensor
        A list of sequences to pad. Each sequence should be a list or a PyTorch tensor of numerical IDs.
    pad_value : int
        The value to use for padding.
    strategy : str, optional
        The padding strategy to use. Can be one of the following:
        - 'longest': Pad all sequences to the length of the longest sequence in the list.
        - 'max_length': Pad all sequences to a specified maximum length. `max_length` must be provided.
        - None: No padding is applied. The input sequences are returned as is.
        (default: 'longest')
    max_length : int, optional
        The maximum length to pad sequences to when `strategy='max_length'`. Must be provided if `strategy='max_length'`. (default: None)

    Returns
    -------
    torch.Tensor or list of torch.Tensor
        A PyTorch tensor containing the padded sequences if padding is applied, or the original list of sequences if `strategy=None`.
        If padding is applied, the tensor has shape (len(sequences), max_len), where max_len is the length of the longest sequence
        in `sequences` or `max_length` if specified.

    Raises
    ------
    ValueError
        If `sequences` is empty.
    ValueError
        If `strategy='max_length'` and `max_length` is not specified.
    ValueError
        If `strategy='max_length'` and any sequence in `sequences` exceeds `max_length`.
    ValueError
        If `strategy` is not one of 'longest', 'max_length', or None.
    """
    if not len(sequences):
        raise ValueError("Input list must not be empty.")
    input_is_numpy = isinstance(sequences[0], np.ndarray)
    tensor_sequences: list[torch.Tensor]
    if input_is_numpy or isinstance(sequences[0], list):
        tensor_sequences = [torch.tensor(seq) for seq in sequences]
    else:
        tensor_sequences = sequences  # type: ignore[assignment]
    padded_sequences: torch.Tensor | np.ndarray[Any, Any] | list[torch.Tensor]
    if strategy == "max_length":
        if max_length is None:
            raise ValueError("max_length must be specified when strategy='max_length'.")
        seq_lengths = torch.tensor([len(seq) for seq in tensor_sequences])
        if any(seq_lengths > max_length):
            raise ValueError(f"Input sequence length {seq_lengths.max()} exceeds `max_length`.")
        # Pad the first sequence to `max_length`
        tensor_sequences[0] = F.pad(
            tensor_sequences[0], (0, max_length - len(tensor_sequences[0])), value=pad_value
        )
        # Pad the remaining sequences to the length of the first sequence
        padded_sequences = pad_sequence(tensor_sequences, batch_first=True, padding_value=pad_value)
    elif strategy == "longest":
        padded_sequences = pad_sequence(tensor_sequences, batch_first=True, padding_value=pad_value)
    elif strategy is None:
        padded_sequences = tensor_sequences
    else:
        raise ValueError(f"Invalid value '{strategy}' for `strategy`.")
    if input_is_numpy and isinstance(padded_sequences, torch.Tensor):
        # Convert the padded tensor back to NumPy
        return padded_sequences.numpy()
    return padded_sequences


def dynamic_pad_collate(
    batch: list[dict[str, Any]],
    pad_token_id: int = 0,
    standardize: bool = True,
) -> dict[str, Any]:
    """
    Collate function for tokenized data which dynamically pads.

    Parameters
    ----------
    batch : list of dict
        A list of dictionaries containing tokenized data.
        Each dictionary should have the same keys, and the values
        should be PyTorch tensors.
    pad_token_id : int, optional
        The token ID to use for padding. Default is 0.
        It is recommended to wrap this function in `functools.partial`
        to set the `pad_token_id` to the tokenizer's pad token ID.
    standardize : bool, optional
        Whether to standardize the batch sizes to powers of 2.
        Default is True.
    Returns
    -------
    dict of str to torch.Tensor
        A dictionary containing the collated data. The keys are the
        same as the input batch, and the values are PyTorch tensors.
    """
    collated: dict[str, Any] = {}
    if batch:
        # Pad and concatenate the arrays at each key in the batch
        pad_values = {"input_ids": pad_token_id} | dict(DEFAULT_PAD_VALUES)
        first_batch = batch[0]
        for key in first_batch:
            first_val = first_batch[key]
            batch_vals = [item[key] for item in batch]
            pad_val = pad_values.get(key, 0)
            if isinstance(first_val, list):
                if standardize:
                    longest_seq = max([len(item[key]) for item in batch])
                    max_length = round_up_to_power_of_2(longest_seq)
                    collated[key] = pad(
                        batch_vals,
                        pad_value=pad_val,
                        strategy="max_length",
                        max_length=max_length,
                    )
                else:
                    collated[key] = pad(
                        batch_vals,
                        pad_value=pad_val,
                        strategy="longest",
                    )
            else:
                collated[key] = batch_vals
                if isinstance(collated[key][0], (int, float)):
                    collated[key] = torch.tensor(collated[key])
        input_ids = collated["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            collated["attention_mask"] = input_ids.ne(pad_token_id).long()
    return collated


def tokenize_docs(
    docs: list[str],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int | None = None,
    truncation: bool = True,
    prechunk: bool = True,
    prechunk_overlap: float | int = 0,
    add_special_tokens: bool = True,
    return_attention_mask: bool = False,
    return_offsets_mapping: bool = False,
    return_tokenized_dataset: bool = False,
    batch_size: int = 10,
    n_jobs: int | None = None,
    show_progress: bool = True,
) -> dict[str, Any] | TokenizedDataset:
    """
    Tokenize documents in a dataset using a tokenizer.

    Parameters
    ----------
    docs : list of str
        The list of documents to be tokenized.
    tokenizer : PreTrainedTokenizer
        The tokenizer to be used for tokenization.
    max_length : int or None, optional
        The maximum length of the tokenized sequences. If None,
        the tokenizer's default max length is used. Default is None.
    truncation : bool, optional
        Whether to truncate sequences that exceed the maximum length.
        Default is True.
    prechunk : bool, optional
        Whether to return overflowing tokens. Default is True.
    prechunk_overlap : float or int, optional
        The overlap between prechunked sequences. Default is 0.
    add_special_tokens : bool, optional
        Whether to add special tokens to the tokenized sequences.
        Default is True.
    return_attention_mask : bool, optional
        Whether to return the attention mask for the tokenized sequences.
        Default is False.
    return_offsets_mapping : bool, optional
        Whether to return the offsets mapping for the tokenized sequences.
        Default is False.
    return_tokenized_dataset : bool, optional
        Whether to return a TokenizedDataset object instead of a dictionary.
    batch_size : int, optional
        The batch size to be used for tokenization. Default is 10.
    n_jobs : int or None, optional
        The number of jobs to be used for parallel processing. If None,
        the number of CPU cores will be used. Default is None.
    show_progress : bool, optional
        Whether to show a progress bar during tokenization. Default is True.

    Returns
    -------
    dict
        A dictionary containing the tokenized sequences and other relevant information.

    Raises
    ------
    ValueError
        If the `max_length` parameter is None and the tokenizer does not
        have a `model_max_length`.
        If the `truncation` parameter is not a boolean.
    """
    if len(docs) == 0:
        raise ValueError("Docs must contain at least one document.")

    # Convert docs to a polars DataFrame and add sample_idx
    data = pl.DataFrame({"text": docs, "sample_idx": range(len(docs))})

    max_length = get_max_length(max_length, tokenizer, required=truncation)
    stride = get_overlap_count(prechunk_overlap, max_length)
    num_batches = math.ceil(len(data) / batch_size)
    batches = tqdm(
        data.iter_slices(batch_size),
        total=num_batches,
        desc="Tokenizing",
        disable=not show_progress,
    )
    # Threading is surprisingly fast because tokenizers are Rust-based and GIL free
    # MP has serialization difficulties and high overhead
    with disable_tokenizer_parallelism():
        tok_batches = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_tokenize_batch)(
                docs=batch["text"].to_list(),
                sample_idx=batch["sample_idx"].to_list(),
                tokenizer=tokenizer,
                max_length=max_length,
                padding=False,
                truncation=truncation,
                return_overflowing_tokens=prechunk,
                stride=stride,
                return_tensors=None,
                add_special_tokens=add_special_tokens,
                return_attention_mask=return_attention_mask,
                return_offsets_mapping=return_offsets_mapping,
            )
            for batch in batches
        )
    # Combine the results into a single dictionary
    combined = {}
    for key, val in tok_batches[0].items():
        if isinstance(val, list):
            combined[key] = [y for x in tok_batches for y in x[key]]
        else:
            warnings.warn(
                f"Unsupported data type {type(val)} for key {key}. Skipping.", stacklevel=2
            )
    combined["sequence_idx"] = list(range(len(combined["input_ids"])))
    return TokenizedDataset(combined) if return_tokenized_dataset else combined
