import logging
import math
import os
import warnings
from collections.abc import Iterator
from functools import cached_property
from types import MappingProxyType

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

from afterthoughts.utils import get_overlap_count, order_by_indices, round_up_to_power_of_2

logger = logging.getLogger(__name__)

DEFAULT_PAD_VALUES = MappingProxyType(
    {
        "attention_mask": 0,
        "token_type_ids": 0,
        "special_tokens_mask": 0,
        "sentence_ids": -1,
    }
)


class TokenizedDataset(Dataset):
    """A dataset class for tokenized data."""

    def __init__(
        self, data: dict[str, list[torch.Tensor]], sort_by_token_count: bool = True
    ) -> None:
        """
        Initialize the TokenizedDataset.

        Parameters
        ----------
        data : dict
            A dictionary containing the tokenized data.
        """
        self.data = data
        self.validate_data()
        self.sort_by_token_count = sort_by_token_count
        self.sort_data()

    def validate_data(self):
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

    def sort_data(self):
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

    def unsort_results(self, results: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Unsort the results of the dataset."""
        unsorted = {}
        for key in results:
            if isinstance(results[key], list):
                if len(results[key]):
                    unsorted[key] = order_by_indices(results[key], self.unsort_idx)
                else:
                    unsorted[key] = []
            elif isinstance(results[key], (torch.Tensor, np.ndarray)):
                if len(results[key]):
                    unsorted[key] = results[key][self.unsort_idx]
                else:
                    unsorted[key] = torch.empty(0)
        return unsorted


class DynamicTokenSampler(Sampler[list[int]]):
    """A sampler that creates batches based on token count.

    This sampler groups sequences into batches such that the total number of tokens
    in each batch does not exceed a specified maximum. It assumes the dataset has been
    pre-sorted by length (longest first) to minimize padding within batches.

    Parameters
    ----------
    data_source : TokenizedDataset
        The dataset containing the tokenized sequences. Should be pre-sorted by length
        (longest first) for optimal batching.
    max_tokens : int
        The maximum number of tokens allowed in each batch.

    Examples
    --------
    >>> dataset = TokenizedDataset(data, sort_by_token_count=True)  # Sort dataset first
    >>> sampler = DynamicTokenSampler(dataset, max_tokens=8192)
    >>> dataloader = DataLoader(dataset, batch_sampler=sampler)
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
        batch_list = []
        current_batch = []
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
    max_length: int | None, tokenizer: PreTrainedTokenizerFast, required: bool = False
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
    tokenizer: PreTrainedTokenizerFast,
    max_length: int | None = None,
    padding: str = False,
    truncation: bool = True,
    return_overflowing_tokens: bool = True,
    stride: int = 0,
    return_tensors: str | None = None,
    add_special_tokens: bool = True,
    return_attention_mask: bool = True,
    return_offsets_mapping: bool = False,
    **kwargs,
) -> dict[str, torch.Tensor]:
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
    if return_tensors == "np":
        sample_idx = np.array(sample_idx)
    elif return_tensors == "pt":
        sample_idx = torch.tensor(sample_idx)
    else:
        sample_idx = list(sample_idx)
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
        if isinstance(sample_idx, list):
            inputs["overflow_to_sample_mapping"] = [
                sample_idx[i] for i in inputs["overflow_to_sample_mapping"]
            ]
        else:
            inputs["overflow_to_sample_mapping"] = sample_idx[inputs["overflow_to_sample_mapping"]]
    # Flatten offset mapping
    if "offset_mapping" in inputs:
        inputs["offset_mapping"] = [
            [tuple(offset) for offset in offsets] for offsets in inputs["offset_mapping"]
        ]
    return inputs


@torch.no_grad()
def pad(
    sequences: list[torch.Tensor],
    pad_value: int,
    strategy: str = "longest",
    max_length: int | None = None,
) -> torch.Tensor | list[torch.Tensor]:
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
    if input_is_numpy or isinstance(sequences[0], list):
        sequences = [torch.tensor(seq) for seq in sequences]
    if strategy == "max_length":
        if max_length is None:
            raise ValueError("max_length must be specified when strategy='max_length'.")
        seq_lengths = torch.tensor([len(seq) for seq in sequences])
        if any(seq_lengths > max_length):
            raise ValueError(f"Input sequence length {seq_lengths.max()} exceeds `max_length`.")
        # Pad the first sequence to `max_length`
        sequences[0] = F.pad(sequences[0], (0, max_length - len(sequences[0])), value=pad_value)
        # Pad the remaining sequences to the length of the first sequence
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_value)
    elif strategy == "longest":
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_value)
    elif strategy is None:
        padded_sequences = sequences
    else:
        raise ValueError(f"Invalid value '{strategy}' for `strategy`.")
    if input_is_numpy:
        # Convert the padded tensor back to NumPy
        padded_sequences = padded_sequences.numpy()
    return padded_sequences


def dynamic_pad_collate(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int = 0,
    standardize: bool = True,
) -> dict[str, torch.Tensor]:
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
    collated = {}
    if batch:
        # Pad and concatenate the arrays at each key in the batch
        pad_values = {"input_ids": pad_token_id} | DEFAULT_PAD_VALUES
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
        collated["attention_mask"] = collated["input_ids"].ne(pad_token_id).long()
    return collated


def tokenize_docs(
    docs: list[str],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int | None = None,
    truncation: bool = True,
    chunk_docs: bool = True,
    overlap: float | int = 0,
    add_special_tokens: bool = True,
    return_attention_mask: bool = False,
    return_offsets_mapping: bool = False,
    return_tokenized_dataset: bool = False,
    batch_size: int = 10,
    n_jobs: int | None = None,
    show_progress: bool = True,
) -> dict | TokenizedDataset:
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
    chunk_docs : bool, optional
        Whether to return overflowing tokens. Default is True.
    overlap : float or int, optional
        The overlap between documents. Default is 0.
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
    stride = get_overlap_count(overlap, max_length)
    num_batches = math.ceil(len(data) / batch_size)
    batches = tqdm(
        data.iter_slices(batch_size),
        total=num_batches,
        desc="Tokenizing",
        disable=not show_progress,
    )
    # Disable built-in parallelism in tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Threading is surprisingly fast because tokenizers are Rust-based and GIL free
    # MP has serialization difficulties and high overhead
    tok_batches = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_tokenize_batch)(
            docs=batch["text"].to_list(),
            sample_idx=batch["sample_idx"].to_list(),
            tokenizer=tokenizer,
            max_length=max_length,
            padding=False,
            truncation=truncation,
            return_overflowing_tokens=chunk_docs,
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
