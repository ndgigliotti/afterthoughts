import math
import os
import warnings

import numpy as np
import polars as pl
import torch
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

from finephrase.utils import get_overlap_count


class TokenizedDataset(Dataset):
    """A dataset class for tokenized data."""

    def __init__(self, data: dict[str, torch.Tensor]):
        """
        Initialize the TokenizedDataset.

        Parameters
        ----------
        data : dict
            A dictionary containing the tokenized data.
        """
        self.data = data

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data["input_ids"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a single item from the dataset."""
        return {key: val[idx] for key, val in self.data.items()}


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
    padding: str = "max_length",
    truncation: bool = True,
    return_overflowing_tokens: bool = True,
    stride: int = 0,
    return_tensors: str = "np",
    add_special_tokens: bool = True,
    return_attention_mask: bool = True,
    return_offsets_mapping: bool = False,
    **kwargs,
) -> dict[str, np.ndarray]:
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
        Padding strategy to use. Default is "max_length".
    truncation : bool, optional
        Whether to truncate sequences to the maximum length. Default is True.
    return_overflowing_tokens : bool, optional
        Whether to return overflowing tokens. Default is True.
    stride : int, optional
        Stride to use when handling overflowing tokens. Default is 0.
    return_tensors : str, optional
        Format of the returned tensors. Default is "np" (NumPy).
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
    dict of str to np.ndarray
        Dictionary containing the tokenized inputs and other relevant information.
    """
    max_length = get_max_length(
        max_length, tokenizer, required=padding is True or padding == "max_length"
    )
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
        inputs["overflow_to_sample_mapping"] = np.asarray(sample_idx)[
            inputs["overflow_to_sample_mapping"]
        ]
    # Flatten offset mapping
    if "offset_mapping" in inputs:
        inputs["offset_mapping"] = [
            [tuple(offset) for offset in offsets]
            for offsets in inputs["offset_mapping"]
        ]
    return inputs


def _pad_and_concat(
    batches: list[dict], key: str, tokenizer: PreTrainedTokenizerFast
) -> np.ndarray:
    """Pad and concatenate the arrays at a certain key in a list of dictionaries."""
    # Find the widths of the arrays in this batch
    widths = np.array(batch[key].shape[1] for batch in batches)
    pad_width = widths.max()
    # Pad all arrays to the maximum width
    pad_diff = pad_width - widths
    for i, (batch, pad_diff) in enumerate(zip(batches, pad_diff)):
        # Determine the padding value based on the key
        pad_values = {
            "input_ids": tokenizer.pad_token_id,
            "attention_mask": 0,
            "token_type_ids": 0,
            "special_tokens_mask": 0,
            "sentence_ids": -1,
        }
        # Pad the array with the appropriate value
        batches[i][key] = np.pad(
            batch[key],
            ((0, 0), (0, pad_diff)),
            constant_values=pad_values.get(key, 0),
            mode="constant",
        )
    # Concatenate the arrays along the first axis
    return np.concatenate([batch[key] for batch in batches], axis=0)


def tokenize_docs(
    data: pl.DataFrame,
    tokenizer: PreTrainedTokenizerFast,
    text_field: str = "text",
    sample_idx_field: str = "sample_idx",
    max_length: int | None = None,
    padding: str = "max_length",
    truncation: bool = True,
    chunk_docs: bool = True,
    overlap: float | int = 0,
    add_special_tokens: bool = True,
    return_attention_mask: bool = True,
    return_offsets_mapping: bool = False,
    batch_size: int = 10,
    n_jobs: int | None = None,
    show_progress: bool = True,
    **kwargs,
) -> dict:
    """
    Tokenize documents in a dataset using a tokenizer.

    Parameters
    ----------
    data : pl.DataFrame
        The dataset containing the documents to be tokenized.
    tokenizer : PreTrainedTokenizer
        The tokenizer to be used for tokenization.
    text_field : str, optional
        The name of the field containing the text to be tokenized.
        Default is "text".
    sample_idx_field : str, optional
        The name of the field containing the sample indices.
        Default is "sample_idx".
    max_length : int or None, optional
        The maximum length of the tokenized sequences. If None,
        the tokenizer's default max length is used. Default is None.
    padding : str, optional
        The padding strategy to be used. Default is "max_length".
    truncation : bool, optional
        Whether to truncate sequences that exceed the maximum length.
        Default is True.
    chunk_docs : bool, optional
        Whether to return overflowing tokens. Default is True.
    overlap : float or int, optional
        The overlap between phrases. Default is 0.
    add_special_tokens : bool, optional
        Whether to add special tokens to the tokenized sequences.
        Default is True.
    return_attention_mask : bool, optional
        Whether to return the attention mask for the tokenized sequences.
        Default is True.
    return_offsets_mapping : bool, optional
        Whether to return the offsets mapping for the tokenized sequences.
        Default is False.
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
        If the `padding` parameter is not one of "max_length", "longest",
        or False.
        If the `truncation` parameter is not a boolean.
    """
    max_length = get_max_length(
        max_length, tokenizer, required=padding is True or padding == "max_length"
    )
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
    batched_results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_tokenize_batch)(
            docs=batch[text_field].to_list(),
            sample_idx=batch[sample_idx_field].to_list(),
            tokenizer=tokenizer,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_overflowing_tokens=chunk_docs,
            stride=stride,
            return_tensors="np",
            add_special_tokens=add_special_tokens,
            return_attention_mask=return_attention_mask,
            return_offsets_mapping=return_offsets_mapping,
            **kwargs,
        )
        for batch in batches
    )
    # Combine the results into a single dictionary
    combined_results = {}
    for key, val in batched_results[0].items():
        if isinstance(val, np.ndarray):
            if val.ndim > 1:
                if padding is True or padding == "longest":
                    combined_results[key] = _pad_and_concat(
                        batched_results, key, tokenizer
                    )
            combined_results[key] = np.concatenate(
                [batch[key] for batch in batched_results], axis=0
            )

        elif isinstance(val, list):
            combined_results[key] = [y for x in batched_results for y in x[key]]
        else:
            warnings.warn(f"Unsupported data type {type(val)} for key {key}. Skipping.")
    return combined_results
