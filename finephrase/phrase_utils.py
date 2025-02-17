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

import numpy as np
import torch
import transformers
from joblib import delayed

from finephrase.utils import get_overlap_count


@delayed
def _tokenize_batch(
    docs: list[str],
    sample_idx: list[int],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int | None = None,
    chunk_docs: bool = True,
    doc_overlap: float = 0.5,
) -> dict[str, np.ndarray]:
    """Tokenize a list of documents into input sequences for the model."""
    if max_length is None:
        max_length = tokenizer.model_max_length
        if max_length is None:
            raise ValueError(
                "The `max_length` parameter must be specified if the tokenizer does not have a `model_max_length`."
            )
    inputs = tokenizer(
        docs,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_overflowing_tokens=chunk_docs,
        stride=get_overlap_count(doc_overlap, max_length),
        return_tensors="np",
        add_special_tokens=True,
        return_attention_mask=True,
    )
    # Globalize overflow_to_sample_mapping
    if "overflow_to_sample_mapping" in inputs:
        inputs["overflow_to_sample_mapping"] = np.asarray(sample_idx)[
            inputs["overflow_to_sample_mapping"]
        ]
    return inputs


def get_phrase_idx(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    phrase_sizes: list | tuple | int,
    overlap: int | float | list | dict = 0.5,
    phrase_min_token_ratio: float = 0.5,
    sequence_idx: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Extract the indices of sub-sequences from input sequences.

    Parameters
    ----------
    input_ids : torch.Tensor
        Tensor containing the tokenized input sequences.
    attention_mask : torch.Tensor
        Tensor containing the attention mask for the input sequences.
    phrase_sizes : list | tuple | int
        Size or list of sizes of the phrases to extract.
    overlap : int | float | list | dict, optional
        Overlap between phrases, by default 0.5.
    phrase_min_token_ratio : float, optional
        Minimum ratio of tokens that must be present in each phrase, by default 0.5.
    sequence_idx : torch.Tensor | None, optional
        Tensor containing the sequence indices, by default None.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing the phrase indices, phrase IDs, valid phrase mask, and sequence indices.

    Raises
    ------
    ValueError
        If `sequence_idx` has a different number of rows than `input_ids`.
    ValueError
        If `overlap` is not in the range [0, 1).
    ValueError
        If `phrase_min_token_ratio` is not greater than 0.
    ValueError
        If `overlap` is not a float, list, tuple, dict, or int.
    """
    if sequence_idx is not None:
        if len(sequence_idx) != input_ids.shape[0]:
            raise ValueError(
                f"`sequence_idx` must have the same number of rows as `input_ids` "
                f"({len(sequence_idx)} != {input_ids.shape[0]})."
            )
        if sequence_idx.device != input_ids.device:
            sequence_idx = sequence_idx.to(input_ids.device)
    results = {
        "phrase_idx": [],
        "phrase_ids": [],
        "phrase_size": [],
        "valid_phrase_mask": [],
        "sequence_idx": [],
    }
    if isinstance(phrase_sizes, int):
        phrase_sizes = [phrase_sizes]
    for i, size in enumerate(phrase_sizes):
        overlap_tokens = get_overlap_count(overlap, size, i)
        # Create generic phrase indices based on the shape of `input_ids`
        start_idx = torch.arange(
            1, input_ids.shape[1] - size + 1, size - overlap_tokens
        )
        stop_idx = start_idx + size
        phrase_idx = torch.stack(
            [torch.arange(start, stop) for start, stop in zip(start_idx, stop_idx)]
        )
        phrase_ids = input_ids[:, phrase_idx]
        min_tokens = math.ceil(phrase_min_token_ratio * size)
        if min_tokens == 0:
            raise ValueError("`phrase_min_token_ratio` must be greater than 0.")
        valid_phrase_mask = attention_mask[:, phrase_idx].sum(dim=2) >= min_tokens
        phrase_ids = phrase_ids[valid_phrase_mask]
        if sequence_idx is None:
            phrase_sequence_idx = torch.arange(
                input_ids.shape[0], device=input_ids.device
            )
        else:
            phrase_sequence_idx = sequence_idx.clone()
        phrase_sequence_idx = phrase_sequence_idx.repeat_interleave(
            valid_phrase_mask.sum(dim=1)
        )
        results["phrase_idx"].append(phrase_idx)
        results["phrase_ids"].append(phrase_ids)
        results["phrase_size"].append(torch.full((phrase_ids.shape[0],), size))
        results["valid_phrase_mask"].append(valid_phrase_mask)
        results["sequence_idx"].append(phrase_sequence_idx)
    return results


def _compute_phrase_embeddings(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_embeds: torch.Tensor,
    sequence_idx: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizer,
    phrase_sizes: int | list | tuple = 12,
    overlap: int | float | list | dict = 0.5,
    phrase_min_token_ratio: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Extract the sub-sequences and sub-sequence embeddings from token embeddings.

    Parameters
    ----------
    input_ids : torch.Tensor
        Tokenized input sequences with no special tokens (except padding).
    attention_mask : torch.Tensor
        Attention mask for the input sequences.
    token_embeds : torch.Tensor
        Token embeddings.
    sequence_idx : torch.Tensor
        Sequence indices.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer used to tokenize the input sequences.
    phrase_sizes : list, tuple, optional
        Sub-sequence size or list of sub-sequence sizes to extract, by default 12.
        For example, if `phrase_sizes` is set to `(12, 24, 48)`, sub-sequences
        of sizes 12, 24, and 48 will be extracted from the input sequences.
    overlap : int, float, list, dict, optional
        Overlap for the sub-sequences, by default 0.5.
        If a float, it is interpreted as a fraction of the phrase size.
        If an integer, it is interpreted as the number of tokens to overlap.
        If a list or tuple, it should contain the overlap for each phrase size.
        If a dictionary, it should map phrase sizes to overlaps.
    phrase_min_token_ratio : float, optional
        Minimum ratio of tokens that must be present in each sub-sequence,
        by default 0.5. This mainly pertains to the last short sub-sequence.
        Typically the default value works well.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing the sample indices, n-grams, and n-gram embeddings.

    Notes
    -----
    1. The phrase embeddings are obtained by averaging the token embeddings for each
    token in each phrase. This is known as the "mean-tokens" embedding method.

    2. Special tokens are not included in the phrase embeddings. This is because special
    tokens such as [CLS], [SEP], and [PAD] do not contribute to the local fine-grained meaning
    of phrases. Even if there was an argument for including tokens such as [UNK], it is easier
    for model compatibility reasons to simply exclude all special tokens.

    3. To extract all possible n-grams, set `overlap` to 0.999999.

    """
    phrase_data = get_phrase_idx(
        input_ids,
        attention_mask,
        phrase_sizes=phrase_sizes,
        overlap=overlap,
        phrase_min_token_ratio=phrase_min_token_ratio,
        sequence_idx=sequence_idx,
    )
    # Mask all special tokens
    attention_mask = torch.isin(
        input_ids,
        torch.tensor(tokenizer.all_special_ids, device=input_ids.device),
        invert=True,
    ).to(torch.uint8)
    results = {
        "sequence_idx": [],
        "phrase_ids": [],
        "phrase_size": [],
        "phrase_embeds": [],
    }
    for i, idx in enumerate(phrase_data["phrase_idx"]):
        attn_factor = attention_mask[:, idx].unsqueeze(3)
        phrase_embeds = torch.sum(token_embeds[:, idx] * attn_factor, dim=2) / (
            torch.clamp(attn_factor.sum(dim=2), min=1)
        )
        phrase_embeds = phrase_embeds[phrase_data["valid_phrase_mask"][i]]
        results["sequence_idx"].append(phrase_data["sequence_idx"][i])
        results["phrase_ids"].append(phrase_data["phrase_ids"][i])
        results["phrase_size"].append(phrase_data["phrase_size"][i])
        results["phrase_embeds"].append(phrase_embeds)

    # Combine results
    results["sequence_idx"] = torch.hstack(results["sequence_idx"])
    results["phrase_size"] = torch.hstack(results["phrase_size"])
    results["phrase_embeds"] = torch.vstack(results["phrase_embeds"])
    return results
