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

"""Phrase Foundry is a library for extracting n-grams from text using transformer models."""

import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm


class TokenizedDataset(Dataset):
    """Dataset class for tokenized input sequences."""

    exclude = frozenset(["overflow_to_sample_mapping"])

    def __init__(self, inputs: dict) -> None:
        """Initialize a TokenizedDataset instance.

        Parameters
        ----------
        inputs : dict
            Tokenized input sequences.
        """
        self.inputs = inputs
        if not isinstance(self.inputs["input_ids"], torch.Tensor):
            raise TypeError("`input_ids` must be a torch.Tensor.")

    def __len__(self) -> int:
        """Returns the number of sequences in the dataset."""
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        """Returns the input sequence at the specified index.

        Parameters
        ----------
        idx : int
            Index of the input sequence to return.

        Returns
        -------
        dict
            Dictionary containing the input sequence.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        return {
            k: v[idx].squeeze(0)
            for k, v in self.inputs.items()
            if k not in self.exclude
        }


def get_ngram_idx(
    input_ids: np.ndarray | torch.Tensor, ngram_range: tuple[int, int] = (4, 6)
) -> list[np.ndarray]:
    """Get n-gram indices for a batch of input sequences.

    Parameters
    ----------
    input_ids : np.ndarray, torch.Tensor
        Tokenized input sequences.
    ngram_range : tuple[int, int], optional
        Range of n-gram sizes to extract, by default (4, 6).

    Returns
    -------
    list[np.ndarray]
        List of n-gram indices for each n-gram size.
    """
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.cpu().numpy()
    ngram_idx = []
    for ngram_size in range(ngram_range[0], ngram_range[1] + 1):
        idx = np.vstack(
            [np.arange(input_ids.shape[1]) + i for i in range(ngram_size)]
        ).T
        idx = idx[(idx[:, -1] < input_ids.shape[1])]
        # Continue if empty
        if idx.size == 0:
            continue
        ngram_idx.append(idx)
    return ngram_idx


class PhraseFoundry:
    def __init__(
        self,
        model_name: str,
        device: torch.device | str | int = "cuda",
        invalid_start_token_pattern: str | None = r"^##",
        exclude_tokens: list[str] | list[int] | None = None,
    ) -> None:
        """Initialize a PhraseFoundry model.

        Parameters
        ----------
        model_name : str
            Name of the pretrained model to use.
        device : torch.device, str, int, optional
            Device to use for inference, by default "cuda".
        invalid_start_token_pattern : str, None, optional
            Regular expression pattern for invalid start tokens, by default r"^##".
        exclude_tokens : list[str], list[int], None, optional
            List of tokens to exclude from n-gram extraction, by default None.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, clean_up_tokenization_spaces=True
        )
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval().to(device)
        if not isinstance(invalid_start_token_pattern, (str, type(None))):
            raise TypeError("`invalid_start_token_pattern` must be a string.")
        self.invalid_start_token_pattern = invalid_start_token_pattern
        if not isinstance(exclude_tokens, (list, tuple, set, np.ndarray, type(None))):
            raise TypeError("`exclude_tokens` must be a list of token IDs or tokens.")
        self.exclude_tokens = exclude_tokens


    @property
    def device(self) -> torch.device:
        """Returns the device the model is on."""
        return self.model.device

    def to(self, device: torch.device | str | int) -> "PhraseFoundry":
        """Move the model to a new device.

        Parameters
        ----------
        device : torch.device, str, int
            Device to move the model to.

        Returns
        -------
        PhraseFoundry
            Returns the model instance.
        """
        self.model.to(device)
        return self

    @property
    def exclude_token_ids(self) -> np.ndarray:
        """Converts `exclude_tokens` to token IDs."""
        if self.exclude_tokens is None:
            ids = self.tokenizer.all_special_ids
        elif isinstance(self.exclude_tokens, (str, int)):
            raise TypeError("`exclude_tokens` must be a list of token IDs or tokens.")
        elif isinstance(self.exclude_tokens[0], str):
            ids = self.tokenizer.convert_tokens_to_ids(self.exclude_tokens)
        elif isinstance(self.exclude_tokens[0], int):
            ids = self.exclude_tokens
        else:
            raise ValueError(f"Unknown `exclude_tokens` value: {self.exclude_tokens}")
        return np.asarray(ids)

    @property
    def invalid_start_token_ids(self) -> np.ndarray:
        """Returns token IDs that cannot be the start of a ngram."""
        ids = []
        if self.invalid_start_token_pattern is not None:
            search = re.compile(self.invalid_start_token_pattern).search
            ids = [v for k, v in self.tokenizer.vocab.items() if search(k)]
        return np.array(ids)

    def extract_ngrams(
        self,
        input_ids: np.ndarray,
        token_embeds: np.ndarray,
        ngram_range: tuple[int, int] = (4, 6),
        overflow_to_sample_mapping: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract the n-grams and n-gram embeddings from token embeddings.

        Parameters
        ----------
        input_ids : np.ndarray
            Tokenized input sequences.
        token_embeds : np.ndarray
            Token embeddings.
        ngram_range : tuple[int, int], optional
            Range of n-gram sizes to extract, by default (4, 6).
        overflow_to_sample_mapping : np.ndarray, optional
            Mapping from overflow indices to sample indices, by default None.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple containing the sequence indices, n-grams, and n-gram embeddings.

        Notes
        -----
        1. The n-gram embeddings are obtained by averaging the token embeddings for each
        token in each n-gram. This is known as the "mean-tokens" embedding method.

        2. The `ngram_range` parameter specifies the range of n-gram sizes to extract.
        For example, if `ngram_range` is set to `(4, 6)`, token n-grams of sizes 4, 5, and 6
        will be extracted from the input sequences.

        3. The `overflow_to_sample_mapping` parameter is used to map overflow indices to sample indices.
        This is useful when chunking documents into overlapping sequences, as the n-grams extracted
        from the overflow sequences need to be mapped back to the original sample indices.
        """
        ngram_idx = get_ngram_idx(input_ids, ngram_range=ngram_range)
        valid_token_mask = np.isin(input_ids, self.exclude_token_ids, invert=True)
        valid_start_token_mask = np.isin(
            input_ids, self.invalid_start_token_ids, invert=True
        )
        global_ngrams = []
        global_ngram_vecs = []
        global_seq_idx = []
        for idx in tqdm(ngram_idx, desc="Extracting"):
            ngram_token_ids = input_ids[:, idx]
            # Create mask of valid ngrams
            # (valid ngrams must not contain any excluded tokens
            #  and must start with a valid start token)
            valid_ngrams = (
                np.all(valid_token_mask[:, idx], axis=2)
                & valid_start_token_mask[:, idx[:, 0]]
            )
            ngram_token_ids = ngram_token_ids[valid_ngrams]
            ngrams = self.tokenizer.batch_decode(ngram_token_ids)
            ngram_vecs = token_embeds[:, idx][valid_ngrams].mean(axis=1)
            seq_idx = np.arange(input_ids.shape[0]).repeat(len(idx))[
                valid_ngrams.ravel()
            ]
            global_ngrams.extend(ngrams)
            global_ngram_vecs.append(ngram_vecs)
            global_seq_idx.append(seq_idx)
        global_ngrams = np.array(global_ngrams, dtype="U")
        global_ngram_vecs = np.vstack(global_ngram_vecs)
        global_seq_idx = np.hstack(global_seq_idx)
        # Map overflow indices to sample indices
        if overflow_to_sample_mapping is not None:
            global_seq_idx = overflow_to_sample_mapping[global_seq_idx]
        return global_seq_idx, global_ngrams, global_ngram_vecs

    def encode(
        self,
        docs: list[str],
        max_length: int | None = 512,
        batch_size: int = 32,
        do_chunking: bool = True,
        stride: int = 128,
        amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
    ) -> tuple[dict, np.ndarray]:
        """Obtain the token embeddings for a list of documents.

        Parameters
        ----------
        docs : list[str]
            List of documents to encode.
        max_length : int, optional
            Maximum length of the input sequences, by default 512.
        batch_size : int, optional
            Batch size for encoding, by default 32.
        do_chunking : bool, optional
            Enable chunking of documents into overlapping sequences, by default True.
        stride : int, optional
            Stride for splitting documents into overlapping sequences, by default 128.
            Only used if `do_chunking` is True.
        amp : bool, optional
            Enable automatic mixed precision, by default False.
        amp_dtype : torch.dtype, optional
            Data type for automatic mixed precision, by default torch.float16.

        Returns
        -------
        tuple[dict, np.ndarray]
            Tuple containing the tokenized inputs and token embeddings.

        Raises
        ------
        ValueError
            If `max_length` is not specified and `tokenizer.model_max_length` is None.

        Notes
        -----
        If `do_chunking` is True, the documents will be split into overlapping sequences
        with an overlap of `stride` tokens. This is useful for processing long documents that exceed
        the maximum sequence length of the model, and is enabled by default.
        """
        if max_length is None:
            if self.tokenizer.model_max_length is None:
                raise ValueError(
                    "max_length must be specified if tokenizer.model_max_length is None"
                )
            max_length = self.tokenizer.model_max_length
            print(f"max_length set to {max_length}")
        inputs = self.tokenizer(
            docs,
            max_length=max_length,
            padding="longest",
            truncation=True,
            return_overflowing_tokens=do_chunking,
            stride=stride,
            return_tensors="pt",
        )
        loader = DataLoader(
            TokenizedDataset(inputs),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
        token_embeds = []
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=amp, dtype=amp_dtype):
                for batch in tqdm(loader, desc="Encoding"):
                    batch = {
                        k: v.to(self.device, non_blocking=True)
                        for k, v in batch.items()
                    }
                    outputs = self.model(**batch)
                    token_embeds.append(outputs.last_hidden_state.cpu().numpy())
        token_embeds = np.concatenate(token_embeds, axis=0)
        return inputs, token_embeds

    def encode_extract(
        self,
        docs: list[str],
        max_length: int | None = 512,
        batch_size: int = 32,
        do_chunking: bool = True,
        stride: int = 128,
        ngram_range: tuple[int, int] = (4, 6),
        amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Obtain the n-grams and n-gram embeddings from a list of documents.

        This is equivalent to calling `encode` followed by `extract_ngrams`.
        It first encodes the input documents, then extracts the n-grams and
        n-gram embeddings from the token embeddings.

        Parameters
        ----------
        docs : list[str]
            List of documents to encode.
        max_length : int, optional
            Maximum length of the input sequences, by default 512.
        batch_size : int, optional
            Batch size for encoding, by default 32.
        do_chunking : bool, optional
            Enable chunking of documents into overlapping sequences, by default True.
        stride : int, optional
            Stride for splitting documents into overlapping sequences, by default 128.
            Only used if `do_chunking` is True.
        ngram_range : tuple[int, int], optional
            Range of n-gram sizes to extract, by default (4, 6).
        amp : bool, optional
            Enable automatic mixed precision, by default False.
        amp_dtype : torch.dtype, optional
            Data type for automatic mixed precision, by default torch.float16.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple containing the sequence indices, n-grams, and n-gram embeddings

        Raises
        ------
        ValueError
            If `max_length` is not specified and `tokenizer.model_max_length` is None.

        Notes
        -----
        The `ngram_range` parameter specifies the range of n-gram sizes to extract.
        For example, if `ngram_range` is set to `(4, 6)`, token n-grams of sizes 4, 5, and 6
        will be extracted from the input sequences.
        """
        inputs, token_embeds = self.encode(
            docs,
            max_length=max_length,
            batch_size=batch_size,
            do_chunking=do_chunking,
            stride=stride,
            amp=amp,
            amp_dtype=amp_dtype,
        )
        seq_idx, ngrams, ngram_vecs = self.extract_ngrams(
            inputs["input_ids"],
            token_embeds,
            ngram_range=ngram_range,
            overflow_to_sample_mapping=inputs["overflow_to_sample_mapping"],
        )
        return seq_idx, ngrams, ngram_vecs

    def encode_queries(
        self,
        queries: list[str],
        max_length: int | None = 512,
        batch_size: int = 32,
        do_chunking: bool = False,
        stride: int = 128,
        amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
    ) -> np.ndarray:
        """Obtain the mean-tokens embeddings for a list of query strings.

        This is a convenient method for embedding query strings into the same space
        as the n-grams extracted from documents. It is mainly useful for doing semantic
        search.

        Parameters
        ----------
        queries : list[str]
            List of queries to encode.
        max_length : int, optional
            Maximum length of the input sequences, by default 512.
        batch_size : int, optional
            Batch size for encoding, by default 32.
        do_chunking : bool, optional
            Enable chunking of documents into overlapping sequences, by default False.
        stride : int, optional
            Stride for splitting documents into overlapping sequences, by default 0.
            Only used if `do_chunking` is True.
        amp : bool, optional
            Enable automatic mixed precision, by default False.
        amp_dtype : torch.dtype, optional
            Data type for automatic mixed precision, by default torch.float16.

        Returns
        -------
        np.ndarray
            Mean-token embeddings for each query.
        """
        inputs, token_embeds = self.encode(
            queries,
            max_length=max_length,
            batch_size=batch_size,
            do_chunking=do_chunking,
            stride=stride,
            amp=amp,
            amp_dtype=amp_dtype,
        )
        valid_token_mask = np.isin(
            inputs["input_ids"], self.tokenizer.all_special_ids, invert=True
        )
        # Extract mean token embeddings for each query
        query_embeds = []
        for i in tqdm(
            range(inputs["input_ids"].shape[0]),
            desc="Extracting",
            total=inputs["input_ids"].shape[0],
        ):
            query_embeds.append(token_embeds[i, valid_token_mask[i], :].mean(axis=0))
        return np.vstack(query_embeds)
