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
from sklearn.decomposition import IncrementalPCA


class TokenizedDataset(Dataset):
    """Dataset class for tokenized input sequences."""

    exclude = frozenset(["overflow_to_sample_mapping"])

    def __init__(
        self, inputs: dict, shuffle: bool = False, random_state: int | None = None
    ) -> None:
        """Initialize a TokenizedDataset instance.

        Parameters
        ----------
        inputs : dict
            Tokenized input sequences.
        shuffle : bool, optional
            Shuffle the input sequences, by default False.
        random_state : int, None, optional
            Random seed for shuffling, by default None.
        """
        self.inputs = inputs
        if not isinstance(self.inputs["input_ids"], torch.Tensor):
            raise TypeError("`input_ids` must be a torch.Tensor.")
        self.shuffle = shuffle
        self.random_state = random_state
        rng = np.random.default_rng(self.random_state)
        self.order_idx = np.arange(len(self.inputs["input_ids"]))
        if self.shuffle:
            rng.shuffle(self.order_idx)

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
        tuple[int, dict]
            Tuple containing the index and input sequence.
            If shuffling is enabled, the index is the true index
            in the dataset.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        true_idx = self.order_idx[idx]
        data = {
            k: v[true_idx].squeeze(0)
            for k, v in self.inputs.items()
            if k not in self.exclude
        }
        return true_idx, data


def get_ngram_idx(
    input_ids: np.ndarray | torch.Tensor, ngram_range: tuple[int, int] = (5, 5)
) -> list[np.ndarray]:
    """Get n-gram indices for a batch of input sequences.

    Parameters
    ----------
    input_ids : np.ndarray, torch.Tensor
        Tokenized input sequences.
    ngram_range : tuple[int, int], optional
        Range of n-gram sizes to extract, by default (5, 5).

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
        amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
        quantize_embeds: bool = True,
        device: torch.device | str | int = "cuda",
        invalid_start_token_pattern: str | None = r"^##",
        exclude_tokens: list[str] | list[int] | None = None,
    ) -> None:
        """Initialize a PhraseFoundry model.

        Parameters
        ----------
        model_name : str
            Name of the pretrained model to use.
        amp : bool, optional
            Enable automatic mixed precision, by default False.
        amp_dtype : torch.dtype, optional
            Data type for automatic mixed precision, by default torch.float16.
        quantize_embeds : bool, optional
            Reduce the embedding precision to float16 if they are float32 or float64,
            by default False.
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
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.quantize_embeds = quantize_embeds
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

    def should_quantize(self, token_embeds: torch.Tensor | np.ndarray) -> bool:
        """Returns True if the embeddings should be quantized."""
        high_dtypes = (torch.float32, torch.float64, np.float32, np.float64)
        return self.quantize_embeds and token_embeds.dtype in high_dtypes

    def quantize_if_needed(
        self, embeds: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Quantize the embeddings if needed."""
        if self.should_quantize(embeds):
            if isinstance(embeds, torch.Tensor):
                embeds = embeds.to(torch.float16)
            else:
                embeds = embeds.astype(np.float16)
        return embeds

    def extract_ngrams(
        self,
        input_ids: np.ndarray,
        token_embeds: np.ndarray,
        ngram_range: tuple[int, int] = (5, 5),
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
            Range of n-gram sizes to extract, by default (5, 5).
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
        for idx in ngram_idx:
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

    def _encode(
        self,
        docs: list[str],
        max_length: int | None = 512,
        batch_size: int = 32,
        do_chunking: bool = True,
        stride: int = 128,
    ) -> dict:
        """Obtain the token embeddings for a list of documents."""
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
        data = TokenizedDataset(inputs)
        loader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
        seq_idx = []
        token_embeds = []
        input_ids = []
        overflow_to_sample_mapping = []
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.amp, dtype=self.amp_dtype):
                for idx, batch in tqdm(loader, desc="Encoding"):
                    batch = {
                        k: v.to(self.device, non_blocking=True)
                        for k, v in batch.items()
                    }
                    outputs = self.model(**batch)
                    seq_idx.append(idx)
                    token_embeds.append(outputs.last_hidden_state.cpu().numpy())
                    input_ids.append(inputs["input_ids"][idx].numpy())
                    if do_chunking:
                        overflow_to_sample_mapping.append(
                            inputs["overflow_to_sample_mapping"][idx].numpy()
                        )
        results = {
            "batch_id": np.arange(len(seq_idx)),
            "seq_idx": seq_idx,
            "input_ids": input_ids,
            "token_embeds": token_embeds,
        }
        if do_chunking:
            results["overflow_to_sample_mapping"] = overflow_to_sample_mapping
        else:
            results["overflow_to_sample_mapping"] = seq_idx
        return results

    def encode_extract(
        self,
        docs: list[str],
        max_length: int | None = 512,
        batch_size: int = 32,
        do_chunking: bool = True,
        stride: int = 128,
        ngram_range: tuple[int, int] = (5, 5),
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
            Range of n-gram sizes to extract, by default (5, 5).

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
        encodings = self._encode(
            docs,
            max_length=max_length,
            batch_size=batch_size,
            do_chunking=do_chunking,
            stride=stride,
        )
        global_samp_idx = []
        global_ngrams = []
        global_ngram_embeds = []

        for batch_id in tqdm(encodings["batch_id"][::-1], desc="Extracting"):
            samp_idx, ngrams, ngram_embeds = self.extract_ngrams(
                encodings["input_ids"].pop(batch_id),
                encodings["token_embeds"].pop(batch_id),
                ngram_range=ngram_range,
                overflow_to_sample_mapping=encodings["overflow_to_sample_mapping"].pop(
                    batch_id
                ),
            )
            ngram_embeds = self.quantize_if_needed(ngram_embeds)
            global_samp_idx.append(samp_idx)
            global_ngrams.append(ngrams)
            global_ngram_embeds.append(ngram_embeds)
        global_samp_idx = np.hstack(global_samp_idx[::-1])
        global_ngrams = np.hstack(global_ngrams[::-1])
        global_ngram_embeds = np.vstack(global_ngram_embeds[::-1])
        return global_samp_idx, global_ngrams, global_ngram_embeds

    def encode_queries(
        self,
        queries: list[str],
        max_length: int | None = 512,
        batch_size: int = 32,
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

        Returns
        -------
        np.ndarray
            Mean-token embeddings for each query.
        """
        encodings = self._encode(
            queries,
            max_length=max_length,
            batch_size=batch_size,
            do_chunking=False,
            stride=0,
        )

        hidden_size = encodings["token_embeds"][0].shape[2]
        query_embeds = []
        for batch_id in tqdm(encodings["batch_id"][::-1], desc="Pooling"):
            token_embeds = encodings["token_embeds"].pop(batch_id)
            input_ids = encodings["input_ids"].pop(batch_id)
            valid_token_mask = np.isin(
                input_ids, self.exclude_token_ids, invert=True
            ).astype("uint8")
            valid_token_weight = valid_token_mask[:, :, np.newaxis].repeat(
                hidden_size, axis=2
            )
            query_embeds.append(
                np.sum(token_embeds * valid_token_weight, axis=1)
                / valid_token_weight.sum(axis=1)
            )
        query_embeds = self.quantize_if_needed(np.vstack(query_embeds[::-1]))
        return query_embeds


class ApproxPhraseFoundry(PhraseFoundry):
    def __init__(
        self,
        model_name: str,
        n_pca_components: int = 64,
        n_pca_training_samples: int = 10**5,
        amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
        quantize_embeds: bool = True,
        device: torch.device | str | int = "cuda",
        invalid_start_token_pattern: str | None = r"^##",
        exclude_tokens: list[str] | list[int] | None = None,
    ) -> None:
        """Initialize an ApproxPhraseFoundry model.

        Parameters
        ----------
        model_name : str
            Name of the pretrained model to use.
        n_pca_components : int, optional
            Number of components for PCA, by default 64.
        n_pca_training_samples : int, optional
            Minimum number of token embeddings to use for training PCA, by default 10**5.
            After PCA has seen at least this number of samples, it will no longer be
            updated and will instead be applied to all token embeddings.
        amp : bool, optional
            Enable automatic mixed precision, by default False.
        amp_dtype : torch.dtype, optional
            Data type for automatic mixed precision, by default torch.float16.
        quantize_embeds : bool, optional
            Reduce the embedding precision to float16 if they are float32 or float64,
            by default True.
        device : torch.device, str, int, optional
            Device to use for inference, by default "cuda".
        invalid_start_token_pattern : str, None, optional
            Regular expression pattern for invalid start tokens, by default r"^##".
        exclude_tokens : list[str], list[int], None, optional
            List of tokens to exclude from n-gram extraction, by default None.
        """
        super().__init__(
            model_name,
            amp=amp,
            amp_dtype=amp_dtype,
            device=device,
            invalid_start_token_pattern=invalid_start_token_pattern,
            exclude_tokens=exclude_tokens,
        )
        self.n_pca_components = n_pca_components
        self.n_pca_training_samples = n_pca_training_samples
        self.quantize_embeds = quantize_embeds

    @property
    def pca_training_complete(self) -> bool:
        """Returns True if PCA has seen enough samples to be applied."""
        return (
            hasattr(self, "pca_")
            and self.pca_.n_samples_seen_ >= self.n_pca_training_samples
        )

    def update_pca(self, token_embeds: np.ndarray) -> None:
        """Update the PCA model with a batch of token embeddings.

        Parameters
        ----------
        token_embeds : np.ndarray
            Token embeddings to update the PCA model with.
        """
        if not hasattr(self, "pca_"):
            self.pca_ = IncrementalPCA(n_components=self.n_pca_components)
        if token_embeds.ndim == 3:
            token_embeds = token_embeds.reshape(-1, token_embeds.shape[2])
        self.pca_.partial_fit(token_embeds)
        if hasattr(self.pca_, "n_batches_seen_"):
            self.pca_.n_batches_seen_ += 1
        else:
            self.pca_.n_batches_seen_ = 1

    def apply_pca(self, token_embeds: np.ndarray) -> np.ndarray:
        """Apply PCA to token embeddings.

        Parameters
        ----------
        token_embeds : np.ndarray
            Token embeddings to apply PCA to.

        Returns
        -------
        np.ndarray
            PCA-transformed token embeddings.
        """
        if not hasattr(self, "pca_"):
            raise AttributeError("PCA must be fitted first.")
        if not self.pca_.n_samples_seen_ >= self.n_pca_training_samples:
            raise RuntimeError("PCA has not seen enough samples to be applied yet.")
        ndim = token_embeds.ndim
        if ndim == 3:
            seq_len = token_embeds.shape[1]
            token_embeds = token_embeds.reshape(-1, token_embeds.shape[2])
        low = self.pca_.transform(token_embeds)
        if ndim == 3:
            low = low.reshape(-1, seq_len, low.shape[1])
        # Apply quantization after PCA if needed
        low = self.quantize_if_needed(low)
        return low

    def clear_pca(self) -> None:
        """Clear the PCA model."""
        if hasattr(self, "pca_"):
            del self.pca_

    def encode_extract(
        self,
        docs: list[str],
        max_length: int | None = 512,
        batch_size: int = 32,
        do_chunking: bool = True,
        stride: int = 128,
        ngram_range: tuple[int, int] = (5, 5),
        random_state: int | None = None,
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
            Range of n-gram sizes to extract, by default (5, 5).
        random_state : int, None, optional
            Random seed for shuffling, by default None. Only used if PCA is not yet trained.

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
        data = TokenizedDataset(
            inputs, shuffle=not self.pca_training_complete, random_state=random_state
        )
        loader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
        samp_idx = []
        ngrams = []
        ngram_embeds = []
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.amp, dtype=self.amp_dtype):
                for batch_id, (idx, batch) in enumerate(tqdm(loader, desc="Encoding")):
                    batch = {
                        k: v.to(self.device, non_blocking=True)
                        for k, v in batch.items()
                    }
                    outputs = self.model(**batch)
                    overflow_to_sample_mapping = None
                    if do_chunking:
                        overflow_to_sample_mapping = inputs[
                            "overflow_to_sample_mapping"
                        ][idx].numpy()
                    token_embeds = outputs.last_hidden_state.cpu().numpy()
                    ng_samp_idx, ng_text, embeds = self.extract_ngrams(
                        inputs["input_ids"][idx].numpy(),
                        token_embeds,
                        ngram_range=ngram_range,
                        overflow_to_sample_mapping=overflow_to_sample_mapping,
                    )
                    if self.pca_training_complete:
                        embeds = self.apply_pca(embeds)
                    else:
                        self.update_pca(embeds)
                        # Apply PCA to all token embeddings if training is complete
                        if (
                            self.pca_training_complete
                            and self.pca_.n_batches_seen_ == (batch_id + 1)
                        ):
                            print(
                                f"PCA training complete after {self.pca_.n_batches_seen_} batches."
                            )
                            print("Applying PCA to all n-gram embeddings.")
                            embeds = self.apply_pca(embeds)
                            ngram_embeds = [self.apply_pca(e) for e in ngram_embeds]
                    ngram_embeds.append(embeds)
                    samp_idx.append(ng_samp_idx)
                    ngrams.append(ng_text)
        if not self.pca_training_complete:
            raise RuntimeError(
                "PCA training failed to complete. Reduce `n_pca_training_samples`."
            )
        samp_idx = np.hstack(samp_idx)
        reorder_idx = np.argsort(samp_idx) if data.shuffle else samp_idx
        ngrams = np.hstack(ngrams)[reorder_idx]
        ngram_embeds = np.vstack(ngram_embeds)[reorder_idx]
        samp_idx = samp_idx[reorder_idx]
        inputs = {k: v.numpy() for k, v in inputs.items()}
        return samp_idx, ngrams, ngram_embeds

    def encode_queries(
        self,
        queries: list[str],
        max_length: int | None = 512,
        batch_size: int = 32,
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

        Returns
        -------
        np.ndarray
            Mean-token embeddings for each query.
        """
        if not self.pca_training_complete:
            raise ValueError("Cannot encode queries if PCA has not been fit.")
        orig_quantize_embeds_setting = self.quantize_embeds
        self.quantize_embeds = False  # Turn quantization off to apply after PCA
        query_embeds = super().encode_queries(
            queries, max_length=max_length, batch_size=batch_size
        )
        self.quantize_embeds = orig_quantize_embeds_setting  # Restore original setting
        return self.apply_pca(query_embeds)  # Apply PCA, quantizing if needed
