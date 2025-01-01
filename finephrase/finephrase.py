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

"""FinePhrase is a library for extracting n-grams from text using transformer models."""

from collections import defaultdict
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize


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


class FinePhrase:
    def __init__(
        self,
        model_name: str,
        amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
        quantize_embeds: bool = True,
        normalize_embeds: bool = False,
        device: torch.device | str | int = "cuda",
        invalid_start_token_pattern: str | None = r"^##",
        exclude_tokens: list[str] | list[int] | None = None,
    ) -> None:
        """Initialize a FinePhrase model.

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
        normalize_embeds : bool, optional
            Normalize the embeddings to unit length, by default False.
            This is useful for quick cosine similarity calculations downstream, since
            the dot product of two unit vectors is equal to the cosine similarity.
            It is also useful if you want downstream Euclidean distance calculations
            to consider only the direction of the vectors, not their magnitude.
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
        self.normalize_embeds = normalize_embeds
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

    def to(self, device: torch.device | str | int) -> "FinePhrase":
        """Move the model to a new device.

        Parameters
        ----------
        device : torch.device, str, int
            Device to move the model to.

        Returns
        -------
        FinePhrase
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

    def normalize_if_needed(self, embeds: np.ndarray, copy=False) -> np.ndarray:
        """Normalize the embeddings if needed.

        Parameters
        ----------
        embeds : np.ndarray
            Embeddings to normalize.
        copy : bool, optional
            Whether to copy the matrix before normalizing, by default False.
            If false, will try to do the normalization in-place. Does nothing
            if normalization is not needed.

        Returns
        -------
        np.ndarray
            Normalized embeddings.
        """
        if embeds.ndim != 2:
            raise ValueError("Embeddings must be 2D.")
        if self.normalize_embeds:
            embeds = normalize(embeds, axis=1, copy=copy, norm="l2")
        return embeds

    def postprocess(self, embeds: np.ndarray) -> np.ndarray:
        """Apply all postprocessing steps to the embeddings.

        The steps are:
        1. Normalize embeddings to unit length, if enabled.
        2. Quantize embeddings to float16, if enabled.

        Parameters
        ----------
        embeds : np.ndarray
            Embeddings to postprocess.

        Returns
        -------
        np.ndarray
            Postprocessed embeddings.
        """
        return self.quantize_if_needed(self.normalize_if_needed(embeds))

    def _extract_ngrams(
        self,
        sequence_idx: np.ndarray,
        input_ids: np.ndarray,
        token_embeds: np.ndarray,
        ngram_range: tuple[int, int] = (5, 5),
    ) -> dict[str, np.ndarray]:
        """Extract the n-grams and n-gram embeddings from token embeddings.

        Parameters
        ----------
        sequence_idx : np.ndarray
            Sequence indices.
        input_ids : np.ndarray
            Tokenized input sequences.
        token_embeds : np.ndarray
            Token embeddings.
        ngram_range : tuple[int, int], optional
            Range of n-gram sizes to extract, by default (5, 5).

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing the sample indices, n-grams, and n-gram embeddings.

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
        results = defaultdict(list)
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
            ngram_embeds = token_embeds[:, idx][valid_ngrams].mean(axis=1)
            results["sequence_idx"].append(
                np.repeat(sequence_idx, valid_ngrams.sum(axis=1))
            )
            results["ngrams"].extend(ngrams)
            results["ngram_embeds"].append(ngram_embeds)

        # Combine results
        results["sequence_idx"] = np.hstack(results["sequence_idx"])
        results["ngrams"] = np.array(results["ngrams"], dtype="U")
        results["ngram_embeds"] = np.vstack(results["ngram_embeds"])
        return dict(results)

    def _generate_embeddings(self, loader: DataLoader):
        """Obtain the token embeddings for a list of documents, one batch at at time."""
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.amp, dtype=self.amp_dtype):
                for batch_id, (idx, batch) in enumerate(tqdm(loader, desc="Encoding")):
                    batch = {
                        k: v.to(self.device, non_blocking=True)
                        for k, v in batch.items()
                    }
                    outputs = self.model(**batch)
                    yield {
                        "batch_id": batch_id,
                        "sequence_idx": idx,
                        "token_embeds": outputs.last_hidden_state.cpu().numpy(),
                    }

    def _encode(
        self,
        docs: list[str],
        max_length: int | None = None,
        batch_size: int = 32,
        do_chunking: bool = True,
        stride: int = 0,
    ) -> dict[str, np.ndarray]:
        """Obtain the token embeddings for a list of documents.

        This is a low-level method that encodes the input documents into token embeddings.

        Parameters
        ----------
        docs : list[str]
            List of documents to encode.
        max_length : int, optional
            Maximum length of the input sequences, by default None.
        batch_size : int, optional
            Batch size for encoding, by default 32.
        do_chunking : bool, optional
            Enable chunking of documents into overlapping sequences, by default True.
        stride : int, optional
            Stride for splitting documents into overlapping sequences, by default 0.
            Only used if `do_chunking` is True.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing the batch indices, sequence indices, input IDs,
            token embeddings, and overflow to sample mapping (if chunking is enabled).
            The batches are left intact, meaning that, for example, the token embeddings
            are stored as a list of arrays, where each array corresponds to a batch.
        """
        if max_length is None:
            if self.tokenizer.model_max_length is None:
                raise ValueError(
                    "max_length must be specified if tokenizer.model_max_length is None"
                )
            max_length = self.tokenizer.model_max_length
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
        results = defaultdict(list)
        for batch in self._generate_embeddings(loader):
            results["batch_id"].append(batch["batch_id"])
            results["sequence_idx"].append(batch["sequence_idx"])
            results["token_embeds"].append(batch["token_embeds"])
            results["input_ids"].append(
                inputs["input_ids"][batch["sequence_idx"]].numpy()
            )
        if do_chunking:
            results["overflow_to_sample_mapping"] = inputs[
                "overflow_to_sample_mapping"
            ].numpy()
        return dict(results)

    def encode_extract(
        self,
        docs: list[str],
        max_length: int | None = None,
        batch_size: int = 32,
        do_chunking: bool = True,
        stride: int = 0,
        ngram_range: tuple[int, int] = (5, 5),
    ) -> dict[str, np.ndarray]:
        """Obtain the n-grams and n-gram embeddings from a list of documents.

        This is equivalent to calling `encode` followed by `extract_ngrams`.
        It first encodes the input documents, then extracts the n-grams and
        n-gram embeddings from the token embeddings.

        Parameters
        ----------
        docs : list[str]
            List of documents to encode.
        max_length : int, optional
            Maximum length of the input sequences, by default None.
        batch_size : int, optional
            Batch size for encoding, by default 32.
        do_chunking : bool, optional
            Enable chunking of documents into overlapping sequences, by default True.
        stride : int, optional
            Stride for splitting documents into overlapping sequences, by default 0.
            Only used if `do_chunking` is True.
        ngram_range : tuple[int, int], optional
            Range of n-gram sizes to extract, by default (5, 5).

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing the sample indices, n-grams, and n-gram embeddings.

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
        results = defaultdict(list)
        # Work backwards and pop from the encodings to conserve memory
        for batch_id in tqdm(encodings["batch_id"][::-1], desc="Extracting"):
            ngram_batch = self._extract_ngrams(
                encodings["sequence_idx"].pop(batch_id),
                encodings["input_ids"].pop(batch_id),
                encodings["token_embeds"].pop(batch_id),
                ngram_range=ngram_range,
            )
            results["sequence_idx"].append(ngram_batch["sequence_idx"])
            results["ngrams"].append(ngram_batch["ngrams"])
            results["ngram_embeds"].append(
                self.postprocess(ngram_batch["ngram_embeds"])
            )
        results["sequence_idx"] = np.hstack(results["sequence_idx"][::-1])
        results["ngrams"] = np.hstack(results["ngrams"][::-1])
        results["ngram_embeds"] = np.vstack(results["ngram_embeds"][::-1])
        if do_chunking:
            mapping = encodings["overflow_to_sample_mapping"]
            results["sample_idx"] = mapping[results["sequence_idx"]]
        else:
            results["sample_idx"] = results["sequence_idx"]
        return dict(results)

    def encode_queries(
        self,
        queries: list[str],
        max_length: int | None = None,
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
            Maximum length of the input sequences, by default None.
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
        # Work backwards and pop from the encodings to conserve memory
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
        query_embeds = self.postprocess(np.vstack(query_embeds[::-1]))
        return query_embeds


class FinePhrasePCA(FinePhrase):
    def __init__(
        self,
        model_name: str,
        n_pca_components: int = 64,
        n_pca_training_batches: float | int = 0.25,
        amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
        quantize_embeds: bool = True,
        normalize_embeds: bool = False,
        device: torch.device | str | int = "cuda",
        invalid_start_token_pattern: str | None = r"^##",
        exclude_tokens: list[str] | list[int] | None = None,
    ) -> None:
        """Initialize an FinePhrasePCA model.

        Parameters
        ----------
        model_name : str
            Name of the pretrained model to use.
        n_pca_components : int, optional
            Number of components for PCA, by default 64.
        n_pca_training_batches : float, int, optional
            Number or fraction of batches to use for training PCA, by default 0.25.
            If a float, it is interpreted as a fraction of the total number of batches
            at the initial calling of `encode_extract`. If an integer, it is interpreted
            as the number of batches. After PCA has seen at least this number of batches,
            it will no longer be updated and will instead be applied to all token embeddings.
            This is true even if `encode_extract` is called a second time on new data.
        amp : bool, optional
            Enable automatic mixed precision, by default False.
        amp_dtype : torch.dtype, optional
            Data type for automatic mixed precision, by default torch.float16.
        quantize_embeds : bool, optional
            Reduce the embedding precision to float16 if they are float32 or float64,
            by default True.
        normalize_embeds : bool, optional
            Normalize the embeddings to unit length, by default False.
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
            quantize_embeds=quantize_embeds,
            normalize_embeds=normalize_embeds,
            device=device,
            invalid_start_token_pattern=invalid_start_token_pattern,
            exclude_tokens=exclude_tokens,
        )
        self.n_pca_components = n_pca_components
        self.n_pca_training_batches = n_pca_training_batches

    @property
    def pca_training_complete(self) -> bool:
        """Returns True if PCA has seen enough samples to be applied."""
        return (
            hasattr(self, "pca_")
            and hasattr(self, "n_pca_training_batches_")
            and hasattr(self.pca_, "n_batches_seen_")
            and self.pca_.n_batches_seen_ >= self.n_pca_training_batches_
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

    def apply_pca(self, embeds: np.ndarray) -> np.ndarray:
        """Apply PCA to embeddings.

        Parameters
        ----------
        embeds : np.ndarray
            Embeddings to apply PCA to.

        Returns
        -------
        np.ndarray
            PCA-transformed embeddings.
        """
        if not hasattr(self, "pca_"):
            raise AttributeError("PCA must be fitted first.")
        if not self.pca_.n_samples_seen_ >= self.n_pca_training_batches:
            raise RuntimeError("PCA has not seen enough samples to be applied yet.")
        ndim = embeds.ndim
        if ndim == 3:
            seq_len = embeds.shape[1]
            embeds = embeds.reshape(-1, embeds.shape[2])
        low = self.pca_.transform(embeds)
        if ndim == 3:
            low = low.reshape(-1, seq_len, low.shape[1])
        return low

    def clear_pca(self) -> None:
        """Clear the PCA model."""
        if hasattr(self, "pca_"):
            del self.pca_
        if hasattr(self, "n_pca_training_batches_"):
            del self.n_pca_training_batches_

    def postprocess(self, embeds):
        """Apply all postprocessing steps to the embeddings.

        The steps are:
        1. Normalize embeddings to unit length, if enabled.
        2. Apply PCA to the embeddings.
        3. Normalize embeddings to unit length again, if enabled.
        4. Quantize embeddings to float16, if enabled.

        Parameters
        ----------
        embeds : np.ndarray
            Embeddings to postprocess.

        Returns
        -------
        np.ndarray
            Postprocessed embeddings.
        """
        steps = (
            self.normalize_if_needed,
            self.apply_pca,
            self.normalize_if_needed,
            self.quantize_if_needed,
        )
        for step in steps:
            embeds = step(embeds)
        return embeds

    def encode_extract(
        self,
        docs: list[str],
        max_length: int | None = None,
        batch_size: int = 32,
        do_chunking: bool = True,
        stride: int = 0,
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
            Maximum length of the input sequences, by default None.
        batch_size : int, optional
            Batch size for encoding, by default 32.
        do_chunking : bool, optional
            Enable chunking of documents into overlapping sequences, by default True.
        stride : int, optional
            Stride for splitting documents into overlapping sequences, by default 0.
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
        # Determine number of batches to use for PCA training
        if isinstance(self.n_pca_training_batches, float):
            self.n_pca_training_batches_ = round(
                self.n_pca_training_batches * len(loader)
            )
        else:
            self.n_pca_training_batches_ = self.n_pca_training_batches
        if self.n_pca_training_batches_ > len(loader):
            raise ValueError(
                f"n_pca_training_batches must be less than or equal to the number of batches ({len(loader)})"
            )
        results = defaultdict(list)
        for batch in self._generate_embeddings(loader):
            ngram_batch = self._extract_ngrams(
                batch["sequence_idx"],
                inputs["input_ids"][batch["sequence_idx"]].numpy(),
                batch["token_embeds"],
                ngram_range=ngram_range,
            )
            if self.pca_training_complete:
                ngram_batch["ngram_embeds"] = self.postprocess(
                    ngram_batch["ngram_embeds"]
                )
            else:
                self.update_pca(self.normalize_if_needed(ngram_batch["ngram_embeds"]))
                # Retroactively apply PCA when training completes
                if self.pca_training_complete:
                    print(
                        f"PCA training complete after {self.pca_.n_batches_seen_} batches."
                    )
                    print("Applying PCA to all n-gram embeddings.")
                    results["ngram_embeds"] = [
                        self.postprocess(e) for e in results["ngram_embeds"]
                    ]
                    ngram_batch["ngram_embeds"] = self.postprocess(
                        ngram_batch["ngram_embeds"]
                    )
            results["sequence_idx"].append(ngram_batch["sequence_idx"])
            results["ngrams"].append(ngram_batch["ngrams"])
            results["ngram_embeds"].append(ngram_batch["ngram_embeds"])
        if not self.pca_training_complete:
            raise RuntimeError("PCA training failed to complete.")
        # Combine results
        results["sequence_idx"] = np.hstack(results["sequence_idx"])
        results["ngrams"] = np.hstack(results["ngrams"])
        results["ngram_embeds"] = np.vstack(results["ngram_embeds"])
        # Reorder results if shuffling was enabled
        if data.shuffle:
            reorder_idx = np.argsort(results["sequence_idx"])
            results["sequence_idx"] = results["sequence_idx"][reorder_idx]
            results["ngrams"] = results["ngrams"][reorder_idx]
            results["ngram_embeds"] = results["ngram_embeds"][reorder_idx]
        if do_chunking:
            mapping = inputs["overflow_to_sample_mapping"].numpy()
            results["sample_idx"] = mapping[results["sequence_idx"]]
        else:
            results["sample_idx"] = results["sequence_idx"]
        return dict(results)
