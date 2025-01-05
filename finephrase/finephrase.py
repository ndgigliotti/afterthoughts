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

import warnings
import numpy as np
import torch
import math
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
from finephrase.pca import IncrementalPCA
from finephrase.utils import normalize, reduce_precision


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


def get_phrase_idx(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    phrase_sizes: list | tuple | int,
    overlap: int | float | list | dict = 0.5,
    phrase_min_token_ratio: float = 0.5,
    sequence_idx: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
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
        "valid_phrase_mask": [],
        "sequence_idx": [],
    }
    if isinstance(phrase_sizes, int):
        phrase_sizes = [phrase_sizes]
    for i, size in enumerate(phrase_sizes):
        # Check if `overlap` is a float in [0, 1)
        if isinstance(overlap, float):
            if overlap < 0 or overlap >= 1:
                raise ValueError("`overlap` must be in [0, 1).")
            # Calculate the number of tokens to overlap
            overlap_tokens = math.ceil(size * overlap)
            if overlap_tokens == size:
                overlap_tokens -= 1
        elif isinstance(overlap, (list, tuple)):
            overlap_tokens = overlap[i]
        elif isinstance(overlap, dict):
            overlap_tokens = overlap[size]
        elif isinstance(overlap, int):
            overlap_tokens = overlap
        else:
            raise ValueError("`overlap` must be a float, list, tuple, dict, or int.")
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
        results["valid_phrase_mask"].append(valid_phrase_mask)
        results["sequence_idx"].append(phrase_sequence_idx)
    return results


def _move_or_convert_results(results, return_tensors="pt", move_results_to_cpu=False):
    if move_results_to_cpu:
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                results[key] = value.cpu()
    if return_tensors == "np":
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                results[key] = value.numpy()
    elif not return_tensors == "pt":
        raise ValueError("`return_tensors` must be 'np' or 'pt'.")
    return results


class FinePhrase:
    def __init__(
        self,
        model_name: str,
        amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
        reduce_precision: bool = False,
        normalize_embeds: bool = False,
        pca: int | None = None,
        pca_fit_batch_count: int | float = 1.0,
        device: torch.device | str | int = "cuda",
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
        reduce_precision : bool, optional
            Reduce the embedding precision to float16 if they are float32 or float64,
            by default False.
        normalize_embeds : bool, optional
            Normalize the embeddings to unit length, by default False.
            This is useful for quick cosine similarity calculations downstream, since
            the dot product of two unit vectors is equal to the cosine similarity.
            It is also useful if you want downstream Euclidean distance calculations
            to consider only the direction of the vectors, not their magnitude.
        pca : int, None, optional
            Number of principal components to keep after PCA, by default None.
            If None, PCA is not fit or applied.
        pca_fit_batch_count : int, float, optional
            Number of batches to use for fitting the PCA model, by default 1.0.
            If an integer, it is the number of batches to use. If a float, it is the
            fraction of the dataset to use (on the first call to `encode()`). If 1.0,
            the entire dataset passed to the `encode()` method is used. Once the PCA
            transformation is fit, it is applied to all embeddings.
        device : torch.device, str, int, optional
            Device to use for inference, by default "cuda".
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, clean_up_tokenization_spaces=True
        )
        self.model = AutoModel.from_pretrained(model_name)
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.quantize_embeds = reduce_precision
        self.normalize_embeds = normalize_embeds
        self.pca = pca
        self.pca_fit_batch_count = pca_fit_batch_count
        self.model.eval().to(device)

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

    def reduce_precision_if_needed(
        self, embeds: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Quantize the embeddings if needed."""
        if self.quantize_embeds:
            embeds = reduce_precision(embeds)
        return embeds

    def normalize_if_needed(
        self, embeds: torch.Tensor | np.ndarray, dim: int = 1
    ) -> torch.Tensor | np.ndarray:
        """Normalize the embeddings if needed.

        Parameters
        ----------
        embeds : torch.Tensor or np.ndarray
            Embeddings to normalize.
        dim : int
            Dimension to normalize.

        Returns
        -------
        torch.Tensor or np.ndarray
            Normalized embeddings.
        """
        if self.normalize_embeds:
            embeds = normalize(embeds, dim=dim)
        return embeds

    def postprocess(self, embeds: np.ndarray) -> np.ndarray:
        """Apply all postprocessing steps to the embeddings.

        The steps are:
        1. Normalize embeddings to unit length, if enabled.
        2. Reduce precision to float16, if enabled.

        Parameters
        ----------
        embeds : np.ndarray
            Embeddings to postprocess.

        Returns
        -------
        np.ndarray
            Postprocessed embeddings.
        """
        return self.reduce_precision_if_needed(self.normalize_if_needed(embeds))

    def _extract_phrases(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_embeds: torch.Tensor,
        sequence_idx: torch.Tensor,
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
            torch.tensor(self.tokenizer.all_special_ids, device=input_ids.device),
            invert=True,
        ).to(torch.uint8)
        results = {"sequence_idx": [], "phrases": [], "phrase_embeds": []}
        for i, idx in enumerate(phrase_data["phrase_idx"]):
            attn_factor = attention_mask[:, idx].unsqueeze(3)
            phrase_embeds = torch.sum(token_embeds[:, idx] * attn_factor, dim=2) / (
                torch.clamp(attn_factor.sum(dim=2), min=1)
            )
            phrase_embeds = phrase_embeds[phrase_data["valid_phrase_mask"][i]]
            phrases = self.tokenizer.batch_decode(
                phrase_data["phrase_ids"][i],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            results["sequence_idx"].append(phrase_data["sequence_idx"][i])
            results["phrases"].extend(phrases)
            results["phrase_embeds"].append(phrase_embeds)

        # Combine results
        results["sequence_idx"] = torch.hstack(results["sequence_idx"])
        results["phrase_embeds"] = torch.vstack(results["phrase_embeds"])
        return dict(results)

    def _tokenize(
        self,
        docs: list[str],
        max_length: int | None = None,
        chunk_docs: bool = True,
        doc_overlap: float | int = 0.5,
    ) -> dict[str, np.ndarray]:
        """Tokenize a list of documents into input sequences for the model.

        Parameters
        ----------
        docs : list[str]
            List of documents to tokenize.
        max_length : int, optional
            Maximum length of the input sequences, by default None.
        chunk_docs : bool, optional
            Enable chunking of documents into overlapping sequences, by default True.
        doc_overlap : float, int, optional
            Overlap for splitting long documents into overlapping sequences due to the
            model's max sequence length limit, by default 0.5. Tokenized documents which fit
            within `max_length` will not be chunked. If a float, it is interpreted as a
            fraction of the maximum sequence length. If an integer, it is interpreted
            as the number of tokens to overlap. Does nothing if `chunk_docs` is False.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing the tokenized input sequences.

        Raises
        ------
        ValueError
            If `max_length` is not specified and `tokenizer.model_max_length` is None.
        """
        if max_length is None:
            if self.tokenizer.model_max_length is None:
                raise ValueError(
                    "`max_length` must be specified if `tokenizer.model_max_length` is None"
                )
            max_length = self.tokenizer.model_max_length
        if chunk_docs:
            if isinstance(doc_overlap, float):
                if doc_overlap < 0 or doc_overlap >= 1:
                    raise ValueError("`doc_overlap` must be in [0, 1).")
                doc_overlap = math.ceil(max_length * doc_overlap)
                if doc_overlap == max_length:
                    doc_overlap -= 1
            elif isinstance(doc_overlap, int):
                if doc_overlap >= max_length:
                    raise ValueError("`doc_overlap` must be less than `max_length`.")
                elif doc_overlap < 0:
                    raise ValueError(
                        "`doc_overlap` must be greater than or equal to 0."
                    )
            else:
                raise ValueError("`doc_overlap` must be a float or an integer.")
        else:
            doc_overlap = 0
        inputs = self.tokenizer(
            docs,
            max_length=max_length,
            padding="longest",
            truncation=True,
            return_overflowing_tokens=chunk_docs,
            stride=doc_overlap,
            return_tensors="pt",
            add_special_tokens=True,
            return_attention_mask=True,
        )
        return inputs

    def _generate_token_embeds(
        self,
        loader: DataLoader,
        move_results_to_cpu: bool = False,
        return_tensors: str = "pt",
    ):
        """Obtain the token embeddings for a list of documents, one batch at at time."""
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.amp, dtype=self.amp_dtype):
                progress_loader = tqdm(loader, desc="Encoding")
                for batch_id, (sequence_idx, batch) in enumerate(progress_loader):
                    batch = {
                        k: v.to(self.device, non_blocking=True)
                        for k, v in batch.items()
                    }
                    outputs = self.model(**batch)
                    results = {
                        "sequence_idx": sequence_idx,
                        "input_ids": batch["input_ids"],
                        "attention_mask": batch["attention_mask"],
                        "token_embeds": outputs.last_hidden_state,
                        "batch_id": torch.full(sequence_idx.shape, batch_id),
                    }
                    _move_or_convert_results(
                        results,
                        return_tensors=return_tensors,
                        move_results_to_cpu=move_results_to_cpu,
                    )
                    yield results

    def _generate_phrase_embeds(
        self,
        loader: DataLoader,
        phrase_sizes: int | list | tuple,
        phrase_overlap: int | float | list | dict,
        phrase_min_token_ratio: float,
        move_results_to_cpu: bool = False,
        return_tensors: str = "pt",
    ):
        """Obtain the phrase embeddings for a list of documents, one batch at at time."""
        batches = self._generate_token_embeds(
            loader, move_results_to_cpu=False, return_tensors="pt"
        )
        for batch in batches:
            results = self._extract_phrases(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_embeds=batch["token_embeds"],
                sequence_idx=batch["sequence_idx"].to(self.device),
                phrase_sizes=phrase_sizes,
                overlap=phrase_overlap,
                phrase_min_token_ratio=phrase_min_token_ratio,
            )
            results["batch_id"] = torch.full(
                results["sequence_idx"].shape, batch["batch_id"][0]
            )
            _move_or_convert_results(
                results,
                return_tensors=return_tensors,
                move_results_to_cpu=move_results_to_cpu,
            )
            yield results

    @property
    def pca_mode(self) -> bool:
        """Returns True if PCA is enabled."""
        return self.pca is not None

    @property
    def pca_is_ready(self) -> bool:
        """Returns True if PCA has seen enough batches to be applied."""
        return (
            hasattr(self, "pca_transform_")
            and hasattr(self, "pca_fit_batch_count_")
            and hasattr(self.pca_transform_, "n_batches_seen_")
            and self.pca_transform_.n_batches_seen_ >= self.pca_fit_batch_count_
        )

    def update_pca(self, phrase_embeds: torch.Tensor) -> None:
        """Update the PCA transformation with a batch of token embeddings.

        Parameters
        ----------
        phrase_embeds : torch.Tensor
            Phrase embeddings to update the PCA model with.
        """
        if not hasattr(self, "pca_transform_"):
            self.pca_transform_ = IncrementalPCA(
                n_components=self.pca, device=self.device
            )
        self.pca_transform_.partial_fit(phrase_embeds)
        if hasattr(self.pca_transform_, "n_batches_seen_"):
            self.pca_transform_.n_batches_seen_ += 1
        else:
            self.pca_transform_.n_batches_seen_ = 1

    def apply_pca(self, phrase_embeds: torch.Tensor) -> torch.Tensor:
        """Apply PCA transformation to embeddings.

        Parameters
        ----------
        phrase_embeds : torch.Tensor
            Phrase embeddings to apply PCA to.

        Returns
        -------
        torch.Tensor
            PCA-transformed embeddings.
        """
        if not hasattr(self, "pca_transform_"):
            raise AttributeError("PCA must be fitted first.")
        if not self.pca_is_ready:
            raise RuntimeError("PCA has not seen enough batches to be applied yet.")
        return self.pca_transform_.transform(phrase_embeds)

    def clear_pca(self) -> None:
        """Clear the fitted PCA transformation."""
        if hasattr(self, "pca_transform_"):
            del self.pca_transform_
        if hasattr(self, "pca_fit_batch_count_"):
            del self.pca_fit_batch_count_

    def encode(
        self,
        docs: list[str],
        max_length: int | None = None,
        batch_size: int = 32,
        phrase_sizes: int | list | tuple = 12,
        phrase_overlap: int | float | list | dict = 0.5,
        phrase_min_token_ratio: float = 0.5,
        chunk_docs: bool = True,
        doc_overlap: float | int = 0.5,
        convert_to_numpy: bool = True,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """Obtain the n-grams and n-gram embeddings from a list of documents.

        This first encodes the input documents, then extracts the n-grams and
        n-gram embeddings from the token embeddings.

        Parameters
        ----------
        docs : list[str]
            List of documents to encode.
        max_length : int, optional
            Maximum length of the input sequences, by default None.
        batch_size : int, optional
            Batch size for encoding, by default 32.
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
        chunk_docs : bool, optional
            Enable chunking of documents into overlapping sequences, by default True.
            This is useful for long documents that exceed the model's maximum sequence length,
            as it allows the model to process the document in overlapping chunks. Documents
            that fit within the maximum sequence length will not be chunked.
        doc_overlap : float or int, optional
            Overlap for splitting long documents into overlapping sequences due to the
            model's max sequence length limit, by default 0.5. Tokenized documents which fit
            within `max_length` will not be chunked. If a float, it is interpreted as a
            fraction of the maximum sequence length. If an integer, it is interpreted
            as the number of tokens to overlap. Does nothing if `chunk_docs` is False.
        convert_to_numpy : bool, optional
            Convert the tensors to numpy arrays before returning, by default True.

        Returns
        -------
        dict[str, np.ndarray or torch.Tensor]
            Dictionary containing the sample indices, phrases, and phrase embeddings.

        Raises
        ------
        ValueError
            If `max_length` is not specified and `tokenizer.model_max_length` is None.

        """
        inputs = self._tokenize(
            docs, max_length=max_length, chunk_docs=chunk_docs, doc_overlap=doc_overlap
        )
        data = TokenizedDataset(inputs)
        loader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
        batches = self._generate_phrase_embeds(
            loader,
            phrase_sizes=phrase_sizes,
            phrase_overlap=phrase_overlap,
            phrase_min_token_ratio=phrase_min_token_ratio,
            move_results_to_cpu=False,
            return_tensors="pt",
        )
        if self.pca_mode:
            if isinstance(self.pca_fit_batch_count, float):
                self.pca_fit_batch_count_ = math.ceil(
                    len(loader) * self.pca_fit_batch_count
                )
            else:
                self.pca_fit_batch_count_ = self.pca_fit_batch_count
        results = {
            "batch_id": [],
            "sequence_idx": [],
            "phrases": [],
            "phrase_embeds": [],
        }
        for batch in batches:
            if not self.pca_mode:
                # Postprocess on the fly to potentially conserve memory
                batch["phrase_embeds"] = self.postprocess(batch["phrase_embeds"])
            else:
                if self.pca_is_ready:
                    # Apply PCA and postprocess to potentially conserve memory
                    batch["phrase_embeds"] = self.postprocess(
                        self.apply_pca(batch["phrase_embeds"])
                    )
                else:
                    # Update PCA if not ready yet
                    self.update_pca(batch["phrase_embeds"])
            # Offload batch to CPU
            batch = _move_or_convert_results(
                batch, return_tensors="pt", move_results_to_cpu=True
            )
            results["batch_id"].append(batch["batch_id"])
            results["sequence_idx"].append(batch["sequence_idx"])
            results["phrases"].extend(batch["phrases"])
            results["phrase_embeds"].append(batch["phrase_embeds"])
        # Process early batches with PCA if necessary
        if self.pca_mode:
            if self.pca_is_ready:
                self.pca_transform_.to("cpu")  # Temporarily move to CPU
                for i in range(self.pca_fit_batch_count_):
                    results["phrase_embeds"][i] = self.postprocess(
                        self.apply_pca(results["phrase_embeds"][i])
                    )
                self.pca_transform_.to(self.device)  # Move back to device
            else:
                warnings.warn("PCA did not finish fitting and will not be applied.")
        # Combine results
        for key, value in results.items():
            if isinstance(value[0], torch.Tensor):
                if value[0].ndim == 1:
                    results[key] = torch.hstack(value)
                elif value[0].ndim == 2:
                    results[key] = torch.vstack(value)
                else:
                    raise ValueError(f"Unsupported dimension {value[0].ndim}.")
        if chunk_docs:
            mapping = inputs["overflow_to_sample_mapping"]
            results["sample_idx"] = mapping[results["sequence_idx"]]
        else:
            results["sample_idx"] = results["sequence_idx"]
        if convert_to_numpy:
            _move_or_convert_results(results, return_tensors="np")
        return results

    def encode_queries(
        self,
        queries: list[str],
        max_length: int | None = None,
        batch_size: int = 32,
        convert_to_numpy: bool = True,
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
        convert_to_numpy : bool, optional
            Convert the tensors to numpy arrays before returning, by default True.

        Returns
        -------
        np.ndarray
            Mean-token embeddings for each query.
        """
        inputs = self._tokenize(queries, max_length=max_length, chunk_docs=False)
        loader = DataLoader(
            TokenizedDataset(inputs),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
        batches = self._generate_token_embeds(
            loader, move_results_to_cpu=False, return_tensors="pt"
        )
        query_embeds = []
        for batch in batches:
            token_embeds = batch["token_embeds"]
            input_ids = batch["input_ids"]
            valid_token_mask = torch.isin(
                input_ids,
                torch.tensor(self.tokenizer.all_special_ids, device=self.device),
                invert=True,
            )
            valid_token_weight = valid_token_mask.unsqueeze(2).float()
            mean_tokens = (token_embeds * valid_token_weight).sum(
                dim=1
            ) / valid_token_weight.sum(dim=1)
            if self.pca_mode and self.pca_is_ready:
                self.pca_transform_.to(self.device)
                mean_tokens = self.apply_pca(mean_tokens)
            mean_tokens = self.postprocess(mean_tokens)
            query_embeds.append(mean_tokens.cpu())
        query_embeds = torch.vstack(query_embeds)
        if convert_to_numpy:
            query_embeds = query_embeds.numpy()
        return query_embeds
