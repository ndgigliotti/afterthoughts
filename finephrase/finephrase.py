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

"""FinePhrase is a library for extracting phrase embeddings using transformer models."""

import math
import os
import warnings
from functools import partial

import numpy as np
import polars as pl
import pyarrow as pa
import torch
from datasets import Dataset
from joblib import Parallel, delayed
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, default_collate
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from finephrase.available import _HAS_FAISS
from finephrase.pca import IncrementalPCA
from finephrase.phrase_utils import _compute_phrase_embeddings, _tokenize_batch
from finephrase.sentence_utils import (
    _compute_sentence_phrase_embeds,
    _compute_sentence_phrase_embeds_slow,
    _tokenize_batch_with_sentence_boundaries,
)
from finephrase.utils import (
    _build_results_dataframe,
    get_overlap_count,
    move_or_convert_results,
    normalize,
    normalize_num_jobs,
    reduce_precision,
    search_phrases,
    timer,
    truncate_dims,
)


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    keys = set(batch[0].keys())
    collated_batch = dict.fromkeys(keys)
    standard_keys = {key for key in keys if key != "sent_boundary_idx"}
    standard_data = [{key: x[key] for key in standard_keys} for x in batch]
    collated_batch.update(default_collate(standard_data))
    if "sent_boundary_idx" in keys:
        collated_batch["sent_boundary_idx"] = pad_sequence(
            [x["sent_boundary_idx"] for x in batch], batch_first=True, padding_value=-1
        )
    return collated_batch


class FinePhrase:
    def __init__(
        self,
        model_name: str,
        model_dtype: torch.dtype = torch.float32,
        amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        reduce_precision: bool = False,
        truncate_dims: int | None = None,
        normalize_embeds: bool = False,
        pca: int | None = None,
        pca_fit_batch_count: int | float = 1.0,
        device: torch.device | str | int = "cuda",
        num_token_jobs: int | None = -1,
        num_loader_jobs: int | None = 4,
    ) -> None:
        """Initialize a FinePhrase model.

        Parameters
        ----------
        model_name : str
            Name of the pretrained model to use.
        model_dtype : torch.dtype, optional
            Data type for the model, by default torch.float32.
        amp : bool, optional
            Enable automatic mixed precision, by default False.
        amp_dtype : torch.dtype, optional
            Data type for automatic mixed precision, by default torch.float16.
        reduce_precision : bool, optional
            Reduce the final embedding precision to float16 if they are float32 or float64,
            by default False. Note that the primary benefit of this is memory savings,
            not speed (on CPU).
        truncate_dims : int, None, optional
            Truncate the dimensions of the embeddings to the specified value, by default None.
            If None, the embeddings are not truncated.
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
        num_token_jobs : int, None, optional
            Number of jobs to use for multiprocessing on tokenization and
            detokenization, by default -1. If None, the number of jobs is
            set to the number of CPU cores. If less than 0, the number
            of jobs is set to `os.cpu_count() + n_jobs + 1`.
        num_loader_jobs : int, None, optional
            Number of jobs to use for multiprocessing on DataLoader, by default 4.
        """
        self.model_name = model_name
        self.model_dtype = model_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            clean_up_tokenization_spaces=True,
        )
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=model_dtype)
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.reduce_precision = reduce_precision
        self.truncate_dims = truncate_dims
        self.normalize_embeds = normalize_embeds
        self.pca = pca
        self.pca_fit_batch_count = pca_fit_batch_count
        self.num_token_jobs = num_token_jobs
        self.num_loader_jobs = num_loader_jobs
        self.model.eval().to(device)
        if truncate_dims is not None and pca is not None:
            if truncate_dims < pca:
                raise ValueError("`truncate_dims` must be greater than `pca`.")

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

    def half(self) -> "FinePhrase":
        """Convert the model to half precision.

        Returns
        -------
        FinePhrase
            Returns the model instance.
        """
        self.model.half()
        self.model_dtype = self.model.dtype
        return self

    @property
    def _num_token_jobs(self) -> int:
        """Returns the number of jobs to use for tokenization."""
        return normalize_num_jobs(self.num_token_jobs)

    @property
    def _num_loader_jobs(self) -> int:
        """Returns the number of jobs to use for the dataloader."""
        return normalize_num_jobs(self.num_loader_jobs)

    def reduce_precision_if_needed(
        self, embeds: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Quantize the embeddings if needed."""
        if self.reduce_precision:
            embeds = reduce_precision(embeds)
        return embeds

    def truncate_dims_if_needed(self, embeds: torch.Tensor | np.ndarray):
        """Truncate the dimensions of the embeddings if needed.

        Parameters
        ----------
        embeds : torch.Tensor or np.ndarray
            Embeddings to truncate.

        Returns
        -------
        torch.Tensor or np.ndarray
            Truncated embeddings.
        """
        if self.truncate_dims is not None:
            embeds = truncate_dims(embeds, dim=self.truncate_dims)
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
        1. Reduce precision to float16, if enabled.
        2. Normalize embeddings to unit length, if enabled.

        Parameters
        ----------
        embeds : np.ndarray
            Embeddings to postprocess.

        Returns
        -------
        np.ndarray
            Postprocessed embeddings.
        """
        steps = [
            self.reduce_precision_if_needed,
            self.normalize_if_needed,
        ]
        for step in steps:
            embeds = step(embeds)
        return embeds

    def _decode_phrases(self, phrase_ids: list[torch.Tensor]) -> pa.Array:
        """Decode the phrase IDs into human-readable phrases.

        Parameters
        ----------
        phrase_ids : list[torch.Tensor]
            List of phrase IDs to decode.

        Returns
        -------
        pa.Array
            PyArrow array containing the decoded phrases.
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        _decode = delayed(
            partial(
                self.tokenizer.batch_decode,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        )
        phrases = Parallel(n_jobs=self._num_token_jobs, prefer="processes")(
            _decode(ids) for ids in tqdm(phrase_ids, desc="Detokenizing")
        )
        return pa.array([y for x in phrases for y in x])

    @timer(readout="Finished preprocessing in {time:.4f} seconds.")
    def _tokenize(
        self,
        docs: list[str],
        sentences: bool = False,
        max_length: int | None = None,
        chunk_docs: bool = True,
        doc_overlap: float | int = 0.5,
        batch_size: int = 10,
        num_jobs: int | None = None,
    ) -> Dataset:
        """Tokenize a list of documents into input sequences for the model.

        Parameters
        ----------
        docs : list[str]
            List of documents to tokenize.
        sentences : bool, optional
            Enable sentence-level tokenization, by default False.
            Sentence boundaries will be located and preserved during tokenization.
            Chunking will preserve sentence boundaries.
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
        batch_size : int, optional
            Batch size for tokenization, by default 10.
        num_jobs : int, optional
            Number of jobs to use for parallel processing on tokenization.
            If None, will default to `self.num_token_jobs`.

        Returns
        -------
        Dataset
            Dataset containing the tokenized input sequences.

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

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # Use `Dataset` for easy batching
        data = Dataset.from_dict({"text": docs, "sample_idx": np.arange(len(docs))})
        num_batches = math.ceil(len(data) / batch_size)
        if num_jobs is None:
            num_jobs = self._num_token_jobs
        if sentences:
            _tokenize = _tokenize_batch_with_sentence_boundaries
        else:
            _tokenize = _tokenize_batch
        prefer = "threads" if sentences else "processes"
        batched_inputs = Parallel(n_jobs=num_jobs, prefer=prefer)(
            _tokenize(
                batch["text"],
                batch["sample_idx"],
                tokenizer=self.tokenizer,
                max_length=max_length,
                chunk_docs=chunk_docs,
                doc_overlap=doc_overlap,
            )
            for batch in tqdm(
                data.iter(batch_size), desc="Tokenizing", total=num_batches
            )
        )
        # Concatenate inputs
        inputs = {k: [] for k in batched_inputs[0].keys()}
        for key in inputs:
            if isinstance(batched_inputs[0][key], np.ndarray):
                inputs[key] = np.concatenate([x[key] for x in batched_inputs])
            if isinstance(batched_inputs[0][key], list):
                inputs[key] = [y for x in batched_inputs for y in x[key]]
                # inputs[key] = pad_sequence([torch.tensor(x) for x in inputs[key]], batch_first=True, padding_value=-1).numpy()
        # Add sequence index
        inputs["sequence_idx"] = np.arange(len(inputs["input_ids"]))
        inputs = Dataset.from_dict(inputs)
        inputs.set_format(type="torch")
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
                for batch_idx, batch in enumerate(progress_loader):
                    batch = {
                        k: v.to(self.device, non_blocking=True)
                        for k, v in batch.items()
                    }
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    results = {
                        "sequence_idx": batch["sequence_idx"],
                        "input_ids": batch["input_ids"],
                        "attention_mask": batch["attention_mask"],
                        "token_embeds": self.truncate_dims_if_needed(
                            outputs.last_hidden_state
                        ),
                        "batch_idx": torch.full(batch["sequence_idx"].shape, batch_idx),
                    }
                    if "sentence_ids" in batch:
                        results["sentence_ids"] = batch["sentence_ids"]
                    yield move_or_convert_results(
                        results,
                        return_tensors=return_tensors,
                        move_results_to_cpu=move_results_to_cpu,
                    )

    def _generate_phrase_embeds(
        self,
        loader: DataLoader,
        sentences: bool,
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
            if sentences:
                results = _compute_sentence_phrase_embeds(
                    batch["input_ids"],
                    token_embeds=batch["token_embeds"],
                    sentence_ids=batch["sentence_ids"],
                    sequence_idx=batch["sequence_idx"].to(self.device),
                    tokenizer=self.tokenizer,
                    phrase_sizes=phrase_sizes,
                    overlap=phrase_overlap,
                )
            else:
                results = _compute_phrase_embeddings(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_embeds=batch["token_embeds"],
                    sequence_idx=batch["sequence_idx"].to(self.device),
                    tokenizer=self.tokenizer,
                    phrase_sizes=phrase_sizes,
                    overlap=phrase_overlap,
                    phrase_min_token_ratio=phrase_min_token_ratio,
                )
            results["batch_idx"] = torch.full(
                results["sequence_idx"].shape, batch["batch_idx"][0]
            )
            yield move_or_convert_results(
                results,
                return_tensors=return_tensors,
                move_results_to_cpu=move_results_to_cpu,
            )

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
        pca_attrs = ["pca_transform_", "pca_fit_batch_count_"]
        for attr in pca_attrs:
            if hasattr(self, attr):
                delattr(self, attr)
        print("PCA cleared.")

    def encode(
        self,
        docs: list[str],
        sentences: bool = False,
        max_length: int | None = None,
        batch_size: int = 32,
        token_batch_size: int = 10,
        phrase_sizes: int | list | tuple = 12,
        phrase_overlap: int | float | list | dict = 0.5,
        phrase_min_token_ratio: float = 0.5,
        chunk_docs: bool = True,
        doc_overlap: float | int = 0.5,
        return_frame: str = "polars",
        convert_to_numpy: bool = True,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """Obtain the phrases and phrase embeddings from a list of documents.

        This first encodes the input documents, then extracts phrase embeddings
        from the token embeddings.

        Parameters
        ----------
        docs : list[str]
            List of documents to encode.
        sentences : bool, optional
            Enable sentence-level phrase extraction, by default False.
            If True, the model will extract phrases while preserving sentence structure,
            and the `phrase_sizes` parameter will control the number of sentences per phrase.
            The `phrase_overlap` parameter will control the overlap between phrases in terms
            of the number of sentences.
        max_length : int, optional
            Maximum length of the input sequences, by default None.
        batch_size : int, optional
            Batch size for encoder, by default 32.
        token_batch_size : int, optional
            Batch size for tokenization, by default 10.
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
        return_frame : str, optional
            The type of DataFrame of phrases and indices to return. Options are
            'polars', 'pandas', or 'arrow'.
        convert_to_numpy : bool, optional
            Convert the tensors to numpy arrays before returning, by default True.

        Returns
        -------
        tuple[pl.DataFrame | pd.DataFrame | pa.Table, np.ndarray | torch.Tensor]
            Tuple containing the DataFrame of phrases and the phrase embeddings.

        Raises
        ------
        ValueError
            If `max_length` is not specified and `tokenizer.model_max_length` is None.

        """
        inputs = self._tokenize(
            docs,
            sentences=sentences,
            max_length=max_length,
            chunk_docs=chunk_docs,
            doc_overlap=doc_overlap,
            batch_size=token_batch_size,
        )
        loader = DataLoader(
            inputs,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self._num_loader_jobs,
        )
        batches = self._generate_phrase_embeds(
            loader,
            sentences=sentences,
            phrase_sizes=phrase_sizes,
            phrase_overlap=phrase_overlap,
            phrase_min_token_ratio=phrase_min_token_ratio,
            move_results_to_cpu=False,
            return_tensors="pt",
        )
        pca_ready_at_start = self.pca_is_ready
        if self.pca_mode:
            if hasattr(self, "pca_transform_"):
                self.pca_transform_.to(self.device)
            if pca_ready_at_start:
                print("PCA is already fit and will be applied to all batches.")
            else:
                if isinstance(self.pca_fit_batch_count, float):
                    self.pca_fit_batch_count_ = math.ceil(
                        len(loader) * self.pca_fit_batch_count
                    )
                else:
                    self.pca_fit_batch_count_ = self.pca_fit_batch_count

        results = {
            "batch_idx": [],
            "sequence_idx": [],
            "phrase_ids": [],
            "phrase_size": [],
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
            batch = move_or_convert_results(
                batch, return_tensors="pt", move_results_to_cpu=True
            )
            results["batch_idx"].append(batch["batch_idx"])
            results["sequence_idx"].append(batch["sequence_idx"])
            if isinstance(batch["phrase_ids"], list):
                results["phrase_ids"].extend(batch["phrase_ids"])
            else:  # If phrase_ids is a tensor, convert to list
                results["phrase_ids"].append(batch["phrase_ids"])
            results["phrase_size"].append(batch["phrase_size"])
            results["phrase_embeds"].append(batch["phrase_embeds"])
        # Process early batches with PCA if necessary
        if self.pca_mode and not pca_ready_at_start:
            if self.pca_is_ready:
                self.pca_transform_.to("cpu")  # Temporarily move to CPU
                for i in range(self.pca_fit_batch_count_):
                    batch_embeds = results["phrase_embeds"][i]
                    if batch_embeds.size(1) != self.pca:
                        results["phrase_embeds"][i] = self.postprocess(
                            self.apply_pca(batch_embeds)
                        )
                self.pca_transform_.to(self.device)  # Move back to device
            else:
                warnings.warn("PCA did not finish fitting and will not be applied.")
        # Decode phrases in existing batches
        results["phrase"] = self._decode_phrases(results.pop("phrase_ids"))
        # Combine results
        for key, value in results.items():
            if isinstance(value[0], torch.Tensor):
                results[key] = torch.cat(value, dim=0)
        if chunk_docs:
            mapping = inputs["overflow_to_sample_mapping"]
            results["sample_idx"] = mapping[results["sequence_idx"]]
        else:
            results["sample_idx"] = results["sequence_idx"]
        results["embed_idx"] = torch.arange(results["phrase_embeds"].shape[0])
        return _build_results_dataframe(
            results,
            convert_to_numpy=convert_to_numpy,
            return_frame=return_frame,
        )

    def encode_queries(
        self,
        queries: list[str],
        max_length: int | None = None,
        batch_size: int = 32,
        token_batch_size: int = 10,
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
            Maximum length of the query sequences, by default None.
            If None, the tokenizer's maximum length will be used.
        batch_size : int, optional
            Batch size for encoding, by default 32.
        token_batch_size : int, optional
            Batch size for tokenization, by default 10.
        convert_to_numpy : bool, optional
            Convert the tensors to numpy arrays before returning, by default True.

        Returns
        -------
        np.ndarray
            Mean-token embeddings for each query.
        """
        small_thresh = 5
        num_token_batches = math.ceil(len(queries) / token_batch_size)
        inputs = self._tokenize(
            queries,
            max_length=max_length,
            chunk_docs=False,
            batch_size=token_batch_size,
            num_jobs=1 if num_token_batches <= small_thresh else self._num_token_jobs,
        )
        num_batches = math.ceil(len(inputs) / batch_size)
        loader = DataLoader(
            inputs,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=1 if num_batches <= small_thresh else self._num_loader_jobs,
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

    def search(
        self,
        queries: list,
        phrase_df: pl.DataFrame,
        phrase_embeds: np.ndarray,
        sim_thresh: float = 0.5,
        query_max_length: int | None = None,
        query_batch_size: int = 32,
        query_token_batch_size: int = 10,
    ) -> dict[str, pl.DataFrame]:
        """
        Search for documents that match the given queries based on their vector representations.

        Parameters
        ----------
        queries : list of str
            List of query strings. If `query_embeds` is None, the queries
            will be encoded on the fly.
        phrase_embeds : np.ndarray or NearestNeighbors
            Matrix of vector representations for the phrases. Alternatively, this can be a
            precomputed search index (of type `sklearn.neighbors.NearestNeighbors`).
        phrase_df : pl.DataFrame
            DataFrame containing phrase information. It should have a column
            "embed_idx" for indexing.
        query_embeds : np.ndarray, optional
            Precomputed query embeddings. If None, the queries will be encoded on the fly.
        sim_thresh : float, optional
            Cosine similarity threshold for the nearest neighbors search. Default is 0.5.
            Will return all results with similarity equal to or above this threshold.
        metric : str, optional
            Distance metric for the nearest neighbors search. Default is "cosine".
        query_max_length : int, optional
            Maximum length of the query sequences. If None, the tokenizer's
            maximum length will be used. Default is None.
        query_batch_size : int, optional
            Batch size for encoding queries. Default is 32.
        query_token_batch_size : int, optional
            Batch size for tokenization of queries. Default is 10.

        Returns
        -------
        query_embeds : np.ndarray
            Embeddings for the queries.
        search_index : NearestNeighbors
            The search index used for finding nearest neighbors.
        hits: dict
            A dictionary where keys are query strings and values are DataFrames
            containing the matching phrases.

        Raises
        ------
        ImportError
            If FAISS is not installed.

        See Also
        --------
        finephrase.utils.build_faiss_index
            Build a FAISS index using cosine similarity.
        finephrase.utils.search_phrases
            Function for searching phrases using FAISS.
        """
        if not _HAS_FAISS:
            raise ImportError("FAISS is not installed.")
        query_embeds = self.encode_queries(
            queries,
            max_length=query_max_length,
            batch_size=query_batch_size,
            token_batch_size=query_token_batch_size,
        )
        _, hits = search_phrases(
            queries,
            query_embeds,
            phrase_df=phrase_df,
            phrase_embeds=phrase_embeds,
            sim_thresh=sim_thresh,
        )
        return hits
