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

"""FinePhrase is a library for extracting sentence-segment embeddings using transformer models."""

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

from finephrase.pca import IncrementalPCA
from finephrase.sentence_utils import (
    _compute_segment_embeds,
    tokenize_with_sentence_boundaries,
)
from finephrase.tokenize import (
    DEFAULT_PAD_VALUES,
    TokenizedDataset,
    DynamicTokenSampler,
    dynamic_pad_collate,
    pad,
    tokenize_docs,
)
from finephrase.utils import (
    _build_results_dataframe,
    get_overlap_count,
    move_or_convert_tensors,
    normalize,
    normalize_num_jobs,
    reduce_precision,
    timer,
    truncate_dims,
    order_by_indices,
)

# def collate_fn(batch):
#     """Custom collate function for DataLoader."""
#     keys = set(batch[0].keys())
#     collated_batch = dict.fromkeys(keys)
#     standard_keys = {key for key in keys if key != "sent_boundary_idx"}
#     standard_data = [{key: x[key] for key in standard_keys} for x in batch]
#     collated_batch.update(default_collate(standard_data))
#     if "sent_boundary_idx" in keys:
#         collated_batch["sent_boundary_idx"] = pad_sequence(
#             [x["sent_boundary_idx"] for x in batch], batch_first=True, padding_value=-1
#         )
#     return collated_batch


class FinePhrase:
    def __init__(
        self,
        model_name: str,
        model_dtype: torch.dtype = torch.float32,
        amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        attn_implementation: str | None = None,
        compile: bool | str = True,
        reduce_precision: bool = False,
        truncate_dims: int | None = None,
        normalize_embeds: bool = False,
        pca: int | None = None,
        pca_fit_batch_count: int | float = 1.0,
        device: torch.device | str | int = "cuda",
        num_token_jobs: int | None = -1,
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
        attn_implementation : str | None, optional
            Attention implementation to use, by default None. If None, the model will use the
            default attention implementation.
        compile : bool | str, optional
            Compile the model, by default True. If True, the model will be compiled using
            torch.compile(mode="reduce-overhead"). If False, the model will not be compiled.
            You can specify the compilation mode using a string:
            - "reduce-overhead"
            - "default"
            - "max-autotune"
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
        """
        self.model_name = model_name
        self.model_dtype = model_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            clean_up_tokenization_spaces=True,
        )
        self.attn_implementation = attn_implementation
        model_kws = {"torch_dtype": self.model_dtype, "device_map": {"": device}}
        if self.attn_implementation is not None:
            model_kws["attn_implementation"] = self.attn_implementation
        self.model = AutoModel.from_pretrained(model_name, **model_kws).eval()
        self.compile = compile
        match compile:
            case True | "reduce-overhead":
                self.model = torch.compile(
                    self.model, mode="reduce-overhead", dynamic=False
                )
            case str():
                self.model = torch.compile(self.model, mode=compile, dynamic=False)
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.reduce_precision = reduce_precision
        self.truncate_dims = truncate_dims
        self.normalize_embeds = normalize_embeds
        self.pca = pca
        self.pca_fit_batch_count = pca_fit_batch_count
        self.num_token_jobs = num_token_jobs

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

    def _decode_segments(self, segment_token_ids: list[torch.Tensor]) -> pa.Array:
        """Decode the segment token IDs into human-readable segments.

        Parameters
        ----------
        segment_token_ids : list[torch.Tensor]
            List of segment token IDs to decode.

        Returns
        -------
        pa.Array
            PyArrow array containing the decoded segments.
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        _decode = delayed(
            partial(
                self.tokenizer.batch_decode,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        )
        segments = Parallel(n_jobs=self._num_token_jobs, prefer="processes")(
            _decode(ids) for ids in tqdm(segment_token_ids, desc="Detokenizing")
        )
        return pa.array([y for x in segments for y in x])

    @timer(readout="Finished preprocessing in {time:.4f} seconds.")
    def _tokenize(
        self,
        docs: list[str],
        max_length: int | None = None,
        chunk_docs: bool = True,
        doc_overlap: float | int = 0.5,
        batch_size: int = 10,
        num_jobs: int | None = None,
    ) -> Dataset:
        """Tokenize a list of documents into input sequences for the model.

        Tokenization preserves sentence boundaries using BlingFire sentence detection.

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
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if num_jobs is None:
            num_jobs = self._num_token_jobs
        inputs = tokenize_with_sentence_boundaries(
            docs,
            self.tokenizer,
            max_length=max_length,
            chunk_docs=chunk_docs,
            overlap=doc_overlap,
            batch_size=batch_size,
            n_jobs=num_jobs,
            return_tokenized_dataset=True,
        )
        return inputs

    @torch.no_grad()
    def _generate_token_embeds(
        self,
        loader: DataLoader,
        move_results_to_cpu: bool = False,
        return_tensors: str = "pt",
    ):
        """Obtain the token embeddings for a list of documents, one batch at at time."""
        with torch.autocast(
            device_type=self.device.type, enabled=self.amp, dtype=self.amp_dtype
        ):
            progress_loader = tqdm(loader, desc="Encoding")
            for batch_idx, batch in enumerate(progress_loader):
                batch = {
                    k: v.to(self.device, non_blocking=True) for k, v in batch.items()
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
                yield move_or_convert_tensors(
                    results,
                    return_tensors=return_tensors,
                    move_to_cpu=move_results_to_cpu,
                )

    def _generate_segment_embeds(
        self,
        loader: DataLoader,
        segment_sizes: int | list | tuple,
        segment_overlap: int | float | list | dict,
        move_results_to_cpu: bool = False,
        return_tensors: str = "pt",
    ):
        """Obtain the segment embeddings for a list of documents, one batch at at time."""
        batches = self._generate_token_embeds(
            loader, move_results_to_cpu=False, return_tensors="pt"
        )
        for batch in batches:
            results = _compute_segment_embeds(
                batch["input_ids"],
                token_embeds=batch["token_embeds"],
                sentence_ids=batch["sentence_ids"],
                sequence_idx=batch["sequence_idx"].to(self.device),
                tokenizer=self.tokenizer,
                segment_sizes=segment_sizes,
                overlap=segment_overlap,
            )
            results["batch_idx"] = torch.full(
                results["sequence_idx"].shape, batch["batch_idx"][0]
            )
            yield move_or_convert_tensors(
                results,
                return_tensors=return_tensors,
                move_to_cpu=move_results_to_cpu,
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

    def update_pca(self, segment_embeds: torch.Tensor) -> None:
        """Update the PCA transformation with a batch of segment embeddings.

        Parameters
        ----------
        segment_embeds : torch.Tensor
            Segment embeddings to update the PCA model with.
        """
        if not hasattr(self, "pca_transform_"):
            self.pca_transform_ = IncrementalPCA(
                n_components=self.pca, device=self.device
            )
        self.pca_transform_.partial_fit(segment_embeds)
        if hasattr(self.pca_transform_, "n_batches_seen_"):
            self.pca_transform_.n_batches_seen_ += 1
        else:
            self.pca_transform_.n_batches_seen_ = 1

    def apply_pca(self, segment_embeds: torch.Tensor) -> torch.Tensor:
        """Apply PCA transformation to embeddings.

        Parameters
        ----------
        segment_embeds : torch.Tensor
            Segment embeddings to apply PCA to.

        Returns
        -------
        torch.Tensor
            PCA-transformed embeddings.
        """
        if not hasattr(self, "pca_transform_"):
            raise AttributeError("PCA must be fitted first.")
        if not self.pca_is_ready:
            raise RuntimeError("PCA has not seen enough batches to be applied yet.")
        return self.pca_transform_.transform(segment_embeds)

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
        max_length: int | None = None,
        batch_max_tokens: int = 16384,
        token_batch_size: int = 10,
        segment_sizes: int | list | tuple = 2,
        segment_overlap: int | float | list | dict = 0.5,
        chunk_docs: bool = True,
        doc_overlap: float | int = 0.5,
        return_frame: str = "polars",
        convert_to_numpy: bool = True,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """Obtain the segments and segment embeddings from a list of documents.

        This first encodes the input documents, then extracts segment embeddings
        from the token embeddings. Segments are groups of consecutive sentences.

        Parameters
        ----------
        docs : list[str]
            List of documents to encode.
        max_length : int, optional
            Maximum length of the input sequences, by default None.
        batch_max_tokens : int, optional
            Maximum tokens per batch for encoder, by default 16384.
        token_batch_size : int, optional
            Batch size for tokenization, by default 10.
        segment_sizes : int, list, or tuple, optional
            Number of sentences per segment, by default 2.
            For example, if `segment_sizes` is set to `(1, 2, 3)`, segments
            of 1, 2, and 3 consecutive sentences will be extracted.
        segment_overlap : int, float, list, or dict, optional
            Overlap between segments (in sentences), by default 0.5.
            If a float, it is interpreted as a fraction of the segment size.
            If an integer, it is interpreted as the number of sentences to overlap.
            If a list or tuple, it should contain the overlap for each segment size.
            If a dictionary, it should map segment sizes to overlaps.
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
            The type of DataFrame of segments and indices to return. Options are
            'polars', 'pandas', or 'arrow'.
        convert_to_numpy : bool, optional
            Convert the tensors to numpy arrays before returning, by default True.

        Returns
        -------
        tuple[pl.DataFrame | pd.DataFrame | pa.Table, np.ndarray | torch.Tensor]
            Tuple containing the DataFrame of segments and the segment embeddings.

        Raises
        ------
        ValueError
            If `max_length` is not specified and `tokenizer.model_max_length` is None.

        """
        inputs = self._tokenize(
            docs,
            max_length=max_length,
            chunk_docs=chunk_docs,
            doc_overlap=doc_overlap,
            batch_size=token_batch_size,
        )
        loader = DataLoader(
            inputs,
            shuffle=False,
            pin_memory=True,
            batch_sampler=DynamicTokenSampler(
                inputs,
                max_tokens=batch_max_tokens,
            ),
            collate_fn=partial(
                dynamic_pad_collate, pad_token_id=self.tokenizer.pad_token_id
            ),
        )
        batches = self._generate_segment_embeds(
            loader,
            segment_sizes=segment_sizes,
            segment_overlap=segment_overlap,
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
            "segment_token_ids": [],
            "segment_size": [],
            "segment_embeds": [],
            "sentence_ids": [],
        }
        for batch in batches:
            if not self.pca_mode:
                # Postprocess on the fly to potentially conserve memory
                batch["segment_embeds"] = self.postprocess(batch["segment_embeds"])
            else:
                if self.pca_is_ready:
                    # Apply PCA and postprocess to potentially conserve memory
                    batch["segment_embeds"] = self.postprocess(
                        self.apply_pca(batch["segment_embeds"])
                    )
                else:
                    # Update PCA if not ready yet
                    self.update_pca(batch["segment_embeds"])
            # Offload batch to CPU
            batch = move_or_convert_tensors(
                batch, return_tensors="pt", move_to_cpu=True
            )
            results["batch_idx"].append(batch["batch_idx"])
            results["sequence_idx"].append(batch["sequence_idx"])
            if isinstance(batch["segment_token_ids"], list):
                results["segment_token_ids"].extend(batch["segment_token_ids"])
            else:  # If segment_token_ids is a tensor, convert to list
                results["segment_token_ids"].append(batch["segment_token_ids"])
            results["segment_size"].append(batch["segment_size"])
            results["segment_embeds"].append(batch["segment_embeds"])
        # Process early batches with PCA if necessary
        if self.pca_mode and not pca_ready_at_start:
            if self.pca_is_ready:
                self.pca_transform_.to("cpu")  # Temporarily move to CPU
                for i in range(self.pca_fit_batch_count_):
                    batch_embeds = results["segment_embeds"][i]
                    if batch_embeds.size(1) != self.pca:
                        results["segment_embeds"][i] = self.postprocess(
                            self.apply_pca(batch_embeds)
                        )
                self.pca_transform_.to(self.device)  # Move back to device
            else:
                warnings.warn("PCA did not finish fitting and will not be applied.")
        # Decode segments in existing batches
        results["segment"] = self._decode_segments(results.pop("segment_token_ids"))
        # Combine results
        for key, value in results.items():
            if len(value) and isinstance(value[0], torch.Tensor):
                results[key] = torch.cat(value, dim=0)
        if chunk_docs:
            mapping = torch.tensor(inputs.data["overflow_to_sample_mapping"])
            results["sample_idx"] = mapping[results["sequence_idx"]]
        else:
            results["sample_idx"] = results["sequence_idx"]
        results["embed_idx"] = torch.arange(results["segment_embeds"].shape[0])
        pdf, vecs = _build_results_dataframe(
            results,
            convert_to_numpy=convert_to_numpy,
            return_frame="polars",
        )
        if inputs.sort_by_token_count:
            pdf = pdf.sort("sequence_idx", descending=False)
            vecs = vecs[pdf["embed_idx"]]
        return pdf, vecs

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
            collate_fn=partial(
                dynamic_pad_collate, pad_token_id=self.tokenizer.pad_token_id
            ),
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
