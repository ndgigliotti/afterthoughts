# Copyright 2024-2026 Nicholas Gigliotti
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

"""Encoder is a library for extracting sentence-segment embeddings using transformer models."""

import logging
import math
import os
import warnings
from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import polars as pl
import torch
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from afterthoughts.chunk import (
    _compute_chunk_embeds,
    tokenize_with_sentence_boundaries,
)
from afterthoughts.pca import IncrementalPCA
from afterthoughts.tokenize import (
    DynamicTokenSampler,
    TokenizedDataset,
    _get_tokenization_batch_size,
    dynamic_pad_collate,
)
from afterthoughts.utils import (
    move_or_convert_tensors,
    normalize,
    normalize_num_jobs,
    reduce_precision,
    timer,
    truncate_dims,
)

logger = logging.getLogger(__name__)

# Minimum number of tokenization batches before enabling parallel processing.
# For small batch counts, the overhead of multiprocessing exceeds the benefit.
_MIN_BATCHES_FOR_PARALLEL = 5


class _EncoderBase(ABC):
    """Abstract base class for Encoder models.

    This class provides shared functionality for model loading, tokenization,
    and embedding generation. Subclasses implement specific encode methods.
    """

    def __init__(
        self,
        model_name: str,
        model_dtype: torch.dtype = torch.float32,
        amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        attn_implementation: str | None = None,
        normalize: bool = False,
        device: torch.device | str | int = "cuda",
        _num_token_jobs: int | None = -1,
    ) -> None:
        """Initialize a Encoder model.

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
        normalize : bool, optional
            Normalize the embeddings to unit length, by default False.
            This is useful for quick cosine similarity calculations downstream, since
            the dot product of two unit vectors is equal to the cosine similarity.
            It is also useful if you want downstream Euclidean distance calculations
            to consider only the direction of the vectors, not their magnitude.
        device : torch.device, str, int, optional
            Device to use for inference, by default "cuda".
        _num_token_jobs : int, None, optional
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
        logger.info("Loading model '%s' on device '%s'", model_name, device)
        self.model = AutoModel.from_pretrained(model_name, **model_kws).eval()
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.normalize = normalize
        self._num_token_jobs = _num_token_jobs

    @property
    def device(self) -> torch.device:
        """Returns the device the model is on."""
        return self.model.device

    def to(self, device: torch.device | str | int) -> "_EncoderBase":
        """Move the model to a new device.

        Parameters
        ----------
        device : torch.device, str, int
            Device to move the model to.

        Returns
        -------
        _EncoderBase
            Returns the model instance.
        """
        self.model.to(device)
        return self

    def half(self) -> "_EncoderBase":
        """Convert the model to half precision.

        Returns
        -------
        _EncoderBase
            Returns the model instance.
        """
        self.model.half()
        self.model_dtype = self.model.dtype
        return self

    def compile(self, mode: str = "reduce-overhead", dynamic: bool = True) -> "_EncoderBase":
        """Compile the model using torch.compile for potential speedups.

        This is an advanced feature. Compilation benefits vary significantly
        depending on GPU hardware, batch sizes, and document workloads.
        Benchmark on your own hardware with representative data before
        relying on compilation for performance gains.

        Parameters
        ----------
        mode : str, optional
            Compilation mode, by default "reduce-overhead".
            Options: "default", "reduce-overhead", "max-autotune".
        dynamic : bool, optional
            Enable dynamic shape support, by default True.
            Should be True when using dynamic batching (varying batch sizes).

        Returns
        -------
        _EncoderBase
            Returns the model instance for method chaining.
        """
        self.model = torch.compile(self.model, mode=mode, dynamic=dynamic)
        return self

    @property
    def __num_token_jobs(self) -> int:
        """Returns the number of jobs to use for tokenization."""
        return normalize_num_jobs(self._num_token_jobs)

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
        if self.normalize:
            embeds = normalize(embeds, dim=dim)
        return embeds

    def _decode_segments(
        self, segment_token_ids: list[torch.Tensor], show_progress: bool = True
    ) -> pl.Series:
        """Decode the segment token IDs into human-readable segments.

        Parameters
        ----------
        segment_token_ids : list[torch.Tensor]
            List of segment token IDs to decode.
        show_progress : bool, optional
            Show progress bar during decoding, by default True.

        Returns
        -------
        pl.Series
            Polars Series containing the decoded segments.
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        _decode = delayed(
            partial(
                self.tokenizer.batch_decode,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        )
        segments = Parallel(n_jobs=self.__num_token_jobs, prefer="processes")(
            _decode(ids)
            for ids in tqdm(segment_token_ids, desc="Detokenizing", disable=not show_progress)
        )
        return pl.Series([y for x in segments for y in x])

    @staticmethod
    def _build_results_dataframe(
        results: dict,
        return_frame: str = "polars",
        as_numpy: bool = True,
        debug: bool = False,
    ) -> tuple[pl.DataFrame, np.ndarray | torch.Tensor]:
        """Consolidate results into a DataFrame and embeddings array.

        Parameters
        ----------
        results : dict
            Dictionary containing 'document_idx', 'sequence_idx', 'batch_idx',
            'chunk_size', 'chunk', and 'chunk_embeds'.
        return_frame : str, optional
            DataFrame type: 'polars' or 'pandas'. Default is 'polars'.
        as_numpy : bool, optional
            Convert embeddings to NumPy arrays. Default is True.
        debug : bool, optional
            Include debug columns (sequence_idx, batch_idx). Default is False.

        Returns
        -------
        tuple
            (DataFrame of chunk metadata, embeddings array)
        """
        df = {}
        base_keys = [
            "embed_idx",
            "sequence_idx",
            "document_idx",
            "chunk_idx",
            "chunk_size",
        ]
        if debug:
            keys = base_keys + ["batch_idx", "chunk"]
        else:
            keys = base_keys + ["chunk"]
        for key in keys:
            if key in results:
                df[key] = results[key]
        df = move_or_convert_tensors(df, return_tensors="np", move_to_cpu=True)
        embeds = results["chunk_embeds"]
        if not isinstance(embeds, torch.Tensor):
            raise TypeError("Chunk embeddings must be torch.Tensor.")
        if as_numpy:
            embeds = embeds.cpu().numpy()
        else:
            embeds = embeds.cpu()
        if return_frame == "polars":
            df = pl.DataFrame(df)
        elif return_frame == "pandas":
            try:
                import pandas as pd
            except ImportError:
                raise ImportError(
                    "pandas is required for return_frame='pandas'. "
                    "Install it with: pip install pandas"
                ) from None
            df = pd.DataFrame(df)
        else:
            raise ValueError(f"Invalid value for return_frame: {return_frame}")
        return df, embeds

    @timer(readout="Finished preprocessing in {time:.4f} seconds.")
    def _tokenize(
        self,
        docs: list[str],
        max_length: int | None = None,
        prechunk: bool = True,
        prechunk_overlap: float | int = 0.5,
        batch_size: int = 10,
        num_jobs: int | None = None,
        show_progress: bool = True,
    ) -> TokenizedDataset:
        """Tokenize a list of documents into input sequences for the model.

        Tokenization preserves sentence boundaries using BlingFire sentence detection.

        Parameters
        ----------
        docs : list[str]
            List of documents to tokenize.
        max_length : int, optional
            Maximum length of the input sequences, by default None.
        prechunk : bool, optional
            Enable chunking of documents into overlapping sequences, by default True.
        prechunk_overlap : float, int, optional
            Overlap for splitting long documents into overlapping sequences due to the
            model's max sequence length limit, by default 0.5. Tokenized documents which fit
            within `max_length` will not be chunked. If a float, it is interpreted as a
            fraction of the maximum sequence length. If an integer, it is interpreted
            as the number of tokens to overlap. Does nothing if `prechunk` is False.
        batch_size : int, optional
            Batch size for tokenization, by default 10.
        num_jobs : int, optional
            Number of jobs to use for parallel processing on tokenization.
            If None, will default to `self._num_token_jobs`.
        show_progress : bool, optional
            Show progress bar during tokenization, by default True.

        Returns
        -------
        TokenizedDataset
            Dataset containing the tokenized input sequences.

        Raises
        ------
        ValueError
            If `max_length` is not specified and `tokenizer.model_max_length` is None.
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if num_jobs is None:
            num_jobs = self.__num_token_jobs
        inputs = tokenize_with_sentence_boundaries(
            docs,
            self.tokenizer,
            max_length=max_length,
            prechunk=prechunk,
            prechunk_overlap=prechunk_overlap,
            batch_size=batch_size,
            n_jobs=num_jobs,
            return_tokenized_dataset=True,
            show_progress=show_progress,
        )
        return inputs

    @torch.no_grad()
    def _generate_token_embeds(
        self,
        loader: DataLoader,
        move_results_to_cpu: bool = False,
        return_tensors: str = "pt",
        truncate_dim: int | None = None,
        show_progress: bool = True,
    ):
        """Obtain the token embeddings for a list of documents, one batch at at time.

        Parameters
        ----------
        loader : DataLoader
            DataLoader containing the tokenized input sequences.
        move_results_to_cpu : bool, optional
            Move results to CPU after processing, by default False.
        return_tensors : str, optional
            Return tensor format, by default "pt".
        truncate_dim : int | None, optional
            Truncate token embeddings to this dimension, by default None.
        show_progress : bool, optional
            Show progress bar during encoding, by default True.
        """
        with torch.autocast(device_type=self.device.type, enabled=self.amp, dtype=self.amp_dtype):
            progress_loader = tqdm(loader, desc="Encoding", disable=not show_progress)
            for batch_idx, batch in enumerate(progress_loader):
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                token_embeds = outputs.last_hidden_state
                if truncate_dim is not None:
                    token_embeds = truncate_dims(token_embeds, dim=truncate_dim)
                results = {
                    "sequence_idx": batch["sequence_idx"],
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "token_embeds": token_embeds,
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
        num_sents: int | list | tuple,
        chunk_overlap: int | float | list | dict,
        move_results_to_cpu: bool = False,
        return_tensors: str = "pt",
        truncate_dim: int | None = None,
        show_progress: bool = True,
    ):
        """Obtain the segment embeddings for a list of documents, one batch at at time.

        Parameters
        ----------
        loader : DataLoader
            DataLoader containing the tokenized input sequences.
        num_sents : int, list, or tuple
            Number of sentences per segment.
        chunk_overlap : int, float, list, or dict
            Overlap between segments (in sentences).
        move_results_to_cpu : bool, optional
            Move results to CPU after processing, by default False.
        return_tensors : str, optional
            Return tensor format, by default "pt".
        truncate_dim : int | None, optional
            Truncate token embeddings to this dimension, by default None.
        show_progress : bool, optional
            Show progress bar during encoding, by default True.
        """
        batches = self._generate_token_embeds(
            loader,
            move_results_to_cpu=False,
            return_tensors="pt",
            truncate_dim=truncate_dim,
            show_progress=show_progress,
        )
        for batch in batches:
            results = _compute_chunk_embeds(
                batch["input_ids"],
                token_embeds=batch["token_embeds"],
                sentence_ids=batch["sentence_ids"],
                sequence_idx=batch["sequence_idx"].to(self.device),
                tokenizer=self.tokenizer,
                num_sents=num_sents,
                chunk_overlap=chunk_overlap,
            )
            results["batch_idx"] = torch.full(results["sequence_idx"].shape, batch["batch_idx"][0])
            yield move_or_convert_tensors(
                results,
                return_tensors=return_tensors,
                move_to_cpu=move_results_to_cpu,
            )

    @abstractmethod
    def encode(
        self,
        docs: list[str],
        max_length: int | None = None,
        batch_tokens: int = 16384,
        num_sents: int | list | tuple = 1,
        chunk_overlap: int | float | list | dict = 0,
        prechunk: bool = True,
        prechunk_overlap: float | int = 0.5,
        return_frame: str = "polars",
        as_numpy: bool = True,
        debug: bool = False,
        return_text: bool = True,
        show_progress: bool = True,
    ):
        """Obtain the segments and segment embeddings from a list of documents."""
        pass

    def _postprocess_query_embeds(self, mean_tokens: torch.Tensor) -> torch.Tensor:
        """Postprocess query embeddings. Override in subclasses for additional processing."""
        return self.normalize_if_needed(mean_tokens)

    def encode_queries(
        self,
        queries: list[str],
        max_length: int | None = None,
        batch_size: int = 32,
        as_numpy: bool = True,
    ) -> np.ndarray:
        """Obtain the mean-tokens embeddings for a list of query strings.

        This is a convenient method for embedding query strings into the same space
        as the segments extracted from documents. It is mainly useful for doing semantic
        search.

        Parameters
        ----------
        queries : list[str]
            List of queries to encode.
        max_length : int, optional
            Maximum length of the query sequences, by default None.
        batch_size : int, optional
            Batch size for encoding, by default 32.
        as_numpy : bool, optional
            Convert the tensors to numpy arrays before returning, by default True.

        Returns
        -------
        np.ndarray
            Mean-token embeddings for each query.
        """
        token_batch_size = _get_tokenization_batch_size(queries)
        num_token_batches = math.ceil(len(queries) / token_batch_size)
        inputs = self._tokenize(
            queries,
            max_length=max_length,
            prechunk=False,
            batch_size=token_batch_size,
            num_jobs=1 if num_token_batches <= _MIN_BATCHES_FOR_PARALLEL else self.__num_token_jobs,
        )
        loader = DataLoader(
            inputs,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            pin_memory_device="",
            collate_fn=partial(dynamic_pad_collate, pad_token_id=self.tokenizer.pad_token_id),
        )
        batches = self._generate_token_embeds(
            loader,
            move_results_to_cpu=False,
            return_tensors="pt",
            truncate_dim=getattr(self, "truncate_dims", None),
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
            mean_tokens = (token_embeds * valid_token_weight).sum(dim=1) / valid_token_weight.sum(
                dim=1
            )
            mean_tokens = self._postprocess_query_embeds(mean_tokens)
            query_embeds.append(mean_tokens.cpu())
        query_embeds = torch.vstack(query_embeds)
        if as_numpy:
            query_embeds = query_embeds.numpy()
        return query_embeds


class Encoder(_EncoderBase):
    """Simple Encoder model for generating sentence-segment embeddings.

    This class provides a straightforward API for extracting segment embeddings
    from documents. For memory-efficient operations with PCA, precision reduction,
    and dimension truncation, use LiteEncoder instead.
    """

    def __init__(
        self,
        model_name: str,
        model_dtype: torch.dtype = torch.float32,
        amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        attn_implementation: str | None = None,
        normalize: bool = False,
        device: torch.device | str | int = "cuda",
        _num_token_jobs: int | None = -1,
    ) -> None:
        """Initialize a Encoder model.

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
        normalize : bool, optional
            Normalize the embeddings to unit length, by default False.
            This is useful for quick cosine similarity calculations downstream, since
            the dot product of two unit vectors is equal to the cosine similarity.
        device : torch.device, str, int, optional
            Device to use for inference, by default "cuda".
        _num_token_jobs : int, None, optional
            Number of jobs to use for multiprocessing on tokenization and
            detokenization, by default -1.
        """
        super().__init__(
            model_name=model_name,
            model_dtype=model_dtype,
            amp=amp,
            amp_dtype=amp_dtype,
            attn_implementation=attn_implementation,
            normalize=normalize,
            device=device,
            _num_token_jobs=_num_token_jobs,
        )

    def encode(
        self,
        docs: list[str],
        max_length: int | None = None,
        batch_tokens: int = 16384,
        num_sents: int | list | tuple = 1,
        chunk_overlap: int | float | list | dict = 0,
        prechunk: bool = True,
        prechunk_overlap: float | int = 0.5,
        return_frame: str = "polars",
        as_numpy: bool = True,
        debug: bool = False,
        return_text: bool = True,
        show_progress: bool = True,
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
        batch_tokens : int, optional
            Maximum tokens per batch for encoder, by default 16384.
        num_sents : int, list, or tuple, optional
            Number of sentences per segment, by default 1.
            For example, if `num_sents` is set to `(1, 2, 3)`, segments
            of 1, 2, and 3 consecutive sentences will be extracted.
        chunk_overlap : int, float, list, or dict, optional
            Overlap between segments (in sentences), by default 0.
            If a float, it is interpreted as a fraction of the segment size.
            If an integer, it is interpreted as the number of sentences to overlap.
            If a list or tuple, it should contain the overlap for each segment size.
            If a dictionary, it should map segment sizes to overlaps.
        prechunk : bool, optional
            Enable chunking of documents into overlapping sequences, by default True.
        prechunk_overlap : float or int, optional
            Overlap for splitting long documents into overlapping sequences, by default 0.5.
        return_frame : str, optional
            The type of DataFrame of segments and indices to return, by default 'polars'.
            Options are 'pandas' or 'polars'.
        as_numpy : bool, optional
            Convert the tensors to numpy arrays before returning, by default True.
        debug : bool, optional
            Include additional columns in the output DataFrame for debugging,
            by default False.
        return_text : bool, optional
            Include decoded text chunks in the output DataFrame, by default True.
            Set to False to skip detokenization for faster processing.
        show_progress : bool, optional
            Show progress bars during encoding, by default True.

        Returns
        -------
        tuple[pd.DataFrame | pl.DataFrame, np.ndarray | torch.Tensor]
            Tuple containing the DataFrame of segments and the segment embeddings.
        """
        # Validate return_frame early to fail fast before expensive computation
        if return_frame == "pandas":
            try:
                import pandas  # noqa: F401
            except ImportError:
                raise ImportError(
                    "pandas is required for return_frame='pandas'. "
                    "Install it with: pip install pandas"
                ) from None
        elif return_frame != "polars":
            raise ValueError(f"Invalid value for return_frame: {return_frame}")

        inputs = self._tokenize(
            docs,
            max_length=max_length,
            prechunk=prechunk,
            prechunk_overlap=prechunk_overlap,
            batch_size=_get_tokenization_batch_size(docs),
            show_progress=show_progress,
        )
        loader = DataLoader(
            inputs,
            shuffle=False,
            pin_memory=True,
            pin_memory_device="",
            batch_sampler=DynamicTokenSampler(
                inputs,
                max_tokens=batch_tokens,
            ),
            collate_fn=partial(dynamic_pad_collate, pad_token_id=self.tokenizer.pad_token_id),
        )
        batches = self._generate_segment_embeds(
            loader,
            num_sents=num_sents,
            chunk_overlap=chunk_overlap,
            move_results_to_cpu=False,
            return_tensors="pt",
            show_progress=show_progress,
        )

        results = {
            "batch_idx": [],
            "sequence_idx": [],
            "chunk_idx": [],
            "chunk_token_ids": [],
            "chunk_size": [],
            "chunk_embeds": [],
            "sentence_ids": [],
        }
        for batch in batches:
            # Apply normalization if needed
            batch["chunk_embeds"] = self.normalize_if_needed(batch["chunk_embeds"])
            # Offload batch to CPU
            batch = move_or_convert_tensors(batch, return_tensors="pt", move_to_cpu=True)
            results["batch_idx"].append(batch["batch_idx"])
            results["sequence_idx"].append(batch["sequence_idx"])
            results["chunk_idx"].append(batch["chunk_idx"])
            if isinstance(batch["chunk_token_ids"], list):
                results["chunk_token_ids"].extend(batch["chunk_token_ids"])
            else:
                results["chunk_token_ids"].append(batch["chunk_token_ids"])
            results["chunk_size"].append(batch["chunk_size"])
            results["chunk_embeds"].append(batch["chunk_embeds"])

        # Decode segments in existing batches
        if return_text:
            results["chunk"] = self._decode_segments(results.pop("chunk_token_ids"), show_progress)
        else:
            results.pop("chunk_token_ids")
        # Combine results
        for key, value in results.items():
            if len(value) and isinstance(value[0], torch.Tensor):
                results[key] = torch.cat(value, dim=0)
        if prechunk:
            seq_to_sample = dict(
                zip(
                    inputs.data["sequence_idx"],
                    inputs.data["overflow_to_sample_mapping"],
                    strict=False,
                )
            )
            results["document_idx"] = torch.tensor(
                [seq_to_sample[s.item()] for s in results["sequence_idx"]]
            )
        else:
            results["document_idx"] = results["sequence_idx"]
        results["embed_idx"] = torch.arange(results["chunk_embeds"].shape[0])
        pdf, vecs = self._build_results_dataframe(
            results,
            as_numpy=as_numpy,
            return_frame="polars",
            debug=debug,
        )
        if inputs.sort_by_token_count:
            pdf = pdf.sort("sequence_idx", "chunk_idx", descending=False)
            vecs = vecs[pdf["embed_idx"]]
        # Handle internal columns
        if debug:
            pdf = pdf.rename({"embed_idx": "orig_embed_idx"})
        else:
            pdf = pdf.drop(["embed_idx", "sequence_idx"])
        # Convert to requested DataFrame format
        if return_frame == "pandas":
            pdf = pdf.to_pandas()
        elif return_frame != "polars":
            raise ValueError(f"Invalid value for return_frame: {return_frame}")
        return pdf, vecs


class LiteEncoder(_EncoderBase):
    """Memory-efficient Encoder variant for advanced users.

    This class includes lossy memory optimizations:
    - PCA dimensionality reduction (GPU-accelerated, incremental fitting)
    - Precision reduction (float32/64 to float16)
    - Dimension truncation

    For simple use cases without these optimizations, use Encoder instead.
    """

    def __init__(
        self,
        model_name: str,
        model_dtype: torch.dtype = torch.float32,
        amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
        attn_implementation: str | None = None,
        half_embeds: bool = True,
        truncate_dims: int | None = None,
        normalize: bool = False,
        pca: int | None = None,
        pca_early_stop: int | float = 1.0,
        device: torch.device | str | int = "cuda",
        _num_token_jobs: int | None = -1,
    ) -> None:
        """Initialize a LiteEncoder model.

        Parameters
        ----------
        model_name : str
            Name of the pretrained model to use.
        model_dtype : torch.dtype, optional
            Data type for the model, by default torch.float32.
        amp : bool, optional
            Enable automatic mixed precision, by default True.
        amp_dtype : torch.dtype, optional
            Data type for automatic mixed precision, by default torch.float16.
        attn_implementation : str | None, optional
            Attention implementation to use, by default None.
        half_embeds : bool, optional
            Reduce the final embedding precision to float16, by default True.
        truncate_dims : int, None, optional
            Truncate the dimensions of the embeddings, by default None.
        normalize : bool, optional
            Normalize the embeddings to unit length, by default False.
        pca : int, None, optional
            Number of principal components to keep after PCA, by default None.
        pca_early_stop : int, float, optional
            Number of batches to use for fitting the PCA model, by default 1.0.
            If a float, it is the fraction of the dataset to use.
        device : torch.device, str, int, optional
            Device to use for inference, by default "cuda".
        _num_token_jobs : int, None, optional
            Number of jobs for tokenization/detokenization, by default -1.
        """
        super().__init__(
            model_name=model_name,
            model_dtype=model_dtype,
            amp=amp,
            amp_dtype=amp_dtype,
            attn_implementation=attn_implementation,
            normalize=normalize,
            device=device,
            _num_token_jobs=_num_token_jobs,
        )
        self.half_embeds = half_embeds
        self.truncate_dims = truncate_dims
        self.pca = pca
        self.pca_early_stop = pca_early_stop

        if truncate_dims is not None and pca is not None and truncate_dims < pca:
            raise ValueError("`truncate_dims` must be greater than `pca`.")

    def half_embeds_if_needed(self, embeds: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Quantize the embeddings if needed."""
        if self.half_embeds:
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
            self.half_embeds_if_needed,
            self.normalize_if_needed,
        ]
        for step in steps:
            embeds = step(embeds)
        return embeds

    @property
    def pca_mode(self) -> bool:
        """Returns True if PCA is enabled."""
        return self.pca is not None

    @property
    def pca_is_ready(self) -> bool:
        """Returns True if PCA has seen enough batches to be applied."""
        return (
            hasattr(self, "pca_transform_")
            and hasattr(self, "pca_early_stop_")
            and hasattr(self.pca_transform_, "n_batches_seen_")
            and self.pca_transform_.n_batches_seen_ >= self.pca_early_stop_
        )

    def update_pca(self, segment_embeds: torch.Tensor) -> None:
        """Update the PCA transformation with a batch of segment embeddings.

        Parameters
        ----------
        segment_embeds : torch.Tensor
            Segment embeddings to update the PCA model with.
        """
        if not hasattr(self, "pca_transform_"):
            self.pca_transform_ = IncrementalPCA(n_components=self.pca, device=self.device)
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
        pca_attrs = ["pca_transform_", "pca_early_stop_"]
        for attr in pca_attrs:
            if hasattr(self, attr):
                delattr(self, attr)
        logger.info("PCA cleared.")

    def encode(
        self,
        docs: list[str],
        max_length: int | None = None,
        batch_tokens: int = 16384,
        num_sents: int | list | tuple = 1,
        chunk_overlap: int | float | list | dict = 0,
        prechunk: bool = True,
        prechunk_overlap: float | int = 0.5,
        return_frame: str = "polars",
        as_numpy: bool = True,
        debug: bool = False,
        return_text: bool = True,
        show_progress: bool = True,
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
        batch_tokens : int, optional
            Maximum tokens per batch for encoder, by default 16384.
        num_sents : int, list, or tuple, optional
            Number of sentences per segment, by default 1.
        chunk_overlap : int, float, list, or dict, optional
            Overlap between segments (in sentences), by default 0.
        prechunk : bool, optional
            Enable chunking of documents into overlapping sequences, by default True.
        prechunk_overlap : float or int, optional
            Overlap for splitting long documents into overlapping sequences, by default 0.5.
        return_frame : str, optional
            The type of DataFrame of segments and indices to return, by default 'polars'.
            Options are 'pandas' or 'polars'.
        as_numpy : bool, optional
            Convert the tensors to numpy arrays before returning, by default True.
        debug : bool, optional
            Include additional columns in the output DataFrame for debugging,
            by default False.
        return_text : bool, optional
            Include decoded text chunks in the output DataFrame, by default True.
            Set to False to skip detokenization for faster processing.
        show_progress : bool, optional
            Show progress bars during encoding, by default True.

        Returns
        -------
        tuple[pd.DataFrame | pl.DataFrame, np.ndarray | torch.Tensor]
            Tuple containing the DataFrame of segments and the segment embeddings.
        """
        # Validate return_frame early to fail fast before expensive computation
        if return_frame == "pandas":
            try:
                import pandas  # noqa: F401
            except ImportError:
                raise ImportError(
                    "pandas is required for return_frame='pandas'. "
                    "Install it with: pip install pandas"
                ) from None
        elif return_frame != "polars":
            raise ValueError(f"Invalid value for return_frame: {return_frame}")

        inputs = self._tokenize(
            docs,
            max_length=max_length,
            prechunk=prechunk,
            prechunk_overlap=prechunk_overlap,
            batch_size=_get_tokenization_batch_size(docs),
            show_progress=show_progress,
        )
        loader = DataLoader(
            inputs,
            shuffle=False,
            pin_memory=True,
            pin_memory_device="",
            batch_sampler=DynamicTokenSampler(
                inputs,
                max_tokens=batch_tokens,
            ),
            collate_fn=partial(dynamic_pad_collate, pad_token_id=self.tokenizer.pad_token_id),
        )
        batches = self._generate_segment_embeds(
            loader,
            num_sents=num_sents,
            chunk_overlap=chunk_overlap,
            move_results_to_cpu=False,
            return_tensors="pt",
            truncate_dim=self.truncate_dims,
            show_progress=show_progress,
        )
        pca_ready_at_start = self.pca_is_ready
        if self.pca_mode:
            if hasattr(self, "pca_transform_"):
                self.pca_transform_.to(self.device)
            if pca_ready_at_start:
                logger.info("PCA is already fit and will be applied to all batches.")
            else:
                if isinstance(self.pca_early_stop, float):
                    self.pca_early_stop_ = math.ceil(len(loader) * self.pca_early_stop)
                else:
                    self.pca_early_stop_ = self.pca_early_stop

        results = {
            "batch_idx": [],
            "sequence_idx": [],
            "chunk_idx": [],
            "chunk_token_ids": [],
            "chunk_size": [],
            "chunk_embeds": [],
            "sentence_ids": [],
        }
        for batch in batches:
            if not self.pca_mode:
                # Postprocess on the fly to potentially conserve memory
                batch["chunk_embeds"] = self.postprocess(batch["chunk_embeds"])
            else:
                if self.pca_is_ready:
                    # Apply PCA and postprocess to potentially conserve memory
                    batch["chunk_embeds"] = self.postprocess(self.apply_pca(batch["chunk_embeds"]))
                else:
                    # Update PCA if not ready yet
                    self.update_pca(batch["chunk_embeds"])
            # Offload batch to CPU
            batch = move_or_convert_tensors(batch, return_tensors="pt", move_to_cpu=True)
            results["batch_idx"].append(batch["batch_idx"])
            results["sequence_idx"].append(batch["sequence_idx"])
            # segment_idx is per-document (resets for each document)
            results["chunk_idx"].append(batch["chunk_idx"])
            if isinstance(batch["chunk_token_ids"], list):
                results["chunk_token_ids"].extend(batch["chunk_token_ids"])
            else:  # If segment_token_ids is a tensor, convert to list
                results["chunk_token_ids"].append(batch["chunk_token_ids"])
            results["chunk_size"].append(batch["chunk_size"])
            results["chunk_embeds"].append(batch["chunk_embeds"])
        # Process early batches with PCA if necessary
        if self.pca_mode and not pca_ready_at_start:
            if self.pca_is_ready:
                self.pca_transform_.to("cpu")  # Temporarily move to CPU
                for i in range(self.pca_early_stop_):
                    batch_embeds = results["chunk_embeds"][i]
                    if batch_embeds.size(1) != self.pca:
                        results["chunk_embeds"][i] = self.postprocess(self.apply_pca(batch_embeds))
                self.pca_transform_.to(self.device)  # Move back to device
            else:
                warnings.warn("PCA did not finish fitting and will not be applied.", stacklevel=2)
        # Decode segments in existing batches
        if return_text:
            results["chunk"] = self._decode_segments(results.pop("chunk_token_ids"), show_progress)
        else:
            results.pop("chunk_token_ids")
        # Combine results
        for key, value in results.items():
            if len(value) and isinstance(value[0], torch.Tensor):
                results[key] = torch.cat(value, dim=0)
        if prechunk:
            # Both sequence_idx and overflow_to_sample_mapping are sorted together,
            # so create a mapping from original sequence index to sample index
            seq_to_sample = dict(
                zip(
                    inputs.data["sequence_idx"],
                    inputs.data["overflow_to_sample_mapping"],
                    strict=False,
                )
            )
            results["document_idx"] = torch.tensor(
                [seq_to_sample[s.item()] for s in results["sequence_idx"]]
            )
        else:
            results["document_idx"] = results["sequence_idx"]
        results["embed_idx"] = torch.arange(results["chunk_embeds"].shape[0])
        pdf, vecs = self._build_results_dataframe(
            results,
            as_numpy=as_numpy,
            return_frame="polars",
            debug=debug,
        )
        if inputs.sort_by_token_count:
            pdf = pdf.sort("sequence_idx", "chunk_idx", descending=False)
            vecs = vecs[pdf["embed_idx"]]
        # Handle internal columns
        if debug:
            # Rename embed_idx to orig_embed_idx for clarity
            pdf = pdf.rename({"embed_idx": "orig_embed_idx"})
        else:
            # Drop internal columns in non-debug mode
            pdf = pdf.drop(["embed_idx", "sequence_idx"])
        # Convert to requested DataFrame format
        if return_frame == "pandas":
            pdf = pdf.to_pandas()
        elif return_frame != "polars":
            raise ValueError(f"Invalid value for return_frame: {return_frame}")
        return pdf, vecs

    def _postprocess_query_embeds(self, mean_tokens: torch.Tensor) -> torch.Tensor:
        """Apply PCA (if ready) and postprocessing to query embeddings."""
        if self.pca_mode and self.pca_is_ready:
            self.pca_transform_.to(self.device)
            mean_tokens = self.apply_pca(mean_tokens)
        return self.postprocess(mean_tokens)
