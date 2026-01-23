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

"""Main encoder class for extracting context-aware sentence-chunk embeddings.

This module provides the Encoder class, which implements late chunking for
transformer-based embedding models. Late chunking processes entire documents
through the model to capture full context, then extracts embeddings for
sentence groups by mean-pooling token embeddings within sentence boundaries.

The encoder supports various optimizations including automatic mixed precision,
model compilation, dynamic batching by token count, optional dimension
truncation for Matryoshka models, and memory-efficient float16 conversion.

Classes
-------
Encoder : Main API for encoding documents and queries

Key Features
------------
- Late chunking: Context-aware embeddings that preserve document-level context
- Flexible chunking: Configure number of sentences per chunk and overlap
- Sentence tokenizers: BlingFire, NLTK, pysbd, or syntok backends
- Memory optimization: Dynamic batching, float16 conversion, dimension truncation
- Performance: AMP support, torch.compile compatibility, GPU acceleration
- Output formats: Polars/pandas DataFrames with NumPy/PyTorch embeddings

Notes
-----
The Encoder class wraps HuggingFace transformers models and adds sentence-aware
chunking on top of the model's token embeddings. This enables semantic search
and retrieval over variable-length text chunks while maintaining full document
context during encoding.
"""

import logging
import math
from collections.abc import Iterator
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, overload

if TYPE_CHECKING:
    import pandas as pd

import numpy as np
import polars as pl
import torch
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from afterthoughts.avail import require_pandas
from afterthoughts.chunk import (
    _compute_chunk_embeds,
    tokenize_with_sentence_boundaries,
)
from afterthoughts.tokenize import (
    DynamicTokenSampler,
    TokenizedDataset,
    _get_tokenization_batch_size,
    dynamic_pad_collate,
)
from afterthoughts.utils import (
    disable_tokenizer_parallelism,
    get_device,
    half_embeds,
    move_or_convert_tensors,
    normalize,
    normalize_num_jobs,
    timer,
    truncate_dims,
)
from afterthoughts.validation import validate_encode_params, validate_encode_queries_params

logger = logging.getLogger(__name__)

# Minimum number of tokenization batches before enabling parallel processing.
# For small batch counts, the overhead of multiprocessing exceeds the benefit.
_MIN_BATCHES_FOR_PARALLEL = 5


class Encoder:
    """Encoder for generating context-aware sentence-chunk embeddings.

    This class implements late chunking: encoding entire documents through a
    transformer model to capture full context, then extracting embeddings for
    groups of consecutive sentences (chunks) via mean-pooling of token embeddings.

    Late chunking preserves document-level context that would be lost with
    traditional chunking approaches that split documents before encoding.

    Attributes
    ----------
    model_name : str
        Name or path of the HuggingFace transformer model.
    model_dtype : torch.dtype
        Data type used for model weights (e.g., torch.float32, torch.float16).
    tokenizer : transformers.PreTrainedTokenizer
        HuggingFace tokenizer for the model.
    model : transformers.PreTrainedModel
        HuggingFace transformer model in evaluation mode.
    attn_implementation : str or None
        Attention implementation to use (e.g., "flash_attention_2", "sdpa").
    amp : bool
        Whether automatic mixed precision is enabled for inference.
    amp_dtype : torch.dtype
        Data type for AMP autocast (typically torch.float16 or torch.bfloat16).
    normalize : bool
        Whether to L2-normalize embeddings to unit length.
    half_embeds : bool
        Whether to convert chunk embeddings to float16 after computation.
    truncate_dims : int or None
        Number of embedding dimensions to keep (for MRL models), or None.
    _num_token_jobs : int or None
        Number of parallel jobs for tokenization/detokenization.

    Examples
    --------
    Basic usage with default settings:

    >>> encoder = Encoder("sentence-transformers/all-MiniLM-L6-v2")
    >>> docs = ["First sentence. Second sentence.", "Another document."]
    >>> df, embeddings = encoder.encode(docs, num_sents=1)

    Encode with multiple chunk sizes and overlap:

    >>> df, embeddings = encoder.encode(
    ...     docs,
    ...     num_sents=[1, 2, 3],
    ...     chunk_overlap=0.5
    ... )

    Encode queries for semantic search:

    >>> query_embeds = encoder.encode_queries(["search query"])
    >>> # Compare with chunk embeddings via dot product (if normalized)
    >>> similarities = query_embeds @ embeddings.T

    Use memory optimization for large datasets:

    >>> encoder = Encoder(
    ...     "sentence-transformers/all-MiniLM-L6-v2",
    ...     half_embeds=True,
    ...     normalize=True
    ... )

    Notes
    -----
    - The encoder automatically handles documents longer than model max_length
      by splitting them into overlapping sequences while preserving sentences.
    - Chunk embeddings are computed by mean-pooling token embeddings within
      sentence boundaries, following the late chunking methodology.
    - For retrieval tasks, set normalize=True to enable fast cosine similarity
      via dot product.
    """

    def __init__(
        self,
        model_name: str,
        model_dtype: torch.dtype = torch.float32,
        amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        attn_implementation: str | None = None,
        normalize: bool = False,
        half_embeds: bool = False,
        truncate_dims: int | None = None,
        device: torch.device | str | int | None = None,
        query_prompt: str | None = None,
        document_prompt: str | None = None,
        _num_token_jobs: int | None = -1,
    ) -> None:
        """Initialize an Encoder model.

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
        half_embeds : bool, optional
            Convert chunk embeddings to float16 for reduced memory, by default False.
        truncate_dims : int | None, optional
            Truncate embedding dimensions to this value, by default None.
            Useful for models trained with Matryoshka Representation Learning (MRL).
            Truncation is applied to token embeddings before pooling.
        device : torch.device, str, int, None, optional
            Device to use for inference. If None (default), auto-detects the best
            available device (CUDA > MPS > CPU).
        query_prompt : str | None, optional
            Prompt to prepend to query strings in encode_queries(), by default None.
            Used for instruct-style embedding models (E5-instruct, BGE, GTE-Qwen2-instruct).
            Example: "Instruct: Given a web search query, retrieve relevant passages\\nQuery: "
        document_prompt : str | None, optional
            Prompt to prepend to documents in encode(), by default None.
            Only needed for some instruct models (e.g., Instructor) that require
            document-side prompts.
        _num_token_jobs : int, None, optional
            Number of jobs to use for multiprocessing on tokenization and
            detokenization, by default -1. If None, the number of jobs is
            set to the number of CPU cores. If less than 0, the number
            of jobs is set to `os.cpu_count() + n_jobs + 1`.
        """
        self.model_name = model_name
        self.model_dtype = model_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            model_name,
            clean_up_tokenization_spaces=True,
        )
        self.attn_implementation = attn_implementation
        if device is None:
            device = get_device()
        model_kws = {"torch_dtype": self.model_dtype, "device_map": {"": device}}
        if self.attn_implementation is not None:
            model_kws["attn_implementation"] = self.attn_implementation
        logger.info("Loading model '%s' on device '%s'", model_name, device)
        self.model = AutoModel.from_pretrained(model_name, **model_kws).eval()
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.normalize = normalize
        self.half_embeds = half_embeds
        self.truncate_dims = truncate_dims
        self.query_prompt = query_prompt
        self.document_prompt = document_prompt
        self._num_token_jobs = _num_token_jobs

    @property
    def device(self) -> torch.device:
        """Returns the device the model is on."""
        device = self.model.device
        assert isinstance(device, torch.device)
        return device

    def to(self, device: torch.device | str | int) -> "Encoder":
        """Move the model to a new device.

        Parameters
        ----------
        device : torch.device, str, int
            Device to move the model to.

        Returns
        -------
        Encoder
            Returns the model instance.
        """
        self.model.to(device)
        return self

    def half(self) -> "Encoder":
        """Convert the model to half precision.

        Returns
        -------
        Encoder
            Returns the model instance.
        """
        self.model.half()
        self.model_dtype = self.model.dtype
        return self

    def compile(self, mode: str = "reduce-overhead", dynamic: bool = True) -> "Encoder":
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
        Encoder
            Returns the model instance for method chaining.
        """
        self.model = torch.compile(self.model, mode=mode, dynamic=dynamic)
        return self

    @property
    def __num_token_jobs(self) -> int:
        """Returns the number of jobs to use for tokenization."""
        return normalize_num_jobs(self._num_token_jobs)

    def _apply_prompt(self, texts: list[str], prompt: str | None) -> list[str]:
        """Apply a prompt prefix to a list of texts.

        Parameters
        ----------
        texts : list[str]
            List of texts to prepend the prompt to.
        prompt : str | None
            Prompt to prepend. If None, returns texts unchanged.

        Returns
        -------
        list[str]
            Texts with prompt prepended (if prompt is not None).
        """
        if prompt is None:
            return texts
        return [prompt + text for text in texts]

    def _get_prompt_length(self, prompt: str | None) -> int:
        """Get the number of tokens in a prompt (excluding special tokens).

        Parameters
        ----------
        prompt : str | None
            Prompt to tokenize. If None, returns 0.

        Returns
        -------
        int
            Number of tokens in the prompt.
        """
        if prompt is None:
            return 0
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        return len(tokens)

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

    def half_embeds_if_needed(self, embeds: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Convert embeddings to float16 if enabled.

        Parameters
        ----------
        embeds : torch.Tensor or np.ndarray
            Embeddings to convert.

        Returns
        -------
        torch.Tensor or np.ndarray
            Embeddings in float16 if half_embeds is enabled, otherwise unchanged.
        """
        if self.half_embeds:
            embeds = half_embeds(embeds)
        return embeds

    def postprocess(self, embeds: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Apply postprocessing steps to embeddings.

        Applies half_embeds (if enabled) then normalization (if enabled).

        Parameters
        ----------
        embeds : torch.Tensor or np.ndarray
            Embeddings to postprocess.

        Returns
        -------
        torch.Tensor or np.ndarray
            Postprocessed embeddings.
        """
        embeds = self.half_embeds_if_needed(embeds)
        embeds = self.normalize_if_needed(embeds)
        return embeds

    def _decode_chunks(
        self, chunk_token_ids: list[torch.Tensor], show_progress: bool = True
    ) -> pl.Series:
        """Decode the chunk token IDs into human-readable chunks.

        Parameters
        ----------
        chunk_token_ids : list[torch.Tensor]
            List of chunk token IDs to decode.
        show_progress : bool, optional
            Show progress bar during decoding, by default True.

        Returns
        -------
        pl.Series
            Polars Series containing the decoded chunks.
        """
        _decode = delayed(
            partial(
                self.tokenizer.batch_decode,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        )
        with disable_tokenizer_parallelism():
            chunks = Parallel(n_jobs=self.__num_token_jobs, prefer="processes")(
                _decode(ids)
                for ids in tqdm(chunk_token_ids, desc="Detokenizing", disable=not show_progress)
            )
        return pl.Series([y for x in chunks for y in x])

    def _reconstruct_chunk_texts(
        self,
        chunk_sentence_ids: list[torch.Tensor],
        chunk_token_ids: list[torch.Tensor],
        sequence_idx: torch.Tensor,
        seq_to_doc: dict[int, int],
        sentence_texts: list[list[str]],
        separator: str = " ",
        show_progress: bool = True,
    ) -> pl.Series:
        """Join original sentence texts by sentence ID instead of detokenizing.

        Parameters
        ----------
        chunk_sentence_ids : list[torch.Tensor]
            List of sentence ID tensors for each chunk (batched).
        chunk_token_ids : list[torch.Tensor]
            List of token ID tensors for each chunk (used for fallback).
        sequence_idx : torch.Tensor
            Tensor mapping each chunk to its sequence index.
        seq_to_doc : dict[int, int]
            Mapping from sequence index to document index.
        sentence_texts : list[list[str]]
            Per-document list of sentence texts.
        separator : str, optional
            String to join sentences with, by default " ".
        show_progress : bool, optional
            Show progress bar during reconstruction, by default True.

        Returns
        -------
        pl.Series
            Polars Series containing the reconstructed chunk texts.
        """
        chunks = []
        fallback_indices = []
        fallback_token_ids = []

        # Flatten batched sentence_ids and pair with sequence indices
        flat_sent_ids = [sid for batch in chunk_sentence_ids for sid in batch]
        flat_token_ids = [tid for batch in chunk_token_ids for tid in batch]

        for i, (sent_ids, token_ids) in enumerate(
            tqdm(
                zip(flat_sent_ids, flat_token_ids, strict=True),
                desc="Reconstructing",
                disable=not show_progress,
                total=len(flat_sent_ids),
            )
        ):
            doc_idx = seq_to_doc[int(sequence_idx[i].item())]
            doc_sentences = sentence_texts[doc_idx]
            # Sentence IDs are contiguous, so just get min/max instead of unique
            valid_mask = sent_ids != -1
            if not valid_mask.any():
                chunks.append("")
                continue
            valid_ids = sent_ids[valid_mask]
            min_id, max_id = int(valid_ids[0].item()), int(valid_ids[-1].item())
            # Handle edge case: sentence was split (ID exceeds original count)
            if max_id >= len(doc_sentences):
                chunks.append("")  # Placeholder for fallback (will be replaced)
                fallback_indices.append(i)
                fallback_token_ids.append(token_ids)
            else:
                chunks.append(
                    separator.join(doc_sentences[sid] for sid in range(min_id, max_id + 1))
                )

        # Fall back to detokenization for chunks with split sentences
        if fallback_token_ids:
            decoded = self.tokenizer.batch_decode(
                fallback_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            for idx, text in zip(fallback_indices, decoded, strict=True):
                chunks[idx] = text

        return pl.Series(chunks)

    @staticmethod
    def _build_results_df(
        results: dict[str, Any],
        return_frame: str = "polars",
        as_numpy: bool = True,
        debug: bool = False,
    ) -> tuple[pl.DataFrame, np.ndarray | torch.Tensor]:
        """Consolidate results into a DataFrame and embeddings array.

        Parameters
        ----------
        results : dict
            Dictionary containing 'document_idx', 'sequence_idx', 'batch_idx',
            'num_sents', 'chunk', and 'chunk_embeds'.
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
        df_dict: dict[str, Any] = {}
        base_keys = [
            "embed_idx",
            "sequence_idx",
            "document_idx",
            "chunk_idx",
            "num_sents",
        ]
        if debug:
            keys = base_keys + ["batch_idx", "chunk"]
        else:
            keys = base_keys + ["chunk"]
        for key in keys:
            if key in results:
                df_dict[key] = results[key]
        df_dict = move_or_convert_tensors(df_dict, return_tensors="np", move_to_cpu=True)
        embeds = results["chunk_embeds"]
        if not isinstance(embeds, torch.Tensor):
            raise TypeError("Chunk embeddings must be torch.Tensor.")
        embeds_result: np.ndarray[Any, Any] | torch.Tensor
        if as_numpy:
            embeds_result = embeds.cpu().numpy()
        else:
            embeds_result = embeds.cpu()
        df: pl.DataFrame | Any
        if return_frame == "polars":
            df = pl.DataFrame(df_dict)
        elif return_frame == "pandas":
            pd = require_pandas()
            df = pd.DataFrame(df_dict)
        else:
            raise ValueError(f"Invalid value for return_frame: {return_frame}")
        return df, embeds_result

    @staticmethod
    def _deduplicate_chunk_embeds(
        results: dict[str, Any],
        method: str = "average",
    ) -> dict[str, Any]:
        """
        Deduplicate chunk embeddings from overlapping pre-chunks.

        When documents exceed max_length, they are split into overlapping pre-chunks.
        This can create duplicate chunk embeddings for the same sentence groups
        (with different attention contexts). This function groups by
        (document_idx, num_sents, first_sent, last_sent) and either averages
        embeddings or keeps the first occurrence.

        Uses vectorized operations (np.unique, torch.scatter_add) for performance
        on large numbers of chunks.

        Parameters
        ----------
        results : dict
            Dictionary containing accumulated batch results with keys:
            - 'document_idx': tensor of document indices
            - 'num_sents': tensor of chunk sizes
            - 'chunk_sentence_ids': list of sentence ID tensors per chunk
            - 'chunk_embeds': tensor of chunk embeddings
            - 'sequence_idx', 'batch_idx', 'chunk_idx', 'chunk_token_ids' (preserved)
        method : str, optional
            Deduplication method: 'average' (default) or 'first'.
            - 'average': compute mean of embeddings in each group
            - 'first': keep first occurrence only

        Returns
        -------
        dict
            Deduplicated results with same structure, chunk_idx reindexed
            to be sequential within each document.
        """
        if method not in ("average", "first"):
            raise ValueError(f"method must be 'average' or 'first', got {method!r}")

        document_idx = results["document_idx"]
        num_sents = results["num_sents"]
        chunk_embeds = results["chunk_embeds"]
        chunk_sentence_ids = results["chunk_sentence_ids"]

        n_chunks = len(document_idx)

        # Compute first and last sentence ID for each chunk
        # Since sentences are consecutive within a chunk, (first, last) uniquely identifies it
        first_sent = torch.zeros(n_chunks, dtype=torch.long)
        last_sent = torch.zeros(n_chunks, dtype=torch.long)

        for i, sent_ids in enumerate(chunk_sentence_ids):
            if isinstance(sent_ids, torch.Tensor):
                valid_mask = sent_ids != -1
                if valid_mask.any():
                    valid_ids_tensor = sent_ids[valid_mask]
                    first_sent[i] = valid_ids_tensor[0]
                    last_sent[i] = valid_ids_tensor[-1]
            else:
                valid_ids_list = [s for s in sent_ids if s != -1]
                if valid_ids_list:
                    first_sent[i] = valid_ids_list[0]
                    last_sent[i] = valid_ids_list[-1]

        # Build compound key and use np.unique for fast grouping
        keys = torch.stack([document_idx, num_sents, first_sent, last_sent], dim=1).numpy()
        _, inverse_indices, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True
        )

        n_groups = len(counts)
        group_indices = torch.from_numpy(inverse_indices).long()

        # Find first occurrence index for each group (for metadata)
        first_occurrence = torch.zeros(n_groups, dtype=torch.long)
        seen = torch.zeros(n_groups, dtype=torch.bool)
        for i in range(n_chunks):
            g = int(group_indices[i].item())
            if not seen[g]:
                first_occurrence[g] = i
                seen[g] = True

        if method == "first":
            # Just keep first occurrence
            sorted_group_order = torch.argsort(first_occurrence)
            dedup_indices = first_occurrence[sorted_group_order]
            dedup_embeds = chunk_embeds[dedup_indices]
        else:
            # Average embeddings per group using scatter_add
            embed_dim = chunk_embeds.shape[1]
            group_sums = torch.zeros(n_groups, embed_dim, dtype=chunk_embeds.dtype)

            # scatter_add to sum embeddings per group
            expanded_indices = group_indices.unsqueeze(1).expand_as(chunk_embeds)
            group_sums.scatter_add_(0, expanded_indices, chunk_embeds)

            # Divide by counts to get means
            group_counts = torch.from_numpy(counts).to(chunk_embeds.dtype).unsqueeze(1)
            group_means = group_sums / group_counts

            # Reorder to match original document order
            sorted_group_order = torch.argsort(first_occurrence)
            dedup_indices = first_occurrence[sorted_group_order]
            dedup_embeds = group_means[sorted_group_order]

        # Build deduplicated results dict
        dedup_indices_list = dedup_indices.tolist()
        dedup_results = {}
        for key in results:
            if key == "chunk_embeds":
                dedup_results[key] = dedup_embeds
            elif key == "chunk_sentence_ids":
                dedup_results[key] = [chunk_sentence_ids[i] for i in dedup_indices_list]
            elif key == "chunk_token_ids":
                dedup_results[key] = [results[key][i] for i in dedup_indices_list]
            elif isinstance(results[key], torch.Tensor):
                dedup_results[key] = results[key][dedup_indices]
            elif isinstance(results[key], list):
                dedup_results[key] = [results[key][i] for i in dedup_indices_list]
            else:
                dedup_results[key] = results[key]

        # Reindex chunk_idx to be sequential within each document (vectorized)
        if "chunk_idx" in dedup_results and "document_idx" in dedup_results:
            doc_idx_arr = dedup_results["document_idx"].numpy()
            new_chunk_idx = np.zeros(len(doc_idx_arr), dtype=np.int64)
            for doc in np.unique(doc_idx_arr):
                mask = doc_idx_arr == doc
                new_chunk_idx[mask] = np.arange(mask.sum())
            dedup_results["chunk_idx"] = torch.from_numpy(new_chunk_idx)

        return dedup_results

    @timer(readout="Finished preprocessing in {time:.4f} seconds.")
    def _tokenize(
        self,
        docs: list[str],
        max_length: int | None = None,
        prechunk: bool = True,
        prechunk_overlap: float | int = 0.5,
        batch_size: int = 10,
        num_jobs: int | None = None,
        sent_tokenizer: str = "blingfire",
        show_progress: bool = True,
        prompt: str | None = None,
    ) -> tuple[TokenizedDataset, list[list[str]]]:
        """Tokenize a list of documents into input sequences for the model.

        Tokenization preserves sentence boundaries using sentence detection.

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
        sent_tokenizer : str, optional
            Sentence tokenizer to use for sentence boundary detection, by default "blingfire".
            Options are "blingfire", "nltk", "pysbd", or "syntok".
        show_progress : bool, optional
            Show progress bar during tokenization, by default True.
        prompt : str | None, optional
            Prompt to prepend to each document, by default None. Sentence detection
            is performed on the original document (without prompt).

        Returns
        -------
        tuple[TokenizedDataset, list[list[str]]]
            Tuple of (dataset containing tokenized input sequences, per-document sentence texts).

        Raises
        ------
        ValueError
            If `max_length` is not specified and `tokenizer.model_max_length` is None.
        """
        if num_jobs is None:
            num_jobs = self.__num_token_jobs
        result = tokenize_with_sentence_boundaries(
            docs,
            self.tokenizer,
            method=sent_tokenizer,
            max_length=max_length,
            prechunk=prechunk,
            prechunk_overlap=prechunk_overlap,
            batch_size=batch_size,
            n_jobs=num_jobs,
            return_tokenized_dataset=True,
            show_progress=show_progress,
            prompt=prompt,
        )
        if not isinstance(result, tuple):
            raise TypeError("Expected tokenize_with_sentence_boundaries to return a tuple")
        inputs, sentence_texts = result
        return inputs, sentence_texts

    @torch.inference_mode()
    def _generate_token_embeds(
        self,
        loader: DataLoader[dict[str, Any]],
        move_results_to_cpu: bool = False,
        return_tensors: str = "pt",
        truncate_dim: int | None = None,
        show_progress: bool = True,
    ) -> Iterator[dict[str, Any]]:
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

    def _generate_chunk_embeds(
        self,
        loader: DataLoader[dict[str, Any]],
        num_sents: int | list[int] | tuple[int, ...] | None,
        chunk_overlap: int | float | list[int] | dict[int, int],
        move_results_to_cpu: bool = False,
        return_tensors: str = "pt",
        truncate_dim: int | None = None,
        exclude_special_tokens: bool = True,
        show_progress: bool = True,
        max_chunk_tokens: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Obtain the chunk embeddings for a list of documents, one batch at at time.

        Parameters
        ----------
        loader : DataLoader
            DataLoader containing the tokenized input sequences.
        num_sents : int, list, tuple, or None
            Number of sentences per chunk. None means no sentence limit.
        chunk_overlap : int, float, list, or dict
            Overlap between chunks (in sentences).
        move_results_to_cpu : bool, optional
            Move results to CPU after processing, by default False.
        return_tensors : str, optional
            Return tensor format, by default "pt".
        truncate_dim : int | None, optional
            Truncate token embeddings to this dimension, by default None.
        exclude_special_tokens : bool, optional
            If True (default), exclude all special tokens from mean pooling.
            If False, include [CLS] in first chunk and [SEP] in last chunk
            of each sequence.
        show_progress : bool, optional
            Show progress bar during encoding, by default True.
        max_chunk_tokens : int | None, optional
            Maximum tokens per chunk, by default None.
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
                exclude_special_tokens=exclude_special_tokens,
                max_chunk_tokens=max_chunk_tokens,
            )
            results["batch_idx"] = torch.full(results["sequence_idx"].shape, batch["batch_idx"][0])
            yield move_or_convert_tensors(
                results,
                return_tensors=return_tensors,
                move_to_cpu=move_results_to_cpu,
            )

    @overload
    def encode(
        self,
        docs: list[str],
        max_length: int | None = ...,
        batch_tokens: int = ...,
        num_sents: int | list[int] | tuple[int, ...] | None = ...,
        chunk_overlap: int | float | list[int] | dict[int, int] = ...,
        prechunk: bool = ...,
        prechunk_overlap: float | int = ...,
        sent_tokenizer: str = ...,
        exclude_special_tokens: bool = ...,
        deduplicate: bool = ...,
        return_frame: Literal["polars"] = ...,
        as_numpy: Literal[True] = ...,
        debug: bool = ...,
        return_text: bool = ...,
        show_progress: bool = ...,
        prompt: str | None = ...,
        max_chunk_tokens: int | None = ...,
    ) -> tuple[pl.DataFrame, np.ndarray[Any, Any]]: ...

    @overload
    def encode(
        self,
        docs: list[str],
        max_length: int | None = ...,
        batch_tokens: int = ...,
        num_sents: int | list[int] | tuple[int, ...] | None = ...,
        chunk_overlap: int | float | list[int] | dict[int, int] = ...,
        prechunk: bool = ...,
        prechunk_overlap: float | int = ...,
        sent_tokenizer: str = ...,
        exclude_special_tokens: bool = ...,
        deduplicate: bool = ...,
        return_frame: Literal["polars"] = ...,
        as_numpy: Literal[False] = ...,
        debug: bool = ...,
        return_text: bool = ...,
        show_progress: bool = ...,
        prompt: str | None = ...,
        max_chunk_tokens: int | None = ...,
    ) -> tuple[pl.DataFrame, torch.Tensor]: ...

    @overload
    def encode(
        self,
        docs: list[str],
        max_length: int | None = ...,
        batch_tokens: int = ...,
        num_sents: int | list[int] | tuple[int, ...] | None = ...,
        chunk_overlap: int | float | list[int] | dict[int, int] = ...,
        prechunk: bool = ...,
        prechunk_overlap: float | int = ...,
        sent_tokenizer: str = ...,
        exclude_special_tokens: bool = ...,
        deduplicate: bool = ...,
        return_frame: Literal["pandas"] = ...,
        as_numpy: Literal[True] = ...,
        debug: bool = ...,
        return_text: bool = ...,
        show_progress: bool = ...,
        prompt: str | None = ...,
        max_chunk_tokens: int | None = ...,
    ) -> tuple["pd.DataFrame", np.ndarray[Any, Any]]: ...

    @overload
    def encode(
        self,
        docs: list[str],
        max_length: int | None = ...,
        batch_tokens: int = ...,
        num_sents: int | list[int] | tuple[int, ...] | None = ...,
        chunk_overlap: int | float | list[int] | dict[int, int] = ...,
        prechunk: bool = ...,
        prechunk_overlap: float | int = ...,
        sent_tokenizer: str = ...,
        exclude_special_tokens: bool = ...,
        deduplicate: bool = ...,
        return_frame: Literal["pandas"] = ...,
        as_numpy: Literal[False] = ...,
        debug: bool = ...,
        return_text: bool = ...,
        show_progress: bool = ...,
        prompt: str | None = ...,
        max_chunk_tokens: int | None = ...,
    ) -> tuple["pd.DataFrame", torch.Tensor]: ...

    def encode(
        self,
        docs: list[str],
        max_length: int | None = None,
        batch_tokens: int = 16384,
        num_sents: int | list[int] | tuple[int, ...] | None = 1,
        chunk_overlap: int | float | list[int] | dict[int, int] = 0,
        prechunk: bool = True,
        prechunk_overlap: float | int = 0.5,
        sent_tokenizer: str = "blingfire",
        exclude_special_tokens: bool = True,
        deduplicate: bool = True,
        return_frame: str = "polars",
        as_numpy: bool = True,
        debug: bool = False,
        return_text: bool = True,
        show_progress: bool = True,
        prompt: str | None = None,
        max_chunk_tokens: int | None = None,
    ) -> (
        tuple[pl.DataFrame, np.ndarray[Any, Any]]
        | tuple[pl.DataFrame, torch.Tensor]
        | tuple["pd.DataFrame", np.ndarray[Any, Any]]
        | tuple["pd.DataFrame", torch.Tensor]
    ):
        """Obtain the chunks and chunk embeddings from a list of documents.

        This first encodes the input documents, then extracts chunk embeddings
        from the token embeddings. Chunks are groups of consecutive sentences.

        Parameters
        ----------
        docs : list[str]
            List of documents to encode.
        max_length : int, optional
            Maximum length of the input sequences, by default None.
        batch_tokens : int, optional
            Maximum tokens per batch for encoder, by default 16384.
        num_sents : int, list, tuple, or None, optional
            Number of sentences per chunk, by default 1. Can be:
            - int: Fixed number of sentences per chunk
            - list/tuple: Extract multiple chunk sizes simultaneously
            - None: No sentence limit (only valid with max_chunk_tokens)
            For example, if `num_sents` is set to `(1, 2, 3)`, chunks
            of 1, 2, and 3 consecutive sentences will be extracted.
        chunk_overlap : int, float, list, or dict, optional
            Overlap between chunks (in sentences), by default 0.
            If a float, it is interpreted as a fraction of the chunk size.
            If an integer, it is interpreted as the number of sentences to overlap.
            If a list or tuple, it should contain the overlap for each chunk size.
            If a dictionary, it should map chunk sizes to overlaps.
        prechunk : bool, optional
            Enable chunking of documents into overlapping sequences, by default True.
        prechunk_overlap : float or int, optional
            Overlap for splitting long documents into overlapping sequences, by default 0.5.
        sent_tokenizer : str, optional
            Sentence tokenizer to use for sentence boundary detection, by default "blingfire".
            Options are "blingfire", "nltk", "pysbd", or "syntok".
        exclude_special_tokens : bool, optional
            If True (default), exclude all special tokens from mean pooling.
            If False, include [CLS] in first chunk and [SEP] in last chunk
            of each sequence.
        deduplicate : bool, optional
            If True (default), average embeddings for duplicate chunks that arise
            from overlapping pre-chunks. Duplicates are identified by matching
            (document_idx, num_sents, sentence_ids). If False, keep all chunks.
        return_frame : str, optional
            The type of DataFrame of chunks and indices to return, by default 'polars'.
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
        prompt : str | None, optional
            Prompt to prepend to documents, by default None. If provided, overrides
            the document_prompt set at initialization. Prompt tokens are excluded from
            chunk mean-pooling. Used for instruct-style embedding models.
        max_chunk_tokens : int or None, optional
            Maximum number of tokens per chunk, by default None. When specified,
            chunks are built by greedily accumulating sentences until the token
            limit is reached, respecting sentence boundaries. Can be used in
            combination with `num_sents`:
            - max_chunk_tokens alone: Greedy accumulation, no sentence limit
            - num_sents alone: Fixed sentence count per chunk (default behavior)
            - Both specified: "At most N sentences AND at most M tokens" - whichever
              limit is hit first stops the chunk

        Returns
        -------
        tuple[pd.DataFrame | pl.DataFrame, np.ndarray | torch.Tensor]
            Tuple containing the DataFrame of chunks and the chunk embeddings.
        """
        # Validate inputs early to fail fast before expensive computation
        validate_encode_params(
            docs=docs,
            num_sents=num_sents,
            chunk_overlap=chunk_overlap,
            prechunk_overlap=prechunk_overlap,
            sent_tokenizer=sent_tokenizer,
            return_frame=return_frame,
            batch_tokens=batch_tokens,
            max_length=max_length,
            max_chunk_tokens=max_chunk_tokens,
        )
        if return_frame == "pandas":
            require_pandas()

        # Determine which prompt to use (per-call override or default)
        effective_prompt = prompt if prompt is not None else self.document_prompt

        inputs, sentence_texts = self._tokenize(
            docs,
            max_length=max_length,
            prechunk=prechunk,
            prechunk_overlap=prechunk_overlap,
            batch_size=_get_tokenization_batch_size(docs),
            sent_tokenizer=sent_tokenizer,
            prompt=effective_prompt,
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
        batches = self._generate_chunk_embeds(
            loader,
            num_sents=num_sents,
            chunk_overlap=chunk_overlap,
            move_results_to_cpu=False,
            return_tensors="pt",
            truncate_dim=self.truncate_dims,
            exclude_special_tokens=exclude_special_tokens,
            show_progress=show_progress,
            max_chunk_tokens=max_chunk_tokens,
        )

        results: dict[str, Any] = {
            "batch_idx": [],
            "sequence_idx": [],
            "chunk_idx": [],
            "chunk_token_ids": [],
            "chunk_sentence_ids": [],
            "num_sents": [],
            "chunk_embeds": [],
        }
        for batch in batches:
            # Apply postprocessing (half_embeds then normalization)
            batch["chunk_embeds"] = self.postprocess(batch["chunk_embeds"])
            # Offload batch to CPU
            batch = move_or_convert_tensors(batch, return_tensors="pt", move_to_cpu=True)
            results["batch_idx"].append(batch["batch_idx"])
            results["sequence_idx"].append(batch["sequence_idx"])
            results["chunk_idx"].append(batch["chunk_idx"])
            if isinstance(batch["chunk_token_ids"], list):
                results["chunk_token_ids"].extend(batch["chunk_token_ids"])
            else:
                results["chunk_token_ids"].append(batch["chunk_token_ids"])
            if isinstance(batch["sentence_ids"], list):
                results["chunk_sentence_ids"].extend(batch["sentence_ids"])
            else:
                results["chunk_sentence_ids"].append(batch["sentence_ids"])
            results["num_sents"].append(batch["num_sents"])
            results["chunk_embeds"].append(batch["chunk_embeds"])

        # Combine tensor results (excluding chunk_sentence_ids and chunk_token_ids
        # which need special handling for text reconstruction)
        keys_to_skip = {"chunk_sentence_ids", "chunk_token_ids"}
        for key, value in results.items():
            if key not in keys_to_skip and len(value) and isinstance(value[0], torch.Tensor):
                results[key] = torch.cat(value, dim=0)

        # Flatten chunk_sentence_ids and chunk_token_ids from batched to per-chunk lists
        # Original format: list of 2D batch tensors [[c1, c2], [c3, c4], ...]
        # Flattened format: list of 1D chunk tensors [c1, c2, c3, c4, ...]
        results["chunk_sentence_ids"] = [
            sid for batch in results["chunk_sentence_ids"] for sid in batch
        ]
        results["chunk_token_ids"] = [tid for batch in results["chunk_token_ids"] for tid in batch]

        # Build seq_to_doc mapping for document index computation and text reconstruction
        seq_to_doc = (
            dict(
                zip(
                    inputs.data["sequence_idx"],
                    inputs.data["overflow_to_sample_mapping"],
                    strict=False,
                )
            )
            if prechunk
            else {i: i for i in inputs.data["sequence_idx"]}
        )

        # Map sequence indices to document indices
        if prechunk:
            results["document_idx"] = torch.tensor(
                [seq_to_doc[s.item()] for s in results["sequence_idx"]]
            )
        else:
            results["document_idx"] = results["sequence_idx"]

        # Deduplicate chunks from overlapping pre-chunks (before text reconstruction)
        if deduplicate and prechunk:
            results = self._deduplicate_chunk_embeds(results)

        # Reconstruct chunk text from original sentences
        if return_text:
            results["chunk"] = self._reconstruct_chunk_texts(
                [results.pop("chunk_sentence_ids")],  # Wrap in list for batched format
                [results.pop("chunk_token_ids")],  # Wrap in list for batched format
                results["sequence_idx"],
                seq_to_doc,
                sentence_texts,
                show_progress=show_progress,
            )
        else:
            results.pop("chunk_token_ids")
            results.pop("chunk_sentence_ids")

        results["embed_idx"] = torch.arange(results["chunk_embeds"].shape[0])
        df, vecs = self._build_results_df(
            results,
            as_numpy=as_numpy,
            return_frame="polars",
            debug=debug,
        )
        if inputs.sort_by_token_count:
            df = df.sort("sequence_idx", "chunk_idx", descending=False)
            vecs = vecs[df["embed_idx"].to_list()]
        # Handle internal columns
        if debug:
            df = df.rename({"embed_idx": "orig_embed_idx"})
        else:
            df = df.drop(["embed_idx", "sequence_idx"])
        # Convert to requested DataFrame format
        if return_frame == "pandas":
            df = df.to_pandas()
        elif return_frame != "polars":
            raise ValueError(f"Invalid value for return_frame: {return_frame}")
        return df, vecs

    @overload
    def encode_queries(
        self,
        queries: list[str],
        max_length: int | None = ...,
        batch_size: int = ...,
        exclude_special_tokens: bool = ...,
        as_numpy: Literal[True] = ...,
        prompt: str | None = ...,
    ) -> np.ndarray[Any, Any]: ...

    @overload
    def encode_queries(
        self,
        queries: list[str],
        max_length: int | None = ...,
        batch_size: int = ...,
        exclude_special_tokens: bool = ...,
        as_numpy: Literal[False] = ...,
        prompt: str | None = ...,
    ) -> torch.Tensor: ...

    def encode_queries(
        self,
        queries: list[str],
        max_length: int | None = None,
        batch_size: int = 32,
        exclude_special_tokens: bool = True,
        as_numpy: bool = True,
        prompt: str | None = None,
    ) -> np.ndarray[Any, Any] | torch.Tensor:
        """Obtain the mean-tokens embeddings for a list of query strings.

        This is a convenient method for embedding query strings into the same space
        as the chunks extracted from documents. It is mainly useful for doing semantic
        search.

        Parameters
        ----------
        queries : list[str]
            List of queries to encode.
        max_length : int, optional
            Maximum length of the query sequences, by default None.
        batch_size : int, optional
            Batch size for encoding, by default 32.
        exclude_special_tokens : bool, optional
            If True (default), exclude all special tokens from mean pooling.
            If False, include [CLS] and [SEP] tokens in mean pooling for queries.
        as_numpy : bool, optional
            Convert the tensors to numpy arrays before returning, by default True.
        prompt : str | None, optional
            Prompt to prepend to queries, by default None. If provided, overrides
            the query_prompt set at initialization. Used for instruct-style embedding
            models that require task-specific prefixes.

        Returns
        -------
        np.ndarray
            Mean-token embeddings for each query.
        """
        # Validate inputs early to fail fast
        validate_encode_queries_params(
            queries=queries,
            batch_size=batch_size,
            max_length=max_length,
        )

        # Determine which prompt to use (per-call override or default)
        effective_prompt = prompt if prompt is not None else self.query_prompt

        # Apply prompt to queries
        queries_with_prompt = self._apply_prompt(queries, effective_prompt)

        token_batch_size = _get_tokenization_batch_size(queries_with_prompt)
        num_token_batches = math.ceil(len(queries_with_prompt) / token_batch_size)
        inputs, _ = self._tokenize(
            queries_with_prompt,
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
            truncate_dim=self.truncate_dims,
        )
        query_embeds_list: list[torch.Tensor] = []
        for batch in batches:
            token_embeds = batch["token_embeds"]
            input_ids = batch["input_ids"]
            if exclude_special_tokens:
                # Default: exclude all special tokens
                valid_token_mask = torch.isin(
                    input_ids,
                    torch.tensor(self.tokenizer.all_special_ids, device=self.device),
                    invert=True,
                )
            else:
                # Include [CLS] and [SEP] for queries (entire sequence is a single "chunk")
                # Only exclude padding tokens
                valid_token_mask = input_ids != self.tokenizer.pad_token_id
            valid_token_weight = valid_token_mask.unsqueeze(2).float()
            mean_tokens = (token_embeds * valid_token_weight).sum(dim=1) / valid_token_weight.sum(
                dim=1
            )
            mean_tokens_processed = self.postprocess(mean_tokens)
            if isinstance(mean_tokens_processed, np.ndarray):
                mean_tokens_processed = torch.from_numpy(mean_tokens_processed)
            query_embeds_list.append(mean_tokens_processed.cpu())
        query_embeds = torch.vstack(query_embeds_list)
        # Restore original order (TokenizedDataset sorts by token count for efficiency)
        if inputs.sort_by_token_count:
            query_embeds = query_embeds[torch.tensor(inputs.unsort_idx)]
        if as_numpy:
            return query_embeds.numpy()
        return query_embeds
