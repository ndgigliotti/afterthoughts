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

"""Encoder is a library for extracting sentence-chunk embeddings using transformer models."""

import logging
import math
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

from afterthoughts.avail import require_pandas
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
    binary_quantize,
    disable_tokenizer_parallelism,
    half_embeds,
    int8_quantize,
    move_or_convert_tensors,
    normalize,
    normalize_num_jobs,
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

    def postprocess(self, embeds: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Apply postprocessing steps to embeddings.

        Base implementation only normalizes if enabled. Subclasses may override
        to add additional steps (e.g., quantization, PCA).

        Parameters
        ----------
        embeds : torch.Tensor or np.ndarray
            Embeddings to postprocess.

        Returns
        -------
        torch.Tensor or np.ndarray
            Postprocessed embeddings.
        """
        return self.normalize_if_needed(embeds)

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
            doc_idx = seq_to_doc[sequence_idx[i].item()]
            doc_sentences = sentence_texts[doc_idx]
            # Sentence IDs are contiguous, so just get min/max instead of unique
            valid_mask = sent_ids != -1
            if not valid_mask.any():
                chunks.append("")
                continue
            valid_ids = sent_ids[valid_mask]
            min_id, max_id = valid_ids[0].item(), valid_ids[-1].item()
            # Handle edge case: sentence was split (ID exceeds original count)
            if max_id >= len(doc_sentences):
                chunks.append(None)  # Placeholder for fallback
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
            pd = require_pandas()
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
        sent_tokenizer: str = "blingfire",
        show_progress: bool = True,
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
        inputs, sentence_texts = tokenize_with_sentence_boundaries(
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
        )
        return inputs, sentence_texts

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

    def _generate_chunk_embeds(
        self,
        loader: DataLoader,
        num_sents: int | list | tuple,
        chunk_overlap: int | float | list | dict,
        move_results_to_cpu: bool = False,
        return_tensors: str = "pt",
        truncate_dim: int | None = None,
        show_progress: bool = True,
    ):
        """Obtain the chunk embeddings for a list of documents, one batch at at time.

        Parameters
        ----------
        loader : DataLoader
            DataLoader containing the tokenized input sequences.
        num_sents : int, list, or tuple
            Number of sentences per chunk.
        chunk_overlap : int, float, list, or dict
            Overlap between chunks (in sentences).
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
        """Obtain the chunks and chunk embeddings from a list of documents."""
        pass

    def encode_queries(
        self,
        queries: list[str],
        max_length: int | None = None,
        batch_size: int = 32,
        as_numpy: bool = True,
    ) -> np.ndarray:
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
        as_numpy : bool, optional
            Convert the tensors to numpy arrays before returning, by default True.

        Returns
        -------
        np.ndarray
            Mean-token embeddings for each query.
        """
        token_batch_size = _get_tokenization_batch_size(queries)
        num_token_batches = math.ceil(len(queries) / token_batch_size)
        inputs, _ = self._tokenize(
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
            mean_tokens = self.postprocess(mean_tokens)
            query_embeds.append(mean_tokens.cpu())
        query_embeds = torch.vstack(query_embeds)
        if as_numpy:
            query_embeds = query_embeds.numpy()
        return query_embeds


class Encoder(_EncoderBase):
    """Simple Encoder model for generating sentence-chunk embeddings.

    This class provides a straightforward API for extracting chunk embeddings
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
        sent_tokenizer: str = "blingfire",
        return_frame: str = "polars",
        as_numpy: bool = True,
        debug: bool = False,
        return_text: bool = True,
        show_progress: bool = True,
    ) -> dict[str, np.ndarray | torch.Tensor]:
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
        num_sents : int, list, or tuple, optional
            Number of sentences per chunk, by default 1.
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

        Returns
        -------
        tuple[pd.DataFrame | pl.DataFrame, np.ndarray | torch.Tensor]
            Tuple containing the DataFrame of chunks and the chunk embeddings.
        """
        # Validate return_frame early to fail fast before expensive computation
        if return_frame == "pandas":
            require_pandas()
        elif return_frame != "polars":
            raise ValueError(f"Invalid value for return_frame: {return_frame}")

        inputs, sentence_texts = self._tokenize(
            docs,
            max_length=max_length,
            prechunk=prechunk,
            prechunk_overlap=prechunk_overlap,
            batch_size=_get_tokenization_batch_size(docs),
            sent_tokenizer=sent_tokenizer,
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
            show_progress=show_progress,
        )

        results = {
            "batch_idx": [],
            "sequence_idx": [],
            "chunk_idx": [],
            "chunk_token_ids": [],
            "chunk_sentence_ids": [],
            "chunk_size": [],
            "chunk_embeds": [],
        }
        for batch in batches:
            # Apply postprocessing (normalization in base Encoder)
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
            results["chunk_size"].append(batch["chunk_size"])
            results["chunk_embeds"].append(batch["chunk_embeds"])

        # Build seq_to_doc mapping for text reconstruction
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

        # Reconstruct chunk text from original sentences (or decode if return_text)
        if return_text:
            results["chunk"] = self._reconstruct_chunk_texts(
                results.pop("chunk_sentence_ids"),
                results.pop("chunk_token_ids"),
                torch.cat(results["sequence_idx"]),
                seq_to_doc,
                sentence_texts,
                show_progress=show_progress,
            )
        else:
            results.pop("chunk_token_ids")
            results.pop("chunk_sentence_ids")
        # Combine results
        for key, value in results.items():
            if len(value) and isinstance(value[0], torch.Tensor):
                results[key] = torch.cat(value, dim=0)
        # Map sequence indices to document indices (reuse seq_to_doc from text reconstruction)
        if prechunk:
            results["document_idx"] = torch.tensor(
                [seq_to_doc[s.item()] for s in results["sequence_idx"]]
            )
        else:
            results["document_idx"] = results["sequence_idx"]
        results["embed_idx"] = torch.arange(results["chunk_embeds"].shape[0])
        df, vecs = self._build_results_df(
            results,
            as_numpy=as_numpy,
            return_frame="polars",
            debug=debug,
        )
        if inputs.sort_by_token_count:
            df = df.sort("sequence_idx", "chunk_idx", descending=False)
            vecs = vecs[df["embed_idx"]]
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


class LiteEncoder(_EncoderBase):
    """Memory-efficient Encoder variant for advanced users.

    This class includes lossy memory optimizations:
    - PCA dimensionality reduction (GPU-accelerated, incremental fitting)
    - Quantization (fp16, int8, or binary)
    - Dimension truncation

    For simple use cases without these optimizations, use Encoder instead.
    """

    QUANTIZE_OPTIONS = (None, "float16", "int8", "binary")

    def __init__(
        self,
        model_name: str,
        model_dtype: torch.dtype = torch.float32,
        amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        attn_implementation: str | None = None,
        truncate_dims: int | None = None,
        normalize: bool = False,
        pca: int | None = None,
        pca_early_stop: int | float = 1.0,
        quantize: str | None = "float16",
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
        truncate_dims : int, None, optional
            Truncate the dimensions of the embeddings, by default None.
        normalize : bool, optional
            Normalize the embeddings to unit length, by default False.
        pca : int, None, optional
            Number of principal components to keep after PCA, by default None.
        pca_early_stop : int, float, optional
            Number of batches to use for fitting the PCA model, by default 1.0.
            If a float, it is the fraction of the dataset to use.
        quantize : str or None, optional
            Quantization method for embeddings, by default "float16".
            - None: No quantization (full float32)
            - "float16": Float16 precision (2x compression)
            - "int8": Per-row uint8 quantization (4x compression).
              Returns (embeds, scales, min_vals) tuple for dequantization.
            - "binary": Packed binary (32x compression). Incompatible with normalize.
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
        self.truncate_dims = truncate_dims
        self.pca = pca
        self.pca_early_stop = pca_early_stop
        self.quantize = quantize

        if truncate_dims is not None and pca is not None and truncate_dims < pca:
            raise ValueError("`truncate_dims` must be greater than `pca`.")
        match quantize:
            case None | "float16":
                pass
            case "int8" | "binary" if normalize:
                raise ValueError(f"`quantize={quantize!r}` is incompatible with `normalize=True`.")
            case "int8" | "binary":
                pass
            case _:
                raise ValueError(
                    f"`quantize` must be one of {self.QUANTIZE_OPTIONS}, got {quantize!r}."
                )

    def quantize_if_needed(self, embeds: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Apply quantization if enabled."""
        match self.quantize:
            case "float16":
                embeds = half_embeds(embeds)
            case "int8":
                embeds = int8_quantize(embeds)
            case "binary":
                embeds = binary_quantize(embeds)
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

    def postprocess(self, embeds: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Apply all postprocessing steps to the embeddings.

        The steps are:
        1. Apply PCA transformation (if enabled).
        2. Apply quantization (float16, int8, or binary).
        3. Normalize embeddings to unit length (if enabled, only for float16/None).

        Parameters
        ----------
        embeds : torch.Tensor or np.ndarray
            Embeddings to postprocess.

        Returns
        -------
        torch.Tensor or np.ndarray
            Postprocessed embeddings.

        Raises
        ------
        RuntimeError
            If PCA mode is enabled but PCA is not ready.
        """
        if self.pca_mode:
            if not self.pca_is_ready:
                raise RuntimeError(
                    "Cannot postprocess in PCA mode when PCA is not ready. "
                    "Fit more batches or disable PCA."
                )
            embeds = self.apply_pca(embeds)
        steps = [
            self.quantize_if_needed,
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
            hasattr(self, "pca_transformer_")
            and hasattr(self, "_n_pca_fit_batches")
            and hasattr(self.pca_transformer_, "n_batches_seen_")
            and self.pca_transformer_.n_batches_seen_ >= self._n_pca_fit_batches
        )

    def update_pca(self, chunk_embeds: torch.Tensor) -> None:
        """Update the PCA transformation with a batch of chunk embeddings.

        Parameters
        ----------
        chunk_embeds : torch.Tensor
            Chunk embeddings to update the PCA model with.
        """
        if not hasattr(self, "pca_transformer_"):
            self.pca_transformer_ = IncrementalPCA(n_components=self.pca, device=self.device)
        self.pca_transformer_.partial_fit(chunk_embeds)

    def apply_pca(self, chunk_embeds: torch.Tensor) -> torch.Tensor:
        """Apply PCA transformation to embeddings.

        Parameters
        ----------
        chunk_embeds : torch.Tensor
            Chunk embeddings to apply PCA to.

        Returns
        -------
        torch.Tensor
            PCA-transformed embeddings.
        """
        if not hasattr(self, "pca_transformer_"):
            raise AttributeError("PCA must be fitted first.")
        if not self.pca_is_ready:
            raise RuntimeError("PCA has not seen enough batches to be applied yet.")
        # Ensure PCA is on the same device as input
        self.pca_transformer_.to(chunk_embeds.device)
        return self.pca_transformer_.transform(chunk_embeds)

    def clear_pca(self) -> None:
        """Clear the fitted PCA transformation."""
        pca_attrs = ["pca_transformer_", "_n_pca_fit_batches"]
        for attr in pca_attrs:
            if hasattr(self, attr):
                delattr(self, attr)
        logger.info("PCA cleared.")

    def _postprocess_early_batches(self, results: dict) -> None:
        """Apply postprocessing to early batches that were used for PCA fitting.

        During the fitting phase, early batches are stored without postprocessing.
        Once PCA fitting is complete, this method goes back and applies
        postprocessing (including PCA) to those batches.

        Parameters
        ----------
        results : dict
            Results dictionary containing chunk_embeds list to process in-place.
        """
        if not self.pca_is_ready:
            warnings.warn("PCA did not finish fitting and will not be applied.", stacklevel=3)
            return
        for i in range(self._n_pca_fit_batches):
            batch_embeds = results["chunk_embeds"][i]
            # Skip if already postprocessed (dimension matches PCA output)
            if batch_embeds.size(1) != self.pca:
                results["chunk_embeds"][i] = self.postprocess(batch_embeds)

    def _new_results_dict(self) -> dict:
        """Create a new results dictionary for batch accumulation."""
        return {
            "batch_idx": [],
            "sequence_idx": [],
            "chunk_idx": [],
            "chunk_token_ids": [],
            "chunk_sentence_ids": [],
            "chunk_size": [],
            "chunk_embeds": [],
        }

    def _accumulate_batch(self, results: dict, batch: dict) -> None:
        """Accumulate a processed batch into the results dictionary."""
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
        results["chunk_size"].append(batch["chunk_size"])
        results["chunk_embeds"].append(batch["chunk_embeds"])

    def _process_batches_simple(self, batches) -> dict:
        """Process batches with immediate postprocessing (non-PCA or PCA-ready mode)."""
        results = self._new_results_dict()
        for batch in batches:
            batch["chunk_embeds"] = self.postprocess(batch["chunk_embeds"])
            self._accumulate_batch(results, batch)
        return results

    def _process_batches_pca_fitting(self, batches, n_fit_batches: int) -> dict:
        """Process batches while fitting PCA on early batches.

        Early batches are used to fit PCA, then PCA is applied to later batches.
        After the loop, early batches are retroactively postprocessed.
        """
        self._n_pca_fit_batches = n_fit_batches
        results = self._new_results_dict()
        for batch in batches:
            if not self.pca_is_ready:
                # Fitting PCA - don't postprocess yet
                self.update_pca(batch["chunk_embeds"])
            else:
                # PCA is ready - postprocess (includes PCA)
                batch["chunk_embeds"] = self.postprocess(batch["chunk_embeds"])
            self._accumulate_batch(results, batch)
        # Retroactively postprocess early batches
        self._postprocess_early_batches(results)
        return results

    def encode(
        self,
        docs: list[str],
        max_length: int | None = None,
        batch_tokens: int = 16384,
        num_sents: int | list | tuple = 1,
        chunk_overlap: int | float | list | dict = 0,
        prechunk: bool = True,
        prechunk_overlap: float | int = 0.5,
        sent_tokenizer: str = "blingfire",
        return_frame: str = "polars",
        as_numpy: bool = True,
        debug: bool = False,
        return_text: bool = True,
        show_progress: bool = True,
    ) -> dict[str, np.ndarray | torch.Tensor]:
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
        num_sents : int, list, or tuple, optional
            Number of sentences per chunk, by default 1.
        chunk_overlap : int, float, list, or dict, optional
            Overlap between chunks (in sentences), by default 0.
        prechunk : bool, optional
            Enable chunking of documents into overlapping sequences, by default True.
        prechunk_overlap : float or int, optional
            Overlap for splitting long documents into overlapping sequences, by default 0.5.
        sent_tokenizer : str, optional
            Sentence tokenizer to use for sentence boundary detection, by default "blingfire".
            Options are "blingfire", "nltk", "pysbd", or "syntok".
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

        Returns
        -------
        tuple[pd.DataFrame | pl.DataFrame, np.ndarray | torch.Tensor]
            Tuple containing the DataFrame of chunks and the chunk embeddings.
        """
        # Validate arguments early to fail fast before expensive computation
        if return_frame == "pandas":
            require_pandas()
        elif return_frame != "polars":
            raise ValueError(f"Invalid value for return_frame: {return_frame}")
        if self.quantize == "binary" and not as_numpy:
            raise ValueError("`quantize='binary'` requires `as_numpy=True`.")

        inputs, sentence_texts = self._tokenize(
            docs,
            max_length=max_length,
            prechunk=prechunk,
            prechunk_overlap=prechunk_overlap,
            batch_size=_get_tokenization_batch_size(docs),
            sent_tokenizer=sent_tokenizer,
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
            show_progress=show_progress,
        )
        # Process batches based on PCA mode
        if self.pca_mode and not self.pca_is_ready:
            # PCA needs fitting - use fitting mode
            if isinstance(self.pca_early_stop, float):
                n_fit_batches = math.ceil(len(loader) * self.pca_early_stop)
            else:
                n_fit_batches = self.pca_early_stop
            results = self._process_batches_pca_fitting(batches, n_fit_batches)
        else:
            # No PCA or PCA already ready - use simple mode
            if self.pca_mode:
                logger.info("PCA is already fit and will be applied to all batches.")
            results = self._process_batches_simple(batches)

        # Build seq_to_doc mapping for text reconstruction
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

        # Reconstruct chunk text from original sentences (or decode if return_text)
        if return_text:
            results["chunk"] = self._reconstruct_chunk_texts(
                results.pop("chunk_sentence_ids"),
                results.pop("chunk_token_ids"),
                torch.cat(results["sequence_idx"]),
                seq_to_doc,
                sentence_texts,
                show_progress=show_progress,
            )
        else:
            results.pop("chunk_token_ids")
            results.pop("chunk_sentence_ids")
        # Combine results
        for key, value in results.items():
            if len(value) and isinstance(value[0], torch.Tensor):
                results[key] = torch.cat(value, dim=0)
        # Map sequence indices to document indices (reuse seq_to_doc from text reconstruction)
        if prechunk:
            results["document_idx"] = torch.tensor(
                [seq_to_doc[s.item()] for s in results["sequence_idx"]]
            )
        else:
            results["document_idx"] = results["sequence_idx"]
        results["embed_idx"] = torch.arange(results["chunk_embeds"].shape[0])
        df, vecs = self._build_results_df(
            results,
            as_numpy=as_numpy,
            return_frame="polars",
            debug=debug,
        )
        if inputs.sort_by_token_count:
            df = df.sort("sequence_idx", "chunk_idx", descending=False)
            vecs = vecs[df["embed_idx"]]
        # Handle internal columns
        if debug:
            # Rename embed_idx to orig_embed_idx for clarity
            df = df.rename({"embed_idx": "orig_embed_idx"})
        else:
            # Drop internal columns in non-debug mode
            df = df.drop(["embed_idx", "sequence_idx"])
        # Convert to requested DataFrame format
        if return_frame == "pandas":
            df = df.to_pandas()
        elif return_frame != "polars":
            raise ValueError(f"Invalid value for return_frame: {return_frame}")
        return df, vecs

    def encode_queries(
        self,
        queries: list[str],
        max_length: int | None = None,
        batch_size: int = 32,
        as_numpy: bool = True,
    ) -> np.ndarray:
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
        as_numpy : bool, optional
            Convert the tensors to numpy arrays before returning, by default True.

        Returns
        -------
        np.ndarray or tuple
            Mean-token embeddings for each query. If quantize='binary', returns
            packed uint8 array. If quantize='int8', returns (quantized, scales,
            min_vals) tuple.
        """
        if self.quantize == "binary" and not as_numpy:
            raise ValueError("`quantize='binary'` requires `as_numpy=True`.")
        return super().encode_queries(
            queries, max_length=max_length, batch_size=batch_size, as_numpy=as_numpy
        )
