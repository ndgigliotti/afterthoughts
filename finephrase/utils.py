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
import os
import time
import warnings
from functools import singledispatch
from typing import Callable

import numpy as np
import polars as pl
import pyarrow as pa
import torch
import torch.nn.functional as F

from finephrase.available import _HAS_FAISS, _HAS_PANDAS

if _HAS_PANDAS:
    import pandas as pd
else:
    pd = None


def norm_jobs(num_jobs: int | None) -> int:
    """Return the normalized number of jobs.

    Parameters
    ----------
    num_jobs : int or None
        Number of jobs. If None, 1 is returned. If 0, 1 is returned. If negative,
        the number of jobs is set to `os.cpu_count() + 1 + num_jobs`. If greater than
        the number of CPU cores, a warning is issued and the number of jobs is clipped to
        the number of CPU cores.

    Returns
    -------
    int
        Absolute number of jobs.
    """
    true_num_jobs = num_jobs
    cpu_count = os.cpu_count()
    if num_jobs is None:
        true_num_jobs = 1
    elif num_jobs == 0:
        warnings.warn("`num_jobs` cannot be 0. Setting `num_jobs` to 1.")
        true_num_jobs = 1
    elif num_jobs < 0:
        true_num_jobs = cpu_count + 1 + num_jobs
    if true_num_jobs > cpu_count:
        warnings.warn(
            f"`num_jobs` ({num_jobs}) exceeds the number of CPU cores ({cpu_count})."
        )
        true_num_jobs = min(num_jobs, cpu_count)
    return true_num_jobs


def timer(readout: str = "Execution time: {time:.4f} seconds") -> Callable:
    """Decorator to time a function execution and print the result.

    Parameters
    ----------
    readout : str
        Format string for the printout. Must contain '{time}'.

    Returns
    -------
    Callable
        Decorated function.
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(readout.format(time=elapsed_time))
            return result

        return wrapper

    return decorator


def get_memory_size(
    a: pa.Array | torch.Tensor | np.ndarray | pl.Series | pl.DataFrame,
) -> int:
    """Get the size of the array in bytes.

    Parameters
    ----------
    a : pa.Array, torch.Tensor, np.ndarray, pl.Series, pl.DataFrame, pd.Series, pd.DataFrame
        Array.

    Returns
    -------
    int
        Array size in bytes.

    Raises
    ------
    TypeError
        If the input is not a supported type.
    """
    if isinstance(a, pa.Array):
        return sum(buf.size for buf in a.buffers() if buf is not None)
    elif isinstance(a, np.ndarray):
        return a.nbytes
    elif isinstance(a, torch.Tensor):
        return a.element_size() * a.numel()
    elif isinstance(a, (pl.Series, pl.DataFrame)):
        return a.estimated_size()
    elif _HAS_PANDAS and isinstance(a, pd.Series):
        return a.memory_usage(index=True, deep=True)
    elif _HAS_PANDAS and isinstance(a, pd.DataFrame):
        return a.memory_usage(index=True, deep=True).sum()
    else:
        raise TypeError(f"Invalid input type {type(a).__name__}.")


def format_memory_size(n_bytes: int) -> str:
    """Format the size of an array in bytes.

    Parameters
    ----------
    n_bytes : int
        Size in bytes.

    Returns
    -------
    str
        Formatted size.
    """
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.2f} PB"


def get_memory_report(
    data: dict[str, pa.Array | torch.Tensor | np.ndarray | pl.Series | pl.DataFrame],
    readable: bool = True,
) -> dict[str, str]:
    """Get the size of the arrays in data.

    Parameters
    ----------
    data : dict[str, pa.Array or torch.Tensor or np.ndarray or pl.Series or pl.DataFrame or pd.Series or pd.DataFrame]
        Dictionary of arrays.
    readable : bool
        Whether to format the sizes in human-readable format, by default True.
        If True, the sizes are formatted in bytes, KB, MB, GB, TB, or PB and
        returned as strings.

    Returns
    -------

    dict[str, str]
        Dictionary of array sizes in human-readable format.
    """
    report = {}
    for name, arr in data.items():
        valid_types = (pa.Array, torch.Tensor, np.ndarray, pl.Series, pl.DataFrame)
        if _HAS_PANDAS:
            valid_types += (pd.Series, pd.DataFrame)
        if not isinstance(arr, valid_types):
            warnings.warn(f"Encountered invalid input type {type(arr).__name__}.")
        else:
            n_bytes = get_memory_size(arr)
            report[name] = n_bytes
    report["_total_"] = sum(report.values())
    if readable:
        report = {name: format_memory_size(size) for name, size in report.items()}
    return report


def normalize(
    embeds: torch.Tensor | np.ndarray, dim: int = 1
) -> torch.Tensor | np.ndarray:
    """Normalize the embeddings to have unit length.

    Parameters
    ----------
    embeds : torch.Tensor or np.ndarray
        Embeddings.
    dim : int
        Dimension to normalize.

    Returns
    -------
    torch.Tensor or np.ndarray
        Normalized embeddings.

    Raises
    ------
    TypeError
        If the input is not a Torch tensor or NumPy array.
    """
    eps = 1e-12
    if isinstance(embeds, np.ndarray):
        norms = np.linalg.norm(embeds, axis=dim, keepdims=True)
        norms[norms == 0] = eps
        return embeds / norms
    elif isinstance(embeds, torch.Tensor):
        return F.normalize(embeds, p=2, dim=dim, eps=eps)
    else:
        raise TypeError(f"Invalid input type {type(embeds).__name__}.")


def get_torch_dtype(dtype: str) -> torch.dtype:
    """Return the Torch data type for the specified string.

    Parameters
    ----------
    dtype : str
        Data type string.

    Returns
    -------
    torch.dtype
        Torch data type.
    """
    err_msg = f"Invalid Torch data type '{dtype}'."
    if hasattr(torch, dtype):
        torch_dtype = getattr(torch, dtype)
        if not isinstance(torch_dtype, torch.dtype):
            raise ValueError(err_msg)
    else:
        raise ValueError(err_msg)
    return torch_dtype


@singledispatch
def reduce_precision(a: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Reduce the precision of the input to float16.

    If the precision is already float16 or below, the input is returned as is.
    If the data type is not floating point, the input is returned as is.

    Parameters
    ----------
    a : torch.Tensor or np.ndarray
        Input data.

    Returns
    -------
    torch.Tensor or np.ndarray
        Data with reduced precision.

    Raises
    ------
    TypeError
        If the input is not a Torch tensor or NumPy array.
    """
    if not isinstance(a, (torch.Tensor, np.ndarray)):
        raise TypeError(f"Invalid input type {type(a).__name__}.")


@reduce_precision.register
def _(a: torch.Tensor) -> torch.Tensor:
    """Sub-function for reducing the precision of Torch tensors."""
    if a.dtype.is_floating_point and a.element_size() > 2:
        a = a.to(torch.float16)
    return a


@reduce_precision.register
def _(a: np.ndarray) -> np.ndarray:
    """Sub-function for reducing the precision of NumPy arrays."""
    if np.issubdtype(a.dtype, np.floating) and a.dtype.itemsize > 2:
        a = a.astype(np.float16)
    return a


@singledispatch
def truncate_dims(
    embeds: torch.Tensor | np.ndarray, dim: int
) -> torch.Tensor | np.ndarray:
    """Truncate the dimensions of the embeddings.

    Parameters
    ----------
    embeds : torch.Tensor or np.ndarray
        Embeddings.
    dim : int
        Number of dimensions to keep.

    Returns
    -------
    torch.Tensor or np.ndarray
        Truncated embeddings.

    Raises
    ------
    TypeError
        If the input is not a Torch tensor or NumPy array.
    """
    if not isinstance(embeds, (torch.Tensor, np.ndarray)):
        raise TypeError(f"Invalid input type {type(embeds).__name__}.")


@truncate_dims.register
def _(embeds: torch.Tensor, dim: int) -> torch.Tensor:
    """Sub-function for truncating the dimensions of Torch tensors."""
    if dim < 1:
        raise ValueError(f"Invalid number of dimensions {dim}.")
    input_axis_count = embeds.ndim
    if input_axis_count > 3:
        raise ValueError(f"Invalid input shape {embeds.shape}.")
    # If in shape (n_batches, n_tokens, n_features), flatten to 2D
    if input_axis_count == 3:
        seq_len = embeds.size(1)
        embeds = embeds.view(-1, embeds.size(2))
    elif input_axis_count == 1:
        embeds = embeds.unsqueeze(0)
    # Truncate the dimensions
    if dim < embeds.size(1):
        embeds = embeds[:, :dim]
    # Reshape back to original number of axes if necessary
    if input_axis_count == 3:
        embeds = embeds.view(-1, seq_len, dim)
    elif input_axis_count == 1:
        embeds = embeds.squeeze(0)
    return embeds.contiguous()


@truncate_dims.register
def _(embeds: np.ndarray, dim: int) -> np.ndarray:
    """Sub-function for truncating the dimensions of NumPy arrays."""
    if dim < 1:
        raise ValueError(f"Invalid number of dimensions {dim}.")
    input_axis_count = embeds.ndim
    if input_axis_count > 3:
        raise ValueError(f"Invalid input shape {embeds.shape}.")
    # If in shape (n_batches, n_tokens, n_features), flatten to 2D
    if input_axis_count == 3:
        seq_len = embeds.shape[1]
        embeds = embeds.reshape(-1, embeds.shape[2])
    elif input_axis_count == 1:
        embeds = np.expand_dims(embeds, 0)
    # Truncate the dimensions
    if dim < embeds.shape[1]:
        embeds = embeds[:, :dim]
    # Reshape back to original number of axes if necessary
    if input_axis_count == 3:
        embeds = embeds.reshape(-1, seq_len, dim)
    elif input_axis_count == 1:
        embeds = np.squeeze(embeds, 0)
    return embeds


@timer(readout="Built FAISS index in {time:.4f} seconds.")
def build_faiss_index(embeds: np.ndarray):
    """Build a FAISS index for the given embeddings.

    Parameters
    ----------
    embeds : np.ndarray
        Matrix of semantic vector representations.

    Returns
    -------
    faiss.Index
        The FAISS index used for finding nearest neighbors.
    """
    if _HAS_FAISS:
        import faiss
    else:
        raise ImportError("FAISS is not installed.")
    embeds = normalize(embeds)
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(embeds.shape[1])  # Inner product index
    index.add(embeds)
    return index


@timer(readout="Search completed in {time:.4f} seconds.")
def search_phrases(
    queries: list,
    query_embeds: np.ndarray,
    phrase_df: pl.DataFrame,
    phrase_embeds: np.ndarray,
    sim_thresh: float = 0.5,
    index=None,
) -> dict[str, pl.DataFrame]:
    """
    Search for documents that match the given queries based on their vector representations using FAISS.

    Parameters
    ----------
    queries : list of str
        List of query strings.
    query_embeds : np.ndarray
        Matrix of vector representations for the queries.
    phrase_df : pl.DataFrame
        DataFrame containing phrase information. It should have a column "embed_idx"
        for indexing.
    phrase_embeds : np.ndarray
        Matrix of vector representations for the phrases.
    sim_thresh : float, optional
        Cosine similarity threshold for the nearest neighbors search. Default is 0.5.
        Will return all results with similarity equal to or above this threshold.
    index : faiss.Index, optional
        Pre-built FAISS index. If not provided, a new index will be created.

    Returns
    -------
    index : faiss.Index
        The FAISS index used for finding nearest neighbors.
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
        Build a FAISS index for the given phrase embeddings.
    finephrase.finephrase.FinePhrase.search
        Convenient search method which embeds query strings on the fly.
    """
    if not _HAS_FAISS:
        raise ImportError("FAISS is not installed.")
    pandas_input = _HAS_PANDAS and isinstance(phrase_df, pd.DataFrame)
    if pandas_input:
        phrase_df = pl.from_pandas(phrase_df)
    query_embeds = normalize(query_embeds)
    if index is None:
        index = build_faiss_index(phrase_embeds)
    print("Searching...")
    similarities, indices = index.search(query_embeds, len(phrase_embeds))

    hits = dict.fromkeys(queries, pl.DataFrame())
    for sims, idx, query in zip(similarities, indices, queries):
        mask = sims >= sim_thresh
        valid_idx, valid_sims = idx[mask], sims[mask]
        if mask.any():
            df_hits = phrase_df.filter(pl.col("embed_idx").is_in(valid_idx))
            df_hits = df_hits.with_columns(pl.Series("query_sim", valid_sims)).sort(
                "query_sim", descending=True
            )
            hits[query] = df_hits
    if pandas_input:
        hits = {k: v.to_pandas() for k, v in hits.items()}
    return index, hits
