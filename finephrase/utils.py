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


import logging
import math
import os
import time
import warnings
from collections.abc import Callable
from functools import singledispatch
from operator import itemgetter

import numpy as np
import polars as pl
import pyarrow as pa
import torch
import torch.nn.functional as F

from finephrase.available import _HAS_PANDAS

if _HAS_PANDAS:
    import pandas as pd
else:
    pd = None


def configure_logging(
    level: str = "INFO",
    stream: bool = True,
    log_file: str | None = None,
    file_mode: str = "a",
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """Configure the logging for the root logger.

    Parameters
    ----------
    level : str
        Logging level. Default is "INFO".
    stream : bool
        Whether to log to the console. Default is True.
    log_file : str or None
        Path to a log file. If None, logging to file is disabled. Default is None.
    file_mode : str
        Mode to open the log file. Default is "a" (append).
    format : str
        Log message format. Default is "%(asctime)s - %(name)s - %(levelname)s - %(message)s".
    datefmt : str
        Date format for log messages. Default is "%Y-%m-%d %H:%M:%S".
    """
    handlers = []
    if stream:
        handlers.append(logging.StreamHandler())
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode=file_mode))

    logging.basicConfig(level=level, format=format, datefmt=datefmt, handlers=handlers)


def get_overlap_count(
    overlap: float | int | list | tuple | dict,
    length: int,
    length_idx: int | None = None,
) -> int:
    """
    Calculate the number of overlap tokens or sentences given a length.

    Parameters
    ----------
    overlap : float, int, list, tuple, or dict
        The overlap specification. It can be:
        - float: A fraction in the range [0, 1) representing the proportion of the sequence length.
        - int: A fixed number of tokens or sentences.
        - list or tuple: A sequence where the overlap is determined by the index `length_idx`.
        - dict: A mapping where the overlap is determined by the `length` key.
    length : int
        The length of the sequence (number of tokens or sentences).
    length_idx : int or None, optional
        The index to use if `overlap` is a list or tuple. Default is None.

    Returns
    -------
    int
        The number of overlap tokens or sentences.

    Raises
    ------
    ValueError
        If `overlap` is a float not in the range [0, 1), or an int not in the range [0, length),
        or if `overlap` is not a float, int, list, tuple, or dict.
    """
    # Check if `overlap` is a float in [0, 1)
    if isinstance(overlap, float):
        if overlap < 0 or overlap >= 1:
            raise ValueError("`overlap` must be within [0, 1).")
        # Calculate the number of tokens or sentences to overlap
        overlap_count = math.ceil(length * overlap)
        if overlap_count == length:
            overlap_count -= 1
    elif isinstance(overlap, (list, tuple)):
        overlap_count = overlap[length_idx]
    elif isinstance(overlap, dict):
        overlap_count = overlap[length]
    elif isinstance(overlap, int):
        if overlap < 0 or overlap >= length:
            raise ValueError("`overlap` must be within [0, length).")
        overlap_count = overlap
    else:
        raise ValueError("`overlap` must be a float, list, tuple, dict, or int.")
    return overlap_count


def normalize_num_jobs(num_jobs: int | None) -> int:
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
        warnings.warn("`num_jobs` cannot be 0. Setting `num_jobs` to 1.", stacklevel=2)
        true_num_jobs = 1
    elif num_jobs < 0:
        true_num_jobs = cpu_count + 1 + num_jobs
    if true_num_jobs > cpu_count:
        warnings.warn(
            f"`num_jobs` ({num_jobs}) exceeds the number of CPU cores ({cpu_count}).", stacklevel=2
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
            warnings.warn(f"Encountered invalid input type {type(arr).__name__}.", stacklevel=2)
        else:
            n_bytes = get_memory_size(arr)
            report[name] = n_bytes
    report["_total_"] = sum(report.values())
    if readable:
        report = {name: format_memory_size(size) for name, size in report.items()}
    return report


def normalize(embeds: torch.Tensor | np.ndarray, dim: int = 1) -> torch.Tensor | np.ndarray:
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
def truncate_dims(embeds: torch.Tensor | np.ndarray, dim: int) -> torch.Tensor | np.ndarray:
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


def move_or_convert_tensors(data: dict, return_tensors: str = "pt", move_to_cpu: bool = False):
    """Move or convert the results to the specified format.

    Parameters
    ----------
    data : dict
        Dictionary containing the tensors or arrays to move or convert.
    return_tensors : str, optional
        Format to return the tensors in, either 'pt' for PyTorch tensors or 'np' for NumPy arrays, by default "pt".
    move_to_cpu : bool, optional
        Whether to move the data to CPU, by default False.

    Returns
    -------
    dict
        Dictionary containing the moved or converted results.

    Raises
    ------
    ValueError
        If `return_tensors` is not 'np' or 'pt'.
    """
    if move_to_cpu or return_tensors == "np":
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.cpu()
            if isinstance(value, list) and len(value) and isinstance(value[0], torch.Tensor):
                data[key] = [v.cpu() for v in value]
    if return_tensors == "np":
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.numpy()
            elif isinstance(value, list) and len(value) and isinstance(value[0], torch.Tensor):
                data[key] = [v.numpy() for v in value]
    elif return_tensors == "pt":
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = torch.from_numpy(value)
            elif isinstance(value, list) and len(value) and isinstance(value[0], np.ndarray):
                data[key] = [torch.from_numpy(v) for v in value]
    else:
        raise ValueError("`return_tensors` must be 'np' or 'pt'.")
    return data


def _build_results_dataframe(
    results: dict,
    return_frame: str = "polars",
    convert_to_numpy: bool = True,
    debug: bool = False,
) -> tuple[pl.DataFrame | pa.Table, np.ndarray | torch.Tensor]:
    """
    Consolidates the results by combining the segment indices and segments into a DataFrame
    and returning it alongside the embeddings.

    Parameters
    ----------
    results : dict
        A dictionary containing the results with keys such as 'document_idx',
        'sequence_idx', 'batch_idx', 'segment_size', 'segment', and 'segment_embeds'.
    return_frame : str, optional
        The type of DataFrame to return. Options are 'polars', 'pandas', or 'arrow'.
        Defaults to 'polars'.
    convert_to_numpy : bool, optional
        Whether to convert the embeddings to NumPy arrays. Defaults to True.
    debug : bool, optional
        Include additional columns for debugging. When False, only essential columns
        are returned: document_idx, segment_idx, segment_size, segment. When True,
        additional columns are included: sequence_idx, batch_idx. Defaults to False.

    Returns
    -------
    tuple
        A tuple with two elements:
        - A DataFrame containing the segment indices and segments.
        - The segment embeddings in the specified format.

    Raises
    ------
    TypeError
        If 'segment_embeds' in results is not a torch.Tensor.
    """
    # Get indices and 1d-arrays
    df = {}
    # embed_idx and sequence_idx are always needed internally for reordering (dropped later by caller)
    # Base columns (segment comes after debug columns when debug=True)
    base_keys = [
        "embed_idx",
        "sequence_idx",
        "document_idx",
        "segment_idx",
        "segment_size",
    ]
    # Build key list with proper ordering (batch_idx before segment when debug=True)
    if debug:
        keys = base_keys + ["batch_idx", "segment"]
    else:
        keys = base_keys + ["segment"]
    for key in keys:
        if key in results:
            df[key] = results[key]
    # Convert 1d arrays to NumPy and move to CPU
    df = move_or_convert_tensors(df, return_tensors="np", move_to_cpu=True)
    # Convert embeddings if needed
    embeds = results["segment_embeds"]
    if not isinstance(embeds, torch.Tensor):
        raise TypeError("Segment embeddings must be torch.Tensor.")
    if convert_to_numpy:
        embeds = embeds.cpu().numpy()
    else:
        embeds = embeds.cpu()
    # Build DataFrame
    if return_frame == "polars":
        df = pl.DataFrame(df)
    elif return_frame == "pandas":
        if _HAS_PANDAS:
            import pandas as pd

            df = pd.DataFrame(df)
        else:
            raise ImportError("Pandas is not installed.")
    elif return_frame == "arrow":
        df = pa.Table.from_pydict(df)
    else:
        raise ValueError(f"Invalid value for return_frame: {return_frame}")
    return df, embeds


def order_by_indices(elements: list, indices: list[int]) -> list:
    """Order a Python list by the given indices."""
    if len(elements) != len(indices):
        raise ValueError(
            f"List and indices must have the same length, but got {len(elements)}, {len(indices)}."
        )
    if len(elements) == 0:
        ordered_elements = elements
    elif len(elements) == 1:
        # itemgetter with single index returns the element, not a tuple
        ordered_elements = [elements[indices[0]]]
    else:
        ordered_elements = list(itemgetter(*indices)(elements))
    return ordered_elements


def round_up_to_power_of_2(x: int) -> int:
    """Round up to the nearest power of 2."""
    return 2 ** math.ceil(math.log2(x))
