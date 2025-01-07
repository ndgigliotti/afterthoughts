import time
import warnings
from functools import singledispatch
from typing import Callable

import numpy as np
import pyarrow as pa
import torch
import torch.nn.functional as F


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


def get_memory_size(a: pa.Array | torch.Tensor | np.ndarray) -> int:
    """Get the size of the array in bytes.

    Parameters
    ----------
    a : pa.Array or torch.Tensor or np.ndarray
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
    results: dict[str, pa.Array | torch.Tensor | np.ndarray]
) -> dict[str, str]:
    """Get the size of the arrays in .

    Parameters
    ----------
    results : dict[str, pa.Array or torch.Tensor]
        Dictionary of arrays.

    Returns
    -------
    dict[str, str]
        Dictionary of array sizes in human-readable format.
    """
    report = {}
    for name, arr in results.items():
        if not isinstance(arr, (pa.Array, torch.Tensor, np.ndarray)):
            warnings.warn(f"Encountered invalid input type {type(arr).__name__}.")
        else:
            n_bytes = get_memory_size(arr)
            report[name] = n_bytes
    report["total"] = sum(report.values())
    return {name: format_memory_size(size) for name, size in report.items()}


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
