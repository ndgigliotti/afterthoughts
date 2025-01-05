import numpy as np
import torch
import torch.nn.functional as F


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
    """
    if isinstance(embeds, np.ndarray):
        norms = np.linalg.norm(embeds, axis=dim, keepdims=True)
        norms[norms == 0] = 1e-12
        norm_embeds = embeds / norms
    elif isinstance(embeds, torch.Tensor):
        norm_embeds = F.normalize(embeds, p=2, dim=dim, eps=1e-12)
    return norm_embeds


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


def reduce_precision(embeds: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Reduce the precision of the embeddings to float16.

    Checks whether the input embeddings are of type float32 or float64 and
    converts them to float16. If the input is of a dtype other than float32
    or float64, it will be returned as is.

    Parameters
    ----------
    embeds : torch.Tensor or np.ndarray
        Embeddings.

    Returns
    -------
    torch.Tensor or np.ndarray
        Embeddings with reduced precision.
    """
    if isinstance(embeds, np.ndarray):
        if embeds.dtype in (np.float32, np.float64):
            embeds = embeds.astype("float16")
    elif isinstance(embeds, torch.Tensor):
        if embeds.dtype in (torch.float32, torch.float64):
            embeds = embeds.to(torch.float16)
    else:
        raise ValueError("Invalid input type.")
    return embeds
