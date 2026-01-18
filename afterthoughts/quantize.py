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

"""Quantization utilities for embedding compression."""

from abc import abstractmethod
from functools import singledispatch
from typing import ClassVar

import numpy as np
import torch


class QuantizedEmbeds(np.ndarray):
    """Abstract base class for quantized embeddings.

    Provides common infrastructure for numpy array subclasses that store
    compressed embeddings and auto-decompress for operations.
    """

    # Functions that require decompression - subclasses can extend
    _HANDLED_FUNCTIONS: ClassVar[set[str]] = {
        "dot",
        "matmul",
        "inner",
        "outer",
        "tensordot",
        "vdot",
        "linalg.norm",
        "linalg.multi_dot",
        "einsum",
        "sum",
        "mean",
        "std",
        "var",
        "min",
        "max",
        "argmin",
        "argmax",
        "sort",
        "argsort",
        "concatenate",
        "stack",
        "vstack",
        "hstack",
    }

    # Functions that should return the same quantized type
    _REQUANTIZE_FUNCTIONS: ClassVar[set[str]] = {
        "concatenate",
        "stack",
        "vstack",
        "hstack",
    }

    @abstractmethod
    def decompress(self) -> np.ndarray:
        """Decompress to full representation.

        Returns
        -------
        np.ndarray
            Decompressed embeddings.
        """
        pass

    @classmethod
    @abstractmethod
    def _compress(cls, data: np.ndarray) -> "QuantizedEmbeds":
        """Compress data back to quantized form.

        Parameters
        ----------
        data : np.ndarray
            Data to compress.

        Returns
        -------
        QuantizedEmbeds
            Compressed embeddings.
        """
        pass

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """Intercept ufuncs to auto-decompress before operations."""
        had_quantized_input = any(isinstance(inp, type(self)) for inp in inputs)

        # Decompress any quantized inputs
        decomp_inputs = []
        for inp in inputs:
            if isinstance(inp, type(self)):
                decomp_inputs.append(inp.decompress())
            else:
                decomp_inputs.append(inp)

        # Handle output arrays
        if out is not None:
            decomp_out = []
            for o in out:
                if isinstance(o, type(self)):
                    decomp_out.append(o.decompress())
                else:
                    decomp_out.append(o)
            kwargs["out"] = tuple(decomp_out)

        result = getattr(ufunc, method)(*decomp_inputs, **kwargs)

        # Requantize if result is 2D and appropriate dtype
        if (
            had_quantized_input
            and isinstance(result, np.ndarray)
            and result.ndim == 2
            and self._should_requantize_result(result)
        ):
            return self._compress(result)

        return result

    def __array_function__(self, func, types, args, kwargs):
        """Intercept array functions to auto-decompress."""
        func_name = func.__name__
        module = getattr(func, "__module__", "")

        if "linalg" in module:
            func_name = f"linalg.{func_name}"

        if func_name not in self._HANDLED_FUNCTIONS:
            return NotImplemented

        should_requantize = func_name in self._REQUANTIZE_FUNCTIONS

        # Decompress any quantized inputs in args
        decomp_args = []
        for arg in args:
            if isinstance(arg, type(self)):
                decomp_args.append(arg.decompress())
            elif isinstance(arg, (list, tuple)):
                decomp_arg = []
                for item in arg:
                    if isinstance(item, type(self)):
                        decomp_arg.append(item.decompress())
                    else:
                        decomp_arg.append(item)
                decomp_args.append(type(arg)(decomp_arg))
            else:
                decomp_args.append(arg)

        # Decompress any quantized inputs in kwargs
        decomp_kwargs = {}
        for key, val in kwargs.items():
            if isinstance(val, type(self)):
                decomp_kwargs[key] = val.decompress()
            else:
                decomp_kwargs[key] = val

        result = func(*decomp_args, **decomp_kwargs)

        if (
            should_requantize
            and isinstance(result, np.ndarray)
            and result.ndim == 2
            and self._should_requantize_result(result)
        ):
            return self._compress(result)

        return result

    def _should_requantize_result(self, result: np.ndarray) -> bool:
        """Check if result should be requantized. Override in subclasses."""
        return True


class BinaryEmbeds(QuantizedEmbeds):
    """NumPy array subclass for packed binary embeddings.

    This class stores packed binary embeddings (uint8, shape (n, ceil(d/8)))
    and automatically unpacks when operations are performed. Provides efficient
    hamming_distance computation directly on packed data.

    Parameters
    ----------
    packed : np.ndarray
        Packed binary embeddings with dtype uint8 and shape (n, ceil(d/8)).
    orig_dim : int
        Original embedding dimension before packing.

    Examples
    --------
    >>> embeds = BinaryEmbeds(packed, orig_dim=768)
    >>> embeds.shape
    (1000, 96)  # 768 / 8 = 96 bytes per row
    >>> embeds.orig_dim
    768
    >>> embeds.unpack()  # Returns (1000, 768) binary matrix
    >>> embeds.hamming_distance(query)  # Efficient on packed data
    >>> embeds @ weights  # Auto-unpacks for matmul
    """

    def __new__(cls, packed: np.ndarray, orig_dim: int) -> "BinaryEmbeds":
        """Create BinaryEmbeds from packed data."""
        obj = np.asarray(packed).view(cls)
        obj._orig_dim = orig_dim
        return obj

    def __array_finalize__(self, obj) -> None:
        """Finalize array creation, preserving original dimension."""
        if obj is None:
            return
        self._orig_dim = getattr(obj, "_orig_dim", None)

    def __reduce__(self):
        """Support pickling by storing original dimension."""
        pickled_state = super().__reduce__()
        new_state = pickled_state[2] + (self._orig_dim,)
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        """Restore from pickle, including original dimension."""
        self._orig_dim = state[-1]
        super().__setstate__(state[:-1])

    def __getitem__(self, idx) -> "BinaryEmbeds":
        """Index or slice the embeddings, preserving metadata."""
        result = np.ndarray.__getitem__(self, idx)

        if not isinstance(result, np.ndarray):
            return result

        # Handle single row indexing
        if result.ndim == 1 and self.ndim == 2:
            result = result[np.newaxis, :]

        obj = result.view(BinaryEmbeds)
        obj._orig_dim = self._orig_dim
        return obj

    def decompress(self) -> np.ndarray:
        """Decompress to binary matrix (alias for unpack)."""
        return self.unpack()

    def unpack(self) -> np.ndarray:
        """Unpack to binary matrix.

        Returns
        -------
        np.ndarray
            Binary matrix with shape (n, orig_dim) and dtype uint8.
        """
        data = self.view(np.ndarray)
        unpacked = np.unpackbits(data, axis=-1)
        # Trim to original dimension (unpackbits pads to multiple of 8)
        return unpacked[:, : self._orig_dim]

    @classmethod
    def _compress(cls, data: np.ndarray) -> "BinaryEmbeds":
        """Compress binary data to BinaryEmbeds."""
        orig_dim = data.shape[1]
        packed = np.packbits(data.astype(np.uint8), axis=-1)
        return cls(packed, orig_dim)

    def _should_requantize_result(self, result: np.ndarray) -> bool:
        """Only requantize integer/binary results."""
        return np.issubdtype(result.dtype, np.integer)

    def hamming_distance(self, other: "BinaryEmbeds | np.ndarray") -> np.ndarray:
        """Compute hamming distance efficiently on packed data.

        Uses XOR + popcount directly on packed bits, avoiding unpacking.

        Parameters
        ----------
        other : BinaryEmbeds or np.ndarray
            Other packed binary embeddings. If 1D, treated as single query.
            Shape should be (m, packed_dim) or (packed_dim,).

        Returns
        -------
        np.ndarray
            Hamming distances with shape (n, m) or (n,) for single query.
        """
        self_data = self.view(np.ndarray)

        if isinstance(other, BinaryEmbeds):
            other_data = other.view(np.ndarray)
        else:
            other_data = np.asarray(other)

        # Handle single query (1D)
        single_query = other_data.ndim == 1
        if single_query:
            other_data = other_data[np.newaxis, :]

        # XOR and count bits: (n, 1, packed_dim) XOR (1, m, packed_dim)
        xor_result = self_data[:, np.newaxis, :] ^ other_data[np.newaxis, :, :]

        # Count set bits using lookup table (faster than unpackbits)
        popcount_table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
        distances = popcount_table[xor_result].sum(axis=-1)

        if single_query:
            return distances.squeeze(axis=1)
        return distances

    def hamming_similarity(self, other: "BinaryEmbeds | np.ndarray") -> np.ndarray:
        """Compute hamming similarity (1 - normalized hamming distance).

        Parameters
        ----------
        other : BinaryEmbeds or np.ndarray
            Other packed binary embeddings.

        Returns
        -------
        np.ndarray
            Hamming similarity scores in [0, 1].
        """
        distances = self.hamming_distance(other)
        return 1.0 - distances / self._orig_dim

    @property
    def orig_dim(self) -> int:
        """Original embedding dimension before packing."""
        return self._orig_dim

    @property
    def packed(self) -> np.ndarray:
        """The underlying packed binary data."""
        return self.view(np.ndarray)

    def __repr__(self) -> str:
        return f"BinaryEmbeds(shape={self.shape}, orig_dim={self._orig_dim})"


@singledispatch
def binary_quantize(embeds: torch.Tensor | np.ndarray) -> BinaryEmbeds:
    """Convert embeddings to packed binary representation.

    Values > 0 become 1, values <= 0 become 0. The result is packed into
    bits using np.packbits, providing 32x compression vs float32.

    Parameters
    ----------
    embeds : torch.Tensor or np.ndarray
        Embeddings to quantize. Shape (n, d) where d is the embedding dimension.

    Returns
    -------
    BinaryEmbeds
        Packed binary embeddings with hamming_distance support.

    Raises
    ------
    TypeError
        If the input is not a Torch tensor or NumPy array.
    """
    if not isinstance(embeds, (torch.Tensor, np.ndarray)):
        raise TypeError(f"Invalid input type {type(embeds).__name__}.")


@binary_quantize.register
def _(embeds: torch.Tensor) -> BinaryEmbeds:
    """Sub-function for binary quantization of Torch tensors."""
    orig_dim = embeds.shape[1]
    binary = (embeds > 0).cpu().numpy().astype(np.uint8)
    packed = np.packbits(binary, axis=-1)
    return BinaryEmbeds(packed, orig_dim)


@binary_quantize.register
def _(embeds: np.ndarray) -> BinaryEmbeds:
    """Sub-function for binary quantization of NumPy arrays."""
    orig_dim = embeds.shape[1]
    binary = (embeds > 0).astype(np.uint8)
    packed = np.packbits(binary, axis=-1)
    return BinaryEmbeds(packed, orig_dim)
