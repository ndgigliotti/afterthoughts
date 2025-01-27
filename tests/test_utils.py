import time

import numpy as np
import pyarrow as pa
import pytest
import torch

from finephrase.utils import (
    format_memory_size,
    get_memory_report,
    get_memory_size,
    get_torch_dtype,
    normalize,
    reduce_precision,
    timer,
    truncate_dims,
)


def test_timer_decorator():
    @timer()
    def dummy_function():
        time.sleep(0.1)
        return "done"

    result = dummy_function()
    assert result == "done"


def test_get_memory_size():
    pa_array = pa.array([1, 2, 3])
    np_array = np.array([1, 2, 3])
    torch_tensor = torch.tensor([1, 2, 3])

    assert get_memory_size(pa_array) == sum(
        buf.size for buf in pa_array.buffers() if buf is not None
    )
    assert get_memory_size(np_array) == np_array.nbytes
    assert (
        get_memory_size(torch_tensor)
        == torch_tensor.element_size() * torch_tensor.numel()
    )


def test_format_memory_size():
    assert format_memory_size(1023) == "1023.00 B"
    assert format_memory_size(1024) == "1.00 KB"
    assert format_memory_size(1048576) == "1.00 MB"


def test_get_memory_report():
    pa_array = pa.array([1, 2, 3])
    np_array = np.array([1, 2, 3])
    torch_tensor = torch.tensor([1, 2, 3])

    results = {"pa_array": pa_array, "np_array": np_array, "torch_tensor": torch_tensor}

    report = get_memory_report(results)
    assert "pa_array" in report
    assert "np_array" in report
    assert "torch_tensor" in report
    assert "_total_" in report


def test_normalize_numpy():
    np_array = np.array([[1, 2, 3], [4, 5, 6]])
    np_normalized = normalize(np_array)
    assert np.allclose(np.linalg.norm(np_normalized, axis=1), 1)


def test_normalize_torch():
    torch_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    torch_normalized = normalize(torch_tensor)
    assert torch.allclose(torch.norm(torch_normalized, dim=1), torch.tensor([1.0, 1.0]))


def test_get_torch_dtype():
    assert get_torch_dtype("float32") == torch.float32
    assert get_torch_dtype("int64") == torch.int64
    with pytest.raises(ValueError):
        get_torch_dtype("invalid_dtype")


def test_reduce_precision_numpy():
    np_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np_reduced = reduce_precision(np_array)
    assert np_reduced.dtype == np.float16


def test_reduce_precision_torch():
    torch_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    torch_reduced = reduce_precision(torch_tensor)
    assert torch_reduced.dtype == torch.float16


def test_truncate_dims_numpy_1_axis():
    np_array = np.random.rand(4)
    np_truncated = truncate_dims(np_array, 2)
    assert np_truncated.shape == (2,)


def test_truncate_dims_torch_1_axis():
    torch_tensor = torch.rand(4)
    torch_truncated = truncate_dims(torch_tensor, 2)
    assert torch_truncated.shape == (2,)


def test_truncate_dims_numpy_1_axis():
    np_array = np.random.rand(4)
    np_truncated = truncate_dims(np_array, 2)
    assert np_truncated.shape == (2,)


def test_truncate_dims_torch_1_axis():
    torch_tensor = torch.rand(4)
    torch_truncated = truncate_dims(torch_tensor, 2)
    assert torch_truncated.shape == (2,)


def test_truncate_dims_numpy_2_axes():
    np_array = np.random.rand(3, 4)
    np_truncated = truncate_dims(np_array, 2)
    assert np_truncated.shape == (3, 2)


def test_truncate_dims_torch_2_axes():
    torch_tensor = torch.rand(3, 4)
    torch_truncated = truncate_dims(torch_tensor, 2)
    assert torch_truncated.shape == (3, 2)


def test_truncate_dims_numpy_3_axes():
    np_array = np.random.rand(2, 3, 4)
    np_truncated = truncate_dims(np_array, 2)
    assert np_truncated.shape == (2, 3, 2)


def test_truncate_dims_torch_3_axes():
    torch_tensor = torch.rand(2, 3, 4)
    torch_truncated = truncate_dims(torch_tensor, 2)
    assert torch_truncated.shape == (2, 3, 2)
