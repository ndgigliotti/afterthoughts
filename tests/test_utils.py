import os
import time

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
import torch

from finephrase.utils import (
    format_memory_size,
    get_memory_report,
    get_memory_size,
    get_torch_dtype,
    normalize,
    normalize_num_jobs,
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
    # Test with supported types
    pa_array = pa.array([1, 2, 3])
    np_array = np.array([1, 2, 3])
    torch_tensor = torch.tensor([1, 2, 3])
    pl_series = pl.Series("a", [1, 2, 3])
    pl_dataframe = pl.DataFrame({"a": [1, 2, 3]})

    assert get_memory_size(pa_array) == sum(
        buf.size for buf in pa_array.buffers() if buf is not None
    )
    assert get_memory_size(np_array) == np_array.nbytes
    assert get_memory_size(torch_tensor) == torch_tensor.element_size() * torch_tensor.numel()
    assert get_memory_size(pl_series) == pl_series.estimated_size()
    assert get_memory_size(pl_dataframe) == pl_dataframe.estimated_size()

    # Test with Pandas
    pd_series = pd.Series([1, 2, 3])
    pd_dataframe = pd.DataFrame({"a": [1, 2, 3]})
    assert get_memory_size(pd_series) == pd_series.memory_usage(index=True, deep=True)
    assert get_memory_size(pd_dataframe) == pd_dataframe.memory_usage(index=True, deep=True).sum()


def test_format_memory_size():
    assert format_memory_size(1023) == "1023.00 B"
    assert format_memory_size(1024) == "1.00 KB"
    assert format_memory_size(1048576) == "1.00 MB"


def test_get_memory_report():
    results = {
        "pa_array": pa.array([1, 2, 3]),
        "np_array": np.array([1, 2, 3]),
        "torch_tensor": torch.tensor([1, 2, 3]),
        "pl_series": pl.Series("a", [1, 2, 3]),
        "pl_dataframe": pl.DataFrame({"a": [1, 2, 3]}),
    }
    results.update(
        {
            "pd_series": pd.Series([1, 2, 3]),
            "pd_dataframe": pd.DataFrame({"a": [1, 2, 3]}),
        }
    )
    report = get_memory_report(results)
    assert "pa_array" in report
    assert "np_array" in report
    assert "torch_tensor" in report
    assert "pl_series" in report
    assert "pl_dataframe" in report
    assert "_total_" in report
    assert "pd_series" in report
    assert "pd_dataframe" in report


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


def test_norm_jobs():
    cpu_count = os.cpu_count()

    # Test with None
    assert normalize_num_jobs(None) == 1

    # Test with 0
    with pytest.warns(UserWarning, match="`num_jobs` cannot be 0."):
        assert normalize_num_jobs(0) == 1

    # Test with positive number less than CPU count
    assert normalize_num_jobs(2) == 2

    # Test with positive number greater than CPU count
    with pytest.warns(
        UserWarning,
        match=f"`num_jobs` \\(.*\\) exceeds the number of CPU cores \\({cpu_count}\\).",
    ):
        assert normalize_num_jobs(cpu_count + 2) == cpu_count

    # Test with negative number
    assert normalize_num_jobs(-1) == cpu_count

    # Test with a smaller negative number
    assert normalize_num_jobs(-3) == cpu_count - 3 + 1
