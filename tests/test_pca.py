import numpy as np
import pytest
import torch

from finephrase.pca import IncrementalPCA, _add_to_diagonal, gen_batches


def test_equivalence_with_sklearn():
    try:
        from sklearn.decomposition import IncrementalPCA as skIncrementalPCA
    except ImportError:
        pytest.skip("scikit-learn is not installed")
    rng = np.random.default_rng(42)
    X = rng.standard_normal((1000, 100), dtype=np.float64)
    chunks = np.array_split(X, 10)
    ipca = IncrementalPCA(n_components=5, whiten=False, device="cpu")
    sk_ipca = skIncrementalPCA(n_components=5, whiten=False)
    for chunk in chunks:
        ipca.partial_fit(torch.tensor(chunk))
        sk_ipca.partial_fit(chunk)
        X_pca = ipca.transform(torch.tensor(X)).numpy()
    X_pca_sk = sk_ipca.transform(X)
    assert np.allclose(X_pca, X_pca_sk)
    assert np.allclose(ipca.components_.numpy(), sk_ipca.components_)
    assert np.allclose(ipca.explained_variance_.numpy(), sk_ipca.explained_variance_)
    assert np.allclose(
        ipca.explained_variance_ratio_.numpy(), sk_ipca.explained_variance_ratio_
    )
    assert np.allclose(ipca.singular_values_.numpy(), sk_ipca.singular_values_)
    assert np.allclose(ipca.mean_.numpy(), sk_ipca.mean_)
    assert np.allclose(ipca.noise_variance_.numpy(), sk_ipca.noise_variance_)


def test_incremental_pca_fit_transform():
    X = torch.randn(100, 20)
    ipca = IncrementalPCA(n_components=5, device="cpu")
    X_transformed = ipca.fit_transform(X)
    assert X_transformed.shape == (100, 5)
    assert hasattr(ipca, "components_")
    assert hasattr(ipca, "explained_variance_")
    assert hasattr(ipca, "explained_variance_ratio_")


def test_incremental_pca_partial_fit():
    X = torch.randn(100, 20)
    ipca = IncrementalPCA(n_components=5, device="cpu")
    ipca.partial_fit(X[:50])
    ipca.partial_fit(X[50:])
    assert hasattr(ipca, "components_")
    assert hasattr(ipca, "explained_variance_")
    assert hasattr(ipca, "explained_variance_ratio_")


def test_incremental_pca_transform():
    X = torch.randn(100, 20)
    ipca = IncrementalPCA(n_components=5, device="cpu")
    ipca.fit(X)
    X_transformed = ipca.transform(X)
    assert X_transformed.shape == (100, 5)


def test_incremental_pca_inverse_transform():
    X = torch.randn(100, 20)
    ipca = IncrementalPCA(n_components=5, device="cpu")
    ipca.fit(X)
    X_transformed = ipca.transform(X)
    X_inverse = ipca.inverse_transform(X_transformed)
    assert X_inverse.shape == (100, 20)


def test_incremental_pca_get_covariance():
    X = torch.randn(100, 20)
    ipca = IncrementalPCA(n_components=5, device="cpu")
    ipca.fit(X)
    covariance = ipca.get_covariance()
    assert covariance.shape == (20, 20)


def test_incremental_pca_get_precision():
    X = torch.randn(100, 20)
    ipca = IncrementalPCA(n_components=5, device="cpu")
    ipca.fit(X)
    precision = ipca.get_precision()
    assert precision.shape == (20, 20)


def test_incremental_pca_whiten():
    X = torch.randn(100, 20)
    ipca = IncrementalPCA(n_components=5, whiten=True, device="cpu")
    X_transformed = ipca.fit_transform(X)
    assert X_transformed.shape == (100, 5)
    assert hasattr(ipca, "components_")
    assert hasattr(ipca, "explained_variance_")
    assert hasattr(ipca, "explained_variance_ratio_")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_incremental_pca_to_device():
    X = torch.randn(100, 20)
    ipca = IncrementalPCA(n_components=5, device="cpu")
    ipca.fit(X)
    ipca.to("cuda")
    assert ipca.device == "cuda"
    assert ipca.components_.device.type == "cuda"
    assert ipca.mean_.device.type == "cuda"
    assert ipca.singular_values_.device.type == "cuda"
    assert ipca.explained_variance_.device.type == "cuda"
    assert ipca.explained_variance_ratio_.device.type == "cuda"
    assert ipca.var_.device.type == "cuda"
    assert ipca.noise_variance_.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_incremental_pca_fit_transform_gpu():
    X = torch.randn(100, 20, device="cuda")
    ipca = IncrementalPCA(n_components=5, device="cuda")
    X_transformed = ipca.fit_transform(X)
    assert X_transformed.shape == (100, 5)
    assert hasattr(ipca, "components_")
    assert hasattr(ipca, "explained_variance_")
    assert hasattr(ipca, "explained_variance_ratio_")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_incremental_pca_partial_fit_gpu():
    X = torch.randn(100, 20, device="cuda")
    ipca = IncrementalPCA(n_components=5, device="cuda")
    ipca.partial_fit(X[:50])
    ipca.partial_fit(X[50:])
    assert hasattr(ipca, "components_")
    assert hasattr(ipca, "explained_variance_")
    assert hasattr(ipca, "explained_variance_ratio_")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_incremental_pca_transform_gpu():
    X = torch.randn(100, 20, device="cuda")
    ipca = IncrementalPCA(n_components=5, device="cuda")
    ipca.fit(X)
    X_transformed = ipca.transform(X)
    assert X_transformed.shape == (100, 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_incremental_pca_inverse_transform_gpu():
    X = torch.randn(100, 20, device="cuda")
    ipca = IncrementalPCA(n_components=5, device="cuda")
    ipca.fit(X)
    X_transformed = ipca.transform(X)
    X_inverse = ipca.inverse_transform(X_transformed)
    assert X_inverse.shape == (100, 20)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_incremental_pca_get_covariance_gpu():
    X = torch.randn(100, 20, device="cuda")
    ipca = IncrementalPCA(n_components=5, device="cuda")
    ipca.fit(X)
    covariance = ipca.get_covariance()
    assert covariance.shape == (20, 20)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_incremental_pca_get_precision_gpu():
    X = torch.randn(100, 20, device="cuda")
    ipca = IncrementalPCA(n_components=5, device="cuda")
    ipca.fit(X)
    precision = ipca.get_precision()
    assert precision.shape == (20, 20)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_incremental_pca_whiten_gpu():
    X = torch.randn(100, 20, device="cuda")
    ipca = IncrementalPCA(n_components=5, whiten=True, device="cuda")
    X_transformed = ipca.fit_transform(X)
    assert X_transformed.shape == (100, 5)
    assert hasattr(ipca, "components_")
    assert hasattr(ipca, "explained_variance_")
    assert hasattr(ipca, "explained_variance_ratio_")


def test_gen_batches():
    # Test case 1: n is divisible by batch_size
    batches = list(gen_batches(10, 2))
    assert batches == [slice(0, 2), slice(2, 4), slice(4, 6), slice(6, 8), slice(8, 10)]

    # Test case 2: n is not divisible by batch_size
    batches = list(gen_batches(7, 3))
    assert batches == [slice(0, 3), slice(3, 6), slice(6, 7)]

    # Test case 3: n is less than batch_size
    batches = list(gen_batches(5, 10))
    assert batches == [slice(0, 5)]


def test_add_to_diagonal():
    # Test case 1: Adding a positive value to the diagonal
    array = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    value = 5.0
    result = _add_to_diagonal(array, value)
    expected = torch.tensor([[6.0, 2.0], [3.0, 9.0]])
    assert torch.allclose(result, expected)

    # Test case 2: Adding a negative value to the diagonal
    array = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    value = -1.0
    result = _add_to_diagonal(array, value)
    expected = torch.tensor([[0.0, 2.0], [3.0, 3.0]])
    assert torch.allclose(result, expected)

    # Test case 3: Adding zero to the diagonal
    array = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    value = 0.0
    result = _add_to_diagonal(array, value)
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.allclose(result, expected)

    # Test case 4: Adding a value to a larger matrix
    array = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    value = 2.0
    result = _add_to_diagonal(array, value)
    expected = torch.tensor([[3.0, 2.0, 3.0], [4.0, 7.0, 6.0], [7.0, 8.0, 11.0]])
    assert torch.allclose(result, expected)
