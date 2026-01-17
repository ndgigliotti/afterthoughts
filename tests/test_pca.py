import numpy as np
import pytest
import torch

from afterthoughts.pca import IncrementalPCA, _add_to_diagonal, gen_batches


@pytest.mark.parametrize(
    "n_components, whiten",
    [
        (2, False),
        (2, True),
        (8, False),
        (8, True),
        (16, False),
        (16, True),
    ],
)
def test_equivalence_with_sklearn(n_components, whiten):
    skIncrementalPCA = pytest.importorskip("sklearn.decomposition").IncrementalPCA
    # Generate random data and split into chunks
    rng = np.random.default_rng(42)
    X = rng.standard_normal((1000, 100), dtype=np.float64)
    chunks = np.array_split(X, 10)

    # Initialize IncrementalPCA for both custom and sklearn implementations
    ipca = IncrementalPCA(n_components=n_components, whiten=whiten, device="cpu")
    sk_ipca = skIncrementalPCA(n_components=n_components, whiten=whiten)

    # Perform partial fit on each chunk
    for chunk in chunks:
        ipca.partial_fit(torch.tensor(chunk))
        sk_ipca.partial_fit(chunk)

    # Transform the data using both implementations
    X_pca = ipca.transform(torch.tensor(X))
    X_pca_sk = sk_ipca.transform(X)

    # Compare transformed data
    assert np.allclose(X_pca, X_pca_sk, atol=1e-6)

    # Compare attributes
    attributes = [
        "components_",
        "explained_variance_",
        "explained_variance_ratio_",
        "singular_values_",
        "mean_",
        "noise_variance_",
    ]
    for attr in attributes:
        ipca_attr = getattr(ipca, attr).numpy()
        sk_ipca_attr = getattr(sk_ipca, attr)
        assert np.allclose(ipca_attr, sk_ipca_attr, atol=1e-6)


@pytest.mark.parametrize(
    "n_components, whiten",
    [
        (2, False),
        (2, True),
        (8, False),
        (8, True),
        (16, False),
        (16, True),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_equivalence_with_sklearn_gpu(n_components, whiten):
    skIncrementalPCA = pytest.importorskip("sklearn.decomposition").IncrementalPCA
    # Generate random data and split into chunks
    rng = np.random.default_rng(42)
    X = rng.standard_normal((1000, 100), dtype=np.float64)
    chunks = np.array_split(X, 10)

    # Initialize IncrementalPCA for both custom and sklearn implementations
    ipca = IncrementalPCA(n_components=n_components, whiten=whiten, device="cuda")
    sk_ipca = skIncrementalPCA(n_components=n_components, whiten=whiten)

    # Perform partial fit on each chunk
    for chunk in chunks:
        ipca.partial_fit(torch.tensor(chunk, device="cuda"))
        sk_ipca.partial_fit(chunk)

    # Transform the data using both implementations
    X_pca = ipca.transform(torch.tensor(X, device="cuda"))
    X_pca_sk = sk_ipca.transform(X)

    # Compare transformed data
    assert np.allclose(X_pca.cpu(), X_pca_sk, atol=1e-6)

    # Compare attributes
    attributes = [
        "components_",
        "explained_variance_",
        "explained_variance_ratio_",
        "singular_values_",
        "mean_",
        "noise_variance_",
    ]
    for attr in attributes:
        ipca_attr = getattr(ipca, attr).cpu().numpy()
        sk_ipca_attr = getattr(sk_ipca, attr)
        assert np.allclose(ipca_attr, sk_ipca_attr, atol=1e-6)


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
    assert ipca.device.type == "cuda"
    assert ipca.components_.device.type == "cuda"
    assert ipca.mean_.device.type == "cuda"
    assert ipca.singular_values_.device.type == "cuda"
    assert ipca.explained_variance_.device.type == "cuda"
    assert ipca.explained_variance_ratio_.device.type == "cuda"
    assert ipca.var_.device.type == "cuda"
    assert ipca.noise_variance_.device.type == "cuda"


def test_incremental_pca_save_load(tmp_path):
    X = torch.randn(100, 20)
    ipca = IncrementalPCA(n_components=5, device="cpu")
    ipca.fit(X)
    X_transformed_original = ipca.transform(X)

    # Save and load
    save_path = tmp_path / "pca.pt"
    ipca.save(str(save_path))
    ipca_loaded = IncrementalPCA.load(str(save_path), device="cpu")

    # Verify config
    assert ipca_loaded.n_components == ipca.n_components
    assert ipca_loaded.whiten == ipca.whiten

    # Verify fitted attributes
    assert torch.allclose(ipca_loaded.components_, ipca.components_)
    assert torch.allclose(ipca_loaded.mean_, ipca.mean_)
    assert torch.allclose(ipca_loaded.explained_variance_, ipca.explained_variance_)

    # Verify transform produces same results
    X_transformed_loaded = ipca_loaded.transform(X)
    assert torch.allclose(X_transformed_original, X_transformed_loaded)


def test_incremental_pca_save_unfitted_raises():
    ipca = IncrementalPCA(n_components=5, device="cpu")
    with pytest.raises(ValueError, match="Cannot save unfitted"):
        ipca.save("test.pt")


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
