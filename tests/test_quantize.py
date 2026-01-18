import pickle

import numpy as np
import pytest
import torch

from afterthoughts.quantize import (
    BinaryEmbeds,
    QuantizedEmbeds,
    binary_quantize,
)

# =============================================================================
# QuantizedEmbeds ABC Tests
# =============================================================================


def test_quantized_embeds_has_abstract_methods():
    """Test that QuantizedEmbeds has abstract methods that must be implemented."""
    import inspect

    assert hasattr(QuantizedEmbeds, "decompress")
    assert hasattr(QuantizedEmbeds, "_compress")

    # Verify subclasses implement the abstract methods
    assert not inspect.isabstract(BinaryEmbeds)


# =============================================================================
# BinaryEmbeds Tests
# =============================================================================


class TestBinaryQuantize:
    """Tests for binary_quantize function."""

    def test_numpy_input(self):
        """Test binary quantization with numpy arrays."""
        embeds = np.array([[0.5, -0.3, 0.1, -0.8]], dtype=np.float32)
        result = binary_quantize(embeds)

        assert isinstance(result, BinaryEmbeds)
        assert result.dtype == np.uint8
        assert result.orig_dim == 4

    def test_torch_input(self):
        """Test binary quantization with torch tensors."""
        embeds = torch.tensor(
            [
                [
                    -0.5,
                    0.3,
                    -0.1,
                    0.8,
                    -0.2,
                    0.1,
                    -0.3,
                    0.9,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    -0.1,
                    -0.2,
                    -0.3,
                    -0.4,
                ]
            ],
            dtype=torch.float32,
        )
        result = binary_quantize(embeds)

        assert isinstance(result, BinaryEmbeds)
        assert result.dtype == np.uint8
        assert result.shape == (1, 2)  # 16 dims -> 2 bytes
        assert result.orig_dim == 16

    def test_packing_correctness(self):
        """Test that bit packing is correct."""
        # 16 dimensions -> 2 bytes per row
        # Values: [0,1,0,1,0,1,0,1, 1,1,1,1,0,0,0,0] -> [0b01010101, 0b11110000]
        embeds = torch.tensor(
            [
                [
                    -0.5,
                    0.3,
                    -0.1,
                    0.8,
                    -0.2,
                    0.1,
                    -0.3,
                    0.9,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    -0.1,
                    -0.2,
                    -0.3,
                    -0.4,
                ]
            ],
            dtype=torch.float32,
        )
        packed = binary_quantize(embeds)

        assert packed[0, 0] == 0b01010101
        assert packed[0, 1] == 0b11110000

    def test_multiple_rows(self):
        """Test binary quantization with multiple rows."""
        embeds = np.random.randn(5, 16).astype(np.float32)
        result = binary_quantize(embeds)

        assert result.shape == (5, 2)
        assert result.orig_dim == 16

    def test_invalid_input_raises(self):
        """Test that invalid input types raise TypeError."""
        with pytest.raises(TypeError, match="Invalid input type"):
            binary_quantize([1, 2, 3])


class TestBinaryEmbeds:
    """Tests for BinaryEmbeds class."""

    @pytest.fixture
    def sample_embeds(self):
        """Create sample embeddings for testing."""
        return np.array(
            [
                [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],  # binary: 11001100
                [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],  # binary: 10101010
                [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],  # binary: 00001111
            ],
            dtype=np.float32,
        )

    @pytest.fixture
    def packed(self, sample_embeds):
        """Create packed binary embeddings."""
        return binary_quantize(sample_embeds)

    def test_unpack(self):
        """Test unpack() returns correct binary matrix."""
        embeds = np.array([[0.5, -0.3, 0.1, -0.8]], dtype=np.float32)
        packed = binary_quantize(embeds)

        unpacked = packed.unpack()

        assert unpacked.shape == (1, 4)
        assert unpacked.dtype == np.uint8
        np.testing.assert_array_equal(unpacked[0], [1, 0, 1, 0])

    def test_decompress_alias(self, packed):
        """Test decompress() is alias for unpack()."""
        np.testing.assert_array_equal(packed.decompress(), packed.unpack())

    def test_unpack_trims_padding(self):
        """Test unpack() trims to original dimension (not multiple of 8)."""
        # 10 dims -> 2 bytes packed, but unpack should return 10 cols
        embeds = np.random.randn(3, 10).astype(np.float32)
        packed = binary_quantize(embeds)

        unpacked = packed.unpack()

        assert unpacked.shape == (3, 10)

    def test_hamming_distance_self(self, packed):
        """Test hamming distance to self is zero."""
        distances = packed.hamming_distance(packed)

        assert distances.shape == (3, 3)
        assert distances[0, 0] == 0
        assert distances[1, 1] == 0
        assert distances[2, 2] == 0

    def test_hamming_distance_symmetric(self, packed):
        """Test hamming distance is symmetric."""
        distances = packed.hamming_distance(packed)

        np.testing.assert_array_equal(distances, distances.T)

    def test_hamming_distance_correctness(self, packed):
        """Test hamming distance computes correctly."""
        distances = packed.hamming_distance(packed)

        # Row 0: 11001100, Row 1: 10101010
        # XOR: 01100110 -> 4 bits different
        assert distances[0, 1] == 4
        assert distances[1, 0] == 4

    def test_hamming_distance_single_query(self, packed):
        """Test hamming_distance with 1D query returns 1D result."""
        query = packed.packed[0]  # Get underlying 1D packed array

        distances = packed.hamming_distance(query)

        assert distances.shape == (3,)
        assert distances[0] == 0  # distance to self

    def test_hamming_similarity(self, sample_embeds):
        """Test hamming_similarity returns normalized scores in [0, 1]."""
        # Create identical embeddings
        embeds = np.array(
            [
                [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
            ],
            dtype=np.float32,
        )
        packed = binary_quantize(embeds)

        similarity = packed.hamming_similarity(packed)

        assert similarity.shape == (2, 2)
        # Identical vectors should have similarity 1.0
        np.testing.assert_allclose(similarity[0, 1], 1.0)
        np.testing.assert_allclose(similarity[1, 0], 1.0)
        # Self-similarity should be 1.0
        np.testing.assert_allclose(similarity[0, 0], 1.0)

    def test_hamming_similarity_range(self, packed):
        """Test hamming_similarity values are in [0, 1]."""
        similarity = packed.hamming_similarity(packed)

        assert np.all(similarity >= 0)
        assert np.all(similarity <= 1)

    def test_slicing_preserves_type(self, packed):
        """Test slicing returns BinaryEmbeds with correct metadata."""
        sliced = packed[0:2]

        assert isinstance(sliced, BinaryEmbeds)
        assert sliced.shape == (2, 1)  # 8 dims -> 1 byte
        assert sliced.orig_dim == 8

    def test_single_row_indexing(self, packed):
        """Test single row indexing keeps 2D shape."""
        row = packed[0]

        assert isinstance(row, BinaryEmbeds)
        assert row.shape == (1, 1)
        assert row.orig_dim == 8

    def test_pickle_roundtrip(self, packed):
        """Test BinaryEmbeds can be pickled and unpickled."""
        pickled = pickle.dumps(packed)
        unpickled = pickle.loads(pickled)

        assert isinstance(unpickled, BinaryEmbeds)
        assert unpickled.shape == packed.shape
        assert unpickled.orig_dim == packed.orig_dim
        np.testing.assert_array_equal(unpickled.packed, packed.packed)

    def test_repr(self, packed):
        """Test __repr__ returns expected format."""
        repr_str = repr(packed)
        assert "BinaryEmbeds" in repr_str
        assert "shape=" in repr_str
        assert "orig_dim=" in repr_str

    def test_properties(self, packed):
        """Test orig_dim and packed properties."""
        assert packed.orig_dim == 8
        assert packed.packed.dtype == np.uint8
        np.testing.assert_array_equal(packed.packed, packed.view(np.ndarray))

    def test_matmul_auto_unpack(self, sample_embeds, packed):
        """Test @ operator auto-unpacks for computation."""
        weights = np.random.randn(8, 4).astype(np.float32)

        result = packed @ weights
        expected = (sample_embeds > 0).astype(np.float32) @ weights

        np.testing.assert_allclose(result, expected, atol=1e-5)


# =============================================================================
# Cross-type Tests
# =============================================================================


class TestBinaryOperations:
    """Tests for binary embedding operations."""

    def test_binary_slicing(self):
        """Test BinaryEmbeds supports consistent slicing."""
        embeds = np.random.randn(5, 16).astype(np.float32)
        binary = binary_quantize(embeds)

        binary_slice = binary[1:3]

        assert isinstance(binary_slice, BinaryEmbeds)
        assert binary_slice.shape[0] == 2

    def test_binary_pickle(self):
        """Test BinaryEmbeds can be pickled."""
        embeds = np.random.randn(3, 16).astype(np.float32)
        binary = binary_quantize(embeds)

        binary_restored = pickle.loads(pickle.dumps(binary))

        assert isinstance(binary_restored, BinaryEmbeds)
        np.testing.assert_array_equal(binary_restored.packed, binary.packed)
