"""Tests for CLIPEmbedder."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image


class TestCLIPEmbedderNormalize:
    """Unit tests for the static _normalize helper – no model required."""

    def test_unit_vector_unchanged(self) -> None:
        from src.embedder import CLIPEmbedder

        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = CLIPEmbedder._normalize(vec)
        np.testing.assert_allclose(result, vec, atol=1e-6)

    def test_arbitrary_vector_is_unit(self) -> None:
        from src.embedder import CLIPEmbedder

        vec = np.array([3.0, 4.0], dtype=np.float32)
        result = CLIPEmbedder._normalize(vec)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_zero_vector_returns_zeros(self) -> None:
        from src.embedder import CLIPEmbedder

        vec = np.zeros(4, dtype=np.float32)
        result = CLIPEmbedder._normalize(vec)
        np.testing.assert_array_equal(result, vec)

    def test_output_is_float32(self) -> None:
        from src.embedder import CLIPEmbedder

        vec = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = CLIPEmbedder._normalize(vec)
        assert result.dtype == np.float32


class TestCLIPEmbedderWithMock:
    """Tests that exercise embed_image / embed_text via a mocked SentenceTransformer.

    We bypass ``__init__`` entirely (which requires the heavy ``sentence-transformers``
    package) and instead build the object by hand.
    """

    def _make_embedder(self, dim: int = 8) -> "CLIPEmbedder":  # noqa: F821
        """Create a CLIPEmbedder whose internal model is a MagicMock."""
        from src.embedder import CLIPEmbedder

        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones(dim, dtype=np.float32)

        # Skip __init__ to avoid the sentence-transformers import.
        embedder = CLIPEmbedder.__new__(CLIPEmbedder)
        embedder.model_name = "test-model"
        embedder._model = mock_model
        embedder.embedding_dim = dim
        return embedder

    def test_embed_text_returns_unit_vector(self) -> None:
        embedder = self._make_embedder()
        result = embedder.embed_text("a cat")
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_embed_text_returns_float32(self) -> None:
        embedder = self._make_embedder()
        result = embedder.embed_text("hello")
        assert result.dtype == np.float32

    def test_embed_image_returns_unit_vector(self) -> None:
        import tempfile

        from PIL import Image

        embedder = self._make_embedder()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            Image.new("RGB", (16, 16)).save(f.name)
            result = embedder.embed_image(f.name)

        assert abs(np.linalg.norm(result) - 1.0) < 1e-6
