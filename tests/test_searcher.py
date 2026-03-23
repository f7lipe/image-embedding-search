"""Tests for ImageSearcher."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.conftest import EMBEDDING_DIM


def _build_searcher(mock_embedder, tmp_dir: Path, num_images: int = 5):
    """Helper: create index, add *num_images*, return (index, searcher)."""
    from src.indexer import ImageIndex
    from src.searcher import ImageSearcher

    idx_dir = tmp_dir / "idx"
    index = ImageIndex(embedder=mock_embedder, index_path=str(idx_dir))

    paths = []
    for i in range(num_images):
        p = tmp_dir / f"img_{i}.jpg"
        Image.new("RGB", (16, 16), color=(i * 20, 0, 0)).save(str(p))
        paths.append(str(p))
    index.add_images(paths)

    searcher = ImageSearcher(index=index, embedder=mock_embedder)
    return index, searcher, paths


class TestImageSearcher:

    def test_search_by_text_returns_results(
        self, mock_embedder, tmp_dir: Path
    ) -> None:
        _, searcher, _ = _build_searcher(mock_embedder, tmp_dir, 5)
        results = searcher.search_by_text("a cat", top_k=3)
        assert len(results) == 3

    def test_search_by_text_result_format(
        self, mock_embedder, tmp_dir: Path
    ) -> None:
        _, searcher, _ = _build_searcher(mock_embedder, tmp_dir, 5)
        results = searcher.search_by_text("sunset", top_k=2)
        for path, score in results:
            assert isinstance(path, str)
            assert isinstance(score, float)

    def test_search_by_image_returns_results(
        self, mock_embedder, tmp_dir: Path, sample_image: str
    ) -> None:
        _, searcher, _ = _build_searcher(mock_embedder, tmp_dir, 5)
        results = searcher.search_by_image(sample_image, top_k=3)
        assert len(results) == 3

    def test_top_k_capped_by_index_size(
        self, mock_embedder, tmp_dir: Path
    ) -> None:
        _, searcher, _ = _build_searcher(mock_embedder, tmp_dir, 3)
        results = searcher.search_by_text("query", top_k=10)
        assert len(results) == 3  # only 3 images in index

    def test_empty_index_returns_empty(
        self, mock_embedder, tmp_dir: Path
    ) -> None:
        from src.indexer import ImageIndex
        from src.searcher import ImageSearcher

        index = ImageIndex(embedder=mock_embedder, index_path=str(tmp_dir / "empty"))
        searcher = ImageSearcher(index=index, embedder=mock_embedder)
        assert searcher.search_by_text("anything", top_k=5) == []
        assert searcher.search_by_image("img.jpg", top_k=5) == []

    def test_scores_in_range(
        self, mock_embedder, tmp_dir: Path
    ) -> None:
        """Cosine similarity of unit vectors is in [-1, 1]."""
        _, searcher, _ = _build_searcher(mock_embedder, tmp_dir, 5)
        results = searcher.search_by_text("test", top_k=5)
        for _, score in results:
            assert -1.0 <= score <= 1.0 + 1e-6

    def test_exact_match_scores_one(
        self, mock_embedder, tmp_dir: Path
    ) -> None:
        """Searching with the same embedding that was indexed should score ~1."""
        from src.indexer import ImageIndex
        from src.searcher import ImageSearcher

        # Fixed deterministic embedding
        fixed_vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        fixed_vec[0] = 1.0  # unit vector

        mock_embedder.embed_image.side_effect = lambda *a, **kw: fixed_vec.copy()
        mock_embedder.embed_text.side_effect = lambda *a, **kw: fixed_vec.copy()

        idx_dir = tmp_dir / "idx_exact"
        index = ImageIndex(embedder=mock_embedder, index_path=str(idx_dir))

        img_path = tmp_dir / "exact.jpg"
        Image.new("RGB", (16, 16)).save(str(img_path))
        index.add_images([str(img_path)])

        searcher = ImageSearcher(index=index, embedder=mock_embedder)
        results = searcher.search_by_text("same", top_k=1)
        assert abs(results[0][1] - 1.0) < 1e-5
