"""Tests for ImageIndex."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.conftest import EMBEDDING_DIM


class TestImageIndex:
    """Tests using a mock embedder (no CLIP model required)."""

    def _make_index(self, mock_embedder, index_path: Path):
        from src.indexer import ImageIndex

        return ImageIndex(embedder=mock_embedder, index_path=str(index_path))

    def _make_images(self, directory: Path, count: int = 3) -> list[str]:
        paths = []
        for i in range(count):
            p = directory / f"img_{i}.jpg"
            Image.new("RGB", (16, 16), color=(i * 30, 0, 0)).save(str(p))
            paths.append(str(p))
        return paths

    # ------------------------------------------------------------------
    # add_images
    # ------------------------------------------------------------------

    def test_add_images_increments_length(
        self, mock_embedder, tmp_dir: Path
    ) -> None:
        index = self._make_index(mock_embedder, tmp_dir / "idx")
        paths = self._make_images(tmp_dir, 4)
        count = index.add_images(paths)
        assert count == 4
        assert len(index) == 4

    def test_add_images_skips_invalid_paths(
        self, mock_embedder, tmp_dir: Path
    ) -> None:
        mock_embedder.embed_image.side_effect = FileNotFoundError("missing")
        index = self._make_index(mock_embedder, tmp_dir / "idx")
        count = index.add_images(["nonexistent.jpg"])
        assert count == 0
        assert len(index) == 0

    def test_add_images_accumulates_across_calls(
        self, mock_embedder, tmp_dir: Path
    ) -> None:
        index = self._make_index(mock_embedder, tmp_dir / "idx")
        paths = self._make_images(tmp_dir, 5)
        index.add_images(paths[:2])
        index.add_images(paths[2:])
        assert len(index) == 5

    # ------------------------------------------------------------------
    # save / load round-trip
    # ------------------------------------------------------------------

    def test_save_load_roundtrip(self, mock_embedder, tmp_dir: Path) -> None:
        idx_dir = tmp_dir / "idx"
        index = self._make_index(mock_embedder, idx_dir)
        paths = self._make_images(tmp_dir, 3)
        index.add_images(paths)
        index.save()

        # Load into a fresh instance.
        index2 = self._make_index(mock_embedder, idx_dir)
        index2.load()
        assert len(index2) == 3
        assert set(index2.image_paths) == set(paths)

    def test_load_raises_when_files_missing(
        self, mock_embedder, tmp_dir: Path
    ) -> None:
        index = self._make_index(mock_embedder, tmp_dir / "empty_idx")
        with pytest.raises(FileNotFoundError):
            index.load()

    # ------------------------------------------------------------------
    # clear
    # ------------------------------------------------------------------

    def test_clear_resets_index(self, mock_embedder, tmp_dir: Path) -> None:
        index = self._make_index(mock_embedder, tmp_dir / "idx")
        paths = self._make_images(tmp_dir, 2)
        index.add_images(paths)
        assert len(index) == 2
        index.clear()
        assert len(index) == 0
        assert index.image_paths == []

    # ------------------------------------------------------------------
    # image_paths property
    # ------------------------------------------------------------------

    def test_image_paths_returns_copy(self, mock_embedder, tmp_dir: Path) -> None:
        index = self._make_index(mock_embedder, tmp_dir / "idx")
        paths = self._make_images(tmp_dir, 2)
        index.add_images(paths)
        retrieved = index.image_paths
        retrieved.clear()  # mutating the returned list should not affect the index
        assert len(index.image_paths) == 2
