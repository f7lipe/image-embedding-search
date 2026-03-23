"""Shared pytest fixtures and helpers for the test suite."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 8  # tiny dimension so tests run fast


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_dir() -> Generator[Path, None, None]:
    """Temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture()
def sample_image(tmp_dir: Path) -> str:
    """A 16x16 white JPEG image in a temporary directory."""
    path = tmp_dir / "sample.jpg"
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(str(path))
    return str(path)


@pytest.fixture()
def mock_embedder() -> MagicMock:
    """A CLIPEmbedder stand-in that returns deterministic unit-norm embeddings."""
    embedder = MagicMock()
    embedder.embedding_dim = EMBEDDING_DIM

    rng = np.random.default_rng(42)

    def _embed(*_args, **_kwargs) -> np.ndarray:
        vec = rng.random(EMBEDDING_DIM).astype(np.float32)
        return vec / np.linalg.norm(vec)

    embedder.embed_image.side_effect = _embed
    embedder.embed_text.side_effect = _embed
    return embedder
