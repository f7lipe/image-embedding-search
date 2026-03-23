"""Tests for the FastAPI application."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.conftest import EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fixed_embedding() -> np.ndarray:
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vec[0] = 1.0
    return vec


def _make_mock_embedder() -> MagicMock:
    e = MagicMock()
    e.embedding_dim = EMBEDDING_DIM
    e.embed_image.return_value = _fixed_embedding()
    e.embed_text.return_value = _fixed_embedding()
    return e


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def api_client(tmp_path: Path) -> Generator[TestClient, None, None]:
    """Provide a TestClient with mocked embedder and a fresh empty index."""
    # We need to reload api.main so environment and module-level state is fresh.
    if "api.main" in sys.modules:
        del sys.modules["api.main"]

    mock_emb = _make_mock_embedder()

    with patch("src.embedder.CLIPEmbedder", return_value=mock_emb):
        with patch.dict("os.environ", {"INDEX_PATH": str(tmp_path / "idx"), "CLIP_MODEL": "test"}):
            import api.main as main_module

            # Replace the module-level embedder/index/searcher so routes use mocks.
            from src.indexer import ImageIndex
            from src.searcher import ImageSearcher

            main_module.embedder = mock_emb
            main_module.index = ImageIndex(
                embedder=mock_emb, index_path=str(tmp_path / "idx")
            )
            main_module.searcher = ImageSearcher(
                index=main_module.index, embedder=mock_emb
            )

            client = TestClient(main_module.app)
            yield client

    # Clean up so other test modules can import api.main fresh.
    if "api.main" in sys.modules:
        del sys.modules["api.main"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStatsEndpoint:
    def test_stats_returns_200(self, api_client: TestClient) -> None:
        resp = api_client.get("/stats")
        assert resp.status_code == 200

    def test_stats_has_required_fields(self, api_client: TestClient) -> None:
        data = api_client.get("/stats").json()
        assert "indexed_images" in data
        assert "model" in data
        assert "index_path" in data


class TestIndexEndpoint:
    def test_index_nonexistent_folder_returns_400(
        self, api_client: TestClient
    ) -> None:
        resp = api_client.post("/index", json={"folder": "/nonexistent/path"})
        assert resp.status_code == 400

    def test_index_valid_folder(
        self, api_client: TestClient, tmp_path: Path
    ) -> None:
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        Image.new("RGB", (16, 16)).save(str(img_dir / "a.jpg"))
        Image.new("RGB", (16, 16)).save(str(img_dir / "b.png"))

        resp = api_client.post("/index", json={"folder": str(img_dir)})
        assert resp.status_code == 200
        data = resp.json()
        assert data["indexed"] == 2
        assert data["total"] == 2

    def test_index_empty_folder(
        self, api_client: TestClient, tmp_path: Path
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        resp = api_client.post("/index", json={"folder": str(empty)})
        assert resp.status_code == 200
        assert resp.json()["indexed"] == 0


class TestSearchTextEndpoint:
    def test_search_text_empty_index(self, api_client: TestClient) -> None:
        resp = api_client.post("/search/text?query=cat")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_search_text_returns_results(
        self, api_client: TestClient, tmp_path: Path
    ) -> None:
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        for i in range(3):
            Image.new("RGB", (16, 16)).save(str(img_dir / f"{i}.jpg"))

        api_client.post("/index", json={"folder": str(img_dir)})
        resp = api_client.post("/search/text?query=dog&top_k=2")
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 2
        for item in results:
            assert "path" in item
            assert "score" in item


class TestSearchImageEndpoint:
    def test_search_image_empty_index(
        self, api_client: TestClient, tmp_path: Path
    ) -> None:
        img_path = tmp_path / "q.jpg"
        Image.new("RGB", (16, 16)).save(str(img_path))
        with open(img_path, "rb") as f:
            resp = api_client.post(
                "/search/image", files={"file": ("q.jpg", f, "image/jpeg")}
            )
        assert resp.status_code == 200
        assert resp.json() == []

    def test_search_image_returns_results(
        self, api_client: TestClient, tmp_path: Path
    ) -> None:
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        for i in range(3):
            Image.new("RGB", (16, 16)).save(str(img_dir / f"{i}.jpg"))
        api_client.post("/index", json={"folder": str(img_dir)})

        query_img = tmp_path / "query.jpg"
        Image.new("RGB", (16, 16)).save(str(query_img))
        with open(query_img, "rb") as f:
            resp = api_client.post(
                "/search/image?top_k=2",
                files={"file": ("query.jpg", f, "image/jpeg")},
            )
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 2
