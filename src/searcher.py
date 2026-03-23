"""Similarity search over an :class:`ImageIndex`."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .embedder import CLIPEmbedder
from .indexer import ImageIndex


class ImageSearcher:
    """Performs cosine-similarity search against a populated :class:`ImageIndex`.

    Parameters
    ----------
    index:
        A loaded (or freshly populated) :class:`ImageIndex`.
    embedder:
        The same :class:`CLIPEmbedder` used when building the index.
    """

    def __init__(self, index: ImageIndex, embedder: CLIPEmbedder) -> None:
        self.index = index
        self.embedder = embedder

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search_by_text(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Return the *top_k* most similar images to a natural-language query.

        Parameters
        ----------
        query:
            Free-form text description.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list of (path, score) tuples
            Sorted by descending cosine similarity score.
        """
        embedding = self.embedder.embed_text(query)
        return self._search(embedding, top_k)

    def search_by_image(
        self,
        image_path: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Return the *top_k* most similar images to a query image.

        Parameters
        ----------
        image_path:
            Path to the query image file.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list of (path, score) tuples
            Sorted by descending cosine similarity score.
        """
        embedding = self.embedder.embed_image(image_path)
        return self._search(embedding, top_k)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search(
        self,
        embedding: np.ndarray,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        if len(self.index) == 0:
            return []

        k = min(top_k, len(self.index))
        query_matrix = np.array([embedding], dtype=np.float32)
        scores, indices = self.index._faiss.search(query_matrix, k)  # noqa: SLF001

        results: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.index.image_paths[idx], float(score)))
        return results
