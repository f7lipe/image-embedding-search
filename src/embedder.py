"""CLIP-based image and text embedder."""

from __future__ import annotations

import numpy as np
from PIL import Image


class CLIPEmbedder:
    """Generates L2-normalized embeddings for images and text using CLIP.

    Parameters
    ----------
    model_name:
        Any sentence-transformers CLIP model identifier, e.g. ``"clip-ViT-B-32"``.
    """

    EMBEDDING_DIM: int = 512  # ViT-B/32 output dimension

    def __init__(self, model_name: str = "clip-ViT-B-32") -> None:
        # Lazy import so the module can be imported without the heavy dependency
        # being loaded (useful for testing with mocks).
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        # Allow subclasses / tests to override the dimension at instance level.
        self.embedding_dim: int = self.EMBEDDING_DIM

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def embed_image(self, image_path: str) -> np.ndarray:
        """Return a normalized float32 embedding for an image file.

        Parameters
        ----------
        image_path:
            Path to a JPEG, PNG, or any PIL-readable image.

        Returns
        -------
        np.ndarray
            1-D float32 array of length ``embedding_dim``.
        """
        image = Image.open(image_path).convert("RGB")
        return self._normalize(self._model.encode(image, convert_to_numpy=True))

    def embed_text(self, text: str) -> np.ndarray:
        """Return a normalized float32 embedding for a text string.

        Parameters
        ----------
        text:
            Natural-language description / query.

        Returns
        -------
        np.ndarray
            1-D float32 array of length ``embedding_dim``.
        """
        return self._normalize(self._model.encode(text, convert_to_numpy=True))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            return vec.astype(np.float32)
        return (vec / norm).astype(np.float32)
