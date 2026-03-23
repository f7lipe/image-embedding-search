"""FAISS-backed vector index for image embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import faiss
import numpy as np

from .embedder import CLIPEmbedder


class ImageIndex:
    """Stores image embeddings in a FAISS ``IndexFlatIP`` index.

    Cosine similarity is achieved by L2-normalizing every vector before
    insertion and using inner-product search (equivalent to cosine similarity
    on unit vectors).

    Parameters
    ----------
    embedder:
        :class:`CLIPEmbedder` (or compatible) used to produce embeddings.
    index_path:
        Directory where ``index.faiss`` and ``paths.json`` will be saved.
    """

    _FAISS_FILE = "index.faiss"
    _PATHS_FILE = "paths.json"

    def __init__(
        self,
        embedder: CLIPEmbedder,
        index_path: str = "data/index",
    ) -> None:
        self.embedder = embedder
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self._faiss: faiss.Index = faiss.IndexFlatIP(embedder.embedding_dim)
        self._paths: List[str] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_images(self, image_paths: List[str]) -> int:
        """Embed and add images to the index.

        Parameters
        ----------
        image_paths:
            Absolute or relative paths to image files.

        Returns
        -------
        int
            Number of images successfully added.
        """
        embeddings: List[np.ndarray] = []
        valid_paths: List[str] = []

        for path in image_paths:
            try:
                vec = self.embedder.embed_image(path)
                embeddings.append(vec)
                valid_paths.append(str(path))
            except (FileNotFoundError, OSError, ValueError) as exc:
                print(f"[index] Skipping '{path}': {exc}")

        if embeddings:
            matrix = np.array(embeddings, dtype=np.float32)
            self._faiss.add(matrix)
            self._paths.extend(valid_paths)

        return len(embeddings)

    def save(self) -> None:
        """Persist the FAISS index and path list to :attr:`index_path`."""
        faiss.write_index(self._faiss, str(self.index_path / self._FAISS_FILE))
        with open(self.index_path / self._PATHS_FILE, "w", encoding="utf-8") as fh:
            json.dump(self._paths, fh)

    def load(self) -> None:
        """Load a previously saved index from :attr:`index_path`.

        Raises
        ------
        FileNotFoundError
            If either the FAISS binary or the paths JSON file is missing.
        """
        faiss_file = self.index_path / self._FAISS_FILE
        paths_file = self.index_path / self._PATHS_FILE
        if not faiss_file.exists() or not paths_file.exists():
            raise FileNotFoundError(f"Index files not found in '{self.index_path}'")
        self._faiss = faiss.read_index(str(faiss_file))
        with open(paths_file, encoding="utf-8") as fh:
            self._paths = json.load(fh)

    def clear(self) -> None:
        """Remove all vectors from the index (without touching disk)."""
        self._faiss = faiss.IndexFlatIP(self.embedder.embedding_dim)
        self._paths = []

    # ------------------------------------------------------------------
    # Properties / dunder
    # ------------------------------------------------------------------

    @property
    def image_paths(self) -> List[str]:
        """Ordered list of image paths matching FAISS vector positions."""
        return list(self._paths)

    def __len__(self) -> int:
        return self._faiss.ntotal
