#!/usr/bin/env python3
"""Index a folder of images into the FAISS vector store.

Usage
-----
    python scripts/index_images.py data/samples
    python scripts/index_images.py /path/to/images --index-path data/my_index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedder import CLIPEmbedder
from src.indexer import ImageIndex

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index images into the FAISS vector database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("folder", help="Path to folder containing images.")
    parser.add_argument(
        "--index-path",
        default="data/index",
        help="Directory where the index will be stored.",
    )
    parser.add_argument(
        "--model",
        default="clip-ViT-B-32",
        help="sentence-transformers CLIP model name.",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        default=True,
        help="Do not search sub-directories.",
    )
    return parser.parse_args()


def collect_images(folder: Path, recursive: bool) -> list[str]:
    glob = folder.rglob if recursive else folder.glob
    return [str(p) for p in glob("*") if p.suffix.lower() in _IMAGE_EXTENSIONS]


def main() -> None:
    args = parse_args()
    folder = Path(args.folder)

    if not folder.exists():
        print(f"[error] Folder not found: '{folder}'", file=sys.stderr)
        sys.exit(1)

    print(f"[index] Loading embedder '{args.model}' …")
    embedder = CLIPEmbedder(model_name=args.model)
    index = ImageIndex(embedder=embedder, index_path=args.index_path)

    try:
        index.load()
        print(f"[index] Loaded existing index with {len(index)} images.")
    except FileNotFoundError:
        print("[index] No existing index found – creating a new one.")

    paths = collect_images(folder, args.recursive)
    print(f"[index] Found {len(paths)} images in '{folder}'.")

    if not paths:
        print("[index] Nothing to index.")
        return

    count = index.add_images(paths)
    index.save()
    print(f"[index] Done. Added {count} images. Total in index: {len(index)}.")


if __name__ == "__main__":
    main()
