#!/usr/bin/env python3
"""Search the indexed images by text or image query.

Usage
-----
    python scripts/search.py --text "a red car"
    python scripts/search.py --image data/samples/cat.jpg --top-k 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedder import CLIPEmbedder
from src.indexer import ImageIndex
from src.searcher import ImageSearcher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search indexed images by text or image query.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Natural-language search query.")
    group.add_argument("--image", help="Path to a query image.")
    parser.add_argument(
        "--index-path",
        default="data/index",
        help="Directory containing the saved index.",
    )
    parser.add_argument(
        "--model",
        default="clip-ViT-B-32",
        help="sentence-transformers CLIP model name.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[search] Loading embedder '{args.model}' …")
    embedder = CLIPEmbedder(model_name=args.model)
    index = ImageIndex(embedder=embedder, index_path=args.index_path)

    try:
        index.load()
        print(f"[search] Loaded index with {len(index)} images.")
    except FileNotFoundError:
        print(
            f"[error] Index not found at '{args.index_path}'. "
            "Run scripts/index_images.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    searcher = ImageSearcher(index=index, embedder=embedder)

    if args.text:
        print(f"[search] Query (text): '{args.text}'")
        results = searcher.search_by_text(args.text, top_k=args.top_k)
    else:
        print(f"[search] Query (image): '{args.image}'")
        results = searcher.search_by_image(args.image, top_k=args.top_k)

    if not results:
        print("[search] No results found.")
        return

    print(f"\nTop {len(results)} result(s):\n")
    for rank, (path, score) in enumerate(results, start=1):
        print(f"  {rank:>3}. score={score:.4f}  {path}")


if __name__ == "__main__":
    main()
