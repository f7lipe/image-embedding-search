# image-embedding-search

Semantic image search powered by [CLIP](https://openai.com/research/clip) embeddings and a [FAISS](https://faiss.ai/) vector store.

Search your image collection using **natural-language text** or a **query image**. Both modalities are mapped into the same CLIP embedding space, enabling zero-shot cross-modal retrieval.

---

## Features

- 🔍 **Text → image** search ("a red sports car in the rain")
- 🖼 **Image → image** search (find visually similar images)
- ⚡ **FAISS inner-product index** with cosine similarity (L2-normalised vectors)
- 🤗 **CLIP via sentence-transformers** – no OpenAI API key needed
- 📂 **Recursive folder indexing** with a one-line CLI command
- 🚀 **FastAPI REST API** – easily integrate with any front-end or pipeline
- 🐳 **Docker / Docker Compose** support
- Configurable `top_k`, model name, and index path

---

## Project structure

```
image-embedding-search/
├── src/
│   ├── __init__.py
│   ├── embedder.py          # CLIPEmbedder – image & text → vector
│   ├── indexer.py           # ImageIndex  – FAISS-backed vector store
│   └── searcher.py          # ImageSearcher – cosine similarity search
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI application
├── scripts/
│   ├── index_images.py      # CLI: index a folder of images
│   ├── search.py            # CLI: search by text or image
│   └── create_sample_data.py  # Generate sample images for testing
├── data/
│   └── samples/             # Eight sample PNG images (shapes & colours)
├── tests/
│   ├── conftest.py
│   ├── test_embedder.py
│   ├── test_indexer.py
│   ├── test_searcher.py
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── pyproject.toml
```

---

## Quick start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> The first run downloads the CLIP model (~340 MB) from Hugging Face and caches it locally.

---

### 2. Index images

```bash
# Index the bundled sample images
python scripts/index_images.py data/samples

# Index your own folder (recursive by default)
python scripts/index_images.py /path/to/my/photos

# Options
python scripts/index_images.py data/samples \
    --index-path data/my_index \
    --model clip-ViT-B-32 \
    --no-recursive
```

---

### 3. Search

**By text:**

```bash
python scripts/search.py --text "a blue geometric shape"
python scripts/search.py --text "something red and round" --top-k 3
```

**By image:**

```bash
python scripts/search.py --image data/samples/red_circle.jpg
python scripts/search.py --image /path/to/query.jpg --top-k 10
```

---

### 4. Start the REST API

```bash
uvicorn api.main:app --reload
# Swagger UI: http://localhost:8000/docs
```

#### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/index` | Index images from a local folder |
| `POST` | `/search/text` | Search by text query |
| `POST` | `/search/image` | Search by uploading a query image |
| `GET` | `/stats` | Index statistics |

**Index a folder:**
```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"folder": "data/samples", "recursive": true}'
```

**Search by text:**
```bash
curl -X POST "http://localhost:8000/search/text?query=red+circle&top_k=3"
```

**Search by image:**
```bash
curl -X POST "http://localhost:8000/search/image?top_k=3" \
  -F "file=@data/samples/red_circle.jpg"
```

#### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INDEX_PATH` | `data/index` | Directory for the FAISS index |
| `CLIP_MODEL` | `clip-ViT-B-32` | sentence-transformers model identifier |

---

## Docker

```bash
# Build & run
docker compose up --build

# Index sample images inside the container
docker compose exec api python scripts/index_images.py data/samples

# API available at http://localhost:8000
```

The `data/` directory is mounted as a volume so the index persists across restarts.

---

## Running tests

Tests use mocked embeddings and do **not** require downloading the CLIP model.

```bash
pip install pytest
pytest
```

---

## How it works

1. **Embedding** – Both images and text are passed through the CLIP encoder
   (`clip-ViT-B-32` via `sentence-transformers`). Each input is projected into
   a shared 512-dimensional embedding space and L2-normalised to unit length.

2. **Indexing** – Normalised image vectors are stored in a FAISS
   `IndexFlatIP` (inner-product) index. Inner product on unit vectors equals
   cosine similarity, so the index provides exact nearest-neighbour search by
   cosine distance.

3. **Search** – The query (text or image) is embedded with the same encoder,
   normalised, and then used to query FAISS. The top-*k* most similar entries
   are returned together with their cosine similarity scores (range –1 to 1,
   higher is more similar).

---

## Configurable parameters

| Parameter | Where | Description |
|-----------|-------|-------------|
| `--model` | CLI / `CLIP_MODEL` env | CLIP model variant (e.g. `clip-ViT-L-14`) |
| `--top-k` | CLI / `top_k` query param | Number of results to return |
| `--index-path` | CLI / `INDEX_PATH` env | Index storage directory |
| `--no-recursive` | CLI | Disable recursive image discovery |
