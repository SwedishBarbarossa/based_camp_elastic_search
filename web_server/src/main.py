import os
import pickle

import numpy as np
from annoy import AnnoyIndex
from fastapi import FastAPI
from fastapi.responses import FileResponse
from sentence_transformers import SentenceTransformer

work_dir = os.path.dirname(os.path.abspath(__file__))
root = work_dir.removesuffix("web_server/src")
static = os.path.join(work_dir, "static")
index_path = os.path.join(static, "index.html")
style_path = os.path.join(static, "style.css")
app = FastAPI()


@app.get("/", response_class=FileResponse)
async def serve_index():
    return index_path


@app.get("/static/style.css", response_class=FileResponse)
async def serve_style():
    return style_path


def build_index(
    dim: int, index_name: str, map_name: str, embeddings_dir: str
) -> tuple[AnnoyIndex, dict[int, str]]:
    index = AnnoyIndex(dim, "angular")

    embeddings = os.listdir(embeddings_dir)
    index_value_map = {}
    for i, embedding in enumerate(embeddings):
        vector = np.load(embeddings_dir + "/" + embedding)
        index.add_item(i, vector)
        index_value_map[i] = embedding.removesuffix(".npy")

    index.build(100)
    index.save(index_name)
    pickle.dump(index_value_map, open(f"{map_name}.pkl", "wb"))

    return index, index_value_map


def get_index(
    dim: int, index_name: str, map_name: str, embeddings_dir: str
) -> tuple[AnnoyIndex, dict[int, str]]:
    os.makedirs(embeddings_dir, exist_ok=True)

    if not os.path.exists(index_name):
        return build_index(dim, index_name, map_name, embeddings_dir)

    index = AnnoyIndex(dim, "angular")
    index.load(index_name)
    index_value_map = pickle.load(open(f"{map_name}.pkl", "rb"))
    return index, index_value_map


os.makedirs(os.path.join(work_dir, "saved"), exist_ok=True)
index_name = os.path.join(work_dir, "saved", "index.ann")
map_name = os.path.join(work_dir, "saved", "index_value_map")
embeddings_dir = os.path.join(root, "embeddings")

dim = 384
index, index_map = get_index(dim, index_name, map_name, embeddings_dir)


# Load the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


@app.get("/search/")
async def search(query: str | None = None):
    if not query:
        query = "total fertility rate collapse"

    if len(query) > 100:
        query = query[:100]

    search_vector = model.encode(query)
    indices, distances = index.get_nns_by_vector(
        search_vector, 100, search_k=10, include_distances=True
    )
    results = [index_map[index] for index in indices]
    return {"query": query, "results": zip(results, distances)}
