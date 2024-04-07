import os
import pickle

import numpy as np
from annoy import AnnoyIndex
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer


def build_index(
    dim: int, index_name: str, map_name: str, embeddings_dir: str
) -> tuple[AnnoyIndex, dict[int, str]]:
    # Assume your embeddings are 128-dimensional
    index = AnnoyIndex(dim, "angular")

    embeddings = os.listdir(embeddings_dir)
    index_value_map = {}
    for i, embedding in enumerate(embeddings):
        vector = np.load(embeddings_dir + "/" + embedding)
        index.add_item(i, vector)
        index_value_map[i] = embedding.removesuffix(".npy")

    index.build(10)  # 10 trees
    index.save(index_name)
    pickle.dump(index_value_map, open(f"{map_name}.pkl", "wb"))

    return index, index_value_map


def get_index(
    dim: int, index_name: str, map_name: str, embeddings_dir: str
) -> tuple[AnnoyIndex, dict[int, str]]:
    if not os.path.exists(index_name):
        return build_index(dim, index_name, map_name, embeddings_dir)

    index = AnnoyIndex(dim, "angular")
    index.load(index_name)
    index_value_map = pickle.load(open(f"{map_name}.pkl", "rb"))
    return index, index_value_map


dim = 128
index_name = "index.ann"
map_name = "index_value_map"
embeddings_dir = "embeddings"
index, index_map = get_index(dim, index_name, map_name, embeddings_dir)

app = FastAPI()

# Load the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


@app.get("/search/")
async def search(query: str):
    search_vector = model.encode(query)
    indices = index.get_nns_by_vector(search_vector, 10, search_k=3)
    results = [index_map[index] for index in indices]
    return {"results": results}
