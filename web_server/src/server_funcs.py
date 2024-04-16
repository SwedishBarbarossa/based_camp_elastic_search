import os
import pickle

import numpy as np
from annoy import AnnoyIndex


def build_index(
    dim: int, index_name: str, map_name: str, embeddings_dir: str
) -> tuple[AnnoyIndex, dict[int, str]]:
    index_folder = os.path.dirname(index_name)
    os.makedirs(index_folder, exist_ok=True)

    os.makedirs(embeddings_dir, exist_ok=True)
    embeddings = os.listdir(embeddings_dir)

    index = AnnoyIndex(dim, "angular")
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
