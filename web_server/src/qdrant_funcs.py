import os
import uuid
from typing import Literal, TypedDict

import numpy as np
import numpy.typing as npt
from qdrant_client import QdrantClient, models


class ModelDims(TypedDict):
    asymmetric: int
    symmetric: int


def query_qdrant(
    channels: list[str],
    vector: npt.NDArray[np.float32],
    client: QdrantClient,
    q_type: Literal["sym", "asym"] = "sym",
) -> list[list]:
    _filter = models.Filter(
        should=[
            models.FieldCondition(
                key="channel",
                match=models.MatchValue(
                    value=x,
                ),
            )
            for x in channels
        ]
    )
    responses: list[models.ScoredPoint] = client.search(
        collection_name=q_type,
        query_vector=vector,
        limit=100,
        with_payload=True,
        query_filter=_filter,
    )
    return [
        (
            [
                x.payload["id"],
                x.payload["start"],
                x.payload["end"],
                round(x.score, 4),
            ]
            if x.payload
            else [{x.id}, 0, 0, round(x.score, 4)]
        )
        for x in responses
    ]


def create_qdrant_collection(collection: str, client: QdrantClient, dim: int) -> None:
    if client.collection_exists(collection):
        return

    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
        hnsw_config=models.HnswConfigDiff(on_disk=True, m=32, ef_construct=256),
    )


def _name_to_payload(name: str) -> tuple[str, str, str, int, int]:
    split = name.removesuffix(".npy").split(" ")
    return split[0], split[1], split[2], int(split[3]), int(split[4])


def add_to_qdrant(
    embeddings_dir: str,
    record_path: str,
    client: QdrantClient,
    dims: ModelDims,
) -> None:

    create_qdrant_collection("asym", client, dims["asymmetric"])
    create_qdrant_collection("sym", client, dims["symmetric"])

    sym_points = []
    asym_points = []

    already_added: set[str] = set()
    if os.path.exists(record_path):
        with open(record_path, "r", encoding="utf-8") as f:
            already_added = {x.strip() for x in f.readlines()}
    else:
        # create the file
        with open(record_path, "w", encoding="utf-8") as f:
            pass

    os.makedirs(embeddings_dir, exist_ok=True)
    embeddings = [
        x
        for x in os.listdir(embeddings_dir)
        if x.endswith(".npy") and x not in already_added
    ]
    payloads = [_name_to_payload(x) for x in embeddings]

    # clear record
    already_added = set()

    for embedding, (q_type, channel, id, start, end) in zip(embeddings, payloads):
        vector = np.load(embeddings_dir + "/" + embedding)

        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"id": id, "start": start, "end": end, "channel": channel},
        )

        if q_type == "sym":
            sym_points.append(point)
        else:
            asym_points.append(point)

    client.upload_points(collection_name="sym", points=sym_points)
    client.upload_points(collection_name="asym", points=asym_points)

    # add to record
    with open(record_path, "a", encoding="utf-8") as f:
        f.writelines(f"{x}\n" for x in embeddings)
