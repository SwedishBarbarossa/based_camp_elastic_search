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
    """Query Qdrant within channels and return the results"""
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
    """Creates a collection if it doesn't exist yet"""
    collection_config = {
        "collection_name": collection,
        "optimizers_config": models.OptimizersConfigDiff(memmap_threshold=20000),
        "hnsw_config": models.HnswConfigDiff(on_disk=True, m=32, ef_construct=256),
        "quantization_config": models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.999,
                always_ram=True,
            ),
        ),
    }
    if client.collection_exists(collection):
        client.update_collection(**collection_config)
    else:
        client.create_collection(
            vectors_config=models.VectorParams(
                size=dim, distance=models.Distance.COSINE
            ),
            **collection_config,
        )


def payload_index_exists(
    client: QdrantClient, collection_name: str, payload_field: str
):
    """Check if payload index exists"""
    collection_info = client.get_collection(collection_name)
    if payload_field in collection_info.payload_schema:
        return True
    return False


def create_qdrant_index(
    payload_key: str, collection: str, client: QdrantClient
) -> None:
    """Creates a payload index for the collection if it doesn't exist yet"""
    if payload_index_exists(client, collection, payload_key):
        return

    client.create_payload_index(
        collection_name=collection,
        field_name=payload_key,
        field_schema=models.PayloadSchemaType.TEXT,
    )


def _name_to_payload(name: str) -> tuple[str, str, str, int, int]:
    """Parse filename into (q_type, channel, id, start, end)"""
    split = name.removesuffix(".npy").split(" ")
    try:
        return split[0], split[1], split[2], int(split[3]), int(split[4])
    except IndexError:
        raise ValueError(f"Invalid name: {name}")


def add_to_qdrant(
    embeddings_dir: str,
    record_path: str,
    client: QdrantClient,
    dims: ModelDims,
) -> None:
    """Add embeddings in embeddings_dir to Qdrant"""
    create_qdrant_collection("asym", client, dims["asymmetric"])
    create_qdrant_index("channel", "asym", client)
    create_qdrant_collection("sym", client, dims["symmetric"])
    create_qdrant_index("channel", "sym", client)

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

    sym_points = []
    asym_points = []
    for i, (embedding, (q_type, channel, id, start, end)) in enumerate(
        zip(embeddings, payloads)
    ):
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

        if i % 1000 == 999:
            client.upload_points(collection_name="sym", points=sym_points)
            client.upload_points(collection_name="asym", points=asym_points)
            sym_points = []
            asym_points = []

    client.upload_points(collection_name="sym", points=sym_points)
    client.upload_points(collection_name="asym", points=asym_points)

    # add to record
    with open(record_path, "a", encoding="utf-8") as f:
        f.writelines(f"{x}\n" for x in sorted(embeddings))
