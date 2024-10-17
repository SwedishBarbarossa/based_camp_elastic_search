import asyncio
import os
import uuid
from typing import Generator, Literal, TypedDict

import numpy as np
import numpy.typing as npt
from qdrant_client import AsyncQdrantClient, models

from services.record_funcs import add_embeddings_to_record, get_files_in_record


class ModelDims(TypedDict):
    asymmetric: int
    symmetric: int


q_types = Literal["sym", "asym"]


class NPFileStruct(TypedDict):
    vector: npt.NDArray[np.float32]
    filename: str
    q_type: q_types
    channel: str
    id: str
    start: int
    end: int


async def create_qdrant_collection(
    collection: str, client: AsyncQdrantClient, dim: int
) -> None:
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
    if await client.collection_exists(collection):
        await client.update_collection(**collection_config)
    else:
        await client.create_collection(
            vectors_config=models.VectorParams(
                size=dim, distance=models.Distance.COSINE
            ),
            **collection_config,
        )


async def payload_index_exists(
    client: AsyncQdrantClient, collection_name: str, payload_field: str
):
    """Check if payload index exists"""
    collection_info = await client.get_collection(collection_name)
    if payload_field in collection_info.payload_schema:
        return True
    return False


async def create_qdrant_index(
    payload_key: str, collection: str, client: AsyncQdrantClient
) -> None:
    """Creates a payload index for the collection if it doesn't exist yet"""
    if await payload_index_exists(client, collection, payload_key):
        return

    await client.create_payload_index(
        collection_name=collection,
        field_name=payload_key,
        field_schema=models.PayloadSchemaType.TEXT,
    )


async def initialize_collection(
    collection: str, client: AsyncQdrantClient, dim: int, index: str
) -> None:
    """Initializes Qdrant collection"""
    await create_qdrant_collection(collection, client, dim)
    await create_qdrant_index(index, collection, client)


async def initialize_qdrant(dims: ModelDims, client: AsyncQdrantClient) -> None:
    await initialize_collection("asym", client, dims["asymmetric"], "channel")
    await initialize_collection("queries_asym", client, dims["asymmetric"], "id")
    await initialize_collection("sym", client, dims["symmetric"], "channel")
    await initialize_collection("queries_sym", client, dims["symmetric"], "id")


async def query_qdrant(
    channels: list[str],
    vector: npt.NDArray[np.float32],
    client: AsyncQdrantClient,
    q_type: q_types = "sym",
) -> list[list[str | int | float]]:
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
    responses: list[models.ScoredPoint] = await client.search(
        collection_name=q_type,
        query_vector=vector,
        limit=100,
        with_payload=True,
        query_filter=_filter,
    )
    return [
        [
            x.payload["id"],
            x.payload["start"],
            x.payload["end"],
            round(x.score, 4),
        ]
        for x in responses
        if x.payload
    ]


def _name_to_filestruct(name: str, embeddings_dir: str) -> NPFileStruct:
    """Parse filename into (q_type, channel, id, start, end)"""
    split = name.removesuffix(".npy").split(" ")
    path = os.path.join(embeddings_dir, split[1], split[2], name)
    vector = np.load(path)

    try:
        return {
            "vector": vector,
            "filename": name,
            "q_type": split[0],  # type: ignore
            "channel": split[1],
            "id": split[2],
            "start": int(split[3]),
            "end": int(split[4]),
        }
    except IndexError:
        raise ValueError(f"Invalid name: {name}")


def _make_pointstruct(payload: NPFileStruct) -> models.PointStruct:
    return models.PointStruct(
        id=str(uuid.uuid4()),
        vector=payload["vector"],  # type: ignore
        payload={
            "id": payload["id"],
            "start": payload["start"],
            "end": payload["end"],
            "channel": payload["channel"],
        },
    )


def _files_to_pointstructs(
    files: list[str], embeddings_dir: str
) -> list[models.PointStruct]:
    return [_make_pointstruct(_name_to_filestruct(x, embeddings_dir)) for x in files]


def remove_added_npy_files(embeddings_dir: str, record_dir: str):
    os.makedirs(embeddings_dir, exist_ok=True)
    file_paths: list[tuple[str, str]] = [
        (x, os.path.join(embeddings_dir, channel_dir, video_dir, x))
        for channel_dir in os.listdir(embeddings_dir)
        for video_dir in os.listdir(os.path.join(embeddings_dir, channel_dir))
        for x in os.listdir(os.path.join(embeddings_dir, channel_dir, video_dir))
        if x.endswith(".npy")
    ]

    file_record = get_files_in_record(record_dir)

    for npy_file, path in file_paths:
        if npy_file in file_record:
            os.remove(path)

    for channel_dir in os.listdir(embeddings_dir):
        for video_dir in os.listdir(os.path.join(embeddings_dir, channel_dir)):
            # remove empty video folders
            if not os.listdir(os.path.join(embeddings_dir, channel_dir, video_dir)):
                os.rmdir(os.path.join(embeddings_dir, channel_dir, video_dir))

        # remove empty channel folders
        if not os.listdir(os.path.join(embeddings_dir, channel_dir)):
            os.rmdir(os.path.join(embeddings_dir, channel_dir))


async def _add_embeddings(
    files: list[str], embeddings_dir: str, record_dir: str, client: AsyncQdrantClient
) -> None:
    """Add pointstructs to Qdrant"""
    asym_files = [x for x in files if x.startswith("asym")]
    asym_points = _files_to_pointstructs(asym_files, embeddings_dir)
    one = client.upload_points(collection_name="asym", points=asym_points, wait=True)
    add_embeddings_to_record(record_dir, asym_files)

    sym_files = [x for x in files if x.startswith("sym")]
    sym_points = _files_to_pointstructs(sym_files, embeddings_dir)
    two = client.upload_points(collection_name="sym", points=sym_points, wait=True)
    add_embeddings_to_record(record_dir, sym_files)

    await asyncio.gather(one, two)
    remove_added_npy_files(embeddings_dir, record_dir)


def _batch_list(files: list[str], batch_size: int) -> Generator[list[str], None, None]:
    for i in range(0, len(files), batch_size):
        yield files[i : i + batch_size]


async def add_to_qdrant(
    embeddings_dir: str,
    record_dir: str,
    client: AsyncQdrantClient,
) -> None:
    """Add embeddings in embeddings_dir to Qdrant"""
    already_added: set[str] = get_files_in_record(record_dir)

    os.makedirs(embeddings_dir, exist_ok=True)
    embeddings = [
        embedding
        for channel_dir in os.listdir(embeddings_dir)
        for video_dir in os.listdir(os.path.join(embeddings_dir, channel_dir))
        for embedding in os.listdir(
            os.path.join(embeddings_dir, channel_dir, video_dir)
        )
        if embedding.endswith(".npy") and embedding not in already_added
    ]
    already_added = set()  # clear record

    for batch in _batch_list(embeddings, 10_000):
        await _add_embeddings(batch, embeddings_dir, record_dir, client)
