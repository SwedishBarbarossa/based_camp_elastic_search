import aiosqlite
import numpy as np
import numpy.typing as npt
from qdrant_client import AsyncQdrantClient, models
from sentence_transformers import SentenceTransformer

from .qdrant_funcs import q_types


# Function to initialize the database
async def init_db(db_path: str) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS query (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phrase TEXT NOT NULL,
                q_type TEXT NOT NULL,
                hits INTEGER DEFAULT 0,
                UNIQUE(phrase, q_type)
            )
            """
        )
        await db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_phrase_q_type ON query (phrase, q_type)
            """
        )
        await db.commit()


# Function to fetch a row with an exact matching phrase and q_type
async def fetch_phrase_id(phrase: str, q_type: q_types, db_path: str) -> int | None:
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT id FROM query WHERE phrase = ? AND q_type = ?",
            (phrase, q_type),
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


# Function to increment hits of a phrase
async def increment_hits(phrase: str, q_type: q_types, db_path: str) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE query SET hits = hits + 1 WHERE phrase = ? AND q_type = ?",
            (phrase, q_type),
        )
        await db.commit()


def create_point(vector: npt.NDArray[np.float32], pk: int) -> models.PointStruct:
    return models.PointStruct(id=pk, vector=vector)  # type: ignore


# Function to insert a phrase and return the primary key of the newly created row
async def insert_phrase(
    phrase: str,
    q_type: str,
    qdrant_client: AsyncQdrantClient,
    encoding_model: SentenceTransformer,
    db_path: str,
):
    vector = encoding_model.encode(phrase)
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "INSERT INTO query (phrase, q_type) VALUES (?, ?)",
            (phrase, q_type),
        )
        await db.commit()
        pk = cursor.lastrowid

    if not isinstance(pk, int):
        raise ValueError(f"Invalid primary key: {pk}")

    await qdrant_client.upload_points(
        collection_name=f"queries_{q_type}",
        points=[create_point(vector, pk)],  # type: ignore
    )


async def fetch_phrase_vector(
    pk: int, q_type: q_types, qdrant_client: AsyncQdrantClient
) -> npt.NDArray[np.float32] | None:
    points = await qdrant_client.retrieve(
        collection_name=f"queries_{q_type}", ids=[pk], with_vectors=True
    )
    if not points:
        return None

    return points[0].vector  # type: ignore
