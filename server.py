import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import numpy as np
import numpy.typing as npt
import uvicorn
from fastapi import FastAPI, Response
from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer

from gRPC.server import serve as grpc_server
from services.record_funcs import calculate_record_file_checksums, split_record_file
from web_server.src.db_handler import (
    fetch_phrase_id,
    fetch_phrase_vector,
    increment_hits,
    init_db,
    insert_phrase,
)
from web_server.src.qdrant_funcs import (
    ModelDims,
    add_to_qdrant,
    initialize_qdrant,
    q_types,
    query_qdrant,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(ROOT, "server_embeddings")
DB_PATH = os.path.join(ROOT, "queries", "queries.db")

WORK_DIR = os.path.join(ROOT, "web_server", "src")
QDRANT_DIR = os.path.join(WORK_DIR, "qdrant_data")

STATIC_PATH = os.path.join(WORK_DIR, "static")
INDEX_PATH = os.path.join(STATIC_PATH, "index.html")
STYLE_PATH = os.path.join(STATIC_PATH, "style.css")
RECORD_PATH = os.path.join(STATIC_PATH, "added.txt")
RECORD_DIR = os.path.join(STATIC_PATH, "added")

SYM_DIM = 384
ASYM_DIM = 384
MODEL_DIMS: ModelDims = {"asymmetric": ASYM_DIM, "symmetric": SYM_DIM}

REBUILD_COOLDOWN = timedelta(seconds=10)
REBUILD_SLEEP = 60 * 5
SHOW_STRIPE = os.getenv("SHOW_STRIPE") != "0" and os.getenv("SHOW_STRIPE") != None
STRIPE_URL = os.getenv("STRIPE_URL")
DONATION_HOOK_MESSAGE = os.getenv("DONATION_HOOK_MESSAGE")

EXISTING_CHANNELS = [
    "based_camp",
    "balaji_srinivasan",
    "hormozis",
    "y_combinator",
    "charter_cities_institute",
    "startup_societies_foundation",
    "free_cities_foundation",
    "james_lindsay",
    "jordan_b_peterson",
    "chris_williamson",
    "numberphile",
    "computerphile",
    "ted",
    "ryan_chapman",
    "veritasium",
    "perun",
    "lennys_podcast",
    "big_think",
    "freethink",
    "diary_of_a_ceo",
    "iai",
    "phil_illy",
    "the_leftist_cooks",
]

last_grpc_call_time: datetime | None = datetime.now()
last_index_rebuild: datetime | None = None
lock = threading.Lock()

SYMMETRIC_MODEL: SentenceTransformer
ASYMMETCIC_MODEL: SentenceTransformer


# Start the Qdrant client
qdant_client = AsyncQdrantClient(host="qdrant", grpc_port=6334, prefer_grpc=True)


def last_grpc_call_time_callback(*args, **kwargs):
    global last_grpc_call_time
    last_grpc_call_time = datetime.now()


def update_qdrant():
    """
    Periodically updates the Qdrant index with new embeddings, ensuring updates
    occur only after a cooldown period. Handles adding new data and cleaning up
    old embeddings in a loop.
    """
    global last_grpc_call_time, last_index_rebuild
    wait_seconds = np.inf

    if os.path.exists(RECORD_PATH):
        split_record_file(RECORD_PATH, RECORD_DIR)
        os.remove(RECORD_PATH)

    while True:
        with lock:
            if last_grpc_call_time is not None:
                wait_time = (last_grpc_call_time + REBUILD_COOLDOWN) - datetime.now()
                wait_seconds = max(0, wait_time.total_seconds())
            else:
                wait_seconds = REBUILD_SLEEP

        if wait_seconds == 0:
            print("Updating Qdrant...")
            asyncio.run(
                add_to_qdrant(
                    EMBEDDINGS_DIR,
                    RECORD_DIR,
                    qdant_client,
                )
            )
            print("Qdrant updated.")

            # Reset the timer
            with lock:
                last_index_rebuild = datetime.now()
                last_grpc_call_time = None

        time.sleep(wait_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the SentenceTransformer model
    global SYMMETRIC_MODEL, ASYMMETCIC_MODEL
    SYMMETRIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    ASYMMETCIC_MODEL = SentenceTransformer("msmarco-MiniLM-L-6-v3")
    sqdb = init_db(DB_PATH)
    qdrant = initialize_qdrant(MODEL_DIMS, qdant_client)

    # Start the gRPC server thread
    grpc_thread = threading.Thread(
        target=grpc_server,
        daemon=True,
        kwargs={
            "embeddings_dir": EMBEDDINGS_DIR,
            "callback": last_grpc_call_time_callback,
        },
    )
    grpc_thread.start()

    # Start the rebuild index thread
    await asyncio.gather(sqdb, qdrant)
    rebuild_thread = threading.Thread(target=update_qdrant, daemon=True)
    rebuild_thread.start()

    yield  # Start the web server
    # Clean up before shutting down


fapi_app = FastAPI(lifespan=lifespan)


if SHOW_STRIPE:

    @fapi_app.get("/donate_url/")
    async def donate_url():
        return {"donate_url": STRIPE_URL, "message": DONATION_HOOK_MESSAGE}

else:

    @fapi_app.get("/donate_url/")
    async def donate_url_not_implemented():
        return Response(status_code=501)


@fapi_app.get("/search/")
async def search(
    query: str | None = None,
    channels: str | None = None,
    q_type: q_types = "sym",
):
    parsed_channels = ["based_camp"]
    if not query:
        query = "How does text retrieval with embeddings work?"
        q_type = "asym"

    if channels:
        parsed_channels = [x for x in channels.split(",") if x in EXISTING_CHANNELS]

    if q_type not in ["sym", "asym"]:
        q_type = "sym"

    if len(query) > 100:
        query = query[:100]

    query = query.strip().lower()

    search_vector: npt.NDArray[np.float32] | None = None
    search_id = await fetch_phrase_id(query, q_type, DB_PATH)
    if search_id is not None:
        search_vector = await fetch_phrase_vector(search_id, q_type, qdant_client)

    if search_vector is None:
        model = SYMMETRIC_MODEL if q_type == "sym" else ASYMMETCIC_MODEL
        search_vector = model.encode(query)  # type: ignore
        nil = insert_phrase(query, q_type, qdant_client, model, DB_PATH)

    else:
        nil = increment_hits(query, q_type, DB_PATH)

    if search_vector is None:
        return Response(status_code=500)

    results = await query_qdrant(parsed_channels, search_vector, qdant_client, q_type)
    return {
        "channels": parsed_channels,
        "query": query,
        "q_type": q_type,
        "results": results,
    }


@fapi_app.get("/latest_rebuild")
async def latest_rebuild():
    global last_index_rebuild
    if not last_index_rebuild:
        return {"last_rebuild": "never"}
    return {"last_rebuild": last_index_rebuild}


@fapi_app.get("/added_hash")
async def added_hash():
    return calculate_record_file_checksums(RECORD_DIR)


# Serve the app with uvicorn
uvicorn.run(fapi_app, host="0.0.0.0", port=8000)
