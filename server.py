import os
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import uvicorn
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from gRPC.server import serve as grpc_server
from web_server.src.server_funcs import build_index, get_index

ROOT = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(ROOT, "server_embeddings")

WORK_DIR = os.path.join(ROOT, "web_server", "src")
INDEX_NAME = os.path.join(WORK_DIR, "saved", "index.ann")
MAP_NAME = os.path.join(WORK_DIR, "saved", "index_value_map")

STATIC_PATH = os.path.join(WORK_DIR, "static")
INDEX_PATH = os.path.join(STATIC_PATH, "index.html")
STYLE_PATH = os.path.join(STATIC_PATH, "style.css")

DIM = 384
REBUILD_COOLDOWN = timedelta(minutes=1)
REBUILD_SLEEP = 600

fapi_app = FastAPI()
last_grpc_call_time: datetime | None = datetime.now()
last_index_rebuild: datetime | None = None
lock = threading.Lock()

global index
index = None
global index_map
index_map = {}

# Load the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


def rebuild_annoy_index():
    global last_grpc_call_time, index, index_map, last_index_rebuild
    wait_seconds = np.inf
    while True:
        with lock:
            if last_grpc_call_time is not None:
                wait_time = (last_grpc_call_time + REBUILD_COOLDOWN) - datetime.now()
                wait_seconds = max(0, wait_time.total_seconds())
            else:
                wait_seconds = REBUILD_SLEEP

        if wait_seconds == 0:
            # Rebuild the Annoy index here
            print("Rebuilding Annoy index...")
            index, index_map = build_index(DIM, INDEX_NAME, MAP_NAME, EMBEDDINGS_DIR)
            last_index_rebuild = datetime.now()
            print("Annoy index rebuilt.")

            # Reset the timer
            with lock:
                last_grpc_call_time = None

        time.sleep(wait_seconds)


@fapi_app.get("/search/")
async def search(query: str | None = None):
    if not query:
        query = "total fertility rate collapse"

    if len(query) > 100:
        query = query[:100]

    global index, index_map
    if index is None:
        print("get index")
        index, index_map = get_index(DIM, INDEX_NAME, MAP_NAME, EMBEDDINGS_DIR)

    search_vector = model.encode(query)
    indices, distances = index.get_nns_by_vector(
        search_vector, 100, search_k=10, include_distances=True
    )
    results = [index_map[index] for index in indices]
    return {"query": query, "results": zip(results, distances)}


# TODO: add api for current embeddings in folder

# TODO: add api for latest index rebuild


def last_grpc_call_time_callback(*args, **kwargs):
    global last_grpc_call_time
    last_grpc_call_time = datetime.now()


# Start the rebuild index thread
rebuild_thread = threading.Thread(target=rebuild_annoy_index, daemon=True)
rebuild_thread.start()

# Start the gRPC server thread
grpc_thread = threading.Thread(
    target=grpc_server,
    daemon=True,
    kwargs={"embeddings_dir": EMBEDDINGS_DIR, "callback": last_grpc_call_time_callback},
)
grpc_thread.start()

# Serve the app with uvicorn
uvicorn.run(fapi_app, host="0.0.0.0", port=8000)
