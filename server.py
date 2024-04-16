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

root = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.join(root, "web_server", "src")
index_name = os.path.join(work_dir, "saved", "index.ann")
map_name = os.path.join(work_dir, "saved", "index_value_map")
embeddings_dir = os.path.join(root, "server_embeddings")
global index
index = None
global index_map
index_map = {}

dim = 384
rebuild_wait = timedelta(minutes=1)  # 5)
rebuild_sleep = 10  # 600


static = os.path.join(work_dir, "static")
index_path = os.path.join(static, "index.html")
style_path = os.path.join(static, "style.css")
fapi_app = FastAPI()
last_grpc_call_time: datetime | None = datetime.now()
lock = threading.Lock()


# Load the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


def rebuild_annoy_index():
    global last_grpc_call_time, index, index_map
    wait_seconds = np.inf
    while True:
        with lock:
            if last_grpc_call_time is not None:
                wait_time = (last_grpc_call_time + rebuild_wait) - datetime.now()
                wait_seconds = max(0, wait_time.total_seconds())
            else:
                wait_seconds = np.inf

        if wait_seconds == 0:
            # Rebuild the Annoy index here
            print("Rebuilding Annoy index...")
            index, index_map = build_index(dim, index_name, map_name, embeddings_dir)

            print("Annoy index rebuilt.")

            # Reset the timer
            with lock:
                last_grpc_call_time = None

        time.sleep(rebuild_sleep)


@fapi_app.get("/search/")
async def search(query: str | None = None):
    if not query:
        query = "total fertility rate collapse"

    if len(query) > 100:
        query = query[:100]

    # index, index_map = get_index(dim, index_name, map_name, embeddings_dir)
    global index, index_map
    if index is None:
        print("get index")
        index, index_map = get_index(dim, index_name, map_name, embeddings_dir)

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

# Start the gRPC server
grpc_thread = threading.Thread(
    target=grpc_server,
    daemon=True,
    kwargs={"embeddings_dir": embeddings_dir, "callback": last_grpc_call_time_callback},
)
grpc_thread.start()

# Serve the app with uvicorn
uvicorn.run(fapi_app, host="0.0.0.0", port=8000)
