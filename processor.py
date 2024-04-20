import os
import time

import dotenv
import numpy as np

root = os.path.dirname(os.path.abspath(__file__))
env_dir = os.path.join(root, "env")
env_files = os.listdir(env_dir)
if "prod.env" in env_files:
    dotenv.load_dotenv(os.path.join(env_dir, "prod.env"))
else:
    dotenv.load_dotenv(os.path.join(env_dir, "dev.env"))


from gRPC.client import send_embedding
from video_processing.src.main import main as processing

if __name__ == "__main__":
    while True:
        processing()
        # TODO: check which embeddings are not on the server

        for embedding in os.listdir("embeddings"):
            arr = np.load(f"embeddings/{embedding}")
            send_embedding(embedding.removesuffix(".npy"), arr)

        time.sleep(60 * 60)
