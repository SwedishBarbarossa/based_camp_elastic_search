import os
import time

import dotenv
import numpy as np
from tqdm import tqdm

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
        print(f"Starting at {time.ctime()}\n--------------------------------------\n")
        previously_uploaded_files = processing()
        current_files = os.listdir("embeddings")
        to_upload = list(set(current_files) - set(previously_uploaded_files))
        if to_upload:
            p_bar = tqdm(to_upload)
            for embedding in p_bar:
                arr = np.load(f"embeddings/{embedding}")
                res = send_embedding(embedding.removesuffix(".npy"), arr)
                p_bar.set_postfix({"res": res})

        print(f"Ending at {time.ctime()}\n--------------------------------------\n")
        time.sleep(60 * 60)
