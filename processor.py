import os
import time

import dotenv
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

root = os.path.dirname(os.path.abspath(__file__))
env_dir = os.path.join(root, "env")
env_files = os.listdir(env_dir)
if "prod.env" in env_files:
    dotenv.load_dotenv(os.path.join(env_dir, "prod.env"))
else:
    dotenv.load_dotenv(os.path.join(env_dir, "dev.env"))


from gRPC.client import send_embeddings
from video_processing.src.main import main as processing

if __name__ == "__main__":
    RIP_AUDIO_FILES = True if int(os.environ["RIP_AUDIO_FILES"]) else False
    ADDED_RECORD = os.path.join(root, "added.txt")
    while True:
        START_TIME = time.time()
        print(f"Starting at {time.ctime()}\n--------------------------------------\n")
        previously_uploaded_files = processing(
            rip=RIP_AUDIO_FILES, added_record=ADDED_RECORD
        )
        current_files = [x for x in os.listdir("embeddings") if x.endswith(".npy")]
        to_upload = list(set(current_files) - set(previously_uploaded_files))
        if to_upload:
            embeddings: dict[str, npt.NDArray[np.float32]] = {}
            for embedding in tqdm(to_upload, desc="Preparing embeddings"):
                try:
                    arr = np.load(f"embeddings/{embedding}")
                except:  # delete corrupted files
                    os.remove(f"embeddings/{embedding}")
                embeddings[embedding.removesuffix(".npy")] = arr

            p_bar = tqdm(to_upload)

            def callback(res: str):
                p_bar.set_postfix({"res": res})
                p_bar.update(1)

            res = send_embeddings(embeddings, callback)
            p_bar.close()

            with open(ADDED_RECORD, "a", encoding="utf-8") as f:
                f.writelines(f"{x}\n" for x in sorted(to_upload))

        print(f"Ending at {time.ctime()}\n--------------------------------------\n")
        if not RIP_AUDIO_FILES:
            break

        # sleep until START_TIME + 3 hours
        time.sleep((START_TIME + 3 * 60 * 60) - time.time())
