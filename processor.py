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
from services.record_funcs import add_embeddings_to_record, split_record_file
from video_processing.src.main import main as processing

if __name__ == "__main__":
    RIP_AUDIO_FILES = True if os.environ["RIP_AUDIO_FILES"] == "1" else False
    TRANSCRIBE_AUDIO_FILES = (
        True if os.environ["TRANSCRIBE_AUDIO_FILES"] == "1" else False
    )
    ENCODE_TRANSCRIPTS = True if os.environ["ENCODE_TRANSCRIPTS"] == "1" else False
    ADDED_RECORD = os.path.join(root, "added.txt")
    ADDED_RECORD_DIR = os.path.join(root, "added")

    if os.path.exists(ADDED_RECORD):
        split_record_file(ADDED_RECORD, ADDED_RECORD_DIR)
        os.remove(ADDED_RECORD)

    while True:
        START_TIME = time.time()
        print(f"Starting at {time.ctime()}\n--------------------------------------\n")
        previously_uploaded_files = processing(
            record_dir=ADDED_RECORD_DIR,
            rip=RIP_AUDIO_FILES,
            transcribe=TRANSCRIBE_AUDIO_FILES,
            encode=ENCODE_TRANSCRIPTS,
        )
        current_files: list[tuple[str, str]] = [
            (file, os.path.join(root, "embeddings", channel, video_id, file))
            for channel in os.listdir(os.path.join(root, "embeddings"))
            for video_id in os.listdir(os.path.join(root, "embeddings", channel))
            for file in os.listdir(os.path.join(root, "embeddings", channel, video_id))
            if file.endswith(".npy")
        ]
        to_upload = [
            (file, path)
            for file, path in current_files
            if file not in previously_uploaded_files
        ]
        if to_upload:
            print(f"\nUploading {len(to_upload)} embeddings")
            embeddings: dict[str, npt.NDArray[np.float32]] = {}
            for filename, embedding_path in tqdm(
                to_upload, desc="Preparing embeddings"
            ):
                try:
                    arr = np.load(embedding_path)
                    embeddings[filename.removesuffix(".npy")] = arr
                except:  # delete corrupted files
                    os.remove(embedding_path)

            p_bar = tqdm(to_upload)

            def callback(res: str):
                cleaned_res = res[23:-6]
                split_str = cleaned_res.split(" ")
                split_str[0] = split_str[0].ljust(4)
                split_str[3] = split_str[3].rjust(5)
                split_str[4] = split_str[4].rjust(5)
                p_bar.set_postfix({"res": (" ").join(split_str)})
                p_bar.update(1)

            res = send_embeddings(embeddings, callback)
            p_bar.close()

            add_embeddings_to_record(ADDED_RECORD_DIR, [x[0] for x in to_upload])

        print(f"Ending at {time.ctime()}\n--------------------------------------\n")
        if not RIP_AUDIO_FILES:
            break

        # sleep until START_TIME + 3 hours
        sleep_time = (START_TIME + 6 * 60 * 60) - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
