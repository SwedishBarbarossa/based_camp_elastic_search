import os


def remove_added_npy_files(embeddings_dir: str, record: str):
    os.makedirs(embeddings_dir, exist_ok=True)
    npy_files = os.listdir(embeddings_dir)

    with open(record, "r", encoding="utf-8") as f:
        file_record = {x.strip() for x in f.readlines()}

    for npy_file in npy_files:
        if npy_file in file_record:
            os.remove(os.path.join(embeddings_dir, npy_file))
