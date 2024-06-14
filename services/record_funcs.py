# In this file we define functions used to synchronize embeddings between the server and the client
import hashlib
import os


def _calculate_checksum(filename: str) -> str:
    """Calculate the sha256 checksum of a file"""
    sha256 = hashlib.sha256()
    with open(filename, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def calculate_record_file_checksums(record_dir: str) -> dict[str, str]:
    """Calculate the sha256 checksums of the record files"""
    checksums: dict[str, str] = {}
    for filename in os.listdir(record_dir):
        if (
            filename.startswith("added_")
            and filename.endswith(".txt")
            and not filename.endswith("_temp.txt")
        ):
            checksums[filename] = _calculate_checksum(
                os.path.join(record_dir, filename)
            )
    return checksums


def _calculate_checksum_of_str(string: str) -> str:
    """Calculate the sha256 checksum of a string"""
    sha256 = hashlib.sha256()
    sha256.update(string.encode("utf-8"))
    return sha256.hexdigest()


def _get_destination_record_file(embedding_name: str) -> str:
    """Get the destination record file based on the embedding name"""
    cleaned_name = embedding_name.removesuffix(".npy")
    sha256 = _calculate_checksum_of_str(cleaned_name)
    first_chars = sha256[:2]
    return f"added_{first_chars}.txt"


def add_embeddings_to_record(record_dir: str, embeddings: list[str]) -> None:
    """Use this when adding embeddings to the record file"""
    file_dict: dict[str, set[str]] = {}

    for embedding in embeddings:
        destination_file = _get_destination_record_file(embedding)
        if destination_file not in file_dict:
            file_dict[destination_file] = set()

        file_dict[destination_file].add(embedding)

    os.makedirs(record_dir, exist_ok=True)
    for dest_file, lines in file_dict.items():
        record_path = os.path.join(record_dir, dest_file)
        temp_record_path = os.path.join(record_dir, f"{dest_file}_temp")
        if not os.path.exists(record_path):
            with open(record_path, "w", encoding="utf-8") as f:
                f.write("\n".join(sorted(lines)))

            continue

        with open(record_path, "r", encoding="utf-8") as f:
            lines_in_file = {x.strip() for x in f.readlines() if x != "\n"}

        lines_to_add = sorted(lines | lines_in_file)
        with open(temp_record_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines_to_add))

        os.rename(temp_record_path, record_path)


def get_record_files() -> list[str]:
    """Get the list of record files"""
    possible_chars = "0123456789abcdef"
    return [f"added_{x}{y}.txt" for x in possible_chars for y in possible_chars]


def get_files_in_record(record_dir: str) -> set[str]:
    """Get the files in the record files"""
    on_record = set()
    os.makedirs(record_dir, exist_ok=True)
    for filename in os.listdir(record_dir):
        if filename.startswith("added_") and filename.endswith(".txt"):
            with open(os.path.join(record_dir, filename), "r", encoding="utf-8") as f:
                on_record.update({x.strip() for x in f.readlines()})

    return on_record


def split_record_file(record_file: str, new_record_dir: str) -> None:
    """Use this when initially splitting the record file into smaller chunks"""
    with open(record_file, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines()]

    new_files: dict[str, list[str]] = {}

    for line in lines:
        dest_file = _get_destination_record_file(line)
        if dest_file not in new_files:
            new_files[dest_file] = []

        new_files[dest_file].append(line)

    for dest_file, lines in new_files.items():
        new_files[dest_file] = sorted(lines)

    os.makedirs(new_record_dir, exist_ok=True)
    for dest_file, lines in new_files.items():
        with open(os.path.join(new_record_dir, dest_file), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
