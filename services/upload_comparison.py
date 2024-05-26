# In this file we define functions used to verify the added.txt on the server and the added.txt on the processor matches
import hashlib


def calculate_checksum(filename):
    sha256 = hashlib.sha256()
    with open(filename, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()
