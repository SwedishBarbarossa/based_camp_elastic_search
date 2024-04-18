import os
import re

import numpy as np
import numpy.typing as npt
import torch
import whisper
from pytube import Playlist
from pytube.exceptions import AgeRestrictedError
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def rip_audio_files(audio_dir: str):
    os.makedirs(audio_dir, exist_ok=True)

    # create age restricted file if it doesn't exist
    age_restricted_path = os.path.join(audio_dir, "age_restricted.txt")
    if not os.path.exists(age_restricted_path):
        open(age_restricted_path, "w")

    age_restricted = set()
    with open(age_restricted_path, "r") as f:
        for line in f:
            age_restricted.add(line.strip())

    files_in_dir = os.listdir(audio_dir)
    # create a list of audio files that already exist and is > 1 mb
    audio_files: set[str] = {
        file
        for file in files_in_dir
        if (
            file.endswith(".mp3")
            and os.path.getsize(os.path.join(audio_dir, file)) > 1000000
        )
    }

    p = Playlist(
        "https://www.youtube.com/watch?v=S9naHQAmrG4&list=PLg8R-fWRBuHEAFE0yEdkcjCij5PwvfHqH&pp=iAQB"
    )
    vids_in_playlist = list(p.videos)
    if len(vids_in_playlist) == len(age_restricted) + len(audio_files):
        print("All videos already downloaded")
        return

    for video in tqdm(vids_in_playlist):
        # save videos in /video/video_id folder
        video_id = video.video_id
        if f"{video_id}.mp3" in audio_files:
            continue

        try:
            audio_obj = video.streams.get_audio_only()
            if not audio_obj:
                raise ValueError("No audio stream found")

            audio_obj.download(
                output_path=audio_dir, filename=f"{video_id}.mp3", skip_existing=True
            )
        except AgeRestrictedError:
            # add to list to download later
            if video_id in age_restricted:
                continue

            with open(age_restricted_path, "a") as f:
                f.write(f"{video_id}\n")


def transcribe_audio_files(audio_dir: str, transcripts_dir: str):
    # Ensure output directory exists
    os.makedirs(transcripts_dir, exist_ok=True)

    # Load the Whisper model
    # You can choose another model size as needed
    has_cuda = torch.cuda.is_available()
    model = whisper.load_model(
        "medium", device="cuda" if has_cuda else "cpu", in_memory=has_cuda
    )

    # Process each MP3 file in the directory
    for filename in tqdm(os.listdir(audio_dir)):
        if not filename.endswith(".mp3"):
            continue

        file_path = os.path.join(audio_dir, filename)
        output_path = os.path.join(transcripts_dir, filename.replace(".mp3", ".txt"))

        # if transcript already exists and is longer than 10 lines, skip
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                if len(f.readlines()) > 10:
                    continue

        with open(output_path, "w") as output_file:
            result = model.transcribe(file_path, word_timestamps=True)
            for segment in result["segments"]:
                text = segment["text"]
                if not text.strip():
                    continue

                start = segment["start"]
                end = segment["end"]
                output_file.write(f"[{start:.3f},{end:.3f}] {text}\n")

    print("Transcriptions completed.")


def _split_long_segments(
    start: float, end: float, text: str, max_length: int
) -> list[tuple[float, float, str]]:
    if end - start < max_length:
        return [(start, end, text)]

    # Split the segment into multiple segments
    midpoint_time = (start + end) / 2
    split_text = text.split(" ")
    arr_midpoint = int(np.round(len(split_text) / 2))
    first_half = _split_long_segments(
        start, midpoint_time, " ".join(split_text[:arr_midpoint]), max_length
    )
    second_half = _split_long_segments(
        midpoint_time, end, " ".join(split_text[arr_midpoint:]), max_length
    )
    return first_half + second_half


def encode_transcripts(transcripts_dir: str, embeddings_dir: str):
    # Transcript line pattern
    pattern = re.compile(r"\[(\d+\.\d+),(\d+\.\d+)\]\s+(.*)")

    segments: list[tuple[str, str]] = []
    for filename in tqdm(os.listdir(transcripts_dir)):
        if not filename.endswith(".txt"):
            continue

        # get the time stamps from file
        file_path = os.path.join(transcripts_dir, filename)
        with open(file_path, "r") as transcript:
            # Step 1: Parse transcript lines
            parsed_transcript = [
                (
                    float(match.group(1)),  # Start time
                    float(match.group(2)),  # End time
                    match.group(3),  # Text
                )
                for line in transcript
                if (match := pattern.match(line))
            ]

            # Step 1.5: Split long segments
            transcript = []
            for start, end, text in parsed_transcript:
                transcript += _split_long_segments(start, end, text, 15)

            transcript.sort(key=lambda x: x[0])

            # Step 2: Determine the time windows based on the timestamps in the data
            start_time = min([entry[0] for entry in transcript])
            end_time = max([entry[1] for entry in transcript])
            windows = [
                (start, start + 30)
                for start in range(int(start_time), int(end_time), 15)
                if start + 15 <= end_time
            ]

            # Step 3: For each window, concatenate texts that fall within the window
            concatenated_texts = []
            for window_start, window_end in windows:
                texts = [
                    text
                    for start, end, text in transcript
                    if start >= window_start and end <= window_end
                ]
                concatenated_text = "\n".join(texts)
                if len(concatenated_text) < 50:
                    continue
                concatenated_texts.append((window_start, window_end, concatenated_text))

            # save the concatenated texts with the proper name
            for start, end, text in concatenated_texts:
                segment_name = f"{filename.removesuffix('.txt')} {start} {end}"
                segments.append((segment_name, text))

    # Load the SentenceTransformer model
    model = SentenceTransformer(
        "all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Encode the segments and save the .npy file
    os.makedirs(embeddings_dir, exist_ok=True)
    for segment_name, text in tqdm(segments):
        seg_path = os.path.join(embeddings_dir, segment_name + ".npy")
        if os.path.exists(seg_path):
            continue

        embeddings = model.encode(text)
        np.save(seg_path, embeddings)


def main():
    root = os.path.dirname(os.path.abspath(__file__)).split("video_processing")[0]
    audio_dir = os.path.join(root, "audio")
    transcripts_dir = os.path.join(root, "transcriptions")
    embeddings_dir = os.path.join(root, "embeddings")

    print("Ripping audio files...")
    rip_audio_files(audio_dir)
    print("Transcribing audio files...")
    transcribe_audio_files(audio_dir, transcripts_dir)
    print("Encoding transcripts...")
    encode_transcripts(transcripts_dir, embeddings_dir)


if __name__ == "__main__":
    main()
