import os
import re
import subprocess
from typing import TypedDict

import numpy as np
import requests
import torch
import whisper
import yaml
from pytube import Channel, Playlist, YouTube
from pytube.exceptions import AgeRestrictedError
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class ConfigLink(TypedDict):
    link: str


class CreatorConfig(TypedDict):
    youtube_playlists: list[ConfigLink] | None
    youtube_channels: list[ConfigLink] | None
    youtube_videos: list[ConfigLink] | None


def load_config(config_path: str) -> dict[str, CreatorConfig]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        init_config = yaml.safe_load(file)

    # format config
    config: dict[str, CreatorConfig] = init_config["creators"]
    keys = config.keys()
    for key in keys:
        for link_type, val in config[key].items():
            if isinstance(val, dict):
                config[key][link_type] = [val]

    return config


def rip_audio_files(
    audio_dir: str,
    transcripts_dir: str,
    age_restricted_path: str,
    config: CreatorConfig,
):
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(transcripts_dir, exist_ok=True)

    # create age restricted file if it doesn't exist
    if not os.path.exists(age_restricted_path):
        open(age_restricted_path, "w")

    age_restricted = set()
    with open(age_restricted_path, "r") as f:
        for line in f:
            age_restricted.add(line.strip())

    # create a list of audio files that already exist and is > 1 mb
    audio_files: set[str] = {
        file.removesuffix(".mp3")
        for file in os.listdir(audio_dir)
        if (
            file.endswith(".mp3")
            and os.path.getsize(os.path.join(audio_dir, file)) > 1000000
        )
    }

    # create a list of transcripts that already exist and are > 1 kb
    transcript_files: set[str] = {
        file.removesuffix(".txt")
        for file in os.listdir(transcripts_dir)
        if file.endswith(".txt")
        and os.path.getsize(os.path.join(transcripts_dir, file)) > 1000
    }

    skip_files = audio_files.union(transcript_files)

    vids: list[YouTube] = []

    playlists = config.get("youtube_playlists")
    if playlists:
        for playlist in playlists:
            link = playlist["link"]
            playlist = Playlist(link)
            vids.extend(list(playlist.videos))

    channels = config.get("youtube_channels")
    if channels:
        for channel in channels:
            link = channel["link"]
            channel = Channel(link)
            vids.extend(list(channel.videos))

    videos = config.get("youtube_videos")
    if videos:
        for video in videos:
            link = video["link"]
            video = YouTube(link)
            vids.append(video)

    # remove duplicate videos
    vids_dict: dict[str, YouTube] = {
        video.video_id: video for video in vids if video.video_id not in skip_files
    }
    vids = list(vids_dict.values())
    vids_dict = {}
    if not vids:
        return

    for video in tqdm(vids):
        # save video audio
        video_id = video.video_id
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


def _get_media_duration(file_path):
    try:
        # Run ffprobe to get media information
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                file_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        duration_seconds = float(result.stdout)

        # Convert duration to hh:mm:ss format
        hours = int(duration_seconds / 3600)
        minutes = int((duration_seconds % 3600) / 60)
        seconds = int(duration_seconds % 60)

        # Format the time string
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return time_str
    except subprocess.CalledProcessError as e:
        print("Error running ffprobe:", e)
        return "NN:NN:NN"


def transcribe_audio_files(audio_dir: str, transcripts_dir: str):
    # Load the Whisper model
    # You can choose another model size as needed
    has_cuda = torch.cuda.is_available()
    model = whisper.load_model(
        "medium", device="cuda" if has_cuda else "cpu", in_memory=has_cuda
    )

    # create a list of audio files that already exist and is > 1 mb
    audio_files: set[str] = {
        file.removesuffix(".mp3")
        for file in os.listdir(audio_dir)
        if (
            file.endswith(".mp3")
            and os.path.getsize(os.path.join(audio_dir, file)) > 10**6
        )
    }

    # create a list of transcripts that already exist and are > 1 kb
    transcript_files: set[str] = {
        file.removesuffix(".txt")
        for file in os.listdir(transcripts_dir)
        if file.endswith(".txt")
        and os.path.getsize(os.path.join(transcripts_dir, file)) > 10**3
    }

    files_to_transcribe = sorted(audio_files - transcript_files)
    if not files_to_transcribe:
        return

    # Process each MP3 files in the directory
    p_bar = tqdm(files_to_transcribe)
    for filename in p_bar:
        duration = _get_media_duration(os.path.join(audio_dir, filename + ".mp3"))
        p_bar.set_description(f"{filename} [{duration}]")
        file_path = os.path.join(audio_dir, filename + ".mp3")
        output_path = os.path.join(transcripts_dir, filename + ".txt")

        # if transcript already exists and is longer than 10 lines, skip
        skip = False
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                if len(f.readlines()) > 10:
                    skip = True

        if not skip:
            with open(output_path, "w") as output_file:
                result = model.transcribe(
                    file_path, word_timestamps=True, language="en"
                )
                for segment in result["segments"]:
                    text = segment["text"]  # type: ignore
                    if not text.strip():
                        continue

                    start = segment["start"]  # type: ignore
                    end = segment["end"]  # type: ignore
                    output_file.write(f"[{start:.3f},{end:.3f}] {text}\n")

        # delete the audio file
        os.remove(file_path)

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


def encode_transcripts(transcripts_dir: str, embeddings_dir: str, channel: str):
    # Transcript line pattern
    pattern = re.compile(r"\[(\d+\.\d+),(\d+\.\d+)\]\s+(.*)")

    segments: list[tuple[str, str]] = []
    files = os.listdir(transcripts_dir)
    if not files:
        return

    p_bar = tqdm(files)
    for filename in p_bar:
        if not filename.endswith(".txt"):
            continue

        p_bar.set_description(filename.removesuffix(".txt"))
        # --- get the time stamps from file ---
        # Step 1: Parse transcript lines
        file_path = os.path.join(transcripts_dir, filename)
        with open(file_path, "r") as f:
            lines = f.readlines()
            if len(lines) < 2:
                continue

            parsed_transcript: list[tuple[float, float, str]] = [
                (
                    float(match.group(1)),  # Start time
                    float(match.group(2)),  # End time
                    match.group(3),  # Text
                )
                for line in lines
                if (match := pattern.match(line))
            ]

        # Step 1.5: Split long segments
        transcript: list[tuple[float, float, str]] = []
        for start, end, text in parsed_transcript:
            transcript.extend(_split_long_segments(start, end, text, 15))

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
        concatenated_texts: list[tuple[float, float, str]] = []
        for window_start, window_end in windows:
            texts = [
                text.strip()
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
    SYMMETRIC_MODEL = SentenceTransformer(
        "all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu"
    )
    ASYMMETRIC_MODEL = SentenceTransformer(
        "msmarco-MiniLM-L-6-v3", device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Encode the segments and save the .npy file
    os.makedirs(embeddings_dir, exist_ok=True)
    for segment_name, text in tqdm(segments):
        # Symmetric encoding
        sym_seg_path = os.path.join(
            embeddings_dir, "sym " + f"{channel} " + segment_name + ".npy"
        )
        if not os.path.exists(sym_seg_path):
            embeddings = SYMMETRIC_MODEL.encode(text)
            np.save(sym_seg_path, embeddings)

        # Asymmetric encoding
        asym_seg_path = os.path.join(
            embeddings_dir, "asym " + f"{channel} " + segment_name + ".npy"
        )
        if not os.path.exists(asym_seg_path):
            embeddings = ASYMMETRIC_MODEL.encode(text)
            np.save(asym_seg_path, embeddings)


def get_uploaded_files(config_path: str) -> list[str]:
    with open(config_path, "r") as f:
        conf = yaml.safe_load(f)

    address: str = conf["address"]
    file = requests.get(f"{address}/added.txt")
    return file.text.split("\n")


def main() -> list[str]:
    root = os.path.dirname(os.path.abspath(__file__)).split("video_processing")[0]
    audio_dir = os.path.join(root, "audio")
    transcripts_dir = os.path.join(root, "transcriptions")
    embeddings_dir = os.path.join(root, "embeddings")
    config_path = os.path.join(
        root, "video_processing", "config", "video_processing_config.yaml"
    )
    age_restricted_path = os.path.join(audio_dir, "age_restricted.txt")

    # load config
    config = load_config(config_path)
    for i, (creator, creator_conf) in enumerate(config.items()):
        if i != 0:
            print("----------------------------------------------------------\n")
        creator_audio_dir = os.path.join(audio_dir, creator)
        creator_transcripts_dir = os.path.join(transcripts_dir, creator)
        print(f"Ripping {creator} audio files...")
        rip_audio_files(
            creator_audio_dir,
            creator_transcripts_dir,
            age_restricted_path,
            creator_conf,
        )
        print(f"Transcribing {creator} audio files...")
        transcribe_audio_files(creator_audio_dir, creator_transcripts_dir)
        print(f"Encoding {creator} transcripts...")
        encode_transcripts(creator_transcripts_dir, embeddings_dir, creator)

    # return the previously uploaded files on the server
    return get_uploaded_files(config_path)


if __name__ == "__main__":
    main()
