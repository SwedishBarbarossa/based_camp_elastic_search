import os
import re
import subprocess
import time
from traceback import print_tb
from typing import Callable, Generator, Iterable, TypedDict

import numpy as np
import pytube
import pytube.exceptions
import requests
import torch
import whisperx
import yaml
import yt_dlp
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from whisperx.alignment import AlignedTranscriptionResult

from services.record_funcs import calculate_record_file_checksums, get_files_in_record


class ConfigLink(TypedDict):
    link: str


class CreatorConfig(TypedDict):
    youtube_playlists: list[ConfigLink] | None
    youtube_channels: list[ConfigLink] | None
    youtube_videos: list[ConfigLink] | None


def load_config(config_path: str) -> dict[str, CreatorConfig]:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as file:
        init_config = yaml.safe_load(file)

    # format config
    config: dict[str, CreatorConfig] = init_config["creators"]
    keys = config.keys()
    for key in keys:
        for link_type, val in config[key].items():
            if isinstance(val, dict):
                config[key][link_type] = [val]

    return config


def verify_audio_file(audio_file_path: str) -> bool:
    """Verify that the audio file is greater than 1 MB"""
    return (
        audio_file_path.endswith(".mp3")
        and os.path.exists(audio_file_path)
        and os.path.getsize(audio_file_path) > 10**6
    )


def verify_transcript_file(transcript_file_path: str) -> bool:
    """Verify that the transcript file is greater than 1 KB"""
    return (
        transcript_file_path.endswith(".txt")
        and os.path.exists(transcript_file_path)
        and os.path.getsize(transcript_file_path) > 10**3
    )


def clean_segment_text(text: str) -> str:
    """Clean transcription segment text"""
    base_clean = text.replace("\n", " ").replace("¶", " ").replace("¦", " ").strip()
    while "  " in base_clean:
        base_clean = base_clean.replace("  ", " ")
    return base_clean if base_clean != "." else ""


def download_audio_with_yt_dlp(video_url: str, audio_dir: str) -> bool:
    """Backup ripper when pytube doesn't work"""
    # Extract the video ID from the URL
    video_id = video_url.split("v=")[-1]
    # Define the output template using the video ID
    output_template = f"{video_id}.%(ext)s"
    output_path_template = os.path.join(audio_dir, output_template)
    output_path = os.path.join(audio_dir, f"{video_id}.mp3")

    # Options for yt-dlp
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "verbose": False,
        "quiet": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        return os.path.exists(output_path)
    except Exception:
        return False


def rip_audio_files(
    audio_dir: str,
    transcripts_dir: str,
    age_restricted_path: str,
    shorts_path: str,
    config: CreatorConfig,
    end_time: float,
):
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(transcripts_dir, exist_ok=True)

    # create age restricted file and shorts file if they doesn't exist
    if not os.path.exists(age_restricted_path):
        with open(age_restricted_path, "w", encoding="utf-8") as f:
            f.write("")

    if not os.path.exists(shorts_path):
        with open(shorts_path, "w", encoding="utf-8") as f:
            f.write("")

    age_restricted: set[str] = set()
    with open(age_restricted_path, "r", encoding="utf-8") as f:
        age_restricted = {x.strip() for x in f.readlines()}

    all_audio = {file for file in os.listdir(audio_dir) if file.endswith(".mp3")}
    current_audio_count = len(all_audio)

    # get small audio files
    small_audio = {
        file
        for file in all_audio
        if not verify_audio_file(os.path.join(audio_dir, file))
    }

    # add small audio to shorts file
    if small_audio:
        with open(shorts_path, "a", encoding="utf-8") as f:
            f.write("\n".join([x.removesuffix(".mp3") for x in small_audio]) + "\n")

    # get files in shorts file
    short_video_files: set[str] = set()
    with open(shorts_path, "r", encoding="utf-8") as f:
        short_video_files = {x.strip() for x in f.readlines()}

    # delete audio that are in the shorts file
    for file in short_video_files:
        path = os.path.join(audio_dir, file + ".mp3")
        if os.path.exists(path):
            os.remove(path)

    # create a list of audio files that already exist and is > 1 mb
    existing_audio: set[str] = {file.removesuffix(".mp3") for file in all_audio}

    # create a list of transcripts that already exist and are > 1 kb
    transcript_files: set[str] = {
        file.removesuffix(".txt")
        for file in os.listdir(transcripts_dir)
        if verify_transcript_file(os.path.join(transcripts_dir, file))
    }

    skip_files = existing_audio.union(
        transcript_files, short_video_files, age_restricted
    )

    vid_sources = []

    playlists = config.get("youtube_playlists")
    if playlists:
        try:
            for playlist in playlists:
                link = playlist["link"]
                playlist = pytube.Playlist(link, use_oauth=True, allow_oauth_cache=True)
                vid_sources.append(playlist.videos)
                time.sleep(2)
        except Exception as e:
            print("Failed to load playlist")
            raise e

    channels = config.get("youtube_channels")
    if channels:
        try:
            for channel in channels:
                link = channel["link"]
                channel = pytube.Channel(link, use_oauth=True, allow_oauth_cache=True)
                vid_sources.append(channel.videos)
                time.sleep(2)
        except Exception as e:
            print("Failed to load channel")
            raise e

    videos = config.get("youtube_videos")
    if videos:
        video_list: list[pytube.YouTube] = []
        try:
            for video in videos:
                link = video["link"]
                video = pytube.YouTube(link, use_oauth=True, allow_oauth_cache=True)
                video_list.append(video)
                time.sleep(2)
        except Exception as e:
            print("Failed to load video")
            raise e
        vid_sources.append(video_list)

    def video_generator(
        sources: list[Iterable[pytube.YouTube]], skip: set[str]
    ) -> Generator[pytube.YouTube, None, None]:
        returned: set[str] = skip
        for video_li in sources:
            success = False
            while not success:
                try:
                    for vid in video_li:
                        time.sleep(0.5)
                        if vid.video_id in returned:
                            continue

                        returned.add(vid.video_id)
                        yield vid
                    success = True
                except Exception as e:
                    print(e)
                    time.sleep(5)

    pbar = tqdm(video_generator(vid_sources, skip_files))
    for video in pbar:
        if time.time() > end_time:
            break

        # save video audio
        video_id = video.video_id
        try:
            audio_obj = video.streams.get_audio_only()
            if not audio_obj:
                raise ValueError("No audio stream found")

            audio_obj.download(output_path=audio_dir, filename=f"{video_id}.mp3")
            pbar.set_description(f"Downloaded {video_id}")
            time.sleep(10)
        except pytube.exceptions.AgeRestrictedError:
            if download_audio_with_yt_dlp(video.watch_url, audio_dir):
                pbar.set_description(f"yt-dlp - Downloaded {video_id}")
                time.sleep(10)
                continue

            # add to list to download later
            if video_id in age_restricted:
                continue

            with open(age_restricted_path, "a", encoding="utf-8") as f:
                f.write(f"{video_id}\n")

            pbar.set_description(f"Age restricted {video_id}")
        except pytube.exceptions.LiveStreamError:
            pbar.set_description(f"Live stream {video_id}")
            continue
        except pytube.exceptions.MembersOnly:
            pbar.set_description(f"Members only {video_id}")
            continue
        except Exception as e:
            if download_audio_with_yt_dlp(video.watch_url, audio_dir):
                pbar.set_description(f"yt-dlp - Downloaded {video_id}")
                time.sleep(10)
                continue

            print(f"Error downloading {video_id}")
            raise e


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


def transcribe_audio_files(
    audio_dir: str,
    transcripts_dir: str,
    get_segments: Callable[[str], AlignedTranscriptionResult],
    end_time: float,
):
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(transcripts_dir, exist_ok=True)

    # create a list of audio files that already exist and is > 1 mb
    audio_files: set[str] = {
        file.removesuffix(".mp3")
        for file in os.listdir(audio_dir)
        if verify_audio_file(os.path.join(audio_dir, file))
    }

    # create a list of transcripts that already exist and are > 1 kb
    transcript_files: set[str] = {
        file.removesuffix(".txt")
        for file in os.listdir(transcripts_dir)
        if verify_transcript_file(os.path.join(transcripts_dir, file))
    }

    files_to_transcribe = sorted(audio_files - transcript_files)
    if not files_to_transcribe:
        return

    # Process each MP3 files in the directory
    p_bar = tqdm(files_to_transcribe)
    for filename in p_bar:
        if time.time() > end_time:
            break

        audio_path = os.path.join(audio_dir, filename + ".mp3")
        output_path = os.path.join(transcripts_dir, filename + ".txt")

        # if transcript already exists and is longer than 10 lines, skip
        skip = False
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                if len(f.readlines()) > 10:
                    skip = True

        if skip:
            # delete the audio file
            os.remove(audio_path)
            continue

        duration = _get_media_duration(os.path.join(audio_dir, filename + ".mp3"))
        p_bar.set_description(f"{filename} [{duration}]")

        try:
            result = get_segments(audio_path)
        except Exception as e:
            print(f"Error transcribing {filename}: {e}")
            time.sleep(5)
            continue

        cleaned_segments = [
            segment
            for segment in [
                (segment["start"], segment["end"], clean_segment_text(segment["text"]))
                for segment in result["segments"]
            ]
            if segment[2] and len(segment[2]) > 1
        ]
        if not cleaned_segments:
            continue

        formatted_segments = [
            f"[{segment[0]:.3f},{segment[1]:.3f}] {segment[2]}"
            for segment in cleaned_segments
        ]

        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write("\n".join(formatted_segments))

        # delete the audio file if the transcription exists
        if verify_transcript_file(output_path):
            os.remove(audio_path)


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


def encode_transcripts(
    transcripts_dir: str, records_dir: str, embeddings_dir: str, channel: str
):
    if not os.path.exists(transcripts_dir):
        return

    files = os.listdir(transcripts_dir)
    if not files:
        return

    # Transcript line pattern
    pattern = re.compile(r"\[(\d+\.\d+),(\d+\.\d+)\]\s+(.*)")
    segments: list[tuple[str, str, str]] = []

    p_bar = tqdm(files)
    for filename in p_bar:
        if not filename.endswith(".txt"):
            continue

        video_id = filename.removesuffix(".txt")
        p_bar.set_description(video_id)
        # --- get the time stamps from file ---
        # Step 1: Parse transcript lines
        file_path = os.path.join(transcripts_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
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
            segment_name = f"{start} {end}"
            segments.append((video_id, segment_name, text))

    # Load the SentenceTransformer model
    SYMMETRIC_MODEL = SentenceTransformer(
        "all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu"
    )
    ASYMMETRIC_MODEL = SentenceTransformer(
        "msmarco-MiniLM-L-6-v3", device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Encode the segments and save the .npy file
    channel_embeddings_dir = os.path.join(embeddings_dir, channel)
    os.makedirs(channel_embeddings_dir, exist_ok=True)
    uploaded_embeddings: set[str] = set()
    for file in os.listdir(records_dir):
        with open(os.path.join(records_dir, file), "r", encoding="utf-8") as f:
            uploaded_embeddings |= set([x.strip() for x in f.readlines()])

    existing_embeddings = set(
        [
            filename
            for video_id in os.listdir(channel_embeddings_dir)
            for filename in os.listdir(os.path.join(channel_embeddings_dir, video_id))
            if filename.endswith(".npy")
        ]
    )
    skip_embeddings = uploaded_embeddings | existing_embeddings
    video_ids = set([segment[0] for segment in segments])
    for video_id in video_ids:
        os.makedirs(os.path.join(channel_embeddings_dir, video_id), exist_ok=True)

    segments_to_encode = [
        (video_id, segment_name, text)
        for video_id, segment_name, text in segments
        if f"sym {channel} {video_id} {segment_name}.npy" not in skip_embeddings
        and f"asym {channel} {video_id} {segment_name}.npy" not in skip_embeddings
    ]

    for video_id, segment_name, text in tqdm(segments_to_encode):
        # Symmetric encoding
        sym_seg_name = f"sym {channel} {video_id} {segment_name}.npy"
        if sym_seg_name not in existing_embeddings:
            embeddings = SYMMETRIC_MODEL.encode(text)
            sym_seg_path = os.path.join(channel_embeddings_dir, video_id, sym_seg_name)
            np.save(sym_seg_path, embeddings)

        # Asymmetric encoding
        asym_seg_name = f"asym {channel} {video_id} {segment_name}.npy"
        if asym_seg_name not in existing_embeddings:
            embeddings = ASYMMETRIC_MODEL.encode(text)
            asym_seg_path = os.path.join(
                channel_embeddings_dir, video_id, asym_seg_name
            )
            np.save(asym_seg_path, embeddings)


def get_uploaded_files(record_dir: str) -> set[str]:
    # get local file hashes
    checksums = calculate_record_file_checksums(record_dir)

    # get remote file hashes
    address: str = os.environ["HOST_ADDRESS"]

    # get remote file hashes, json dict
    remote_hash_response = requests.get(f"{address}/added_hash")

    if remote_hash_response.status_code != 200:
        raise Exception("Failed to get remote hash")

    remote_checksums: dict[str, list[str]] = remote_hash_response.json()

    # compare
    not_matching: list[str] = []

    for filename in checksums.keys() | remote_checksums.keys():
        if checksums.get(filename) != remote_checksums.get(filename):
            not_matching.append(filename)

    # print not matching files
    if not_matching:
        print("Not matching files:")
        for filename in not_matching:
            print(
                f"{filename}",
                f"local: {checksums.get(filename)}",
                f"remote: {remote_checksums.get(filename)}",
                sep="\n",
            )

    # download the non-matching files
    if not_matching:
        for filename in not_matching:
            res = requests.get(f"{address}/added/{filename}")

            if res.status_code == 200:
                with open(os.path.join(record_dir, filename), "wb") as f:
                    f.write(res.content)

                continue

            if res.status_code == 404:
                with open(os.path.join(record_dir, filename), "w") as f:
                    f.write("")

                continue

            raise Exception("Failed to download file")

    return get_files_in_record(record_dir)


def main(record_dir: str, rip=False) -> set[str]:
    root = os.path.dirname(os.path.abspath(__file__)).split("video_processing")[0]
    audio_dir = os.path.join(root, "audio")
    transcripts_dir = os.path.join(root, "transcriptions")
    embeddings_dir = os.path.join(root, "embeddings")
    records_dir = os.path.join(root, "added")
    config_path = os.path.join(
        root, "video_processing", "config", "video_processing_config.yaml"
    )
    age_restricted_path = os.path.join(audio_dir, "age_restricted.txt")
    shorts_path = os.path.join(audio_dir, "shorts.txt")
    END_TIME = time.time() + 5.5 * 60 * 60

    # Load the Whisper model
    # You can choose another model size as needed
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    model = whisperx.load_model(
        "large-v3",
        device=device,
        compute_type="float16",
        language="en",
    )
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)

    def transcribe_audio_file(audio_file_path: str) -> AlignedTranscriptionResult:
        """Transcription hook"""
        audio = whisperx.load_audio(audio_file_path)
        time.sleep(1)
        result = model.transcribe(audio, batch_size=12, language="en")
        cleaned_segments = []
        for segment in result["segments"]:
            x = segment
            x["text"] = clean_segment_text(x["text"])
            if x["text"] == "":
                continue

            cleaned_segments.append(x)

        time.sleep(1)
        return whisperx.align(
            cleaned_segments,
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

    # load config
    config = load_config(config_path)
    if any(" " in creator for creator in config.keys()):
        raise ValueError("Creator names cannot contain spaces")

    rip_audio = rip
    transcribe_audio = rip
    for i, (creator, creator_conf) in enumerate(config.items()):
        if END_TIME < time.time():
            break

        mid_str = f"({i + 1}/{len(config.keys())})"
        mid_str_len = len(mid_str)
        print("\n" + mid_str.rjust(30 + mid_str_len // 2, "-").ljust(60, "-"))

        creator_audio_dir = os.path.join(audio_dir, creator)
        creator_transcripts_dir = os.path.join(transcripts_dir, creator)
        if transcribe_audio:
            print(f"Transcribing {creator} audio files...")
            transcribe_audio_files(
                creator_audio_dir,
                creator_transcripts_dir,
                transcribe_audio_file,
                END_TIME,
            )

        if rip_audio:
            print(f"Ripping {creator} audio files...")
            try:
                rip_audio_files(
                    creator_audio_dir,
                    creator_transcripts_dir,
                    age_restricted_path,
                    shorts_path,
                    creator_conf,
                    END_TIME,
                )
            except Exception as e:
                print(f"Exception occurred while ripping {creator} audio files:")
                print(e)
                print_tb(e.__traceback__)
                rip_audio = False

        if transcribe_audio:
            print(f"Transcribing {creator} audio files...")
            transcribe_audio_files(
                creator_audio_dir,
                creator_transcripts_dir,
                transcribe_audio_file,
                END_TIME,
            )
        print(f"Encoding {creator} transcripts...")
        encode_transcripts(
            creator_transcripts_dir, records_dir, embeddings_dir, creator
        )

    # return the previously uploaded files on the server
    return get_uploaded_files(record_dir)
