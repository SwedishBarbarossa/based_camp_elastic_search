import os

import whisper
from pytube import Playlist
from pytube.exceptions import AgeRestrictedError
from tqdm import tqdm


def rip_audio_files():
    output_dir = "videos"
    os.makedirs(output_dir, exist_ok=True)

    # create age restricted file if it doesn't exist
    age_restricted_path = f"{output_dir}/age_restricted.txt"
    if not os.path.exists(age_restricted_path):
        open(age_restricted_path, "w")

    age_restricted = set()
    with open(age_restricted_path, "r") as f:
        for line in f:
            age_restricted.add(line.strip())

    files_in_dir = os.listdir(output_dir)
    audio_files = set([file for file in files_in_dir if file.endswith(".mp3")])

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
        try:
            audio_obj = video.streams.get_audio_only()
            if not audio_obj:
                raise ValueError("No audio stream found")

            audio_obj.download(
                output_path=output_dir, filename=f"{video_id}.mp3", skip_existing=True
            )
        except AgeRestrictedError:
            # add to list to download later
            if video_id in age_restricted:
                continue

            with open(age_restricted_path, "a") as f:
                f.write(f"{video_id}\n")


def transcribe_audio_files():
    # Define the directory containing the MP3 files
    audio_dir = "videos"
    output_dir = "transcriptions"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the Whisper model
    model = whisper.load_model("small")  # You can choose another model size as needed

    # Process each MP3 file in the directory
    for filename in tqdm(os.listdir(audio_dir)):
        if not filename.endswith(".mp3"):
            continue

        file_path = os.path.join(audio_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".mp3", ".txt"))

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


def main():
    rip_audio_files()
    transcribe_audio_files()


if __name__ == "__main__":
    main()
