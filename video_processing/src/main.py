import os

import whisper
from pytube import Playlist
from pytube.exceptions import AgeRestrictedError
from tqdm import tqdm


def rip_audio_files():
    if not os.path.exists("videos"):
        os.mkdir("videos")

    # create age restricted file if it doesn't exist
    if not os.path.exists("videos/age_restricted.txt"):
        open("videos/age_restricted.txt", "w")

    age_restricted = set()
    with open("videos/age_restricted.txt", "r") as f:
        for line in f:
            age_restricted.add(line.strip())

    files_in_dir = os.listdir("videos")
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
                output_path="videos", filename=f"{video_id}.mp3", skip_existing=True
            )
        except AgeRestrictedError:
            # add to list to download later
            if video_id in age_restricted:
                continue

            with open("videos/age_restricted.txt", "a") as f:
                f.write(f"{video_id}\n")


def transcribe_audio_files():
    # Define the directory containing the MP3 files
    audio_dir = "videos"
    output_dir = "transcriptions"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the Whisper model
    model = whisper.load_model("tiny")  # You can choose another model size as needed

    # Function to transcribe a segment of audio
    def transcribe_segment(model, audio_path, start, end):
        result = model.transcribe(audio_path, start=start, end=end)
        return result["text"]

    # Process each MP3 file in the directory
    for filename in os.listdir(audio_dir):
        if filename.endswith(".mp3"):
            file_path = os.path.join(audio_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".mp3", ".txt"))

            with open(output_path, "w") as output_file:
                audio_info = whisper.load_audio(file_path)
                audio_duration = whisper.get_audio_duration(audio_info)
                start = 0
                end = 60  # 1 minute in seconds

                while start < audio_duration:
                    text = transcribe_segment(
                        model, file_path, start, min(end, audio_duration)
                    )
                    output_file.write(text + "\n\n")
                    # Move the window forward by 30 seconds
                    start += 30
                    end += 30

    print("Transcription completed.")


def main():
    rip_audio_files()


if __name__ == "__main__":
    main()
