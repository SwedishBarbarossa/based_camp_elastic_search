### Setup

1. Extract the transcriptions_archive.tar.gz file into root folder. It should be `_root_/transcripts/...` not `_root_/transcripts/transcripts/...`.

2. Ensure Docker is installed (https://docs.docker.com/engine/install/).

3. If you don't wish to rip and transcribe new videos, which can be very slow, set `RIP_AUDIO_FILES=False` in `processor.py` in the root folder. When set to `True` It does three things: downloads and converts the entire video catalogue into audio, then uses an AI model (OpenAIs Whisper) to create transcripts for each video, and finally uses another AI model (all-MiniLM-L6-v2 and msmarco-MiniLM-L-6-v3) to assign concepts to 30 second windows in the transcripts.

4. Run the 'run_locally.sh' file. The easiest way to run a shell script like this on Windows is to have Git Bash installed. It usually comes with Git. You can install Git and Git Bash here: https://git-scm.com/downloads

5. Once this process is over, there should be thousands of `.npy` files in the `embeddings` folder in the \_root\_ folder (not in src). They will be copied over to `server_embeddings`, read into Qdrant (the database) and subsequently be deleted from `server_embeddings`. This is because the processes are meant to be split between two machines, even though it works on a single machine.

6. The URL of the app is at `http://localhost`. Look at your terminal and wait until the server responds with `Updating Qdrant...` and finally `Qdrant updated.` This can take a few minutes.

7. Afterwards, you should be able to use the search bar and get results back. Don't hesitate to use natural language and multiple keywords in the search bar. Results are sorted according to proximity to the query. Higher scores indicate more proximity and appear at the top.

### Starting Setup For Windows

1. Download the ffmpeg zip file from https://github.com/BtbN/FFmpeg-Builds/releases (or from unverified location here: https://www.videohelp.com/software/ffmpeg) and extract 'ffmpeg.exe' into the root folder of this repository.

2. Finish the ordinary setup
