### Setup

1. Extract the transcriptions_archive.tar.gz file into root folder. It should be `_root_/transcripts/...` not `_root_/transcripts/transcripts/...`.

2. Ensure Docker is installed (https://docs.docker.com/engine/install/).

3. If you wish to rip and transcribe new videos, which can be very slow, set `RIP_AUDIO_FILES=1` in `_root_/env/dev.env`. When set to `1` It does three things: downloads and converts the entire video catalogue into audio, then uses an AI model (OpenAIs Whisper) to create transcripts for each video, and finally uses another AI model (all-MiniLM-L6-v2 and msmarco-MiniLM-L-6-v3) to assign concepts to 30 second windows in the transcripts.

Also implement this fix in Pytube if you wish to rip videos: https://github.com/pytube/pytube/pull/1409/commits/42a7d8322dd7749a9e950baf6860d115bbeaedfc

4. Run the 'run_locally.sh' file on Linux, or 'run_locally.bat' on windows.

5. Once the processing is done, there should be thousands of `.npy` files in the `embeddings` folder in the \_root\_ folder. They will be copied over to `server_embeddings`, read into Qdrant (the database) and subsequently be deleted from `server_embeddings`. This is because the processes are meant to be split between two machines, even though it works on a single machine.

6. The URL of the app is at `http://localhost`. It can take a few minutes between the index page loading and the search working. This is due to the application loading slower than the reverse proxy.

7. Afterwards, you should be able to use the search bar and get results back. Don't hesitate to use natural language and multiple keywords in the search bar. Results are sorted according to proximity to the query. Higher scores indicate more proximity and appear at the top.

### Starting Setup For Windows

1. Download the ffmpeg zip file from https://github.com/BtbN/FFmpeg-Builds/releases (or from unverified location here: https://www.videohelp.com/software/ffmpeg) and extract 'ffmpeg.exe' into the root folder of this repository.

2. Finish the ordinary setup
