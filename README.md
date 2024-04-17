Bring up docker-compose to serve site.
Run processor.py to create the embeddings and send them to server with gRPC.

To run locally swap this line in "docker-compose.yml":
```
- ./server_embeddings/:/src/server_embeddings/
# swap above line into the below line
- ./embeddings/:/src/server_embeddings/
```
and run "run_locally.sh".

For windows, add ffmpeg.exe to the project folder root.

Safest download is probably here: https://github.com/BtbN/FFmpeg-Builds/releases

Unverified download here https://www.videohelp.com/software/ffmpeg
