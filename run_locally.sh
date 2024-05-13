# Build server
sudo docker compose build

# Launch server
sudo docker compose up -d

# Activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip3 install wheel
pip3 install -r video_processing/requirements.txt

# Launch processor
python3 processor.py