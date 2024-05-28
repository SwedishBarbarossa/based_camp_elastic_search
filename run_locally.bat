@echo off
rem Build server
docker compose build

rem Launch server
docker compose --profile nossl up -d

rem Activate virtual environment
python -m venv .venv
call .venv\Scripts\activate

rem Install dependencies
pip install wheel
pip install -r video_processing\requirements.txt

rem Launch processor
python processor.py
