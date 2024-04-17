python3 -m venv .venv

if [ -f .venv/bin/activate ]; then # Linux system
    source .venv/bin/activate
elif [ -f .venv/Scripts/activate.bat ]; then # Windows system
    source .venv/Scripts/activate
else
    echo "Failed to activate virtual environment"
    exit 1
fi

pip3 install wheel
pip3 install -r video_processing/requirements.txt