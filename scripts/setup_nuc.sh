#!/usr/bin/env bash
# One-shot setup for running Bird Translation AI inference on an x86 Linux box
# (Intel NUC, Linux Mint). Run FROM THE REPO ROOT:
#     bash scripts/setup_nuc.sh
# Python 3.10–3.12 are all fine. Needs internet (BirdNET downloads its model on first run).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "Repo root: $(pwd)"

echo ""
echo "== 1/4 system packages (needs sudo) =="
sudo apt update
sudo apt install -y python3-venv python3-pip python3-dev build-essential \
     libsndfile1 ffmpeg espeak-ng git \
     libportaudio2 portaudio19-dev

echo ""
echo "== 2/4 python virtual environment =="
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

echo ""
echo "== 3/4 python dependencies (trimmed runtime set) =="
pip install -r requirements-deploy.txt
pip install "birdnet-analyzer[embeddings]"

echo ""
echo "== 4/4 sanity checks =="
set +e
python -c "import numpy, soundfile, librosa, torch, tensorflow; print('core imports OK')"
[ -x ".venv/bin/birdnet-analyze" ] && echo "birdnet-analyze OK" || echo "WARN: birdnet-analyze not found in .venv/bin"
[ -x ".venv/bin/birdnet-embeddings" ] && echo "birdnet-embeddings OK" || echo "WARN: birdnet-embeddings not found"
command -v espeak-ng >/dev/null && echo "espeak-ng OK (TTS)" || echo "WARN: espeak-ng missing (--speak will not work)"
set -e

echo ""
echo "Setup complete. Test the demo (from the repo root) with a blackbird clip:"
echo "  ./.venv/bin/python scripts/infer_clip.py --clip path/to/blackbird.wav --speak"
echo ""
echo "No clip on the box yet? Copy one over from your Mac, e.g.:"
echo "  scp 'some_blackbird.wav' user@<nuc-ip>:$(pwd)/"
