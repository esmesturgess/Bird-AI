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
echo "== 3/5 python dependencies (trimmed runtime set) =="
pip install -r requirements-deploy.txt
pip install "birdnet-analyzer[embeddings]"

echo ""
echo "== 4/5 pre-download the model weights =="
# Both encoders fetch large checkpoints on FIRST USE. Doing that here means the exhibition
# never waits on a download — and a missing network on the night is not fatal.
#   AVES2  ~400 MB -> ~/.cache/esp
#   BirdNET        -> its own cache, triggered by the first analyse call
set +e
python - <<'PY'
import sys
try:
    from avex import load_model
    print("  downloading / verifying AVES2 encoder (~400 MB, once) ...", flush=True)
    load_model("esp_aves2_sl_beats_bio", device="cpu", return_features_only=True)
    print("  AVES2 encoder OK")
except Exception as e:
    print(f"  WARN: AVES2 encoder not ready: {type(e).__name__}: {e}")
    print("       the classifier will NOT run until this succeeds — re-run with internet.")
    sys.exit(0)
PY
set -e

echo ""
echo "== 5/5 sanity checks =="
set +e
python -c "import numpy, soundfile, librosa, torch, tensorflow; print('core imports OK')"
[ -x ".venv/bin/birdnet-analyze" ] && echo "birdnet-analyze OK" || echo "WARN: birdnet-analyze not found in .venv/bin"
[ -x ".venv/bin/birdnet-embeddings" ] && echo "birdnet-embeddings OK" || echo "WARN: birdnet-embeddings not found"
command -v espeak-ng >/dev/null && echo "espeak-ng OK (TTS)" || echo "WARN: espeak-ng missing (spoken response will not work)"
python - <<'PY'
import json, glob, os
found = sorted(glob.glob("models/*/adapter.pt"))
if not found:
    print("WARN: no models/*/adapter.pt — did the repo clone include models/ ?")
else:
    for f in found:
        d = os.path.dirname(f)
        m = json.load(open(os.path.join(d, "adapter_meta.json")))
        print(f"  model OK: {os.path.basename(d):11s} classes={m['classes']}")
PY
python -c "import sounddevice as sd; print('audio devices visible:', len(sd.query_devices()))" 2>/dev/null \
  || echo "WARN: sounddevice cannot see any audio device (check PortAudio / user in 'audio' group)"
set -e

echo ""
echo "Setup complete. Next, in order:"
echo "  1. classification only, no audio out:"
echo "     ./.venv/bin/python scripts/live_soundscape.py --soundscape <file>.wav --dry_run"
echo "  2. full loop with spoken responses:"
echo "     ./.venv/bin/python scripts/live_soundscape.py --soundscape <file>.wav"
echo "  3. exhibition mode (repeat forever, birds on the left channel):"
echo "     ./.venv/bin/python scripts/live_soundscape.py --soundscape <file>.wav --loop --left_only"
echo ""
echo "No soundscape on the box yet? Copy one from your Mac:"
echo "  scp data/soundscapes/v1.wav user@<nuc-ip>:$(pwd)/data/soundscapes/"
