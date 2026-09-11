#!/usr/bin/env bash
# Start the pre-rendered VIDEO and the LIVE classifier at the same instant, so you can
# watch them side by side and see whether real-time classification on this machine keeps
# pace with the video's timeline. This is a bench comparison, not the exhibition setup —
# it plays the soundscape TWICE (once baked into the video, once live), so expect doubled/
# echoing audio; that's expected and fine for a timing test.
#
#   bash scripts/compare_timing.sh
#
# Reads: the video's own on-screen "t = ##.# s" clock.
# Reads: this terminal's "[wall ##.#s | nominal ##s]" lines from the live classifier —
#   the "(this is +NNs off)" note on each line says how far behind (or ahead) it is of
#   where the video would be at that same nominal segment. Ctrl+C stops both.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VIDEO="${VIDEO:-data/soundscapes/exhibition_v2.mp4}"
SOUNDSCAPE="${SOUNDSCAPE:-data/soundscapes/exhibition_v2.flac}"
SEGMENT="${SEGMENT:-25}"
PY="./.venv/bin/python"; [ -x "$PY" ] || PY="python3"
BN="./.venv/bin/birdnet-analyze"; [ -x "$BN" ] || BN="birdnet-analyze"

[ -f "$VIDEO" ] || { echo "video not found: $VIDEO"; exit 1; }
[ -f "$SOUNDSCAPE" ] || { echo "soundscape not found: $SOUNDSCAPE"; exit 1; }

START=$("$PY" -c "import time; print(time.time())")   # portable: bash `date +%s.%N` is a
                                                       # GNU-only extension and silently
                                                       # fails on macOS/BSD date
echo "starting video + live classifier together at epoch $START ..."
echo "(the classifier still needs ~30-90s to load its models first — that startup cost"
echo " is real and the video pays none of it; watch the first 'wall' timestamp to see it)"
echo ""

mpv --no-fs "$VIDEO" &
MPV_PID=$!

trap 'kill "$MPV_PID" 2>/dev/null' EXIT INT TERM

"$PY" scripts/live_soundscape.py \
  --soundscape "$SOUNDSCAPE" \
  --segment_seconds "$SEGMENT" \
  --birdnet_binary "$BN" \
  --dry_run \
  --start_epoch "$START"

wait "$MPV_PID" 2>/dev/null
