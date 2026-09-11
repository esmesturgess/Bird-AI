#!/usr/bin/env bash
# Start the LIVE classifier immediately, and the pre-rendered VIDEO after a delay that
# accounts for the classifier's one-time model-loading + first-segment cost — so the two
# line up at segment 0 and stay lined up from there. (Measured on the NUC, 2026-09-11:
# ~52s from launch to the first result; every segment after that took ~20s to classify
# against a 25s slot, so nothing accumulates once past that first one.)
#
#   bash scripts/compare_timing.sh
#   STARTUP_DELAY=60 bash scripts/compare_timing.sh     # if your machine's slower/faster
#
# Reads: the video's own on-screen "t = ##.# s" clock, and this terminal's
# "[wall ##.#s | nominal ##s]" lines — the "(this is +-Ns off)" note on each should now
# sit close to the STARTUP_DELAY's own margin of error, not grow. --dry_run plays no
# audio itself, so only the video's sound is heard — nothing doubled. Ctrl+C stops both.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VIDEO="${VIDEO:-data/soundscapes/exhibition_v2.mp4}"
SOUNDSCAPE="${SOUNDSCAPE:-data/soundscapes/exhibition_v2.flac}"
SEGMENT="${SEGMENT:-25}"
STARTUP_DELAY="${STARTUP_DELAY:-52}"
PY="./.venv/bin/python"; [ -x "$PY" ] || PY="python3"
BN="./.venv/bin/birdnet-analyze"; [ -x "$BN" ] || BN="birdnet-analyze"

[ -f "$VIDEO" ] || { echo "video not found: $VIDEO"; exit 1; }
[ -f "$SOUNDSCAPE" ] || { echo "soundscape not found: $SOUNDSCAPE"; exit 1; }

START=$("$PY" -c "import time; print(time.time())")   # portable: bash `date +%s.%N` is a
                                                       # GNU-only extension and silently
                                                       # fails on macOS/BSD date
echo "starting live classifier now; video starts in ${STARTUP_DELAY}s to match its"
echo "model-loading + first-segment cost, so segment 0 of both line up ..."
echo ""

( sleep "$STARTUP_DELAY"; mpv --no-fs "$VIDEO" ) &
MPV_PID=$!
trap 'kill "$MPV_PID" 2>/dev/null' EXIT INT TERM

"$PY" scripts/live_soundscape.py \
  --soundscape "$SOUNDSCAPE" \
  --segment_seconds "$SEGMENT" \
  --birdnet_binary "$BN" \
  --dry_run \
  --start_epoch "$START"

wait "$MPV_PID" 2>/dev/null
