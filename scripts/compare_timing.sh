#!/usr/bin/env bash
# Start the LIVE classifier immediately, and the pre-rendered VIDEO after a delay that
# accounts for the classifier's one-time model-loading + first-segment cost — so segment 0
# of both line up, and STAY lined up from there.
#
# Runs the classifier in its REAL playback mode, not --dry_run. This matters: --dry_run
# has no pacing and finishes a segment in ~20s flat-out, while the video is fixed at 25s/
# segment (that's how long its audio actually plays) — those are two different underlying
# RATES, and a fixed head start only aligns segment 0, drifting further apart every
# segment after that (~5s/segment, confirmed on the NUC, 2026-09-11). Real playback mode
# calls sd.wait() and blocks for the segment's true ~25s of audio before advancing, so it
# is paced at the SAME rate as the video (both come from the same 25s-segment soundscape)
# — once the one-time ~52s startup cost (model load + first classification) is accounted
# for, nothing races anything and they stay together indefinitely.
#
#   bash scripts/compare_timing.sh
#   STARTUP_DELAY=60 bash scripts/compare_timing.sh     # if your machine's slower/faster
#
# You WILL hear the soundscape twice, close together (baked into the video, plus the
# classifier's own real playback) — that's expected and useful here: well-aligned, it
# should sound like one thickened sound rather than a distinct echo. If you hear a clear
# echo, that's your ear telling you STARTUP_DELAY needs tuning. --no_speak keeps the
# spoken responses out of the way so this is a clean sync check, not extra noise.
# Reads: the video's own on-screen "t = ##.# s" clock against this terminal's
# "[wall ##.#s | nominal ##s]" lines. Ctrl+C stops both.
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
echo "starting live classifier now (real playback mode); video starts in"
echo "${STARTUP_DELAY}s to match its model-loading + first-segment cost ..."
echo ""

( sleep "$STARTUP_DELAY"; mpv --no-fs "$VIDEO" ) &
MPV_PID=$!
trap 'kill "$MPV_PID" 2>/dev/null' EXIT INT TERM

"$PY" scripts/live_soundscape.py \
  --soundscape "$SOUNDSCAPE" \
  --segment_seconds "$SEGMENT" \
  --birdnet_binary "$BN" \
  --no_speak \
  --start_epoch "$START"

wait "$MPV_PID" 2>/dev/null
