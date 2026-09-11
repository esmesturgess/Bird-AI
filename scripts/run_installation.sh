#!/usr/bin/env bash
# Run the installation: play the soundscape and classify it live.
#
#   bash scripts/run_installation.sh                 # play it once, with spoken answers
#   bash scripts/run_installation.sh --dry_run       # classify only, no audio, runs flat out
#   bash scripts/run_installation.sh --loop          # exhibition mode, repeats forever
#   bash scripts/run_installation.sh --loop --left_only   # birds on the left channel only
#
# Any extra arguments are passed straight through to live_soundscape.py.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SOUNDSCAPE="${SOUNDSCAPE:-data/soundscapes/exhibition_v2.flac}"
SEGMENT="${SEGMENT:-25}"

# prefer the venv python, fall back to whatever is active
PY="./.venv/bin/python"; [ -x "$PY" ] || PY="python3"
BN="./.venv/bin/birdnet-analyze"; [ -x "$BN" ] || BN="birdnet-analyze"

[ -f "$SOUNDSCAPE" ] || { echo "soundscape not found: $SOUNDSCAPE"; exit 1; }

echo "soundscape : $SOUNDSCAPE"
echo "segments   : ${SEGMENT}s"
echo "python     : $PY"
echo ""

exec "$PY" scripts/live_soundscape.py \
  --soundscape "$SOUNDSCAPE" \
  --segment_seconds "$SEGMENT" \
  --birdnet_binary "$BN" \
  "$@"
