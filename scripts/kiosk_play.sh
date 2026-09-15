#!/usr/bin/env bash
# The installation, unattended, from one command: play the sync countdown once (your cue
# to start the separate raw-sound speaker at GO), then loop the main exhibition video
# full-screen forever, no controls, no screensaver interrupting it. This is what autostart
# launches on power-on — see install_autostart.sh. Run it by hand to test:
#
#   bash scripts/kiosk_play.sh
#
# Ctrl+C to stop when testing (autostart normally has no way to interrupt it, by design).
# Set SKIP_COUNTDOWN=1 to jump straight to the loop (e.g. re-testing just the exhibition).
# Set AUDIO_DEVICE to pin the output (e.g. the 3.5mm jack rather than a monitor's HDMI
# speakers) — list the names with:  mpv --audio-device=help
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

COUNTDOWN="${COUNTDOWN:-data/soundscapes/countdown.mp4}"
VIDEO="${VIDEO:-data/soundscapes/exhibition_v2.mp4}"

# Right after boot the desktop's audio server can take a while to appear; if mpv starts
# first it plays the video SILENT and never recovers. Wait for it (up to 60s) instead of
# guessing a fixed delay, then give the session a couple of seconds more to settle.
for _ in $(seq 1 60); do
  if command -v pactl >/dev/null && pactl info >/dev/null 2>&1; then break; fi
  command -v pactl >/dev/null || { sleep 8; break; }   # no pactl: fall back to a fixed wait
  sleep 1
done
sleep 2

# stop the screen blanking / lock / power-saving that would otherwise black out an
# unattended display after a few minutes. Harmless to run repeatedly; ignore failures
# on setups that lack one of these (e.g. no Cinnamon session, or Wayland).
xset s off -dpms s noblank 2>/dev/null
gsettings set org.cinnamon.desktop.screensaver lock-enabled false 2>/dev/null
gsettings set org.cinnamon.desktop.session idle-delay 0 2>/dev/null
gsettings set org.cinnamon.settings-daemon.plugins.power sleep-display-ac 0 2>/dev/null

[ -f "$VIDEO" ] || { echo "kiosk_play.sh: video not found: $VIDEO" >&2; exit 1; }

INPUT_CONF="$(pwd)/scripts/mpv_kiosk.conf"
# --no-input-default-bindings turns off everything mpv normally binds (seek, quit,
# fullscreen toggle, volume...); --input-conf then re-adds exactly the bindings in
# mpv_kiosk.conf (space = pause, q/Esc = quit) and nothing else. Deliberately NOT passing
# --input-vo-keyboard=no here: on X11 (Linux Mint's default) that's often the only path
# keypresses reach mpv at all in fullscreen, so disabling it would silently break space too.
MPV_KIOSK=(--fullscreen --no-osc --no-input-default-bindings --input-conf="$INPUT_CONF" --really-quiet)
[ -n "${AUDIO_DEVICE:-}" ] && MPV_KIOSK+=(--audio-device="$AUDIO_DEVICE")

if [ "${SKIP_COUNTDOWN:-0}" != "1" ] && [ -f "$COUNTDOWN" ]; then
  # plays ONCE (no --loop-file) and returns when it finishes — this is the one moment
  # someone needs to be there to press play on the separate raw-sound speaker, at GO
  mpv "${MPV_KIOSK[@]}" "$COUNTDOWN"
fi

exec mpv --loop-file=inf "${MPV_KIOSK[@]}" "$VIDEO"
