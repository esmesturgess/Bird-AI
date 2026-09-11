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
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

COUNTDOWN="${COUNTDOWN:-data/soundscapes/countdown.mp4}"
VIDEO="${VIDEO:-data/soundscapes/exhibition_v2.mp4}"

# give the desktop session (and its audio server) a moment to finish coming up before
# mpv tries to grab a display and an audio device — matters most right after boot
sleep 8

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
# fullscreen toggle, volume...); --input-conf then re-adds exactly one binding (space =
# pause, see mpv_kiosk.conf) and nothing else. Deliberately NOT passing
# --input-vo-keyboard=no here: on X11 (Linux Mint's default) that's often the only path
# keypresses reach mpv at all in fullscreen, so disabling it would silently break space too.
MPV_KIOSK=(--fullscreen --no-osc --no-input-default-bindings --input-conf="$INPUT_CONF" --really-quiet)

if [ "${SKIP_COUNTDOWN:-0}" != "1" ] && [ -f "$COUNTDOWN" ]; then
  # plays ONCE (no --loop-file) and returns when it finishes — this is the one moment
  # someone needs to be there to press play on the separate raw-sound speaker, at GO
  mpv "${MPV_KIOSK[@]}" "$COUNTDOWN"
fi

exec mpv --loop-file=inf "${MPV_KIOSK[@]}" "$VIDEO"
