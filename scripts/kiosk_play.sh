#!/usr/bin/env bash
# The installation, unattended: play the video full-screen, forever, no controls, no
# screensaver interrupting it. This is what autostart launches on power-on — see
# install_autostart.sh. Run it by hand to test before wiring it to boot:
#
#   bash scripts/kiosk_play.sh
#
# Ctrl+C to stop when testing (autostart normally has no way to interrupt it, by design).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

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

exec mpv --fullscreen --loop-file=inf --no-osc --no-input-default-bindings \
         --input-vo-keyboard=no --really-quiet "$VIDEO"
