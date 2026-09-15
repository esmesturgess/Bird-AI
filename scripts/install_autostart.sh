#!/usr/bin/env bash
# Make the installation start itself when the NUC is switched on — no keyboard, no
# monitor-as-control-surface, no command to remember. Run ONCE, on the NUC itself,
# logged into the graphical desktop (not over SSH — it needs your desktop session):
#
#   bash scripts/install_autostart.sh
#   AUDIO_DEVICE='pipewire/alsa_output...analog-stereo' bash scripts/install_autostart.sh
#
# The second form pins the sound to one output (e.g. the 3.5mm jack rather than a
# monitor's HDMI speakers) for every boot — `mpv --audio-device=help` lists the names.
#
# This only wires up the "start on login" half. The other half — logging in
# automatically with no password prompt — has to be done by hand in the Login Window
# settings (see the printed instructions below); this script prints them but does not
# touch that setting itself, since it's a system login file this project has never
# tested editing blind.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"

# settings baked into the boot entry, so they survive without anyone typing them
ENV_PREFIX=""
[ -n "${AUDIO_DEVICE:-}" ] && ENV_PREFIX="env AUDIO_DEVICE=${AUDIO_DEVICE} "

mkdir -p "$HOME/.config/autostart"
cat > "$HOME/.config/autostart/bird-installation.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Bird Installation
Comment=Plays the soundscape + visualisation on a loop
Exec=${ENV_PREFIX}/bin/bash "$REPO/scripts/kiosk_play.sh"
X-GNOME-Autostart-enable=true
Hidden=false
NoDisplay=false
EOF

echo "Wrote $HOME/.config/autostart/bird-installation.desktop"
echo "  -> runs: ${ENV_PREFIX}bash \"$REPO/scripts/kiosk_play.sh\""
[ -z "${AUDIO_DEVICE:-}" ] && echo "  (sound goes to the system's default output — set it once in Menu -> Sound)"
echo ""
echo "This starts the video whenever THIS USER logs into the desktop. Two more settings"
echo "get you from 'power on' to 'video playing' with nobody touching anything — both by"
echo "hand, since they're system settings a script shouldn't edit blind:"
echo ""
echo "  1. Automatic login:"
echo "     Menu -> Login Window -> Automatic Login -> ON -> select this user: $(whoami)"
echo ""
echo "  2. Boot when the power comes on (only if it's switched on at the wall):"
echo "     restart, tap F2 at the Intel logo -> Power -> After Power Failure -> Power On"
echo "     -> F10 to save"
echo ""
echo "Then: power on -> boots -> logs in -> countdown -> video loops forever."
echo "Test the playback part right now without rebooting:"
echo ""
echo "  bash scripts/kiosk_play.sh    # q or Esc to quit"
echo ""
echo "To undo: delete $HOME/.config/autostart/bird-installation.desktop, and turn"
echo "Automatic Login back off in the same Login Window settings."
