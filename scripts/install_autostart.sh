#!/usr/bin/env bash
# Make the installation start itself when the NUC is switched on — no keyboard, no
# monitor-as-control-surface, no command to remember. Run ONCE, on the NUC itself,
# logged into the graphical desktop (not over SSH — it needs your desktop session):
#
#   bash scripts/install_autostart.sh
#
# This only wires up the "start on login" half. The other half — logging in
# automatically with no password prompt — has to be done by hand in the Login Window
# settings (see the printed instructions below); this script prints them but does not
# touch that setting itself, since it's a system login file this project has never
# tested editing blind.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"

mkdir -p "$HOME/.config/autostart"
cat > "$HOME/.config/autostart/bird-installation.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Bird Installation
Comment=Plays the soundscape + visualisation on a loop
Exec=/bin/bash "$REPO/scripts/kiosk_play.sh"
X-GNOME-Autostart-enable=true
Hidden=false
NoDisplay=false
EOF

echo "Wrote $HOME/.config/autostart/bird-installation.desktop"
echo "  -> runs: bash \"$REPO/scripts/kiosk_play.sh\""
echo ""
echo "This starts the video whenever THIS USER logs into the desktop. To get from"
echo "'power on' to 'video playing' with nobody touching the keyboard, one more"
echo "setting is needed — automatic login. Do this by hand (safer than a script"
echo "editing a system login file on hardware nobody has tested it on):"
echo ""
echo "  Menu -> Login Window (or System Settings -> Login Window)"
echo "  -> Automatic Login -> ON -> select this user: $(whoami)"
echo ""
echo "Once both are set: power on -> auto-login -> desktop loads -> this video starts"
echo "-> loops forever. Test the autostart part right now without rebooting:"
echo ""
echo "  bash scripts/kiosk_play.sh    # Ctrl+C to stop"
echo ""
echo "To undo: delete $HOME/.config/autostart/bird-installation.desktop, and turn"
echo "Automatic Login back off in the same Login Window settings."
