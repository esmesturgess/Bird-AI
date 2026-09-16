#!/usr/bin/env bash
# Which speaker is each sound actually coming out of?
#
#   bash scripts/which_output.sh          # print once
#   bash scripts/which_output.sh --watch  # refresh every 2s, Ctrl+C to stop
#
# Shows every output the machine has, which one is the default, and what is currently
# playing on each. Use it when the bass is on the wrong speaker, or the video is silent —
# it answers "where is the sound going" without guesswork.
#
# The graphical equivalent is pavucontrol (sudo apt install pavucontrol), whose Playback
# tab shows the same thing with live meters and lets you drag a stream to another output.
set -u

show() {
  local default; default="$(pactl get-default-sink 2>/dev/null)"
  echo "OUTPUTS"
  pactl list short sinks 2>/dev/null | while read -r idx name rest; do
    local mark=" "; [ "$name" = "$default" ] && mark="*"
    local kind="wired"
    case "$name" in *bluez*) kind="BLUETOOTH";; *hdmi*) kind="HDMI (monitor)";; *usb*) kind="USB";; esac
    printf "  %s [%s] %-52s %s\n" "$mark" "$idx" "$name" "$kind"
  done
  echo "    (* = default: anything not pinned to a specific output goes here)"
  echo
  echo "PLAYING NOW"
  local any=0
  # each sink-input block names the sink index it is attached to, plus the app that owns it
  while IFS= read -r line; do
    case "$line" in
      "Sink Input #"*) sink=""; app="";;
      *"Sink: "*)      sink="${line##*: }";;
      *application.name*) app="${line#*= }"; app="${app//\"/}"
                       name="$(pactl list short sinks 2>/dev/null | awk -v i="$sink" '$1==i {print $2}')"
                       printf "  %-22s -> %s\n" "$app" "${name:-sink $sink}"; any=1;;
    esac
  done < <(pactl list sink-inputs 2>/dev/null)
  [ "$any" = 0 ] && echo "  (nothing is playing)"
}

if [ "${1:-}" = "--watch" ]; then
  while true; do clear; date '+%H:%M:%S'; echo; show; sleep 2; done
else
  show
fi
