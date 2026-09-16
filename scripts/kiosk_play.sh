#!/usr/bin/env bash
# The installation, unattended, from one command: play the sync countdown once (your cue
# to start the separate raw-sound speaker at GO), then loop the main exhibition video
# full-screen forever, no controls, no screensaver interrupting it. This is what autostart
# launches on power-on — see install_autostart.sh. Run it by hand to test:
#
#   bash scripts/kiosk_play.sh
#   BASS_BT_MAC=AA:BB:CC:DD:EE:FF bash scripts/kiosk_play.sh   # + bass on a Bluetooth speaker
#
# q or Esc to quit. Set SKIP_COUNTDOWN=1 to jump straight to the loop.
# Set AUDIO_DEVICE to pin the video's sound (e.g. the 3.5mm jack rather than a monitor's
# HDMI speakers) — list the names with:  mpv --audio-device=help
#
# BASS_BT_MAC (optional) is a paired Bluetooth speaker's address. When set, the bass track
# (data/soundscapes/bass_track.flac) plays on that speaker alongside the video, and the
# video is kept on the headphone jack — otherwise Linux tends to make a newly connected
# Bluetooth speaker the default output and pull the video's sound onto it too.
#
# The bass track is the same length as the video and silent during every analysis page, so
# bass is heard ONLY under a translation. It is started at the video's current position
# (not from the beginning), so a speaker that connects late or drops out mid-show comes
# back in the right place. BASS_LEAD_S nudges it earlier to offset Bluetooth's lag.
#
# When launched by autostart (no terminal), everything is logged to
# ~/.cache/bird-installation.log — read it with:  cat ~/.cache/bird-installation.log
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOG="${LOG:-$HOME/.cache/bird-installation.log}"
if [ ! -t 1 ]; then mkdir -p "$(dirname "$LOG")"; exec >>"$LOG" 2>&1; fi
log() { printf '%s  %s\n' "$(date '+%F %T')" "$*"; }
log "---- kiosk_play.sh starting ----"

COUNTDOWN="${COUNTDOWN:-data/soundscapes/countdown.mp4}"
VIDEO="${VIDEO:-data/soundscapes/exhibition_v2.mp4}"
BASS="${BASS:-data/soundscapes/bass_track.flac}"
BASS_LEAD_S="${BASS_LEAD_S:-0.2}"      # Bluetooth plays late; start the bass this much early
BASS_BT_MAC="$(printf '%s' "${BASS_BT_MAC:-}" | tr '[:lower:]' '[:upper:]')"

# Right after boot the desktop's audio server can take a while to appear; if mpv starts
# first it plays the video SILENT and never recovers. Wait for it (up to 60s) instead of
# guessing a fixed delay, then give the session a couple of seconds more to settle.
for _ in $(seq 1 60); do
  if command -v pactl >/dev/null && pactl info >/dev/null 2>&1; then break; fi
  command -v pactl >/dev/null || { sleep 8; break; }   # no pactl: fall back to a fixed wait
  sleep 1
done
sleep 2
log "audio server up"

# stop the screen blanking / lock / power-saving that would otherwise black out an
# unattended display after a few minutes. Harmless to run repeatedly; ignore failures
# on setups that lack one of these (e.g. no Cinnamon session, or Wayland).
xset s off -dpms s noblank 2>/dev/null
gsettings set org.cinnamon.desktop.screensaver lock-enabled false 2>/dev/null
gsettings set org.cinnamon.desktop.session idle-delay 0 2>/dev/null
gsettings set org.cinnamon.settings-daemon.plugins.power sleep-display-ac 0 2>/dev/null

[ -f "$VIDEO" ] || { log "video not found: $VIDEO"; exit 1; }

# ---- keep the video on the headphone jack when a Bluetooth speaker is involved ----
jack_sink() { pactl list short sinks 2>/dev/null | awk '$2 ~ /analog/ {print $2; exit}'; }
jack=""
if [ -n "$BASS_BT_MAC" ]; then
  # At boot the audio server answers BEFORE the built-in sound card has registered its
  # outputs, so looking once finds nothing — that let the video follow the Bluetooth
  # speaker when it connected (worked when run by hand, failed from autostart). Wait for
  # the jack properly, then make it the DEFAULT as well as pinning the video to it, so
  # nothing can drift onto the speaker.
  for _ in $(seq 1 30); do jack="$(jack_sink)"; [ -n "$jack" ] && break; sleep 1; done
  if [ -n "$jack" ]; then
    pactl set-default-sink "$jack"
    [ -z "${AUDIO_DEVICE:-}" ] && AUDIO_DEVICE="pulse/$jack"
    log "headphone jack: $jack (video pinned to it, and made the default output)"
  else
    log "WARNING no headphone-jack output after 30s — video will use the default. Outputs seen:"
    pactl list short sinks
  fi
fi
log "video audio device: ${AUDIO_DEVICE:-system default}"

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
  log "countdown"
  mpv "${MPV_KIOSK[@]}" "$COUNTDOWN"
fi

# ---- the exhibition video (wired) ----
log "exhibition loop"
mpv --loop-file=inf "${MPV_KIOSK[@]}" "$VIDEO" &
VIDEO_PID=$!
VIDEO_START="$(date +%s)"

# ---- the bass track (Bluetooth), held in step with the video ----
bt_sink() {   # the speaker's output name, empty while it isn't connected
  pactl list short sinks 2>/dev/null |
    awk -v m="${BASS_BT_MAC//:/_}" '$2 ~ /bluez/ && index($2, m) {print $2; exit}'
}
bass_loop() {
  local period="$1" sink offset pid
  bluetoothctl power on >/dev/null 2>&1
  while true; do
    sink="$(bt_sink)"
    if [ -z "$sink" ]; then                 # (re)connect — speakers drop out, get switched off
      bluetoothctl connect "$BASS_BT_MAC" >/dev/null 2>&1
      sleep 5
      continue
    fi
    # a speaker (re)connecting can grab the default output — put it back on the jack
    [ -n "$jack" ] && pactl set-default-sink "$jack"
    # start where the video is now, so a late or reconnecting speaker lands in step
    offset="$(awk -v s="$VIDEO_START" -v n="$(date +%s)" -v p="$period" -v l="$BASS_LEAD_S" \
              'BEGIN{o=(n-s+l)%p; if(o<0)o+=p; printf "%.2f", o}')"
    log "bass: playing on $sink from ${offset}s"
    mpv --no-video --loop-file=inf --really-quiet --no-terminal \
        --start="$offset" --audio-device="pulse/$sink" "$BASS" &
    pid=$!
    # If the speaker disappears, Linux moves the stream to the default output — the
    # headphone jack — so kill it at once rather than let bass leak into the wired speakers.
    while kill -0 "$pid" 2>/dev/null; do
      sleep 3
      [ -n "$(bt_sink)" ] || kill "$pid" 2>/dev/null
    done
    wait "$pid" 2>/dev/null
    log "bass: speaker gone, waiting for it to come back"
    sleep 2
  done
}

if [ -n "$BASS_BT_MAC" ]; then
  if [ -f "$BASS" ]; then
    period="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$BASS" 2>/dev/null)"
    period="${period:-480}"
    set -m                                  # own process group, so one kill stops it all
    bass_loop "$period" &
    BASS_PID=$!
    set +m
    trap 'kill -- -"$BASS_PID" 2>/dev/null; kill "$VIDEO_PID" 2>/dev/null' EXIT INT TERM
  else
    log "WARNING bass file not found: $BASS"
  fi
fi

wait "$VIDEO_PID"
log "video stopped"
