# Running the demo on the Intel NUC (x86 Linux / Linux Mint)

Goal: get `scripts/infer_clip.py` running so a blackbird clip in → the NUC **says
"Blackbird"** aloud. Because the NUC is x86, the whole stack installs cleanly (same as
the Mac) — no ARM issues.

## 1. Get the repo onto the NUC

Either clone it (if you've added an SSH key / it's accessible), or copy it over.

```bash
# Option A — clone (repo is at github.com/esmesturgess/Bird-AI)
git clone https://github.com/esmesturgess/Bird-AI.git
cd Bird-AI

# Option B — copy from the Mac over the network
#   (on the Mac)  scp -r "/Users/esmesturgess/Bird Translation AI" user@<nuc-ip>:~/bird-ai
#   (on the NUC)  cd ~/bird-ai
```

Note: `data/`, `.venv/`, and the large audio/run folders are gitignored, so a fresh clone
won't include audio — that's expected. You'll copy a test clip in step 3.

## 2. Install everything (one command)

From the repo root:

```bash
bash scripts/setup_nuc.sh
```

This installs system packages (Python venv, ffmpeg, libsndfile, **espeak-ng** for speech),
creates `.venv`, installs the trimmed runtime deps (`requirements-deploy.txt`) plus
`birdnet-analyzer[embeddings]`, and runs sanity checks. Python 3.10–3.12 all work.
First run needs internet — BirdNET downloads its model.

## 3. Get a blackbird clip onto the NUC

A clean, mostly-blackbird clip works best (raw multi-bird recordings give noisy detections).
Copy one from the Mac:

```bash
# on the Mac
scp "/Users/esmesturgess/Bird Translation AI/data/raw/blackbird_xeno_canto_v1/turdus_merula/XC128838_Turdus_merula.mp3" user@<nuc-ip>:~/bird-ai/
```

## 4. Run it

```bash
./.venv/bin/python scripts/infer_clip.py --clip XC128838_Turdus_merula.mp3 --speak
```

Expected: prints the detected species and **speaks "Blackbird"** through the NUC's audio out.
On the known-good clip above it detects `Turdus merula` at ~0.995 confidence.

## 5. Record your own clips from a microphone

`scripts/record_clip.py` captures 48 kHz mono WAVs that step 4 (and the offline pipeline)
can read directly. Plug in a **USB mic** — laptop/built-in mics noise-gate and high-pass
aggressively, which eats exactly the bird audio you want.

```bash
# which mic?
./.venv/bin/python scripts/record_clip.py --list_devices

# calibrate: 15s of level metering -> room noise floor + a suggested VOX threshold
./.venv/bin/python scripts/record_clip.py --meter --device usb

# one 5s clip, then identify it
./.venv/bin/python scripts/record_clip.py --seconds 5 --device usb --analyse

# leave it listening: one file per sound event, until Ctrl+C
./.venv/bin/python scripts/record_clip.py --auto --threshold 0.018 --device usb
```

Files land in `data/raw/mic_recordings/` (gitignored) alongside `recordings_log.csv`,
which records level and duration per clip — that log is what you tune the threshold from.
`--analyse` adds ~3-10 s per clip because BirdNET reloads its model on every call.

Run `--meter` in the actual room before `--auto`: the default threshold is a guess, the
meter's suggestion is not.

## Troubleshooting

- **No sound:** confirm `espeak-ng` works standalone: `espeak-ng "hello"`. Check the NUC's
  audio output device / volume. (For the installation you'll wire real speakers to the 3.5mm out.)
- **`birdnet-analyze: not found`:** run from the repo root, or pass
  `--birdnet_binary .venv/bin/birdnet-analyze`.
- **mp3 won't load:** ensure `ffmpeg` installed (setup script does this).
- **pip can't find tensorflow:** check `python3 --version` is 3.10–3.12.
- **`sounddevice` import fails / PortAudio missing:** `sudo apt install -y libportaudio2
  portaudio19-dev` (the setup script does this), then `pip install sounddevice`.
- **No devices listed / no audio when connected over SSH:** the sound server (PulseAudio or
  PipeWire) belongs to the *graphical* login session, and an SSH shell isn't in it. Either
  run from a terminal on the box itself, or as the same user who is logged in graphically:
  `export XDG_RUNTIME_DIR=/run/user/$(id -u)` before running. Cross-check what the kernel
  sees with `arecord -l` (mics) and `aplay -l` (outputs) — those work over SSH regardless.
- **Mic records silence:** check the device index with `--list_devices` and `arecord -l`,
  and check input gain in the system sound settings. `--meter` tells you within seconds.

## 6. The actual installation: video on loop, no keyboard needed

The exhibition doesn't run `infer_clip.py` or any live classifier on the NUC — it plays
one pre-rendered video (`data/soundscapes/exhibition_v2.mp4`, soundscape + on-screen
visualisation, already classified and baked in) full-screen, forever. This is deliberate:
this NUC's i3-5010U is too slow to classify live and draw a live display at once, and an
unattended installation should sit at ~2% CPU playing video, not pinned at 100% for weeks.

Test it manually first:

```bash
bash scripts/kiosk_play.sh
```

Full-screen video, looping, screensaver disabled, no on-screen controls. `Ctrl+C` to stop.

Then wire it to start itself on power-on:

```bash
bash scripts/install_autostart.sh
```

This prints the one manual step it can't safely do for you — turning on Automatic Login
in the Login Window settings (a system login file, worth doing by hand rather than
scripted blind on hardware nobody's tested it on). Once that's set: power on → desktop
loads → video starts → loops. No monitor-as-keyboard, no command to remember.

To swap in a different video later (a new soundscape, or once the sound artist's 12
response clips are mixed in), either replace `data/soundscapes/exhibition_v2.mp4` in
place, or point `kiosk_play.sh` at a new file with `VIDEO=path/to/new.mp4 bash
scripts/kiosk_play.sh`.

## What this is / isn't

- ✅ Milestone 0: clip → species, spoken aloud. Runs on the NUC.
- ⏳ Milestone 1: clip → species + **call-type** ("Blackbird. Alarm."). The phrase builder
  already supports it; needs the reviewed blackbird anchor prototypes wired in. The call-type
  label is **not accurate yet** (the open data/label problem) — plumbing/demo only.
- ⏳ Milestone 2: full installation loop (play clip → classify → trigger second speaker).
