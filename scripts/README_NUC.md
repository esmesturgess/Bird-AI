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

## Troubleshooting

- **No sound:** confirm `espeak-ng` works standalone: `espeak-ng "hello"`. Check the NUC's
  audio output device / volume. (For the installation you'll wire real speakers to the 3.5mm out.)
- **`birdnet-analyze: not found`:** run from the repo root, or pass
  `--birdnet_binary .venv/bin/birdnet-analyze`.
- **mp3 won't load:** ensure `ffmpeg` installed (setup script does this).
- **pip can't find tensorflow:** check `python3 --version` is 3.10–3.12.

## What this is / isn't

- ✅ Milestone 0: clip → species, spoken aloud. Runs on the NUC.
- ⏳ Milestone 1: clip → species + **call-type** ("Blackbird. Alarm."). The phrase builder
  already supports it; needs the reviewed blackbird anchor prototypes wired in. The call-type
  label is **not accurate yet** (the open data/label problem) — plumbing/demo only.
- ⏳ Milestone 2: full installation loop (play clip → classify → trigger second speaker).
