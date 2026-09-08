# Mic PoC Plan — "bird sings into a mic → the box plays that bird a song"

Status: PLAN ONLY (written 2026-08-09). Nothing below is built yet.
Scope: a **bench proof-of-concept on the x86 Linux PC**, not the final installation.

---

## 0. What this PoC is, and two things to read first

**Goal:** a Linux PC with a microphone plugged in. A bird sound happens in the room
(live bird, phone/speaker playing a recording, a person whistling). Within a few seconds
the PC identifies the species and plays an audio response mapped to that species through
a speaker. It logs every decision to CSV. It says/plays nothing when unsure.

### Flag 1 — this contradicts a locked decision, deliberately

`CLAUDE.md` records: **"no microphone in the live installation"** (decided 2026-07-14) —
the gallery piece is meant to play a known clip to Speaker A and classify its own clean
in-memory buffer, precisely to dodge gallery-noise SNR and the Speaker-B→mic feedback loop.

This PoC reintroduces the mic on purpose, because a mic PoC answers a question the digital
path cannot: *does BirdNET actually work on real acoustic audio in a real room?* That is
worth knowing regardless of which path the installation ships.

**It should not be read as reversing the installation decision.** Both feedback and SNR
are handled explicitly in Stage 3 below. If the PoC works well, *then* it's a real
conversation about whether the install goes mic-based.

### Flag 2 — species only, no call type

The PoC will say **"Blackbird"**, not **"Blackbird, alarm call"**. The semantic call-type
layer is the project's known-unsolved problem (per `CLAUDE.md`: adapter overfits at ~0.55
vs a 0.60 majority floor; Blue Tit labels ~91% ambiguous-margin; labels themselves are
cluster-derived and untrusted). Wiring it into a live demo would produce confident-sounding
labels that are not real. Species detection, by contrast, is **already proven working** —
0.995 on a clean blackbird clip via `scripts/infer_clip.py`.

Keep the response map keyed on species. Call type is a later swap-in, not a PoC dependency.

### Assumption to confirm

"Return a song for that bird" is read as: **play an audio file mapped to the detected
species**. The plan makes that mapping a config file (`config/response_map.yaml`), so the
same code covers all three readings — a real recording of that species' song, a composed
musical cue, or spoken text — by editing one line per species. If you meant something
narrower, only the contents of `data/responses/` change, not the build.

---

## 1. What already exists (reuse, don't rebuild)

| Piece | File | State |
|---|---|---|
| BirdNET species inference | [src/birdnet_infer.py](src/birdnet_infer.py) | Works. **Subprocess/CLI-based** — see the latency finding below. |
| Clip → species → spoken name | [scripts/infer_clip.py](scripts/infer_clip.py) | Works end-to-end on the NUC (Milestone 0). `detect_species()`, `friendly()`, `speak()` are all directly reusable. |
| Linux one-shot install | [scripts/setup_nuc.sh](scripts/setup_nuc.sh) | Works (apt-based: Mint/Ubuntu/Debian). Needs 2 lines added for mic support. |
| Trimmed runtime deps | [requirements-deploy.txt](requirements-deploy.txt) | Good. Missing only the audio-I/O package. |
| NUC runbook | [scripts/README_NUC.md](scripts/README_NUC.md) | Accurate; extend with a mic section. |
| Species name table | [scripts/infer_clip.py:23](scripts/infer_clip.py#L23) | 8 species. Extend as the response library grows. |

**Nothing in `live_*.py` exists** — `REPO_ZONES.md` sketches five live files, none written.
This PoC writes the first three of them for real.

### The one real technical finding

`src/birdnet_infer.py` runs BirdNET by **spawning the `birdnet-analyze` CLI as a
subprocess** ([src/birdnet_infer.py:333](src/birdnet_infer.py#L333)) and reading its CSV/JSON
output back off disk. Every single call reloads the model from scratch.

That is fine for `infer_clip.py` (one clip, one run). It is **not real-time** — expect
roughly 3–10 s per detection, dominated by model load, not by audio length.

Consequence for the plan: **the PoC is "chunked", not streaming.** It listens for a few
seconds, thinks for a few seconds, then responds. For a demo where a bird sings and the box
answers, that is acceptable and honest. Stage 5 removes it if the lag is annoying — but do
not start there, because a working slow loop is worth more than a fast loop that doesn't
exist yet.

---

## 2. Architecture of the PoC

```
USB mic
   │  sounddevice, 48 kHz mono, rolling 3 s windows (1 s hop)
   ▼
[ energy gate ]  RMS below room-noise floor → discard, never invoke BirdNET
   │
   ▼
[ BirdNET species ]  reuse detect_species(); top-1 label + score
   │
   ▼
[ decision gate ]  score ≥ min_conf  AND  same species in 2 consecutive windows
   │                       │
   │ pass                  │ fail → log "unknown", stay silent
   ▼
[ response map ]  config/response_map.yaml : species → audio file
   │
   ▼
[ play + guard ]  MUTE MIC during playback + 1 s tail, then cooldown N s
   │
   ▼
   speaker            + append row to events CSV
```

Three gates, in this order, are what stop the PoC from being a random noise machine:
**energy gate** (don't classify silence), **confidence + debounce gate** (don't act on one
lucky window), **mic mute + cooldown** (don't let the response retrigger the system).

### Files to create

| New file | Job |
|---|---|
| `src/live_audio.py` | Mic capture, device selection/listing, ring buffer, window slicing, RMS energy gate, temp-wav writer. |
| `src/live_species.py` | Thin wrapper over `detect_species()`: window → (species, score). One place to later swap in an in-process model. |
| `src/live_respond.py` | Load `response_map.yaml`, resolve species → file, play it, own the mute/cooldown state machine. |
| `scripts/live_demo.py` | The loop + CLI. Flags: `--list_devices`, `--device`, `--min_conf`, `--cooldown`, `--dry_run`, `--log_csv`. |
| `config/response_map.yaml` | Tracked in git. Species → response file + friendly name. |
| `data/responses/` | The response audio itself. **Add to `.gitignore`** (per `STORAGE_POLICY.md`). |

`--dry_run` (detect and log, never play) is not optional polish — it's how you tune
thresholds in the actual room without the speaker interfering.

---

## 3. Stages

Each stage ends in something demonstrable. Do not skip ahead; each one de-risks the next.

### Stage 0 — Hardware + environment (~1 h, mostly waiting)

1. Confirm the Linux PC boots and is on the network. `scripts/setup_nuc.sh` assumes
   apt (Mint/Ubuntu/Debian). If it's a different distro, only step 1 of that script changes.
2. Add to the apt line in `setup_nuc.sh`: `portaudio19-dev libportaudio2` (needed to build
   and run `sounddevice`).
3. Add to `requirements-deploy.txt`: `sounddevice>=0.4` and `PyYAML>=6.0`.
4. Run `bash scripts/setup_nuc.sh`. First run downloads the BirdNET model (needs internet).
5. Verify Milestone 0 still passes: copy a clean blackbird clip over and run
   `./.venv/bin/python scripts/infer_clip.py --clip <clip>.mp3 --speak`.

**Hardware needed:** a USB microphone (a cheap USB condenser or USB lavalier is fine — do
**not** rely on a laptop's built-in mic, they aggressively noise-gate and high-pass), and a
powered speaker or the PC's 3.5 mm out.

**Done when:** the PC speaks "Blackbird" from a file, and `arecord -l` lists the USB mic.

### Stage 1 — Mic capture proven, zero new ML code (~1–2 h)

**BUILT (2026-08-09):** `src/live_audio.py` + `scripts/record_clip.py`. Four modes:
`--list_devices`, `--meter` (noise floor + suggested threshold), `--seconds` (fixed clip),
`--auto` (sound-activated, one file per event). `--analyse` chains straight into the
existing `infer_clip.py` detection. Output: 48 kHz mono PCM_16 WAV in
`data/raw/mic_recordings/` plus `recordings_log.csv` with per-clip levels.

Capture runs at the device's native rate and resamples once at save time (scipy
`resample_poly`), so 44.1 kHz-only mics work without per-block filter discontinuities.

Remaining for this stage: run it on the target PC with a real USB mic, then feed a
captured clip to the **existing** `infer_clip.py`.

This is the highest-value step in the plan: it tests microphone → BirdNET on real acoustic
audio while touching none of the inference code. If species detection collapses here, you
learn it on day one, and every later stage is affected.

**Test:** play a known-good XC blackbird clip from a decent speaker ~1 m from the mic.

**Done when:** mic-captured audio of a blackbird recording is detected as `Turdus merula`.
Record the confidence — it *will* be lower than the 0.995 file-based number, and that drop
is the single most important measurement in this PoC. It sets `min_conf` for everything after.

> If confidence lands below ~0.3 here, stop and fix acoustics (mic gain, distance, speaker
> quality, room reverb) before writing the loop. Wood pigeon in particular is low-frequency
> and will not reproduce from a phone speaker — use a real speaker for test stimuli.

### Stage 2 — Continuous detection loop, no response yet (~half a day)

`src/live_species.py` + `scripts/live_demo.py --dry_run`. Rolling windows, energy gate,
BirdNET per accepted window, print + log `timestamp, rms, species, score, decision`.

Calibrate here, in the actual room:
- **RMS floor:** record 10 s of empty-room noise; set the gate ~3× that RMS.
- **`min_conf`:** from the Stage 1 measurement, minus headroom.
- **Debounce:** require the same top-1 species across 2 consecutive accepted windows.

**Done when:** it runs for 10 minutes in a normal room, prints `unknown` for speech,
typing, chairs and silence, and prints the right species when you play a bird clip at it.
Expect junk species on messy audio — that's the documented lesson from `infer_clip.py`
testing (a raw multi-bird XC recording once topped out as Chiffchaff). The gates exist
precisely for this.

### Stage 3 — Response playback: the actual deliverable (~half a day)

`src/live_respond.py` + `config/response_map.yaml`. Detected species → play its file.

Curate `data/responses/` — start with 3 species you have good audio for (blackbird, wren,
wood pigeon are the best-supported in this repo) plus a fallback. Each entry:

```yaml
species:
  "Turdus merula":
    common_name: Blackbird
    response: data/responses/blackbird_response.wav
  "Troglodytes troglodytes":
    common_name: Wren
    response: data/responses/wren_response.wav
unknown:
  response: null          # stay silent; or point at a neutral cue
cooldown_seconds: 15
```

**Feedback handling (mandatory, not optional):** stop the input stream before playback
starts, resume it 1 s after playback ends, then hold a cooldown. Without this the response
audio is heard by the mic, classified, and triggers another response — the exact failure
mode the "no mic" decision was made to avoid. Test it deliberately: put the speaker facing
the mic and confirm it fires **once**.

**Done when:** the demo test below passes.

### Stage 4 — Soak + tune (~half a day, do it before showing anyone)

Run for an hour with people talking nearby. Count false triggers. Tighten `min_conf` or
debounce (to 3 windows) until false triggers are rare. Log everything; the CSV is the
evidence. Write up the numbers in `scripts/README_NUC.md`.

### Stage 5 — OPTIONAL: kill the latency

Only if the 3–10 s response lag is unacceptable in practice. Replace the CLI subprocess
with an in-process BirdNET that stays loaded (BirdNET-Analyzer's Python API, or the
lightweight `birdnet` PyPI package which accepts numpy arrays directly). This is contained
entirely inside `src/live_species.py` if Stage 2 is written with that boundary respected —
which is the whole reason that file exists as a separate module.

Budget a day, and treat it as a spike: if the in-process path fights the dependency stack,
abandon it and keep the working subprocess loop.

---

## 4. Definition of done for the PoC

A single scripted demo, run on the Linux PC, witnessed:

1. Start `./.venv/bin/python scripts/live_demo.py --device <mic>`.
2. Stay quiet 30 s → nothing plays; log shows energy-gated windows only.
3. Talk near the mic 30 s → nothing plays; log shows `unknown` or gated.
4. Play a blackbird recording from a speaker → within ~10 s, the blackbird response plays.
5. Play a wren recording → the wren response plays, not the blackbird one.
6. Let the response finish → it does **not** retrigger itself.
7. `events.csv` contains a readable row per decision.

That is a genuine end-to-end perception→response loop on real acoustic audio, which is
exactly the thing the project has never yet demonstrated.

---

## 5. Risks, honestly

| Risk | Likelihood | Handling |
|---|---|---|
| Mic-captured confidence far below file-based | **High** | Measured in Stage 1 before anything depends on it; drives `min_conf`. Fix acoustically (mic, distance, speaker) not in code. |
| 3–10 s latency feels dead in a demo | Medium | Accepted for the PoC; Stage 5 is the escape hatch. Consider a quiet "thinking" cue. |
| Response audio retriggers the system | **High if unhandled** | Mic mute during playback + tail + cooldown. Explicitly tested in Stage 3. |
| Room noise → false species | Medium | Three gates: energy, confidence, debounce. Tuned in the real room in Stage 2/4. |
| `sounddevice`/PortAudio device selection pain on Linux | Medium | `--list_devices` from day one; `arecord -l` cross-check. Common and well-documented territory. |
| Distro isn't apt-based | Low | Only step 1 of `setup_nuc.sh` changes. |
| Scope creep into call-type | Medium | Explicitly out of scope (§0, Flag 2). The response map is keyed on species; call type slots in later without redesign. |

## 6. What this PoC does *not* prove

- Nothing about **call types / semantics**. That remains data-limited and unsolved.
- Nothing about **the installation's chosen digital path** — it tests the mic path that
  was deliberately dropped for the gallery piece.
- Nothing about **generalisation to unseen species**; the response map is a fixed lookup
  over species BirdNET already knows.

It proves the loop closes on real audio. That's the point.

## 7. Suggested order of work

1. Stage 0 (env + mic hardware) — blocking, do first, has a shipping delay if a mic must be bought.
2. Stage 1 (mic → existing infer_clip) — **the decisive measurement**.
3. Stages 2 → 3 → 4.
4. Stage 5 only if latency actually hurts.

Realistic total: **2–3 focused days** once the mic is in hand, assuming Stage 1 measures well.
