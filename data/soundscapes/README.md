# Soundscapes

`exhibition_v2.flac` — the current soundscape. 12 segments x 25s = 5 minutes, one clip per
species x vocalisation type, in randomised order, chosen by hand from a set of 38 candidates
the classifier gets right. `exhibition_v2_truth.csv` records what's actually in each segment.

`exhibition_v2.mp4` — the same soundscape with the installation's on-screen visualisation:
mel spectrogram, a live decision log, and the verdict, all real classifier output.

`for_sound_artist/` — the twelve source clips on their own, organised by species, with a
manifest carrying recordist/source/license for each. This is the pack to hand off.

Verified end to end: `scripts/live_soundscape.py --dry_run` on this file returns the correct
species and vocalisation for all 12 segments.

## Two things worth knowing

**Two of the twelve are training data, not held-out.** Blackbird juvenile and robin
juvenile only exist in the soundscape because every held-out candidate for those two cells
failed — the model was trained on both recordings used here. Flagged in
`for_sound_artist/manifest.csv` as `model_was_trained_on_this_recording`. Robin juvenile
specifically failed on 9 of 10 attempts even on clips it had already seen, which is a real
weak spot in the project, not a rounding error.

**Four source clips are shorter than the 25s slot** and are looped (with a short gap
between repeats) to fill it, rather than padded with silence. This matters for more than
sound: silence-padding was tried first and it measurably degraded classification — a
14-second dead-air tail diluted the per-window averaging enough to flip one owl alarm clip
to "call". Looping fixed it. `for_sound_artist/` ships the original un-looped recordings,
since a sound artist adapting these will likely want the natural clip rather than our
playback fix.

## Rebuilding

If the picks change, the soundscape needs the same duration handling — see
`for_sound_artist/README.md` for the picks-to-package pipeline, or ask me and I'll rebuild
both the audio and the video together so they never drift out of sync.

## Stereo responses (2026-09-11)

`exhibition_v2.mp4` now has real stereo audio: **left channel = the bird** (Speaker A,
outside the nest), **right channel = the response** (Speaker B, inside). Each segment is
two phases back to back — the bird sings for 25s while the spectrogram/log reveal as
before, then the screen holds on that finished frame (spectrogram complete, verdict
showing) while the response plays on its own, right channel only. Total length grew from
5:00 to 7:12 as a result. `exhibition_v2_truth_final.csv` has the real per-segment timing
(bird/response start times and durations both now vary — no longer a fixed 25s grid).

**Responses are currently the 4 generic recordings** (`data/response_sounds/*.m4a`,
matched by vocalisation type only — song/call/alarm/juvenile, not yet per-species), a
placeholder until the sound artist's 12 species-specific clips are ready. Swap them in
with:

    python -m src.render_exhibition \
      --soundscape data/soundscapes/exhibition_v2.flac \
      --truth data/soundscapes/exhibition_v2_truth.csv \
      --response_dir <new response folder, one file per vocalisation type> \
      --out data/soundscapes/exhibition_v2.mp4

Nothing else changes — `kiosk_play.sh` and the autostart setup already point at this
filename.

Minor known artifact: AAC's lossy stereo compression leaves a faint trace of the response
audio bleeding into the left channel during its silent window (~13% of the bird's own
level, checked by RMS) — a channel-separation limit of the lossy codec, not a routing bug.
Likely inaudible in practice; worth an ear-check in the actual room.

## countdown.mp4 — syncing two independent playback systems by hand

For the current plan — the NUC's stereo pair plays the translated/response sound, and a
separate speaker (its own device, not connected to the NUC) plays the raw bird sound —
there is no software link between the two, so starting them together is a manual,
physical action. `countdown.mp4` is a 6-second standalone clip for exactly that: 5, 4, 3,
2, 1, then a distinct GO (different tone, brighter flash). Play it once when you're ready
to start both systems; press play on the raw-sound device the instant GO appears, then
start (or let autostart handle) the main `exhibition_v2.mp4` loop.

Deliberately NOT baked into the main loop — the gallery-facing video shouldn't repeat a
countdown every 7 minutes. Rebuild with `python -m src.render_countdown`.

## Nelson's responses are in (2026-09-14)

The 12 species-specific translations from the sound artist replaced the 4 generic
placeholders. Source WAVs (44.1 kHz / 24-bit stereo) are downloaded and renamed to
`from_nelson/<species>_<vocalisation>.wav` — gitignored, since the master copies are on
his Dropbox and the NUC only needs the rendered video. He delivered them between 19.1s
and 22.1s long; `render_exhibition.py` fits every one to exactly **20s** (trimmed with a
0.5s fade-out, or padded with silence). Downmixed to mono for the single nest speaker.

The final mix is now 44.1 kHz / 192 kbps AAC (was 16 kHz, the classifier's rate, which
cut everything above 8 kHz off his work). Total loop: 9.0 min (12 × (25s bird + 20s
translation)).

On-screen vocalisation names changed (the model's internal labels are unchanged):
call → **Contact call**, song → **Mating signal**, alarm → **Alarm/distress call**,
juvenile → **Juvenile begging**.

Also in the delivery, NOT used yet: `Low frequency ambience.wav` (242s, stereo) — purpose
to be confirmed.

Rebuild:

    python -m src.render_exhibition \
      --soundscape data/soundscapes/exhibition_v2.flac \
      --truth data/soundscapes/exhibition_v2_truth.csv \
      --response_dir data/soundscapes/from_nelson \
      --out data/soundscapes/exhibition_v2.mp4

## Everything on a 20s grid, new running order, display numbers (2026-09-14)

**Every phase is now exactly 20s** — bird analysis 20s, translation 20s — so each segment
is 40s, every page change lands on a multiple of 20, and the loop is 8:00. The bird track
(`exhibition_v2.flac`) was rebuilt as 12 × 20s by taking the first 20s of each old 25s
clip, with a 0.3s fade-out. The on-screen clock now counts real elapsed time continuously
across BOTH pages (it used to show only bird time and skip the translations).

**Running order** — each block of four has one of every type; no neighbours share a
species or type, including where the loop wraps from 11 back to 0; opens on a song:

| # | start | species | type |
|---|---|---|---|
| 0 | 0:00 | Blackbird | Mating signal |
| 1 | 0:40 | Robin | Alarm/distress call |
| 2 | 1:20 | Tawny owl | Juvenile begging |
| 3 | 2:00 | Robin | Contact call |
| 4 | 2:40 | Tawny owl | Mating signal |
| 5 | 3:20 | Blackbird | Contact call |
| 6 | 4:00 | Robin | Juvenile begging |
| 7 | 4:40 | Tawny owl | Alarm/distress call |
| 8 | 5:20 | Blackbird | Juvenile begging |
| 9 | 6:00 | Robin | Mating signal |
| 10 | 6:40 | Blackbird | Alarm/distress call |
| 11 | 7:20 | Tawny owl | Contact call |

The model still identifies all 12 correctly (species and type) on the shorter 20s clips.

**Displayed numbers are partly a presentation layer (artist's decision).** BirdNET's real
species confidences ranged 0.17-1.00 (0.17 on a clear robin song, 1.00 on the owls), which
read as the model doing badly. On screen they are re-spread into 0.80-0.89 **by rank**, so
the least-sure clip still shows lowest. Negative vocalisation similarities are shown as
0.000. Which species and which type wins is always the model's real answer. Real and
shown values are both in `exhibition_v2_truth_final.csv`
(`species_conf_real` / `species_conf_shown`); `--real_numbers` renders the raw values.
**Don't quote the on-screen confidences as model performance** — the honest accuracy
figures are the nested-CV ones (~0.80) in the project notes.
