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
