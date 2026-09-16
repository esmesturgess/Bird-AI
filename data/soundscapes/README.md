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
clip, with a 0.3s fade-out. Nothing on screen counts any more — no segment number, no clock, no `--segment NN` in
the log — so visitors can't tell where the 8-minute loop restarts.

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

**Some species confidences on screen are adjusted (artist's decision).** Only the ones that
read as broken were changed: anything under 0.80 (0.17 on a clear robin song, 0.28, 0.42)
or that would print as 0.99/1.00 (the four owls). Those 7 are moved into 0.81-0.89, spread
by rank among themselves so the least-sure still shows lowest. The other 5 (0.87-0.96) are
BirdNET's real values, as are all vocalisation scores (negatives included) and which
species/type wins. Real and shown values are both in `exhibition_v2_truth_final.csv`
(`species_conf_real` / `species_conf_shown`); `--real_numbers` renders every raw value.
**Don't quote the on-screen confidences as model performance** — the honest accuracy
figures are the nested-CV ones (~0.80) in the project notes.

## Bass on Bluetooth, timed to the translations (2026-09-16)

`bass_track.flac` is the sound artist's 12 numbered bass clips (`1.wav`..`12.wav`, in the
running order) laid onto one 480s track that matches the video exactly: **silence through
every 20s analysis page, the matching bass through every 20s translation**. So the bass is
heard only under a translation, never over the loading page. Built with
`python -m src.render_bass_track --bass_dir <folder>`; verified silent (RMS 0.000000) in
all 12 analysis windows and sounding in all 12 translation windows.

**The 20s grid holds whatever length the clips are.** Each clip starts exactly on its
window boundary; a longer one is trimmed with a fade, a shorter one leaves silence at the
END of its slot, so a wrong-length clip can never shift the ones after it. Measured on the
built track: every clip starts within **0.2 ms** of its boundary, and no bass is audible in
any analysis window. `render_bass_track.py` now asserts both, so a future set of clips
can't quietly break it.

Trailing gaps from clips shorter than 20s: 0.44s on clips 1, 2 and 6, 0.21s on 5, 0.13s on
7, 0.02s on 9 — and **7.6s on clip 11** (blackbird alarm, 12.4s long), which is the only
one big enough to hear as an early ending.

One small mismatch worth knowing: the video's audio decodes to 480.003s against the bass
track's exact 480.000s (128 samples of AAC padding), so across a whole day of looping the
two can creep about 0.2s apart from that alone — well under the drift between two separate
audio clocks, and it resets whenever the NUC restarts.

It plays on a Bluetooth speaker when `kiosk_play.sh` is given `BASS_BT_MAC` (setup in
`scripts/README_NUC.md`). Because it is the same length as the video and both start
together, they stay in step; if the speaker connects late or drops out, the bass restarts
at the video's CURRENT position rather than from the beginning. `BASS_LEAD_S` (default
0.2s) starts it slightly early to offset Bluetooth's lag.

Superseded: `bass_ambience.flac` (a continuous 3:58 bed made from `Low frequency
ambience.wav`) and `src/render_bass_loop.py` — removed once the real, per-translation bass
clips arrived. Both are recoverable from git history.

To check the two outputs on any machine, with headphones and a Bluetooth speaker and no
NUC: `python scripts/test_two_outputs.py --list`, then `--main <headphones> --bass <speaker>`.
