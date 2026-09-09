# Soundscapes

`exhibition_v1.flac` — 12 segments × 25 s = 5 minutes. FLAC is lossless, so classification
is identical to the WAV it was made from, at half the size.

`exhibition_v1_truth.csv` — what is actually in each segment. Use it to check the runtime:
every row should match what `live_soundscape.py` prints at that timestamp.

`verification.csv` — the full audition. 72 candidate clips were classified and only the 36
the system got right were kept; these 12 are drawn from those.

## How it was built

Clips come from Xeno-canto and the Macaulay Library, labelled by the recordists. Each was
trimmed to 25 s, loudness-normalised so volume is not a cue, given 50 ms fades to avoid
clicks at the joins, and padded to exactly 25 s so the runtime's fixed segmentation lines up.
Species alternate, and no two neighbouring segments are the same bird.

## Honest note

These clips were **chosen because the system classifies them correctly** — 12/12 on the
dry run. That is legitimate curation for a demonstration, but it is not the system's
accuracy. On arbitrary recordings it manages roughly 50%, because it was trained on tidy
phrase segments and a whole 25 s recording contains wind, gaps and silence it never saw
in training. The ~0.80 figure in `models/README.md` refers to properly segmented material.

## Coverage

10 of 12 species × vocalisation combinations. Missing: blackbird juvenile and robin
juvenile — the two weakest cells in the project (0.690 and 0.493), where no candidate
classified correctly.

## The screen — `exhibition_v1.mp4`

1280×720, greyscale, 5:00, audio muxed in. Top: the segment's mel spectrogram with a
playhead sweeping across its 25 s. Bottom: the decision log filling in as the bird sings,
ending in the verdict.

**The similarity bars are real.** `src/render_visualisation.py` runs the actual classifier
on each segment and keeps all four prototype scores, so the screen shows genuine reasoning
— including the near-misses. Segment 04 reads song 0.906 against call 0.796, which is an
honest picture of how close some of these are.

Rebuild after changing the soundscape:

    python -m src.render_visualisation \
      --soundscape data/soundscapes/exhibition_v1.flac \
      --truth data/soundscapes/exhibition_v1_truth.csv \
      --out data/soundscapes/exhibition_v1.mp4

Delete `_sims_cache.json` first, or it will reuse the previous run's scores.

## Playing it on the NUC

    sudo apt install -y mpv
    mpv --fullscreen --loop-file=inf --no-osc data/soundscapes/exhibition_v1.mp4

That is the whole installation output — picture and sound, always in sync, no models
running. Note this makes the screen a faithful *recording* of the decision rather than one
being made live; the computation is identical either way because the soundscape is fixed.
Use `scripts/run_installation.sh` instead if the machine should genuinely classify on the night.
