"""Lay the sound artist's 12 bass clips onto one 8-minute track that lines up with the video.

The bass must sound ONLY while a translation is playing, never over the analysis page. So
this builds a full-length track the same duration as exhibition_v2.mp4: silence through
every 20s bird/analysis phase, and clip N through every 20s translation phase, in the
running order (1 = the first segment of the video, 2 = the second, and so on).

Because it is the same length as the video, the two stay lined up when both are started
together — the video (and birds) on the headphone jack, this on a Bluetooth speaker.

THE GRID IS FIXED, WHATEVER LENGTH THE CLIPS ARE. Every clip starts exactly on its 20s
boundary; a clip longer than the window is trimmed with a fade, a shorter one leaves
silence at the END of its slot. A wrong-length clip can therefore never shift the ones
after it. This is checked, not assumed — see the assertions at the end.

LEVELS are set in two steps (artist's direction, 2026-09-16):

  1. MATCH every clip to the same loudness, so a call type sounds identical whichever bird
     it belongs to. As delivered they ranged over 4.0 dB, and within one type by up to
     3.7 dB (blackbird contact call vs robin contact call), which is plainly audible.
     Loudness is measured as RMS over the sounding part of the clip, ignoring any padded
     silence. The target is the median of the twelve, so overall loudness barely moves.

  2. GRADE by call type on top of that: juvenile well down, contact call and mating signal
     down, alarm up — a deliberate 10 dB spread from the softest type to the loudest.
     Retune with --gain_db, e.g.
     `--gain_db juvenile=-7,alarm=+2`; anything not named keeps its default below.
     --no_match skips step 1 and leaves the artist's own relative levels alone.

    python -m src.render_bass_track \
        --bass_dir <folder of 1.wav ... 12.wav> \
        --truth data/soundscapes/exhibition_v2_truth_final.csv \
        --out data/soundscapes/bass_track.flac
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SR = 44100
FADE_IN_S = 0.02
TRIM_FADE_S = 0.5
AUDIBLE = 1e-5
SOUNDING = 1e-4          # above this counts as "the clip is playing", not padded silence

# on-screen names: juvenile = "Juvenile begging", call = "Contact call",
# song = "Mating signal", alarm = "Alarm/distress call"
GAIN_DB = {"juvenile": -16.0, "call": -3.0, "song": -3.0, "alarm": +2.0}
# Widened 2026-09-16 at the artist's request (alarm bassier, juvenile weaker). The whole
# set sits 1 dB lower than the literal +3/-7 asked for: that version peaked at -0.1 dB on
# the tawny owl alarm, i.e. no headroom at all. Contrast is identical either way — alarm
# 5 dB above contact/mating, 10 dB above juvenile — so this trades 1 dB of loudness,
# recoverable on the speaker volume, for 1 dB of safety.


def parse_gains(spec: str | None) -> dict[str, float]:
    gains = dict(GAIN_DB)
    for part in filter(None, (spec or "").split(",")):
        k, _, v = part.partition("=")
        k = k.strip().lower()
        if k not in gains:
            raise SystemExit(f"unknown call type {k!r} — expected one of {', '.join(gains)}")
        gains[k] = float(v)
    return gains


def db(x: float) -> float:
    return 20 * np.log10(max(x, 1e-9))


def sounding_rms(y: np.ndarray) -> float:
    """Loudness of the part that is actually playing, ignoring padded silence."""
    live = y[np.abs(y).max(axis=1) > SOUNDING]
    return float(np.sqrt(np.mean(live ** 2))) if len(live) else 0.0


def fit_window(y: np.ndarray, n: int) -> np.ndarray:
    """Trim (with a fade-out) or pad with silence so the clip fits its window exactly."""
    out = np.zeros((n, y.shape[1]), dtype=np.float32)
    if len(y) > n:
        y = y[:n].copy()
        f = min(n, int(TRIM_FADE_S * SR))
        y[-f:] *= np.linspace(1, 0, f)[:, None]
    out[:len(y)] = y
    f = min(len(out), int(FADE_IN_S * SR))
    out[:f] *= np.linspace(0, 1, f)[:, None]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bass_dir", type=Path, required=True)
    p.add_argument("--truth", type=Path, default=Path("data/soundscapes/exhibition_v2_truth_final.csv"))
    p.add_argument("--out", type=Path, default=Path("data/soundscapes/bass_track.flac"))
    p.add_argument("--gain_db", help="per call type, e.g. juvenile=-7,alarm=+2")
    p.add_argument("--base_db", type=float, help="loudness to match every clip to (default: their median)")
    p.add_argument("--no_match", action="store_true", help="keep the artist's own relative levels")
    args = p.parse_args()

    import librosa, soundfile as sf
    gains = parse_gains(args.gain_db)
    truth = pd.read_csv(args.truth)

    clips = []                                               # load and measure first
    for row in truth.itertuples():
        src = args.bass_dir / f"{row.segment + 1}.wav"        # clips are numbered from 1
        if not src.exists():
            raise SystemExit(f"missing bass clip for segment {row.segment}: {src}")
        y, _ = librosa.load(str(src), sr=SR, mono=False)
        y = np.atleast_2d(y).T if y.ndim == 1 else y.T        # librosa gives (channels, samples)
        if y.shape[1] == 1:
            y = np.repeat(y, 2, axis=1)
        y = y.astype(np.float32)
        clips.append((row, src, y, db(sounding_rms(y))))

    base = args.base_db if args.base_db is not None else float(np.median([c[3] for c in clips]))
    print(f"  matching every clip to {base:.1f} dB, then: "
          + ", ".join(f"{k} {v:+.1f}" for k, v in gains.items()) + " dB\n")
    print(f"  {'#':>2} {'species':<10} {'call type':<9} {'as sent':>8} {'match':>7} {'type':>6} {'final':>7} {'peak':>7}")

    total = float(truth.response_start_s.iloc[-1] + truth.response_duration_s.iloc[-1])
    track = np.zeros((int(round(total * SR)), 2), dtype=np.float32)
    windows = []

    for row, src, y, measured in clips:
        match = 0.0 if args.no_match else base - measured
        total_db = match + gains[row.vocalisation]
        y = y * (10 ** (total_db / 20))
        start = int(round(row.response_start_s * SR))
        n = int(round(row.response_duration_s * SR))
        assert start + n <= len(track), f"segment {row.segment} runs past the end of the track"
        assert not windows or start >= windows[-1][1], f"segment {row.segment} overlaps the previous one"
        track[start:start + n] = fit_window(y, n)
        windows.append((start, start + n))
        placed = track[start:start + n]
        print(f"  {row.segment + 1:>2} {row.species:<10} {row.vocalisation:<9} {measured:>8.1f} "
              f"{match:>+7.1f} {gains[row.vocalisation]:>+6.1f} {db(sounding_rms(placed)):>7.1f} "
              f"{db(float(np.abs(placed).max())):>7.1f}", flush=True)

    # --- prove the grid held, rather than trusting that it did ---
    loud = np.abs(track).max(axis=1)
    worst_ms = 0.0
    for (start, end), row in zip(windows, truth.itertuples()):
        nz = np.nonzero(loud[start:end] > AUDIBLE)[0]
        assert len(nz), f"segment {row.segment + 1} has no bass in its window"
        worst_ms = max(worst_ms, nz[0] / SR * 1000)
    for row in truth.itertuples():                            # analysis pages must be silent
        a = int(round(row.bird_start_s * SR))
        b = a + int(round(row.bird_duration_s * SR))
        assert loud[a:b].max() <= AUDIBLE, f"bass leaks into segment {row.segment + 1}'s analysis page"
    assert len(track) == int(round(total * SR)), "track length drifted from the video's"
    peak = float(np.abs(track).max())
    assert peak < 1.0, f"raising a level clipped the track (peak {peak:.3f})"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.out), track, SR, format="FLAC", subtype="PCM_16")
    print(f"\nalignment checked: every clip starts within {worst_ms:.1f} ms of its 20s boundary, "
          f"and no bass sounds during any analysis page")
    print(f"wrote {args.out}  ({len(track)/SR/60:.2f} min, peak {db(peak):.1f} dB, "
          f"{args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
