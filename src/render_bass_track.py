"""Lay the sound artist's 12 bass clips onto one 8-minute track that lines up with the video.

The bass must sound ONLY while a translation is playing, never over the analysis page. So
this builds a full-length track the same duration as exhibition_v2.mp4: silence through
every 20s bird/analysis phase, and clip N through every 20s translation phase, in the
running order (1 = the first segment of the video, 2 = the second, and so on).

Because it is the same length as the video, the two stay lined up when both are started
together — the video (and birds) on the headphone jack, this on a Bluetooth speaker.

Clips longer than the translation window are trimmed with a fade; shorter ones start with
the translation and leave silence at the end of the window.

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
    args = p.parse_args()

    import librosa, soundfile as sf
    truth = pd.read_csv(args.truth)
    total = float(truth.response_start_s.iloc[-1] + truth.response_duration_s.iloc[-1])
    track = np.zeros((int(round(total * SR)), 2), dtype=np.float32)

    for row in truth.itertuples():
        src = args.bass_dir / f"{row.segment + 1}.wav"      # clips are numbered from 1
        if not src.exists():
            raise SystemExit(f"missing bass clip for segment {row.segment}: {src}")
        y, _ = librosa.load(str(src), sr=SR, mono=False)
        y = np.atleast_2d(y).T if y.ndim == 1 else y.T       # librosa gives (channels, samples)
        if y.shape[1] == 1:
            y = np.repeat(y, 2, axis=1)
        natural = len(y) / SR
        start, n = int(round(row.response_start_s * SR)), int(round(row.response_duration_s * SR))
        track[start:start + n] = fit_window(y.astype(np.float32), n)
        note = "trimmed" if natural > row.response_duration_s + 0.01 else (
               f"{row.response_duration_s - natural:.1f}s silence at the end"
               if natural < row.response_duration_s - 0.01 else "exact")
        print(f"  {row.segment + 1:2d}. {row.species:10s} {row.vocalisation:9s} "
              f"{src.name:7s} {natural:5.2f}s -> {row.response_start_s:5.0f}-"
              f"{row.response_start_s + row.response_duration_s:5.0f}s  ({note})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.out), track, SR, format="FLAC", subtype="PCM_16")
    peak = float(np.abs(track).max())
    print(f"\nwrote {args.out}  ({len(track)/SR/60:.2f} min, peak {peak:.2f}, "
          f"{args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
