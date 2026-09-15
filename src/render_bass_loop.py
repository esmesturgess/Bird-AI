"""Turn the sound artist's low-frequency ambience into a file that loops with no seam.

The bass bed plays on its own (a Bluetooth speaker, via kiosk_play.sh), looping
independently of the video — it's an ambient layer, so it doesn't need to line up with
the birds. What it does need is a clean loop point: the source starts louder than it ends,
so a plain loop would audibly jump every four minutes. Fix: crossfade the last
--crossfade seconds over the first (equal-power), so the end flows straight back into the
start. The file gets that much shorter.

Written as FLAC, NOT AAC: lossy AAC pads the start and end of a file (encoder priming and
frame padding), which re-breaks the seam — measured 2026-09-15, the AAC version jumped
0.0145 at the loop point against a typical sample step of 0.0009, i.e. a tick every loop.

    python -m src.render_bass_loop \
        --src "path/to/Low frequency ambience.wav" \
        --out data/soundscapes/bass_ambience.flac
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def seamless_loop(y: np.ndarray, sr: int, crossfade_s: float) -> np.ndarray:
    """Fold the tail over the head with an equal-power crossfade; result loops cleanly."""
    x = int(crossfade_s * sr)
    t = np.linspace(0, np.pi / 2, x, dtype=np.float32)[:, None]
    head, tail = y[:x], y[-x:]
    out = y[:-x].copy()
    out[:x] = tail * np.cos(t) + head * np.sin(t)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("data/soundscapes/bass_ambience.flac"))
    p.add_argument("--crossfade", type=float, default=4.0, help="seconds")
    args = p.parse_args()

    import soundfile as sf
    y, sr = sf.read(str(args.src), dtype="float32", always_2d=True)
    out = seamless_loop(y, sr, args.crossfade)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.out), out, sr, format="FLAC", subtype="PCM_16")
    print(f"wrote {args.out}  ({len(out)/sr:.1f}s loop from {len(y)/sr:.1f}s source, "
          f"{args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
