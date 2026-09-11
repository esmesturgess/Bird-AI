"""Build a standalone sync countdown: 5, 4, 3, 2, 1, GO — for lining up two independent
playback systems (e.g. the NUC's translated-sound video and a separate speaker playing
the raw bird sound) that have no way to be started together in software.

This is deliberately its OWN small file, not baked into exhibition_v2.mp4 — the main
gallery loop shouldn't repeat a countdown every 7 minutes. Play this once, by hand, at
the moment you're about to start both systems: press play on the raw-sound device the
instant "GO" appears/sounds, then start (or let autostart handle) the main exhibition loop.

    python -m src.render_countdown --out data/soundscapes/countdown.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
FPS = 12
SR = 16000
FONT_PATH = "/System/Library/Fonts/Menlo.ttc"


def font(size: int):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


F_HUGE, F_LABEL = font(260), font(20)


def beep(freq: float, dur_s: float, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, dur_s, int(sr * dur_s), endpoint=False)
    env = np.minimum(1.0, np.minimum(t / 0.01, (dur_s - t) / 0.01))  # 10ms fade in/out, no clicks
    return (0.5 * np.sin(2 * np.pi * freq * t) * env).astype(np.float32)


def frame(text: str, sub: str = "") -> Image.Image:
    im = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(im)
    dr.text((46, 34), "SYNC COUNTDOWN", font=F_LABEL, fill=140)
    if sub:
        tw = dr.textlength(sub, font=F_LABEL)
        dr.text(((W - tw) / 2, H - 70), sub, font=F_LABEL, fill=140)
    tw = dr.textlength(text, font=F_HUGE)
    bbox = dr.textbbox((0, 0), text, font=F_HUGE)
    th = bbox[3] - bbox[1]
    dr.text(((W - tw) / 2, (H - th) / 2 - bbox[1]), text, font=F_HUGE, fill=255)
    return im


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/soundscapes/countdown.mp4"))
    p.add_argument("--go_hold_s", type=float, default=1.0, help="how long the GO frame stays up")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import imageio.v2 as imageio, imageio_ffmpeg, soundfile as sf

    tmp_video = Path(tempfile.mkstemp(suffix=".mp4")[1])
    writer = imageio.get_writer(tmp_video, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    audio_chunks = []

    counts = ["5", "4", "3", "2", "1"]
    for n in counts:
        im = frame(n, sub="press play on the other device when GO appears")
        for _ in range(FPS):                       # 1.0s each
            writer.append_data(np.array(im.convert("RGB")))
        audio_chunks.append(beep(880, 0.15))
        audio_chunks.append(np.zeros(SR - int(0.15 * SR), dtype=np.float32))

    go_im = frame("GO", sub="")
    go_frames = max(1, int(args.go_hold_s * FPS))
    for _ in range(go_frames):
        writer.append_data(np.array(go_im.convert("RGB")))
    audio_chunks.append(beep(1320, 0.35))
    audio_chunks.append(np.zeros(max(0, int(args.go_hold_s * SR) - int(0.35 * SR)), dtype=np.float32))

    writer.close()
    mono = np.concatenate(audio_chunks)
    stereo = np.stack([mono, mono], axis=1)         # same cue on both channels/speakers
    audio_tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    sf.write(audio_tmp, stereo, SR)

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(tmp_video), "-i", str(audio_tmp),
                    "-c:v", "copy", "-c:a", "aac", "-shortest", str(args.out)], check=True)
    tmp_video.unlink(missing_ok=True); audio_tmp.unlink(missing_ok=True)
    total = len(counts) + args.go_hold_s
    print(f"wrote {args.out}  ({args.out.stat().st_size/1e6:.2f} MB, {total:.1f}s)")


if __name__ == "__main__":
    main()
