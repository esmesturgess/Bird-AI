"""Build the final exhibition asset: stereo audio (bird on the left, response on the
right) + the visualisation, muxed into one video.

Each segment now has two phases, back to back:
  1. BIRD (25s) — left channel only, right silent. Unchanged from render_visualisation.py:
     the mel spectrogram reveals progressively, the decision log fills in, ending on the
     verdict line.
  2. RESPONSE (however long that clip is) — right channel only, left silent. The screen
     holds on phase 1's final frame (spectrogram fully revealed, verdict showing) while
     the answer plays, so the viewer is looking at "what it decided" exactly while
     hearing the reply to it.

Response clips are matched by VOCALISATION TYPE only (4 generic recordings), not yet by
species — a placeholder until the sound artist's 12 species-specific clips replace them.
Swap them in later with --response_dir pointing at the new set; nothing else changes.

    python -m src.render_exhibition \
        --soundscape data/soundscapes/exhibition_v2.flac \
        --truth data/soundscapes/exhibition_v2_truth.csv \
        --response_dir data/response_sounds \
        --out data/soundscapes/exhibition_v2.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .render_visualisation import (
    SR, FPS, W, H, MARGIN, SPEC_TOP, SPEC_BOT, TERM_TOP, LINE_H,
    BAR_CELLS, BAR_START, BAR_END, F_SM, F_MD,
    mel_image, decision_lines, gather_similarities,
)

FADE_S = 0.05


def _fade(y: np.ndarray) -> np.ndarray:
    n = int(FADE_S * SR)
    if len(y) > 2 * n:
        y = y.copy()
        y[:n] *= np.linspace(0, 1, n)
        y[-n:] *= np.linspace(1, 0, n)
    return y


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--soundscape", type=Path, required=True, help="the BIRD-only track (left channel source)")
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument("--response_dir", type=Path, default=Path("data/response_sounds"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--out_truth", type=Path, default=None,
                   help="where to write the updated per-segment timing (default: alongside --truth, _final suffix)")
    p.add_argument("--models_dir", type=Path, default=Path("models"))
    p.add_argument("--segment_seconds", type=float, default=25.0)
    p.add_argument("--birdnet_binary", default="./.venv/bin/birdnet-analyze")
    p.add_argument("--sims_cache", type=Path, default=Path("data/soundscapes/_sims_cache.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import librosa, imageio.v2 as imageio, imageio_ffmpeg, soundfile as sf

    truth = pd.read_csv(args.truth)
    audio = librosa.load(str(args.soundscape), sr=SR, mono=True)[0]
    n = int(args.segment_seconds * SR)
    bird_segments = [audio[i:i + n] for i in range(0, len(audio), n)]
    bird_segments = [s for s in bird_segments if len(s) >= SR]
    assert len(bird_segments) == len(truth), \
        f"{len(bird_segments)} audio segments vs {len(truth)} truth rows — do these match?"

    print("classifying each segment for the real decision log ...", flush=True)
    info = gather_similarities(args, bird_segments)

    responses = {}
    for voc in truth["vocalisation"].unique():
        f = args.response_dir / f"{voc}.m4a"
        if not f.exists():
            raise SystemExit(f"no response clip for vocalisation '{voc}': expected {f}")
        y = librosa.load(str(f), sr=SR, mono=True)[0]
        responses[voc] = _fade(y)
        print(f"  response[{voc}] = {f.name} ({len(y)/SR:.1f}s)", flush=True)

    specs = [mel_image(s) for s in bird_segments]
    logs = [decision_lines(i, d["sci"] or "no match", d.get("species"), d["voc"] or "-", d["sims"], d["conf"])
            for i, d in enumerate(info)]

    tmp_video = Path(tempfile.mkstemp(suffix=".mp4")[1])
    writer = imageio.get_writer(tmp_video, fps=FPS, codec="libx264", quality=7, macro_block_size=1)
    spec_w = W - 2 * MARGIN
    frames_per_bird_seg = int(args.segment_seconds * FPS)

    left_chunks, right_chunks, timing_rows = [], [], []
    t_cursor = 0.0

    for si, bird in enumerate(bird_segments):
        d, lines = info[si], logs[si]
        voc = truth.iloc[si]["vocalisation"]
        resp = responses[voc]
        resp_frames = max(1, int(round(len(resp) / SR * FPS)))

        final_frame = None
        for f in range(frames_per_bird_seg):
            prog = f / frames_per_bird_seg
            im = Image.new("L", (W, H), 0)
            dr = ImageDraw.Draw(im)
            dr.text((MARGIN, 34), f"SEGMENT {si:02d} / {len(bird_segments)-1:02d}", font=F_MD, fill=150)
            dr.text((W - MARGIN - 210, 34), f"t = {si*args.segment_seconds + prog*args.segment_seconds:6.1f} s",
                    font=F_MD, fill=150)
            revealed_w = max(1, int(prog * spec_w))
            im.paste(specs[si].crop((0, 0, revealed_w, specs[si].height)), (MARGIN, SPEC_TOP))
            dr.rectangle([MARGIN, SPEC_TOP, W - MARGIN, SPEC_BOT], outline=90)
            x = MARGIN + int(prog * spec_w)
            dr.line([x, SPEC_TOP, x, SPEC_BOT], fill=255, width=2)
            dr.text((MARGIN, SPEC_BOT + 8), "mel spectrogram  ·  8 kHz", font=F_SM, fill=110)
            dr.line([MARGIN, TERM_TOP - 18, W - MARGIN, TERM_TOP - 18], fill=70)
            y = TERM_TOP
            for frac, text in lines:
                if prog >= frac:
                    dr.text((MARGIN, y), text, font=F_MD, fill=205)
                    y += LINE_H
                if frac == 0.08 and prog >= BAR_START:
                    fill_frac = min(1.0, (prog - BAR_START) / (BAR_END - BAR_START))
                    done = int(round(fill_frac * BAR_CELLS))
                    bar = "█" * done + "░" * (BAR_CELLS - done)
                    dr.text((MARGIN, y), f"  [{bar}] {fill_frac*100:3.0f}%", font=F_MD, fill=205)
                    y += LINE_H
            frame = np.array(im.convert("RGB"))
            writer.append_data(frame)
            final_frame = frame

        # phase 2: hold the finished frame while the response plays on the right channel
        for _ in range(resp_frames):
            writer.append_data(final_frame)

        left_chunks.append(_fade(bird)); left_chunks.append(np.zeros_like(resp))
        right_chunks.append(np.zeros_like(bird)); right_chunks.append(resp)
        timing_rows.append({"segment": si, "bird_start_s": t_cursor, "bird_duration_s": len(bird)/SR,
                            "response_start_s": t_cursor + len(bird)/SR, "response_duration_s": len(resp)/SR,
                            "species": truth.iloc[si]["species"], "vocalisation": voc})
        t_cursor += len(bird)/SR + len(resp)/SR
        print(f"  rendered segment {si} ({voc}, +{len(resp)/SR:.1f}s response)", flush=True)

    writer.close()

    left = np.concatenate(left_chunks); right = np.concatenate(right_chunks)
    stereo = np.stack([left, right], axis=1).astype(np.float32)
    audio_tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    sf.write(audio_tmp, stereo, SR)

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(tmp_video), "-i", str(audio_tmp),
                    "-c:v", "copy", "-c:a", "aac", "-shortest", str(args.out)], check=True)
    tmp_video.unlink(missing_ok=True); audio_tmp.unlink(missing_ok=True)

    out_truth = args.out_truth or args.truth.with_name(args.truth.stem + "_final.csv")
    pd.DataFrame(timing_rows).to_csv(out_truth, index=False)

    mb = args.out.stat().st_size / 1e6
    print(f"\nwrote {args.out}  ({mb:.1f} MB, {t_cursor/60:.1f} min)")
    print(f"wrote {out_truth}")


if __name__ == "__main__":
    main()
