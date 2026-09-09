"""Render the installation's screen: mel spectrogram + live decision log, muxed to the audio.

Produces a single self-contained video file. Playing a video rather than driving a live
display is deliberate — on the night nothing can drift out of sync, drop frames, or crash,
and any media player on the NUC can show it fullscreen.

The decision log is not decorative: the prototype similarity bars are the real numbers the
classifier produced for that segment, so the screen shows the actual reasoning.

    python -m src.render_visualisation \
        --soundscape data/soundscapes/exhibition_v1.flac \
        --truth data/soundscapes/exhibition_v1_truth.csv \
        --out data/soundscapes/exhibition_v1.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
FPS = 12
SR = 16000
MARGIN = 46
SPEC_TOP, SPEC_BOT = 88, 368
TERM_TOP = 414
LINE_H = 19
BAR_CELLS, BAR_START, BAR_END = 34, 0.12, 0.52
RESULT_GAP = 14          # extra breathing room above the final result line
FONT_PATH = "/System/Library/Fonts/Menlo.ttc"


def font(size: int):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


F_SM, F_MD, F_RESULT = font(15), font(17), font(26)


def mel_image(y: np.ndarray) -> Image.Image:
    """One segment as a grayscale mel spectrogram, quiet = black, loud = white."""
    import librosa
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=128, fmax=8000)
    db = librosa.power_to_db(m, ref=np.max)
    lo, hi = -70.0, 0.0
    norm = np.clip((db - lo) / (hi - lo), 0, 1)
    img = (norm * 255).astype(np.uint8)[::-1]                       # low frequencies at the bottom
    return Image.fromarray(img, mode="L").resize((W - 2 * MARGIN, SPEC_BOT - SPEC_TOP),
                                                 Image.BILINEAR)


def decision_lines(seg: int, species_sci: str, species_slug: str | None, voc: str,
                   sims: dict[str, float], species_conf: float) -> list[tuple[float, str, bool]]:
    """(fraction of the segment elapsed, line, is_the_final_result) — the log fills in as
    the bird sings, ending with the result on its own line, same font family as everything
    above it, just larger."""
    order = sorted(sims, key=lambda k: -sims[k])
    L: list[tuple[float, str, bool]] = [
        (0.02, f"$ classify --segment {seg:02d}", False),
        (0.08, f"  {species_sci:<18s} {species_conf:.2f}", False),
        (0.55, "  nearest prototype:", False),
    ]
    t = 0.60
    for k in order:
        bar = "#" * int(round(sims[k] * 34))
        mark = "<--" if k == voc else "   "
        L.append((t, f"      {k:<9s} {sims[k]:.3f}  {bar:<34s} {mark}", False))
        t += 0.05

    species_disp = species_slug.replace("_", " ").title() if species_slug else "Unknown"
    voc_disp = voc.title() if voc and voc != "-" else "Unknown"
    L.append((min(t + 0.03, 0.94), f"Species: {species_disp}  |  Vocalisation type: {voc_disp}", True))
    return L


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--soundscape", type=Path, required=True)
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--models_dir", type=Path, default=Path("models"))
    p.add_argument("--segment_seconds", type=float, default=25.0)
    p.add_argument("--birdnet_binary", default="./.venv/bin/birdnet-analyze")
    p.add_argument("--sims_cache", type=Path, default=Path("data/soundscapes/_sims_cache.json"))
    return p.parse_args()


def gather_similarities(args, segments) -> list[dict]:
    """Run the real classifier once per segment and keep ALL four prototype scores."""
    if args.sims_cache.exists():
        return json.loads(args.sims_cache.read_text())
    import soundfile as sf
    from .classify_clip import Classifier, COMMON_NAMES, _l2
    clf = Classifier(args.models_dir, birdnet_binary=args.birdnet_binary)
    out = []
    with tempfile.TemporaryDirectory() as td:
        for i, seg in enumerate(segments):
            p = Path(td) / f"s{i}.wav"
            sf.write(p, seg, SR)
            sci, conf = clf.detect_species(p, 0.10)
            if sci is None:
                out.append({"species": "unknown", "sci": "", "conf": float(conf),
                            "sims": {}, "voc": None}); continue
            slug = COMMON_NAMES[sci]
            import librosa
            y = librosa.load(str(p), sr=SR, mono=True)[0]
            feats = clf._embed_windows(y, 6.0, 3.0)
            head = clf.heads[slug]
            import torch
            with torch.no_grad():
                z = head["model"](torch.tensor(feats)).numpy()
            s = (_l2(z) @ head["prototypes"].T).mean(axis=0)
            sims = {lab: float(v) for lab, v in zip(head["labels"], s)}
            out.append({"species": slug, "sci": sci, "conf": float(conf), "sims": sims,
                        "voc": max(sims, key=sims.get)})
            print(f"  segment {i}: {slug} {out[-1]['voc']}", flush=True)
    args.sims_cache.parent.mkdir(parents=True, exist_ok=True)
    args.sims_cache.write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    args = parse_args()
    import librosa, imageio.v2 as imageio, imageio_ffmpeg

    audio = librosa.load(str(args.soundscape), sr=SR, mono=True)[0]
    n = int(args.segment_seconds * SR)
    segments = [audio[i:i + n] for i in range(0, len(audio), n)]
    segments = [s for s in segments if len(s) >= SR]
    truth = pd.read_csv(args.truth)
    print(f"{len(segments)} segments", flush=True)

    print("running the classifier to capture real prototype scores ...", flush=True)
    info = gather_similarities(args, segments)

    specs = [mel_image(s) for s in segments]
    logs = [decision_lines(i, d["sci"] or "no match", d.get("species"), d["voc"] or "-",
                           d["sims"], d["conf"])
            for i, d in enumerate(info)]

    frames_per_seg = int(args.segment_seconds * FPS)
    tmp_video = Path(tempfile.mkstemp(suffix=".mp4")[1])
    writer = imageio.get_writer(tmp_video, fps=FPS, codec="libx264",
                                quality=7, macro_block_size=1)
    spec_w = W - 2 * MARGIN

    for si, seg in enumerate(segments):
        d, lines = info[si], logs[si]
        for f in range(frames_per_seg):
            prog = f / frames_per_seg
            im = Image.new("L", (W, H), 0)
            dr = ImageDraw.Draw(im)

            dr.text((MARGIN, 34), f"SEGMENT {si:02d} / {len(segments)-1:02d}", font=F_MD, fill=150)
            dr.text((W - MARGIN - 210, 34),
                    f"t = {si*args.segment_seconds + prog*args.segment_seconds:6.1f} s", font=F_MD, fill=150)

            im.paste(specs[si], (MARGIN, SPEC_TOP))
            dr.rectangle([MARGIN, SPEC_TOP, W - MARGIN, SPEC_BOT], outline=90)
            x = MARGIN + int(prog * spec_w)
            dr.line([x, SPEC_TOP, x, SPEC_BOT], fill=255, width=2)
            dr.text((MARGIN, SPEC_BOT + 8), "mel spectrogram  ·  8 kHz", font=F_SM, fill=110)

            dr.line([MARGIN, TERM_TOP - 18, W - MARGIN, TERM_TOP - 18], fill=70)
            y = TERM_TOP
            for frac, text, is_result in lines:
                if prog >= frac:
                    if is_result:
                        y += RESULT_GAP
                        dr.text((MARGIN, y), text, font=F_RESULT, fill=205)
                        y += F_RESULT.size + 6
                    else:
                        dr.text((MARGIN, y), text, font=F_MD, fill=205)
                        y += LINE_H
                if frac == 0.08 and prog >= BAR_START:      # the bar sits under the read line
                    fill_frac = min(1.0, (prog - BAR_START) / (BAR_END - BAR_START))
                    done = int(round(fill_frac * BAR_CELLS))
                    bar = "\u2588" * done + "\u2591" * (BAR_CELLS - done)
                    dr.text((MARGIN, y), f"  [{bar}] {fill_frac*100:3.0f}%", font=F_MD, fill=205)
                    y += LINE_H

            writer.append_data(np.array(im.convert("RGB")))
        print(f"  rendered segment {si}", flush=True)
    writer.close()

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(tmp_video),
                    "-i", str(args.soundscape), "-c:v", "copy", "-c:a", "aac",
                    "-shortest", str(args.out)], check=True)
    tmp_video.unlink(missing_ok=True)
    mb = args.out.stat().st_size / 1e6
    print(f"\nwrote {args.out}  ({mb:.1f} MB, {len(segments)*args.segment_seconds/60:.1f} min)")


if __name__ == "__main__":
    main()
