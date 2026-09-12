"""Build the final exhibition asset: stereo audio + narrated visualisation, muxed to video.

Each segment has two phases, back to back:

  1. BIRD (25s) — the raw call plays on the LEFT channel while the analysis runs on screen.
     A synthesised voice narrates it, timed to the visuals and centred across both
     channels: "Detecting species..." over the loading bar, then the species said
     colloquially ("Tawny owl", not Strix aluco) as it resolves, then "Adapting sounds to
     the bird's emotions..." once the vocalisation-type scores are in. The bird audio ducks
     under each spoken line so the voice stays intelligible.

  2. TRANSLATION (however long that clip is) — the response plays on the RIGHT channel
     (Speaker B, inside the nest) and the screen clears to a status page reading
     "Translation playing inside the nest...".

Narration is generated with macOS `say`, so this must be RENDERED ON A MAC — the NUC only
ever plays the finished file.

    python -m src.render_exhibition \
        --soundscape data/soundscapes/exhibition_v2.flac \
        --truth data/soundscapes/exhibition_v2_truth.csv \
        --response_dir data/response_sounds \
        --out data/soundscapes/exhibition_v2.mp4
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .render_visualisation import (
    SR, FPS, W, H, MARGIN, SPEC_TOP, SPEC_BOT, TERM_TOP, LINE_H,
    BAR_CELLS, F_SM, F_MD, mel_image, gather_similarities,
)

FADE_S = 0.05
NARRATION_VOICE = "Daniel"          # en_GB — these are all British birds
NARRATION_GAIN = 0.95
DUCK_LEVEL = 0.35                   # how far the bird drops under a spoken line
DUCK_RAMP_S = 0.12

COLLOQUIAL = {"blackbird": "Blackbird", "robin": "Robin", "tawny_owl": "Tawny owl"}

# Fractions of the 25s bird phase. Retimed so the narration lands ON the thing it names:
# the species is SPOKEN as its line appears, not before the bar that finds it.
T_CMD, T_VOICE_DETECT = 0.02, 0.06
T_BAR_START, T_BAR_END = 0.10, 0.42
T_SPECIES = 0.44                    # species line appears + spoken colloquially
T_PROTO_HDR, T_SCORE_0, T_SCORE_STEP = 0.52, 0.56, 0.04
T_VERDICT = 0.74                    # verdict line + "Adapting sounds..."

_TTS_CACHE: dict[str, np.ndarray] = {}


def _fade(y: np.ndarray) -> np.ndarray:
    n = int(FADE_S * SR)
    if len(y) > 2 * n:
        y = y.copy()
        y[:n] *= np.linspace(0, 1, n)
        y[-n:] *= np.linspace(1, 0, n)
    return y


def tts(text: str) -> np.ndarray:
    """Synthesise one spoken line. Cached — the same phrases repeat every segment."""
    if text in _TTS_CACHE:
        return _TTS_CACHE[text]
    if not shutil.which("say"):
        raise SystemExit("`say` not found — narration needs macOS. Render on the Mac, "
                         "not the NUC (the NUC only plays the finished video).")
    import librosa
    tmp = Path(tempfile.mkstemp(suffix=".aiff")[1])
    subprocess.run(["say", "-v", NARRATION_VOICE, "-o", str(tmp), text], check=True)
    y = _fade(librosa.load(str(tmp), sr=SR, mono=True)[0].astype(np.float32))
    tmp.unlink(missing_ok=True)
    _TTS_CACHE[text] = y
    return y


def decision_lines(seg: int, species_sci: str, species_slug: str | None, voc: str,
                   sims: dict[str, float], species_conf: float) -> list[tuple[float, str]]:
    """(fraction elapsed, line) — the log, retimed to match the spoken narration."""
    order = sorted(sims, key=lambda k: -sims[k])
    L = [(T_CMD, f"$ classify --segment {seg:02d}"),
         (T_SPECIES, f"  {species_sci:<18s} {species_conf:.2f}"),
         (T_PROTO_HDR, "  Nearest vocalisation type:")]
    for i, k in enumerate(order):
        bar = "#" * int(round(sims[k] * 34))
        mark = "<--" if k == voc else "   "
        L.append((T_SCORE_0 + i * T_SCORE_STEP,
                  f"      {k:<9s} {sims[k]:.3f}  {bar:<34s} {mark}"))
    species_disp = COLLOQUIAL.get(species_slug or "", "Unknown")
    voc_disp = voc.title() if voc and voc != "-" else "Unknown"
    L.append((T_VERDICT, f"Species: {species_disp}  |  Vocalisation type: {voc_disp}"))
    return L


def translation_page(si: int, n_segments: int, prog: float) -> Image.Image:
    """Phase 2: analysis panel cleared, one status line, progress through the response."""
    im = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(im)
    dr.text((MARGIN, 34), f"SEGMENT {si:02d} / {n_segments-1:02d}", font=F_MD, fill=150)
    dr.line([MARGIN, TERM_TOP - 18, W - MARGIN, TERM_TOP - 18], fill=70)
    y = TERM_TOP
    dr.text((MARGIN, y), "$ translate --to nest", font=F_MD, fill=205); y += LINE_H + 8
    dr.text((MARGIN, y), "Translation playing inside the nest...", font=F_MD, fill=205)
    y += LINE_H + 4
    done = int(round(prog * BAR_CELLS))
    dr.text((MARGIN, y), f"  [{'█'*done}{'░'*(BAR_CELLS-done)}] {prog*100:3.0f}%",
            font=F_MD, fill=205)
    return im


def duck(bird: np.ndarray, spans: list[tuple[int, int]]) -> np.ndarray:
    """Drop the bird under each spoken line, with short ramps so it doesn't click."""
    gain = np.ones(len(bird), dtype=np.float32)
    ramp = max(1, int(DUCK_RAMP_S * SR))
    for a, b in spans:
        a, b = max(0, a), min(len(bird), b)
        if b <= a:
            continue
        gain[a:b] = DUCK_LEVEL
        pre = slice(max(0, a - ramp), a)
        post = slice(b, min(len(bird), b + ramp))
        if pre.stop > pre.start:
            gain[pre] = np.linspace(1.0, DUCK_LEVEL, pre.stop - pre.start)
        if post.stop > post.start:
            gain[post] = np.linspace(DUCK_LEVEL, 1.0, post.stop - post.start)
    return bird * gain


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--soundscape", type=Path, required=True, help="the BIRD-only track (left channel source)")
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument("--response_dir", type=Path, default=Path("data/response_sounds"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--out_truth", type=Path, default=None)
    p.add_argument("--models_dir", type=Path, default=Path("models"))
    p.add_argument("--segment_seconds", type=float, default=25.0)
    p.add_argument("--birdnet_binary", default="./.venv/bin/birdnet-analyze")
    p.add_argument("--sims_cache", type=Path, default=Path("data/soundscapes/_sims_cache.json"))
    p.add_argument("--no_narration", action="store_true", help="build without the spoken voice")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import librosa, imageio.v2 as imageio, imageio_ffmpeg, soundfile as sf

    truth = pd.read_csv(args.truth)
    audio = librosa.load(str(args.soundscape), sr=SR, mono=True)[0]
    n = int(args.segment_seconds * SR)
    birds = [audio[i:i + n] for i in range(0, len(audio), n)]
    birds = [s for s in birds if len(s) >= SR]
    assert len(birds) == len(truth), f"{len(birds)} audio segments vs {len(truth)} truth rows"

    print("classifying each segment for the real decision log ...", flush=True)
    info = gather_similarities(args, birds)

    responses = {}
    for voc in truth["vocalisation"].unique():
        f = args.response_dir / f"{voc}.m4a"
        if not f.exists():
            raise SystemExit(f"no response clip for '{voc}': expected {f}")
        responses[voc] = _fade(librosa.load(str(f), sr=SR, mono=True)[0].astype(np.float32))
        print(f"  response[{voc}] = {f.name} ({len(responses[voc])/SR:.1f}s)", flush=True)

    specs = [mel_image(s) for s in birds]
    logs = [decision_lines(i, d["sci"] or "no match", d.get("species"), d["voc"] or "-",
                           d["sims"], d["conf"]) for i, d in enumerate(info)]

    tmp_video = Path(tempfile.mkstemp(suffix=".mp4")[1])
    writer = imageio.get_writer(tmp_video, fps=FPS, codec="libx264", quality=7, macro_block_size=1)
    spec_w = W - 2 * MARGIN
    bird_frames = int(args.segment_seconds * FPS)

    left_chunks, right_chunks, rows = [], [], []
    t_cursor = 0.0

    for si, bird in enumerate(birds):
        d, lines = info[si], logs[si]
        voc = truth.iloc[si]["vocalisation"]
        resp = responses[voc]

        # ---- phase 1 frames: analysis, unchanged except retimed to the narration ----
        for f in range(bird_frames):
            prog = f / bird_frames
            im = Image.new("L", (W, H), 0)
            dr = ImageDraw.Draw(im)
            dr.text((MARGIN, 34), f"SEGMENT {si:02d} / {len(birds)-1:02d}", font=F_MD, fill=150)
            dr.text((W - MARGIN - 210, 34),
                    f"t = {si*args.segment_seconds + prog*args.segment_seconds:6.1f} s", font=F_MD, fill=150)
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
                if frac == T_CMD and prog >= T_BAR_START:
                    ff = min(1.0, (prog - T_BAR_START) / (T_BAR_END - T_BAR_START))
                    done = int(round(ff * BAR_CELLS))
                    dr.text((MARGIN, y), f"  [{'█'*done}{'░'*(BAR_CELLS-done)}] {ff*100:3.0f}%",
                            font=F_MD, fill=205)
                    y += LINE_H
            writer.append_data(np.array(im.convert("RGB")))

        # ---- phase 2 frames: the translation status page ----
        resp_frames = max(1, int(round(len(resp) / SR * FPS)))
        for f in range(resp_frames):
            writer.append_data(np.array(
                translation_page(si, len(birds), f / resp_frames).convert("RGB")))

        # ---- narration, placed on the same clock as the visuals ----
        narration = np.zeros(len(bird), dtype=np.float32)
        spans: list[tuple[int, int]] = []
        if not args.no_narration:
            slug = d.get("species")
            cues = [(T_VOICE_DETECT, "Detecting species"),
                    (T_SPECIES, COLLOQUIAL.get(slug or "", "Unknown species")),
                    (T_VERDICT, "Adapting sounds to the bird's emotions")]
            for frac, phrase in cues:
                clip = tts(phrase)
                start = int(frac * len(bird))
                end = min(len(bird), start + len(clip))
                if end > start:
                    narration[start:end] += clip[:end - start] * NARRATION_GAIN
                    spans.append((start, end))

        left = duck(_fade(bird), spans) + narration      # bird + voice, outside the nest
        right = narration.copy()                          # voice only; response comes next
        peak = max(np.abs(left).max(), np.abs(right).max(), 1e-9)
        if peak > 0.99:                                   # guard the mix, don't clip
            left, right = left * (0.99 / peak), right * (0.99 / peak)

        left_chunks += [left, np.zeros_like(resp)]
        right_chunks += [right, resp]
        rows.append({"segment": si, "bird_start_s": t_cursor, "bird_duration_s": len(bird)/SR,
                     "response_start_s": t_cursor + len(bird)/SR, "response_duration_s": len(resp)/SR,
                     "species": truth.iloc[si]["species"], "vocalisation": voc})
        t_cursor += len(bird)/SR + len(resp)/SR
        print(f"  rendered segment {si} ({truth.iloc[si]['species']} {voc}, "
              f"+{len(resp)/SR:.1f}s translation)", flush=True)

    writer.close()

    stereo = np.stack([np.concatenate(left_chunks), np.concatenate(right_chunks)], axis=1).astype(np.float32)
    audio_tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    sf.write(audio_tmp, stereo, SR)

    ff_exe = imageio_ffmpeg.get_ffmpeg_exe()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ff_exe, "-y", "-loglevel", "error", "-i", str(tmp_video), "-i", str(audio_tmp),
                    "-c:v", "copy", "-c:a", "aac", "-shortest", str(args.out)], check=True)
    tmp_video.unlink(missing_ok=True); audio_tmp.unlink(missing_ok=True)

    out_truth = args.out_truth or args.truth.with_name(args.truth.stem + "_final.csv")
    pd.DataFrame(rows).to_csv(out_truth, index=False)
    print(f"\nwrote {args.out}  ({args.out.stat().st_size/1e6:.1f} MB, {t_cursor/60:.1f} min)")
    print(f"wrote {out_truth}")


if __name__ == "__main__":
    main()
