"""Build the final exhibition asset: stereo audio + narrated visualisation, muxed to video.

Every phase is exactly 20s. NOTHING ON SCREEN COUNTS — no segment number, no clock, no
`--segment NN` — so visitors can't tell where the loop restarts. Each segment is two phases, back to back:

  1. BIRD (20s) — the raw call plays on the LEFT channel while the analysis runs on screen.
     A synthesised voice narrates it, timed to the visuals and centred across both
     channels: "Detecting species..." over the loading bar, then the species said
     colloquially ("Tawny owl", not Strix aluco) as it resolves, then "Adapting sounds to
     the bird's emotions..." once the vocalisation-type scores are in. The bird audio ducks
     under each spoken line so the voice stays intelligible.

  2. TRANSLATION (20s) — the response plays on the RIGHT channel (Speaker B, inside the
     nest) and the screen clears to a status page reading "Translation playing inside the
     nest...".

SOME DISPLAYED SPECIES CONFIDENCES ARE ADJUSTED (artist's decision, 2026-09-14). BirdNET's
species confidence is shown as-is when it looks plausible (0.80-0.98). The ones that read as
broken on screen — 0.17 on a clear robin song, or 1.00 on the owls — are moved into
0.81-0.89, spread by rank among themselves so the least-sure still shows lowest. Every other
number (all vocalisation scores, negatives included) and which species/type wins is the
model's real output. Real and shown species values are both written to the *_final.csv;
`--real_numbers` renders every raw value.

Responses are looked up per species AND vocalisation first (`robin_alarm.wav`, anywhere
under --response_dir — the sound artist's 12), falling back to a generic per-type clip
(`alarm.m4a`) with a printed WARNING. Each is trimmed (with a fade-out) or padded with
silence to exactly --response_seconds.

Analysis runs at 16 kHz (what the classifier needs); the final MIX is 44.1 kHz so the
response clips keep their full bandwidth rather than being cut off at 8 kHz.

Narration is generated with macOS `say`, so this must be RENDERED ON A MAC — the NUC only
ever plays the finished file.

    python -m src.render_exhibition \
        --soundscape data/soundscapes/exhibition_v2.flac \
        --truth data/soundscapes/exhibition_v2_truth.csv \
        --response_dir data/soundscapes/from_nelson \
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

OUT_SR = 44100                      # final mix rate; SR (16k) is for analysis only
FADE_S = 0.05
TRIM_FADE_S = 0.5                   # fade-out when a response is cut short
NARRATION_VOICE = "Daniel"          # en_GB — these are all British birds
NARRATION_GAIN = 0.95
DUCK_LEVEL = 0.35                   # how far the bird drops under a spoken line
DUCK_RAMP_S = 0.12
AUDIO_EXTS = {".wav", ".m4a", ".mp3", ".aif", ".aiff", ".flac"}
PLAUSIBLE_SPECIES_CONF = (0.80, 0.98)  # species confidences outside this look broken on screen
SHOWN_SPECIES_CONF = (0.81, 0.89)      # ...and are moved into this range (see docstring)

COLLOQUIAL = {"blackbird": "Blackbird", "robin": "Robin", "tawny_owl": "Tawny owl"}
# on-screen names for the four vocalisation types (the model's labels stay the short keys)
VOC_DISPLAY = {"call": "Contact call", "song": "Mating signal",
               "alarm": "Alarm/distress call", "juvenile": "Juvenile begging"}
VOC_W = max(len(v) for v in VOC_DISPLAY.values())

# Fractions of the bird phase. Retimed so the narration lands ON the thing it names:
# the species is SPOKEN as its line appears, not before the bar that finds it.
T_CMD, T_VOICE_DETECT = 0.02, 0.06
T_BAR_START, T_BAR_END = 0.10, 0.42
T_SPECIES = 0.44                    # species line appears + spoken colloquially
T_PROTO_HDR, T_SCORE_0, T_SCORE_STEP = 0.52, 0.56, 0.04
T_VERDICT = 0.74                    # verdict line + "Adapting sounds..."

_TTS_CACHE: dict[str, np.ndarray] = {}


def _fade(y: np.ndarray, sr: int = OUT_SR) -> np.ndarray:
    n = int(FADE_S * sr)
    if len(y) > 2 * n:
        y = y.copy()
        y[:n] *= np.linspace(0, 1, n)
        y[-n:] *= np.linspace(1, 0, n)
    return y


def fit_length(y: np.ndarray, seconds: float, sr: int = OUT_SR) -> np.ndarray:
    """Trim (fading out the tail) or pad with silence to exactly `seconds`."""
    n = int(round(seconds * sr))
    if len(y) >= n:
        y = y[:n].copy()
        f = min(n, int(TRIM_FADE_S * sr))
        y[-f:] *= np.linspace(1, 0, f)
        return y
    return np.concatenate([y, np.zeros(n - len(y), dtype=y.dtype)])


def shown_species_conf(real: list[float]) -> list[float]:
    """Keep plausible confidences; move only the implausible ones (too low, or anything that
    would print as 0.99/1.00) into SHOWN_SPECIES_CONF, by rank among themselves."""
    ok_lo, ok_hi = PLAUSIBLE_SPECIES_CONF
    odd = sorted((i for i, c in enumerate(real) if c < ok_lo or round(c, 2) > ok_hi),
                 key=lambda i: real[i])
    out = [round(c, 2) for c in real]
    lo, hi = SHOWN_SPECIES_CONF
    for rank, i in enumerate(odd):
        out[i] = round(lo + (hi - lo) * rank / max(1, len(odd) - 1), 2)
    return out


def find_response(response_dir: Path, species: str, voc: str) -> tuple[Path | None, bool]:
    """(path, is_species_specific). Prefers the artist's per-species clip."""
    for stem, specific in ((f"{species}_{voc}", True), (voc, False)):
        for p in sorted(response_dir.rglob(f"{stem}.*")):
            if p.suffix.lower() in AUDIO_EXTS and not p.name.startswith("._"):
                return p, specific
    return None, False


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
    y = _fade(librosa.load(str(tmp), sr=OUT_SR, mono=True)[0].astype(np.float32))
    tmp.unlink(missing_ok=True)
    _TTS_CACHE[text] = y
    return y


def decision_lines(seg: int, species_sci: str, species_slug: str | None, voc: str,
                   sims: dict[str, float], species_conf: float) -> list[tuple[float, str]]:
    """(fraction elapsed, line) — the log, retimed to match the spoken narration."""
    order = sorted(sims, key=lambda k: -sims[k])
    L = [(T_CMD, "$ classify"),
         (T_SPECIES, f"  {species_sci:<18s} {species_conf:.2f}"),
         (T_PROTO_HDR, "  Nearest vocalisation type:")]
    for i, k in enumerate(order):
        bar = "#" * int(round(max(0.0, sims[k]) * 30))
        mark = "<--" if k == voc else "   "
        L.append((T_SCORE_0 + i * T_SCORE_STEP,
                  f"      {VOC_DISPLAY.get(k, k):<{VOC_W}s} {sims[k]:.3f}  {bar:<30s} {mark}"))
    species_disp = COLLOQUIAL.get(species_slug or "", "Unknown")
    voc_disp = VOC_DISPLAY.get(voc, "Unknown")
    L.append((T_VERDICT, f"Species: {species_disp}  |  Vocalisation type: {voc_disp}"))
    return L


def translation_page(prog: float) -> Image.Image:
    """Phase 2: analysis panel cleared, one status line, progress through the response."""
    im = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(im)
    dr.line([MARGIN, TERM_TOP - 18, W - MARGIN, TERM_TOP - 18], fill=70)
    y = TERM_TOP
    dr.text((MARGIN, y), "$ translate --to nest", font=F_MD, fill=205); y += LINE_H + 8
    dr.text((MARGIN, y), "Translation playing inside the nest...", font=F_MD, fill=205)
    y += LINE_H + 4
    done = int(round(prog * BAR_CELLS))
    dr.text((MARGIN, y), f"  [{'█'*done}{'░'*(BAR_CELLS-done)}] {prog*100:3.0f}%",
            font=F_MD, fill=205)
    return im


def duck(bird: np.ndarray, spans: list[tuple[int, int]], sr: int = OUT_SR) -> np.ndarray:
    """Drop the bird under each spoken line, with short ramps so it doesn't click."""
    gain = np.ones(len(bird), dtype=np.float32)
    ramp = max(1, int(DUCK_RAMP_S * sr))
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
    p.add_argument("--response_seconds", type=float, default=20.0,
                   help="trim/pad every response to this length (0 = keep natural length)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--out_truth", type=Path, default=None)
    p.add_argument("--models_dir", type=Path, default=Path("models"))
    p.add_argument("--segment_seconds", type=float, default=20.0)
    p.add_argument("--birdnet_binary", default="./.venv/bin/birdnet-analyze")
    p.add_argument("--sims_cache", type=Path, default=Path("data/soundscapes/_sims_cache.json"))
    p.add_argument("--no_narration", action="store_true", help="build without the spoken voice")
    p.add_argument("--real_numbers", action="store_true",
                   help="show the model's raw confidences instead of the display layer")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import librosa, imageio.v2 as imageio, imageio_ffmpeg, soundfile as sf

    truth = pd.read_csv(args.truth)
    # 16k copy drives the classifier + spectrograms; 44.1k copy goes into the mix
    audio = librosa.load(str(args.soundscape), sr=SR, mono=True)[0]
    audio_hi = librosa.load(str(args.soundscape), sr=OUT_SR, mono=True)[0].astype(np.float32)
    n, n_hi = int(args.segment_seconds * SR), int(args.segment_seconds * OUT_SR)
    birds = [audio[i:i + n] for i in range(0, len(audio), n)]
    birds = [s for s in birds if len(s) >= SR]
    birds_hi = [audio_hi[i * n_hi:(i + 1) * n_hi] for i in range(len(birds))]
    assert len(birds) == len(truth), f"{len(birds)} audio segments vs {len(truth)} truth rows"

    print("classifying each segment for the real decision log ...", flush=True)
    info = gather_similarities(args, birds)

    real_conf = [float(d["conf"]) for d in info]
    conf_shown = real_conf if args.real_numbers else shown_species_conf(real_conf)
    sims_shown = [d["sims"] for d in info]

    responses: dict[tuple[str, str], np.ndarray] = {}
    sources: dict[tuple[str, str], str] = {}
    for sp, voc in truth[["species", "vocalisation"]].drop_duplicates().itertuples(index=False):
        path, specific = find_response(args.response_dir, sp, voc)
        if path is None:
            raise SystemExit(f"no response for {sp} {voc}: expected {sp}_{voc}.<ext> or "
                             f"{voc}.<ext> under {args.response_dir}")
        y = _fade(librosa.load(str(path), sr=OUT_SR, mono=True)[0].astype(np.float32))
        natural = len(y) / OUT_SR
        if args.response_seconds > 0:
            y = fit_length(y, args.response_seconds)
        responses[(sp, voc)] = y
        sources[(sp, voc)] = str(path)
        tag = "" if specific else "   WARNING: generic placeholder, no per-species clip found"
        print(f"  {sp:10s} {voc:9s} <- {path.name} ({natural:.1f}s -> {len(y)/OUT_SR:.1f}s){tag}",
              flush=True)

    specs = [mel_image(s) for s in birds]
    logs = [decision_lines(i, d["sci"] or "no match", d.get("species"), d["voc"] or "-",
                           sims_shown[i], conf_shown[i]) for i, d in enumerate(info)]

    tmp_video = Path(tempfile.mkstemp(suffix=".mp4")[1])
    writer = imageio.get_writer(tmp_video, fps=FPS, codec="libx264", quality=7, macro_block_size=1)
    spec_w = W - 2 * MARGIN
    bird_frames = int(round(args.segment_seconds * FPS))

    left_chunks, right_chunks, rows = [], [], []
    t_cursor = 0.0                                   # real elapsed time; drives the clock

    for si, bird in enumerate(birds_hi):
        d, lines = info[si], logs[si]
        sp, voc = truth.iloc[si]["species"], truth.iloc[si]["vocalisation"]
        resp = responses[(sp, voc)]
        bird_s, resp_s = len(bird) / OUT_SR, len(resp) / OUT_SR

        # ---- phase 1 frames: analysis, timed to the narration ----
        for f in range(bird_frames):
            prog = f / bird_frames
            im = Image.new("L", (W, H), 0)
            dr = ImageDraw.Draw(im)
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
        resp_frames = max(1, int(round(resp_s * FPS)))
        for f in range(resp_frames):
            prog = f / resp_frames
            writer.append_data(np.array(translation_page(prog).convert("RGB")))

        # ---- narration, placed on the same clock as the visuals ----
        narration = np.zeros(len(bird), dtype=np.float32)
        spans: list[tuple[int, int]] = []
        if not args.no_narration:
            cues = [(T_VOICE_DETECT, "Detecting species"),
                    (T_SPECIES, COLLOQUIAL.get(d.get("species") or "", "Unknown species")),
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
        rows.append({"segment": si, "bird_start_s": t_cursor, "bird_duration_s": bird_s,
                     "response_start_s": t_cursor + bird_s, "response_duration_s": resp_s,
                     "species": sp, "vocalisation": voc,
                     "predicted_species": d.get("species"), "predicted_vocalisation": d["voc"],
                     "species_conf_real": round(real_conf[si], 3),
                     "species_conf_shown": conf_shown[si],
                     "response_file": Path(sources[(sp, voc)]).name})
        t_cursor += bird_s + resp_s
        print(f"  rendered segment {si} ({sp} {voc})", flush=True)

    writer.close()

    stereo = np.stack([np.concatenate(left_chunks), np.concatenate(right_chunks)], axis=1).astype(np.float32)
    audio_tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    sf.write(audio_tmp, stereo, OUT_SR)

    ff_exe = imageio_ffmpeg.get_ffmpeg_exe()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ff_exe, "-y", "-loglevel", "error", "-i", str(tmp_video), "-i", str(audio_tmp),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(args.out)],
                   check=True)
    tmp_video.unlink(missing_ok=True); audio_tmp.unlink(missing_ok=True)

    out_truth = args.out_truth or args.truth.with_name(args.truth.stem + "_final.csv")
    pd.DataFrame(rows).to_csv(out_truth, index=False)
    print(f"\nwrote {args.out}  ({args.out.stat().st_size/1e6:.1f} MB, {t_cursor/60:.1f} min)")
    print(f"wrote {out_truth}")


if __name__ == "__main__":
    main()
