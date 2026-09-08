"""Minimal single-clip inference — the seed of the live runtime.

Milestone 0: audio clip in -> BirdNET species out.
Milestone 1 (optional, --adapter_checkpoint): also assign nearest semantic
call-type (song/call/alarm/juvenile) via the trained SupCon adapter + prototypes,
then play a matching response sound from --response_sounds_dir (demo of the full
species -> call-type -> triggered-sound process). If a species pitch profile is
found under --pitch_profiles_dir (see src/compute_species_pitch_profile.py), the
response sound is pitch-shifted by that species' semitone offset before playback --
one shift per species, applied to whichever response track is triggered.

Runs identically on x86 Mac and the Intel NUC.

    ./.venv/bin/python scripts/infer_clip.py --clip path/to/clip.wav
    ./.venv/bin/python scripts/infer_clip.py --clip clip.wav \
        --adapter_checkpoint data/runs/blackbird_supcon_adapter_v1/adapter.pt \
        --play_response
"""
from __future__ import annotations
import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# scientific -> friendly spoken name (extend as needed; falls back to scientific)
COMMON_NAMES = {
    "Turdus merula": "Blackbird",
    "Turdus philomelos": "Song thrush",
    "Turdus iliacus": "Redwing",
    "Erithacus rubecula": "Robin",
    "Cyanistes caeruleus": "Blue tit",
    "Columba palumbus": "Wood pigeon",
    "Troglodytes troglodytes": "Wren",
    "Phylloscopus collybita": "Chiffchaff",
}


def friendly(sci: str) -> str:
    return COMMON_NAMES.get(sci, sci)


def species_slug(friendly_name: str) -> str:
    return friendly_name.lower().replace(" ", "_")


def load_pitch_profile(profiles_dir: Path, slug: str) -> dict | None:
    path = profiles_dir / f"{slug}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def pitch_shift_file(src: Path, semitones: float) -> Path:
    """Pitch-shift an audio file by N semitones, return a path to a temp copy."""
    from pedalboard import Pedalboard, PitchShift
    from pedalboard.io import AudioFile

    with AudioFile(str(src)) as f:
        audio = f.read(f.frames)
        sr = f.samplerate
    board = Pedalboard([PitchShift(semitones=semitones)])
    shifted = board(audio, sr)
    out_path = Path(tempfile.mkstemp(suffix=".wav", prefix="pitch_shifted_")[1])
    with AudioFile(str(out_path), "w", sr, shifted.shape[0]) as f:
        f.write(shifted)
    return out_path


def speak(text: str) -> None:
    """Speak text aloud using the platform's built-in TTS (macOS `say`, Linux `espeak-ng`)."""
    if platform.system() == "Darwin":
        subprocess.run(["say", text], check=False)
        return
    for cmd in (["spd-say", "--wait", text], ["espeak-ng", text], ["espeak", text]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, check=False)
            return
    print(f"[speak] no TTS engine found — on the NUC run: sudo apt install espeak-ng   (wanted to say: {text!r})")


def play_sound(path: Path) -> None:
    """Play an audio file using the platform's built-in player (macOS afplay / Linux)."""
    if platform.system() == "Darwin":
        subprocess.run(["afplay", str(path)], check=False)
        return
    for cmd in (["ffplay", "-nodisp", "-autoexit", str(path)], ["mpg123", str(path)], ["aplay", str(path)]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, check=False)
            return
    print(f"[play_sound] no audio player found (tried ffplay/mpg123/aplay) — wanted to play: {path}")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.birdnet_infer import (  # noqa: E402
    BirdNETConfig,
    ensure_birdnet_available,
    infer_recordings_with_birdnet,
    _extract_label,
    _extract_score,
)
from src.embed import birdnet_embedding_array  # noqa: E402
from src.transform_embeddings import _load_checkpoint  # noqa: E402


def classify_call_type(
    clip: Path,
    checkpoint_path: Path,
    prototypes_path: Path,
    prototype_labels_path: Path,
    birdnet_embeddings_binary: str,
) -> tuple[str, float]:
    """Extract a BirdNET embedding for the clip, run it through the trained adapter,
    and return the nearest prototype's (label, cosine_similarity)."""
    with tempfile.TemporaryDirectory(prefix="infer_clip_embed_") as tmp:
        tmp_dir = Path(tmp)
        shutil.copy2(clip, tmp_dir / clip.name)
        clips_df = pd.DataFrame([{"event_wav": clip.name}])
        embeddings, _meta = birdnet_embedding_array(
            clips_df, events_dir=tmp_dir, binary=birdnet_embeddings_binary,
        )

    _checkpoint, model = _load_checkpoint(checkpoint_path)
    import torch

    with torch.no_grad():
        z = model(torch.tensor(embeddings[:1])).numpy()

    protos = np.load(prototypes_path)
    labels = json.loads(prototype_labels_path.read_text())["labels"]
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-9)
    protos = protos / (np.linalg.norm(protos, axis=-1, keepdims=True) + 1e-9)
    sims = (z @ protos.T)[0]
    best = int(sims.argmax())
    return labels[best], float(sims[best])


def detect_species(clip: Path, binary: str, top_k: int) -> list[tuple[str, float]]:
    ensure_birdnet_available(binary)
    cfg = BirdNETConfig(binary=binary, top_k=top_k)
    preds, _version = infer_recordings_with_birdnet(audio_paths=[clip.resolve()], cfg=cfg)
    # each item is one 3s window: {start_s, end_s, predictions:[{label,score},...]}
    # aggregate -> best score per species across all windows
    best: dict[str, float] = {}
    n_windows = 0
    for _rec, items in preds.items():
        for window in items:
            n_windows += 1
            for p in window.get("predictions", []):
                label = _extract_label(p) or str(p.get("label", "")).strip()
                score = _extract_score(p) if _extract_score(p) else float(p.get("score", 0.0) or 0.0)
                if label:
                    best[label] = max(best.get(label, 0.0), score)
    print(f"analysed {n_windows} x 3s windows")
    return sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k]


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-clip BirdNET species (+ optional call-type) inference.")
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--birdnet_binary", default="./.venv/bin/birdnet-analyze")
    ap.add_argument("--birdnet_embeddings_binary", default="./.venv/bin/birdnet-embeddings")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--adapter_checkpoint", type=Path, default=None,
                    help="Trained SupCon adapter .pt (milestone 1, call-type). "
                         "e.g. data/runs/blackbird_supcon_adapter_v1/adapter.pt")
    ap.add_argument("--prototypes", type=Path, default=None,
                    help="prototypes.npy next to the checkpoint (defaults to the checkpoint's sibling file).")
    ap.add_argument("--prototype_labels", type=Path, default=None,
                    help="prototype_labels.json next to the checkpoint (defaults to the checkpoint's sibling file).")
    ap.add_argument("--response_sounds_dir", type=Path, default=Path("data/response_sounds"),
                    help="Folder with <label>.<ext> files to play on a call-type match.")
    ap.add_argument("--play_response", action="store_true",
                    help="Play the response sound matching the detected call-type.")
    ap.add_argument("--pitch_profiles_dir", type=Path, default=Path("data/species_pitch_profiles"),
                    help="Folder with <species_slug>.json pitch profiles (see compute_species_pitch_profile.py).")
    ap.add_argument("--no_pitch_shift", action="store_true",
                    help="Play response sounds at their original pitch, even if a species profile exists.")
    ap.add_argument("--speak", action="store_true", help="Speak the result aloud (macOS say / Linux espeak-ng).")
    ap.add_argument("--min_conf", type=float, default=0.10,
                    help="Below this top-species confidence, speak 'unknown' instead of a species.")
    args = ap.parse_args()

    if not args.clip.exists():
        raise SystemExit(f"clip not found: {args.clip}")

    print(f"\n=== clip: {args.clip.name} ===")
    ranked = detect_species(args.clip, args.birdnet_binary, args.top_k)
    if not ranked:
        print("No species detected.")
        return
    print("\nTop species:")
    for label, score in ranked:
        print(f"  {score:5.3f}  {label}")
    top_label, top_score = ranked[0]
    print(f"\nDETECTED: {top_label}  (confidence {top_score:.3f})")

    # build the spoken phrase: species now; call-type appended once milestone 1 is wired
    call_type = None
    if args.adapter_checkpoint:
        prototypes = args.prototypes or args.adapter_checkpoint.with_name("prototypes.npy")
        proto_labels = args.prototype_labels or args.adapter_checkpoint.with_name("prototype_labels.json")
        call_type, sim = classify_call_type(
            args.clip, args.adapter_checkpoint, prototypes, proto_labels,
            args.birdnet_embeddings_binary,
        )
        print(f"\n[call-type step] nearest prototype: {call_type}  (cosine sim {sim:.3f})")

        if args.play_response:
            candidates = sorted(args.response_sounds_dir.glob(f"{call_type}.*"))
            if not candidates:
                print(f"[play_response] no sound file found for label '{call_type}' "
                      f"in {args.response_sounds_dir} (expected {call_type}.<ext>)")
            else:
                sound_path = candidates[0]
                slug = species_slug(friendly(top_label))
                profile = None if args.no_pitch_shift else load_pitch_profile(args.pitch_profiles_dir, slug)
                if profile:
                    shift = profile["semitone_shift_vs_reference"]
                    print(f"[pitch_shift] species profile '{slug}': {shift:+.2f} semitones "
                          f"(species median F0 {profile['species_median_f0_hz']:.0f} Hz)")
                    sound_path = pitch_shift_file(sound_path, shift)
                print(f"[play_response] playing {sound_path}")
                play_sound(sound_path)

    if top_score < args.min_conf:
        phrase = "Unknown"
    else:
        phrase = friendly(top_label)
        if call_type:
            phrase = f"{phrase}. {call_type}"
    print(f"SPOKEN OUTPUT: \"{phrase}\"")
    if args.speak:
        speak(phrase)


if __name__ == "__main__":
    main()
