"""Minimal single-clip inference — the seed of the live runtime.

Milestone 0: audio clip in -> BirdNET species out.
Milestone 1 (optional, --anchors_manifest_csv): also assign nearest semantic
call-type (song/call/alarm...) via reviewed anchor prototypes in BirdNET embedding
space. NOTE: the call-type layer is not yet accurate — this is a plumbing/demo path.

Runs identically on x86 Mac and the Intel NUC.

    ./.venv/bin/python scripts/infer_clip.py --clip path/to/clip.wav
    ./.venv/bin/python scripts/infer_clip.py --clip clip.wav \
        --anchors_manifest_csv data/anchors/blackbird_anchor_manifest_v1.csv
"""
from __future__ import annotations
import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.birdnet_infer import (  # noqa: E402
    BirdNETConfig,
    ensure_birdnet_available,
    infer_recordings_with_birdnet,
    _extract_label,
    _extract_score,
)


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
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--anchors_manifest_csv", type=Path, default=None,
                    help="Optional: reviewed anchor prototypes for nearest call-type (milestone 1).")
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
    if args.anchors_manifest_csv:
        print("\n[call-type step] --anchors_manifest_csv given; "
              "reuse src.assign_anchor_labels for prototype matching (milestone 1, accuracy caveats apply).")
        # call_type = assign_nearest_prototype(...)   # milestone 1 hook

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
