"""Live microphone demo: play a bird call near the mic -> classify -> speak the name.

NOTE: the gallery installation is designed mic-LESS (see CLAUDE.md, 2026-07-14) to avoid
gallery-noise/feedback problems. This script is for TESTING/DEMO — play a clean call close
to the mic in a quietish room.

Modes:
  record-on-demand (default): press Enter -> record N seconds -> classify -> speak
      ./.venv/bin/python scripts/live_mic.py --seconds 5 --speak
  continuous listen: react whenever a loud-enough sound occurs
      ./.venv/bin/python scripts/live_mic.py --listen --speak

Setup: pip install sounddevice   (Linux/NUC also: sudo apt install libportaudio2)
"""
from __future__ import annotations
import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_clip import (  # noqa: E402  (reuse existing pipeline)
    classify_call_type,
    detect_species,
    friendly,
    load_pitch_profile,
    pitch_shift_file,
    play_sound,
    speak,
    species_slug,
)

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    sys.exit("Missing audio deps. Run:  pip install sounddevice soundfile\n"
             "(On Linux/NUC also: sudo apt install libportaudio2)")

SR = 48000  # BirdNET resamples internally; 48k mono is a safe capture rate


def classify_and_announce(
    audio: np.ndarray,
    binary: str,
    top_k: int,
    min_conf: float,
    do_speak: bool,
    *,
    adapter_checkpoint: Path | None = None,
    prototypes: Path | None = None,
    prototype_labels: Path | None = None,
    birdnet_embeddings_binary: str = "./.venv/bin/birdnet-embeddings",
    response_sounds_dir: Path = Path("data/response_sounds"),
    play_response: bool = False,
    pitch_profiles_dir: Path = Path("data/species_pitch_profiles"),
    no_pitch_shift: bool = False,
) -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, SR)
        clip = Path(tmp.name)
    ranked = detect_species(clip, binary, top_k)
    if not ranked:
        print("  (no species detected)")
        clip.unlink(missing_ok=True)
        return
    top_label, top_score = ranked[0]
    phrase = "Unknown" if top_score < min_conf else friendly(top_label)
    print(f"  -> {top_label}  ({top_score:.3f})   SAY: \"{phrase}\"")

    call_type = None
    if adapter_checkpoint and top_score >= min_conf:
        proto_path = prototypes or adapter_checkpoint.with_name("prototypes.npy")
        proto_labels_path = prototype_labels or adapter_checkpoint.with_name("prototype_labels.json")
        call_type, sim = classify_call_type(
            clip, adapter_checkpoint, proto_path, proto_labels_path, birdnet_embeddings_binary,
        )
        print(f"  -> call-type: {call_type}  (cosine sim {sim:.3f})")
        phrase = f"{phrase}. {call_type}"

        if play_response:
            candidates = sorted(response_sounds_dir.glob(f"{call_type}.*"))
            if not candidates:
                print(f"  (no response sound for '{call_type}' in {response_sounds_dir})")
            else:
                sound_path = candidates[0]
                slug = species_slug(friendly(top_label))
                profile = None if no_pitch_shift else load_pitch_profile(pitch_profiles_dir, slug)
                if profile:
                    shift = profile["semitone_shift_vs_reference"]
                    print(f"  -> pitch shift for '{slug}': {shift:+.2f} semitones")
                    sound_path = pitch_shift_file(sound_path, shift)
                print(f"  -> playing {sound_path}")
                play_sound(sound_path)

    clip.unlink(missing_ok=True)
    if do_speak:
        speak(phrase)


def record(seconds: float) -> np.ndarray:
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    return audio.reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Live mic -> species -> spoken name.")
    ap.add_argument("--seconds", type=float, default=5.0, help="Seconds to record per capture (>=3).")
    ap.add_argument("--listen", action="store_true", help="Continuous mode: react to loud-enough sounds.")
    ap.add_argument("--rms_threshold", type=float, default=0.01, help="Listen mode: min RMS to trigger.")
    ap.add_argument("--birdnet_binary", default="./.venv/bin/birdnet-analyze")
    ap.add_argument("--birdnet_embeddings_binary", default="./.venv/bin/birdnet-embeddings")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--min_conf", type=float, default=0.10)
    ap.add_argument("--speak", action="store_true", help="Speak the result aloud.")
    ap.add_argument("--adapter_checkpoint", type=Path, default=None,
                    help="Trained SupCon adapter .pt for call-type classification, "
                         "e.g. data/runs/blackbird_supcon_adapter_v1/adapter.pt")
    ap.add_argument("--prototypes", type=Path, default=None)
    ap.add_argument("--prototype_labels", type=Path, default=None)
    ap.add_argument("--response_sounds_dir", type=Path, default=Path("data/response_sounds"))
    ap.add_argument("--play_response", action="store_true",
                    help="Play the response sound matching the detected call-type.")
    ap.add_argument("--pitch_profiles_dir", type=Path, default=Path("data/species_pitch_profiles"))
    ap.add_argument("--no_pitch_shift", action="store_true")
    args = ap.parse_args()
    secs = max(3.0, args.seconds)
    classify_kwargs = dict(
        adapter_checkpoint=args.adapter_checkpoint,
        prototypes=args.prototypes,
        prototype_labels=args.prototype_labels,
        birdnet_embeddings_binary=args.birdnet_embeddings_binary,
        response_sounds_dir=args.response_sounds_dir,
        play_response=args.play_response,
        pitch_profiles_dir=args.pitch_profiles_dir,
        no_pitch_shift=args.no_pitch_shift,
    )

    try:
        print(f"Input device: {sd.query_devices(kind='input')['name']}")
    except Exception as e:
        sys.exit(f"No microphone available: {e}")

    if args.listen:
        print(f"Listening (RMS>{args.rms_threshold}). Ctrl+C to stop. "
              "A cooldown after speaking avoids reacting to its own voice.")
        try:
            while True:
                chunk = record(secs)
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                if rms >= args.rms_threshold:
                    print(f"[sound detected, rms={rms:.3f}] classifying...")
                    classify_and_announce(chunk, args.birdnet_binary, args.top_k, args.min_conf, args.speak,
                                          **classify_kwargs)
                    if args.speak:
                        time.sleep(2.0)  # cooldown so it doesn't hear its own output
                else:
                    print(f"[quiet, rms={rms:.3f}]")
        except KeyboardInterrupt:
            print("\nstopped.")
    else:
        try:
            while True:
                input(f"\nPress Enter to record {secs:.0f}s (Ctrl+C to quit)... ")
                print("recording...")
                chunk = record(secs)
                classify_and_announce(chunk, args.birdnet_binary, args.top_k, args.min_conf, args.speak,
                                      **classify_kwargs)
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
