"""Check the two-output setup on any machine — no NUC, no wiring needed.

Plays the exhibition audio out of one device (headphones, standing in for the wired
speakers) and the bass track out of another (a Bluetooth speaker), started together and
looping, exactly as the installation does. Prints what SHOULD be audible each second, so
you can check by ear that the bass only ever appears under a translation.

    python scripts/test_two_outputs.py --list
    python scripts/test_two_outputs.py --main "Headphones" --bass "JBL"

--main/--bass take a device number from --list, or any part of its name.
Ctrl+C to stop.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

SR = 44100


def list_devices() -> None:
    import sounddevice as sd
    print(f"{'#':>3}  {'outputs':>7}  name")
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0:
            print(f"{i:>3}  {d['max_output_channels']:>7}  {d['name']}")


def pick(spec: str) -> int:
    import sounddevice as sd
    if spec.isdigit():
        return int(spec)
    hits = [i for i, d in enumerate(sd.query_devices())
            if d["max_output_channels"] > 0 and spec.lower() in d["name"].lower()]
    if not hits:
        raise SystemExit(f"no output device matching {spec!r} — run --list")
    if len(hits) > 1:
        import sounddevice as sd
        names = ", ".join(f"{i} ({sd.query_devices()[i]['name']})" for i in hits)
        raise SystemExit(f"{spec!r} matches several devices: {names} — use the number")
    return hits[0]


def load_audio(path: Path) -> np.ndarray:
    """Any file (including the .mp4) as float32 stereo at 44.1 kHz."""
    import soundfile as sf
    if path.suffix.lower() in {".mp4", ".m4a", ".mov"}:
        import imageio_ffmpeg
        tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path),
                        "-vn", "-ac", "2", "-ar", str(SR), str(tmp)], check=True)
        y, sr = sf.read(str(tmp), dtype="float32", always_2d=True); tmp.unlink(missing_ok=True)
    else:
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        raise SystemExit(f"{path} is {sr} Hz, expected {SR}")
    return np.repeat(y, 2, axis=1) if y.shape[1] == 1 else y[:, :2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="show the output devices and exit")
    p.add_argument("--main", help="device for the exhibition audio (headphones / wired speakers)")
    p.add_argument("--bass", help="device for the bass track (Bluetooth speaker)")
    p.add_argument("--main_file", type=Path, default=Path("data/soundscapes/exhibition_v2.mp4"))
    p.add_argument("--bass_file", type=Path, default=Path("data/soundscapes/bass_track.flac"))
    p.add_argument("--truth", type=Path, default=Path("data/soundscapes/exhibition_v2_truth_final.csv"))
    p.add_argument("--start_at", type=float, default=0.0, help="skip ahead, in seconds")
    args = p.parse_args()

    if args.list:
        list_devices(); return
    if not (args.main and args.bass):
        raise SystemExit("need --main and --bass (or --list to see the devices)")

    import sounddevice as sd
    import pandas as pd

    main_audio, bass_audio = load_audio(args.main_file), load_audio(args.bass_file)
    if abs(len(main_audio) - len(bass_audio)) > SR:
        print(f"WARNING the two tracks are different lengths "
              f"({len(main_audio)/SR:.1f}s vs {len(bass_audio)/SR:.1f}s) — they will drift apart",
              file=sys.stderr)
    truth = pd.read_csv(args.truth)

    pos = {"main": int(args.start_at * SR), "bass": int(args.start_at * SR)}
    lock = threading.Lock()

    def make_cb(key: str, audio: np.ndarray):
        def cb(out, frames, _t, status):
            if status:
                print(f"  [{key} {status}]", file=sys.stderr)
            with lock:
                i = pos[key] % len(audio)
                end = i + frames
                if end <= len(audio):
                    out[:] = audio[i:end]
                else:                                   # wrap round for a seamless loop
                    first = len(audio) - i
                    out[:first], out[first:] = audio[i:], audio[:frames - first]
                pos[key] += frames
        return cb

    streams = [sd.OutputStream(device=pick(args.main), samplerate=SR, channels=2,
                               callback=make_cb("main", main_audio)),
               sd.OutputStream(device=pick(args.bass), samplerate=SR, channels=2,
                               callback=make_cb("bass", bass_audio))]
    print(f"main : {sd.query_devices()[pick(args.main)]['name']}  <- birds, narration, translations")
    print(f"bass : {sd.query_devices()[pick(args.bass)]['name']}  <- bass only")
    print("\nBluetooth runs ~0.1-0.3s behind a wired output; a small, steady lag is expected.\n")

    for s in streams:                                   # start together
        s.start()
    try:
        while True:
            with lock:
                t = pos["main"] / SR % (len(main_audio) / SR)
            row = truth[(truth.bird_start_s <= t)].tail(1)
            if len(row):
                r = row.iloc[0]
                phase = ("TRANSLATION — bass should be playing"
                         if t >= r.response_start_s else "analysis — bass should be SILENT")
                print(f"\r  {int(t)//60}:{int(t)%60:02d}  segment {int(r.segment)+1:2d} "
                      f"{r.species} {r.vocalisation:9s} | {phase}   ", end="", flush=True)
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        for s in streams:
            s.stop(); s.close()


if __name__ == "__main__":
    main()
