"""The installation runtime: play a soundscape, classify it live, answer each bird.

No microphone. The device plays the soundscape out of Speaker A and analyses that same
clean audio in memory — the decision recorded in CLAUDE.md (2026-07-14), which avoids
gallery noise and any feedback loop between the two speakers.

Latency is hidden by PIPELINING: segment N+1 is classified in a background thread while
segment N is still playing, so the answer is ready the moment it is needed.

    python scripts/live_soundscape.py --soundscape data/soundscapes/exhibition_v1.wav
    python scripts/live_soundscape.py --soundscape x.wav --dry_run     # no audio, prints only

Speaker routing: --left_only sends the soundscape to the left channel alone, so Speaker A
carries the birds. Once the twelve species+vocalisation response recordings exist they
should play on the right channel; until then the response is spoken via system TTS.
"""
from __future__ import annotations

import argparse
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.classify_clip import Classifier, Result  # noqa: E402

SR = 16000


def speak(text: str) -> None:
    import platform, shutil, subprocess
    if platform.system() == "Darwin":
        subprocess.run(["say", text], check=False); return
    for cmd in (["spd-say", "--wait", text], ["espeak-ng", text], ["espeak", text]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, check=False); return
    print(f"   [no TTS installed — would say: {text!r}]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--soundscape", type=Path, required=True)
    p.add_argument("--models_dir", type=Path, default=Path("models"))
    p.add_argument("--segment_seconds", type=float, default=20.0)
    p.add_argument("--birdnet_binary", default="birdnet-analyze")
    p.add_argument("--min_species_conf", type=float, default=0.10)
    p.add_argument("--loop", action="store_true", help="repeat forever (exhibition mode)")
    p.add_argument("--left_only", action="store_true", help="soundscape on the left channel only")
    p.add_argument("--dry_run", action="store_true", help="classify and print, play nothing")
    p.add_argument("--no_speak", action="store_true")
    p.add_argument("--talk_over", action="store_true",
                   help="answer while the bird is still playing, instead of waiting for it to finish")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.soundscape.exists():
        raise SystemExit(f"soundscape not found: {args.soundscape}")

    import librosa
    print(f"loading soundscape {args.soundscape.name} ...", flush=True)
    audio = librosa.load(str(args.soundscape), sr=SR, mono=True)[0]
    seg_len = int(args.segment_seconds * SR)
    segments = [audio[i:i + seg_len] for i in range(0, len(audio), seg_len)]
    segments = [s for s in segments if len(s) >= SR]          # drop a stub tail
    print(f"  {len(audio)/SR:.0f}s -> {len(segments)} segments of {args.segment_seconds:.0f}s", flush=True)

    print("loading models (this takes ~30 s, once) ...", flush=True)
    t0 = time.time()
    clf = Classifier(args.models_dir, birdnet_binary=args.birdnet_binary)
    print(f"  ready in {time.time()-t0:.0f}s · heads: {', '.join(sorted(clf.heads))}", flush=True)

    if not args.dry_run:
        import sounddevice as sd

    tmpdir = Path(tempfile.mkdtemp(prefix="soundscape_"))
    import soundfile as sf

    def classify_segment(i: int) -> Result:
        p = tmpdir / f"seg_{i:03d}.wav"
        if not p.exists():
            sf.write(p, segments[i], SR)
        return clf.classify(p, min_species_conf=args.min_species_conf)

    # background worker stays one segment ahead of playback
    results: "queue.Queue[tuple[int, Result]]" = queue.Queue()
    stop = threading.Event()

    def worker() -> None:
        for i in range(len(segments)):
            if stop.is_set():
                return
            try:
                results.put((i, classify_segment(i)))
            except Exception as e:                      # never let one bad segment kill the show
                print(f"   ! segment {i} failed: {type(e).__name__}: {e}", flush=True)
                results.put((i, Result(None, None, 0.0, None, 0.0)))

    try:
        while True:
            stop.clear()
            th = threading.Thread(target=worker, daemon=True); th.start()
            pending: dict[int, Result] = {}

            for i, seg in enumerate(segments):
                # make sure this segment's verdict has arrived
                while i not in pending:
                    j, r = results.get()
                    pending[j] = r
                res = pending.pop(i)

                if args.dry_run:
                    print(f"[{i*args.segment_seconds:6.0f}s] {res.spoken():24s} "
                          f"species {res.species_confidence:.2f} · match {res.vocalisation_confidence:.2f}", flush=True)
                    continue

                out = seg if not args.left_only else np.column_stack([seg, np.zeros_like(seg)])
                print(f"[{i*args.segment_seconds:6.0f}s] ♪ {res.spoken():24s} "
                      f"species {res.species_confidence:.2f} · match {res.vocalisation_confidence:.2f}", flush=True)
                sd.play(out, SR)
                if args.talk_over:
                    if not args.no_speak:
                        speak(res.spoken())           # answer while the bird is still singing
                    sd.wait()
                else:
                    sd.wait()                         # let the bird finish, THEN answer
                    if not args.no_speak:
                        speak(res.spoken())

            stop.set()
            if not args.loop:
                break
    except KeyboardInterrupt:
        stop.set()
        print("\nstopped.")


if __name__ == "__main__":
    main()
