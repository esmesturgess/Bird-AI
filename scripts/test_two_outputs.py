"""Check the whole installation on any machine — picture, both outputs, no NUC needed.

Shows the video in a window, plays the exhibition audio out of one device (headphones,
standing in for the wired speakers) and the bass out of another (a Bluetooth speaker),
all started together and looping, exactly as the installation does. The picture follows
the audio, so what you see is always what you are hearing.

    python scripts/test_two_outputs.py --list
    python scripts/test_two_outputs.py --main "External Headphones" --bass "JBL"
    python scripts/test_two_outputs.py --main "External Headphones" --bass "JBL" --mix

--main/--bass take a device number from --list, any part of its name, or the word
`default`, meaning whatever the system's sound settings are currently sending audio to.
On macOS, picking a Bluetooth speaker by name sometimes opens without error yet produces
no sound; setting it as the system output and passing `--bass default` goes through the
same path as every other app and works when the by-name route doesn't.

--mix matters on HEADPHONES. The exhibition is genuinely split left/right — birds and
narration on the left (the speaker outside the nest), translations on the right (the
speaker inside it) — so on headphones the birds are in your left ear only, and during a
translation the left ear is silent. That can sound like the birds have vanished. --mix
sums both sides into both ears so you hear everything; leave it off to check the split.

--no_video plays audio only. Close the window or press Ctrl+C to stop.
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


def dev_name(idx: int | None) -> str:
    """Name of a device index, or of whatever macOS/Linux is currently sending sound to."""
    import sounddevice as sd
    if idx is None:
        return f"{sd.query_devices(kind='output')['name']} [system default output]"
    return sd.query_devices()[idx]["name"]


def pick(spec: str) -> int | None:
    """Device index, or None meaning 'whatever the system is set to'."""
    import sounddevice as sd
    if spec.strip().lower() == "default":
        return None
    if spec.isdigit():
        return int(spec)
    hits = [i for i, d in enumerate(sd.query_devices())
            if d["max_output_channels"] > 0 and spec.lower() in d["name"].lower()]
    if not hits:
        raise SystemExit(f"no output device matching {spec!r} — run --list")
    if len(hits) > 1:
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
    p.add_argument("--mix", action="store_true", help="sum left+right into both ears (for headphones)")
    p.add_argument("--no_video", action="store_true", help="audio only, no window")
    p.add_argument("--scale", type=float, default=0.75, help="window size, 1.0 = full 1280x720")
    args = p.parse_args()

    if args.list:
        list_devices(); return
    if not (args.main and args.bass):
        raise SystemExit("need --main and --bass (or --list to see the devices)")

    import sounddevice as sd
    import pandas as pd

    main_audio, bass_audio = load_audio(args.main_file), load_audio(args.bass_file)
    if args.mix:                                      # both ears hear everything
        mono = main_audio.mean(axis=1, keepdims=True)
        main_audio = np.repeat(mono, 2, axis=1).astype(np.float32)
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
                else:                                 # wrap round for a seamless loop
                    first = len(audio) - i
                    out[:first], out[first:] = audio[i:], audio[:frames - first]
                pos[key] += frames
        return cb

    main_dev, bass_dev = pick(args.main), pick(args.bass)
    streams = [sd.OutputStream(device=main_dev, samplerate=SR, channels=2,
                               callback=make_cb("main", main_audio)),
               sd.OutputStream(device=bass_dev, samplerate=SR, channels=2,
                               callback=make_cb("bass", bass_audio))]
    print(f"main : {dev_name(main_dev)}  <- birds, narration, translations"
          f"{' (mixed to both ears)' if args.mix else ' (birds LEFT, translations RIGHT)'}")
    print(f"bass : {dev_name(bass_dev)}  <- bass only, under translations")
    print("\nBluetooth runs ~0.1-0.3s behind a wired output; a small, steady lag is expected.")

    def phase_at(t: float) -> str:
        row = truth[truth.bird_start_s <= t].tail(1)
        if not len(row):
            return ""
        r = row.iloc[0]
        what = ("TRANSLATION — bass should be playing" if t >= r.response_start_s
                else "analysis — bass should be SILENT")
        return f"{int(t)//60}:{int(t)%60:02d}   {int(r.segment)+1:2d}/12  {r.species} {r.vocalisation}   |   {what}"

    for s in streams:                                 # start together
        s.start()

    if args.no_video:
        try:
            while True:
                with lock:
                    t = pos["main"] / SR % (len(main_audio) / SR)
                print("\r  " + phase_at(t) + "    ", end="", flush=True)
                time.sleep(0.25)
        except KeyboardInterrupt:
            print("\nstopped.")
        finally:
            for s in streams:
                s.stop(); s.close()
        return

    # ---- picture, driven by the audio position so the two can't drift ----
    import tkinter as tk
    import imageio.v2 as imageio
    from PIL import Image, ImageTk

    reader = imageio.get_reader(str(args.main_file))
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 12))
    size = (int(1280 * args.scale), int(720 * args.scale))

    root = tk.Tk()
    root.title("Bird Translation — installation test")
    root.configure(bg="black")
    # a python window on macOS opens BEHIND everything else, which looks like "no video"
    root.lift(); root.attributes("-topmost", True); root.focus_force()
    root.after(800, lambda: root.attributes("-topmost", False))
    panel = tk.Label(root, bg="black"); panel.pack()
    caption = tk.Label(root, bg="black", fg="white", font=("Menlo", 13), pady=6)
    caption.pack(fill="x")
    state = {"frame": -1, "img": None, "running": True}

    def close():
        state["running"] = False
        root.after(50, root.destroy)
    root.protocol("WM_DELETE_WINDOW", close)
    root.bind("<Escape>", lambda _e: close())

    def tick():
        if not state["running"]:
            return
        with lock:
            t = pos["main"] / SR % (len(main_audio) / SR)
        want = int(t * fps)
        if want != state["frame"]:
            try:
                frame = reader.get_data(want)         # seeks only when the position jumps
                img = Image.fromarray(frame).resize(size, Image.BILINEAR)
                state["img"] = ImageTk.PhotoImage(img)     # keep a reference or it vanishes
                panel.configure(image=state["img"])
                state["frame"] = want
            except Exception as e:
                print(f"\n  [video {type(e).__name__}: {e}]", file=sys.stderr)
        caption.configure(text=phase_at(t))
        root.after(int(1000 / fps / 2), tick)         # poll twice per frame, no need to be exact

    tick()
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        state["running"] = False
        for s in streams:
            s.stop(); s.close()
        reader.close()
        print("stopped.")


if __name__ == "__main__":
    main()
