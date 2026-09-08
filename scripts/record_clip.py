#!/usr/bin/env python3
"""Record microphone audio into WAV files for analysis.

The input end of the mic PoC (MIC_POC_PLAN.md stage 1). Produces 48 kHz mono
16-bit WAVs that `scripts/infer_clip.py` and the offline pipeline can read directly.

Four modes:

    # 1. which microphone?
    ./.venv/bin/python scripts/record_clip.py --list_devices

    # 2. calibrate: measure the room noise floor, get a suggested VOX threshold
    ./.venv/bin/python scripts/record_clip.py --meter 15 --device usb

    # 3. record one fixed-length clip (default 5 s), optionally identify it
    ./.venv/bin/python scripts/record_clip.py --seconds 5 --device usb --analyse

    # 4. leave it listening: save a file per sound event until Ctrl+C
    ./.venv/bin/python scripts/record_clip.py --auto --threshold 0.02 --device usb

Every saved file is appended to `recordings_log.csv` in the output directory, with
its levels — that log is what you tune the energy gate against later.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.live_audio import (  # noqa: E402
    TARGET_SR,
    CaptureConfig,
    VoxConfig,
    dbfs,
    describe_device,
    list_input_devices,
    mic_blocks,
    record_fixed,
    resolve_device,
    rms,
    save_recording,
    timestamped_name,
    vox_events,
)

DEFAULT_OUT_DIR = Path("data/raw/mic_recordings")
LOG_NAME = "recordings_log.csv"
LOG_FIELDS = [
    "file",
    "started_at",
    "mode",
    "duration_s",
    "active_s",
    "rms",
    "rms_dbfs",
    "peak",
    "peak_dbfs",
    "species",
    "score",
]


def log_recording(out_dir: Path, row: dict[str, object]) -> None:
    path = out_dir / LOG_NAME
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in LOG_FIELDS})


def analyse(clip: Path, birdnet_binary: str) -> tuple[str, float] | None:
    """Run the existing BirdNET species detection on a saved clip."""
    from infer_clip import detect_species, friendly  # noqa: PLC0415 - lazy, heavy import

    try:
        ranked = detect_species(clip, birdnet_binary, top_k=3)
    except Exception as exc:  # noqa: BLE001 - the recording is the deliverable; never lose it to a failed analysis
        print(f"  [analyse] skipped: {exc}")
        return None
    if not ranked:
        print("  [analyse] no species detected")
        return None
    label, score = ranked[0]
    print(f"  [analyse] {friendly(label)}  ({label}, confidence {score:.3f})")
    return label, score


def describe_level(label: str, level: float, peak: float) -> str:
    return f"{label} rms {level:.5f} ({dbfs(level):6.1f} dBFS)  peak {peak:.3f} ({dbfs(peak):6.1f} dBFS)"


def cmd_list_devices() -> None:
    devices = list_input_devices()
    if not devices:
        raise SystemExit("No input devices found. Is a microphone plugged in?")
    print("Input devices:")
    for device in devices:
        print(f"  [{device.index}]  {device.name}  ({device.max_channels} ch, {device.default_samplerate:.0f} Hz)")
    print("\nPass --device with the index or any unique part of the name, e.g. --device usb")


def cmd_meter(cfg: CaptureConfig, seconds: float) -> None:
    """Print live levels, then report the noise floor and a suggested threshold."""
    print(f"Metering for {seconds:.0f}s — stay quiet to measure the room noise floor, "
          f"then make some noise to check headroom.\n")
    levels: list[float] = []
    deadline = time.time() + seconds
    with mic_blocks(cfg) as mic:
        for block in mic.blocks:
            level = rms(block)
            levels.append(level)
            bars = int(np.clip((dbfs(level) + 60.0) / 60.0, 0.0, 1.0) * 40)
            print(f"\r  {dbfs(level):6.1f} dBFS |{'#' * bars:<40}|", end="", flush=True)
            if time.time() >= deadline:
                break
    print("\n")

    if not levels:
        raise SystemExit("No audio captured — check the device selection.")
    values = np.array(levels)
    floor = float(np.percentile(values, 20))  # quiet end = the room, not your voice
    suggested = max(floor * 3.0, 1e-4)
    print(f"  noise floor (p20) : {floor:.5f}  ({dbfs(floor):6.1f} dBFS)")
    print(f"  median            : {np.median(values):.5f}  ({dbfs(float(np.median(values))):6.1f} dBFS)")
    print(f"  loudest block     : {values.max():.5f}  ({dbfs(float(values.max())):6.1f} dBFS)")
    print(f"\n  suggested VOX threshold: --threshold {suggested:.4f}   (3x the noise floor)")
    if values.max() > 0.99:
        print("  WARNING: clipping detected — turn the input gain down.")
    if values.max() < 0.02:
        print("  WARNING: very quiet — turn the input gain up, or move the mic closer.")


def cmd_fixed(cfg: CaptureConfig, args: argparse.Namespace, out_dir: Path) -> None:
    print(f"Recording {args.seconds:.1f}s from {describe_device(cfg.device)} ...")
    audio, samplerate = record_fixed(cfg, args.seconds)
    if audio.size == 0:
        raise SystemExit("No audio captured — check the device selection.")

    started = time.time() - args.seconds
    path = save_recording(out_dir / timestamped_name(args.prefix, when=started), audio, samplerate)
    level, peak = rms(audio), float(np.max(np.abs(audio)))
    duration = len(audio) / samplerate
    print(f"Saved {path}  ({duration:.1f}s @ {TARGET_SR} Hz"
          + (f", captured at {samplerate} Hz)" if samplerate != TARGET_SR else ")"))
    print("  " + describe_level("level:", level, peak))
    if peak > 0.99:
        print("  WARNING: clipping — turn the input gain down.")
    elif peak < 0.02:
        print("  WARNING: very quiet — turn the input gain up, or move the mic closer.")

    result = analyse(path, args.birdnet_binary) if args.analyse else None
    log_recording(out_dir, {
        "file": path.name,
        "started_at": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        "mode": "fixed",
        "duration_s": round(duration, 3),
        "active_s": "",
        "rms": round(level, 6),
        "rms_dbfs": round(dbfs(level), 1),
        "peak": round(peak, 6),
        "peak_dbfs": round(dbfs(peak), 1),
        "species": result[0] if result else "",
        "score": round(result[1], 4) if result else "",
    })


def cmd_auto(cfg: CaptureConfig, args: argparse.Namespace, out_dir: Path) -> None:
    vox = VoxConfig(
        threshold=args.threshold,
        preroll=args.preroll,
        hang=args.hang,
        min_active=args.min_active,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )
    limit = f"{args.max_events} event(s)" if args.max_events else "Ctrl+C to stop"
    print(f"Listening on {describe_device(cfg.device)}")
    print(f"  threshold {vox.threshold:.4f} ({dbfs(vox.threshold):.1f} dBFS), "
          f"preroll {vox.preroll:.1f}s, hang {vox.hang:.1f}s — {limit}")
    print("  (no suitable threshold yet? run --meter first)\n")

    count = 0
    try:
        with mic_blocks(cfg) as mic:
            for event in vox_events(mic, vox):
                count += 1
                path = save_recording(
                    out_dir / timestamped_name(args.prefix, when=event.started_at),
                    event.audio,
                    event.samplerate,
                )
                print(f"[{count}] {path.name}  {event.duration:.1f}s "
                      f"({event.active_seconds:.1f}s above threshold)")
                print("      " + describe_level("level:", event.level, event.peak))

                result = analyse(path, args.birdnet_binary) if args.analyse else None
                log_recording(out_dir, {
                    "file": path.name,
                    "started_at": datetime.fromtimestamp(event.started_at).isoformat(timespec="seconds"),
                    "mode": "auto",
                    "duration_s": round(event.duration, 3),
                    "active_s": round(event.active_seconds, 3),
                    "rms": round(event.level, 6),
                    "rms_dbfs": round(dbfs(event.level), 1),
                    "peak": round(event.peak, 6),
                    "peak_dbfs": round(dbfs(event.peak), 1),
                    "species": result[0] if result else "",
                    "score": round(result[1], 4) if result else "",
                })
                if args.max_events and count >= args.max_events:
                    break
    except KeyboardInterrupt:
        print("\nStopped.")
    print(f"\n{count} recording(s) in {out_dir}  (levels logged to {LOG_NAME})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Record microphone audio into WAV files for analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--list_devices", action="store_true", help="List input devices and exit.")
    ap.add_argument("--device", default=None, help="Input device index or unique name substring.")
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--prefix", default="mic", help="Filename prefix (default: mic).")

    ap.add_argument("--meter", type=float, nargs="?", const=15.0, default=None,
                    metavar="SECONDS", help="Level meter / noise-floor calibration (default 15s).")
    ap.add_argument("--seconds", type=float, default=5.0, help="Fixed recording length (default 5).")
    ap.add_argument("--auto", action="store_true",
                    help="Sound-activated: save a file per event until Ctrl+C.")

    ap.add_argument("--threshold", type=float, default=0.01,
                    help="--auto trigger level, linear RMS (get one from --meter).")
    ap.add_argument("--preroll", type=float, default=1.0, help="--auto seconds kept before the trigger.")
    ap.add_argument("--hang", type=float, default=1.0, help="--auto quiet seconds that end an event.")
    ap.add_argument("--min_active", type=float, default=0.15,
                    help="--auto minimum above-threshold audio; rejects clicks.")
    ap.add_argument("--min_duration", type=float, default=3.0,
                    help="--auto shorter events are zero-padded to this (BirdNET wants >= 3s).")
    ap.add_argument("--max_duration", type=float, default=15.0, help="--auto maximum event length.")
    ap.add_argument("--max_events", type=int, default=0, help="--auto stop after N events (0 = unlimited).")

    ap.add_argument("--analyse", action="store_true",
                    help="Run BirdNET species detection on each saved clip (slow: ~3-10s per clip).")
    ap.add_argument("--birdnet_binary", default="./.venv/bin/birdnet-analyze")
    args = ap.parse_args()

    if args.list_devices:
        cmd_list_devices()
        return

    cfg = CaptureConfig(device=resolve_device(args.device))
    if args.meter is not None:
        cmd_meter(cfg, args.meter)
        return

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.auto:
        cmd_auto(cfg, args, out_dir)
    else:
        cmd_fixed(cfg, args, out_dir)


if __name__ == "__main__":
    main()
