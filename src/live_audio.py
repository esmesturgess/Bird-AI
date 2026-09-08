"""Microphone capture primitives for the live runtime.

This is the input half of the mic PoC (see MIC_POC_PLAN.md, stage 1): it turns a
microphone into 48 kHz mono WAV files that the existing offline/BirdNET code can
already read. It deliberately knows nothing about BirdNET, species or responses —
`scripts/record_clip.py` is the CLI on top, and later `live_species.py` /
`live_respond.py` sit downstream.

Audio is captured at `TARGET_SR` (48 kHz) mono when the device supports it, and
resampled on the way out when it doesn't (cheap USB mics are often 44.1 kHz only).

Recordings are saved WITHOUT peak normalization, unlike `utils.save_wav` — absolute
level is the thing you calibrate the energy gate against, so it must survive to disk.
"""
from __future__ import annotations

import queue
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf

TARGET_SR = 48_000  # BirdNET operates on 48 kHz mono
DEFAULT_BLOCK_SECONDS = 0.1
SOUNDDEVICE_HINT = (
    "sounddevice is required for microphone capture.\n"
    "  pip install sounddevice\n"
    "and on Debian/Ubuntu/Mint also: sudo apt install -y libportaudio2 portaudio19-dev"
)


def import_sounddevice():
    """Import sounddevice with an actionable message when PortAudio is missing."""
    try:
        import sounddevice as sd
    except ImportError as exc:  # package not installed
        raise SystemExit(f"{SOUNDDEVICE_HINT}\n\n(original error: {exc})") from exc
    except OSError as exc:  # package installed, PortAudio shared library missing
        raise SystemExit(f"{SOUNDDEVICE_HINT}\n\n(original error: {exc})") from exc
    return sd


# ---------------------------------------------------------------- levels


def rms(block: np.ndarray) -> float:
    if block.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))


def dbfs(level: float) -> float:
    """Linear amplitude -> dBFS. Silence returns -inf-ish rather than raising."""
    return 20.0 * np.log10(level) if level > 1e-12 else -240.0


# ---------------------------------------------------------------- devices


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    max_channels: int
    default_samplerate: float


def list_input_devices() -> list[InputDevice]:
    sd = import_sounddevice()
    devices: list[InputDevice] = []
    for index, info in enumerate(sd.query_devices()):
        if int(info.get("max_input_channels", 0)) > 0:
            devices.append(
                InputDevice(
                    index=index,
                    name=str(info.get("name", f"device {index}")),
                    max_channels=int(info["max_input_channels"]),
                    default_samplerate=float(info.get("default_samplerate", 0.0)),
                )
            )
    return devices


def resolve_device(spec: str | None) -> int | None:
    """Accept an index ("3"), a case-insensitive name substring ("usb"), or None."""
    if spec is None or str(spec).strip() == "":
        return None

    spec = str(spec).strip()
    devices = list_input_devices()
    if not devices:
        raise SystemExit("No input devices found. Is a microphone plugged in?")

    if spec.isdigit():
        index = int(spec)
        if not any(d.index == index for d in devices):
            available = "\n".join(f"  [{d.index}] {d.name}" for d in devices)
            raise SystemExit(f"Device {index} is not an input device. Input devices:\n{available}")
        return index

    matches = [d for d in devices if spec.lower() in d.name.lower()]
    if not matches:
        available = "\n".join(f"  [{d.index}] {d.name}" for d in devices)
        raise SystemExit(f"No input device matching {spec!r}. Input devices:\n{available}")
    if len(matches) > 1:
        ambiguous = "\n".join(f"  [{d.index}] {d.name}" for d in matches)
        raise SystemExit(f"{spec!r} matches several input devices — be more specific:\n{ambiguous}")
    return matches[0].index


def describe_device(index: int | None) -> str:
    sd = import_sounddevice()
    info = sd.query_devices(index, "input") if index is not None else sd.query_devices(kind="input")
    return f"{info['name']} ({int(info['max_input_channels'])} ch, {float(info['default_samplerate']):.0f} Hz default)"


def _negotiate(sd, device: int | None) -> tuple[int, int]:
    """Pick a (samplerate, channels) pair the device will actually accept.

    Preference order: 48 kHz mono -> 48 kHz stereo -> device default mono/stereo.
    Anything that is not 48 kHz mono gets fixed up in `_to_mono_48k`.
    """
    info = sd.query_devices(device, "input") if device is not None else sd.query_devices(kind="input")
    device_sr = int(round(float(info.get("default_samplerate", TARGET_SR))))
    max_channels = max(1, int(info.get("max_input_channels", 1)))

    for samplerate in (TARGET_SR, device_sr):
        for channels in (1, 2):
            if channels > max_channels:
                continue
            try:
                sd.check_input_settings(device=device, samplerate=samplerate, channels=channels)
            except Exception:  # noqa: BLE001 - PortAudio raises assorted types here
                continue
            return samplerate, channels
    raise SystemExit(
        f"Could not open the microphone at any usable setting "
        f"(tried {TARGET_SR} Hz and {device_sr} Hz, mono and stereo)."
    )


def _to_mono(block: np.ndarray) -> np.ndarray:
    if block.ndim > 1:
        block = block.mean(axis=1)
    return block.astype(np.float32, copy=False)


def resample_to_target(audio: np.ndarray, samplerate: int) -> np.ndarray:
    """Resample a whole clip to TARGET_SR. No-op when the device already runs at 48 kHz.

    Applied once per finished clip rather than per block: a polyphase filter has no
    carried state, so resampling block-by-block would leave a discontinuity at every
    block boundary. Uses scipy rather than librosa — 44.1 -> 48 kHz is a rational
    160/147 conversion, and it keeps librosa/numba out of the live runtime.
    """
    if samplerate == TARGET_SR or audio.size == 0:
        return audio.astype(np.float32, copy=False)

    from math import gcd

    from scipy.signal import resample_poly

    divisor = gcd(TARGET_SR, samplerate)
    return resample_poly(audio, TARGET_SR // divisor, samplerate // divisor).astype(np.float32)


# ---------------------------------------------------------------- capture


@dataclass(frozen=True)
class CaptureConfig:
    device: int | None = None
    block_seconds: float = DEFAULT_BLOCK_SECONDS


@dataclass(frozen=True)
class MicSource:
    """An open microphone: its native rate, and a real-time stream of mono blocks."""

    samplerate: int
    blocks: Iterator[np.ndarray]


@contextmanager
def mic_blocks(cfg: CaptureConfig) -> Iterator[MicSource]:
    """Open the mic and yield a MicSource of mono float32 blocks at the DEVICE rate.

        with mic_blocks(CaptureConfig(device=3)) as mic:
            for block in mic.blocks:
                ...

    Blocks arrive in real time; the stream never ends on its own, so callers are
    expected to break out or be interrupted. Audio stays at the device's native rate —
    call `resample_to_target` once on the finished clip before saving.
    """
    sd = import_sounddevice()
    samplerate, channels = _negotiate(sd, cfg.device)
    blocksize = max(1, int(round(cfg.block_seconds * samplerate)))
    pending: queue.Queue[np.ndarray] = queue.Queue()
    overflows = 0

    def callback(indata, _frames, _time_info, status):
        nonlocal overflows
        if status and status.input_overflow:
            overflows += 1
        pending.put(indata.copy())

    stream = sd.InputStream(
        device=cfg.device,
        samplerate=samplerate,
        channels=channels,
        blocksize=blocksize,
        dtype="float32",
        callback=callback,
    )

    def generator() -> Iterator[np.ndarray]:
        while True:
            try:
                raw = pending.get(timeout=1.0)
            except queue.Empty:
                continue
            yield _to_mono(raw)

    with stream:
        try:
            yield MicSource(samplerate=samplerate, blocks=generator())
        finally:
            if overflows:
                print(f"[live_audio] warning: {overflows} input overflow(s) — some audio was dropped")


def record_fixed(cfg: CaptureConfig, seconds: float) -> tuple[np.ndarray, int]:
    """Block for `seconds`; return (mono audio, device samplerate)."""
    chunks: list[np.ndarray] = []
    with mic_blocks(cfg) as mic:
        target = int(round(seconds * mic.samplerate))
        captured = 0
        for block in mic.blocks:
            chunks.append(block)
            captured += len(block)
            if captured >= target:
                break
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return audio[:target], mic.samplerate


@dataclass(frozen=True)
class VoxConfig:
    """Sound-activated ('VOX') capture settings. Durations in seconds."""

    threshold: float = 0.01  # linear RMS; calibrate with the meter
    preroll: float = 1.0  # audio kept from *before* the trigger
    hang: float = 1.0  # quiet time that ends an event
    min_active: float = 0.15  # reject clicks: minimum above-threshold audio
    min_duration: float = 3.0  # zero-pad shorter events (BirdNET wants >= 3 s)
    max_duration: float = 15.0  # force-close runaway events


@dataclass(frozen=True)
class VoxEvent:
    audio: np.ndarray
    samplerate: int
    started_at: float
    duration: float
    active_seconds: float
    peak: float
    level: float


def vox_events(mic: MicSource, cfg: VoxConfig) -> Iterator[VoxEvent]:
    """Turn a live mic stream into discrete above-threshold events.

    Each event carries `cfg.preroll` seconds of lead-in, so the attack of a call is
    not clipped off by the trigger latency. Audio stays at the device rate.
    """
    sr = mic.samplerate
    preroll_target = int(round(cfg.preroll * sr))
    hang_target = int(round(cfg.hang * sr))
    max_target = int(round(cfg.max_duration * sr))
    min_samples = int(round(cfg.min_duration * sr))
    min_active_samples = int(round(cfg.min_active * sr))

    preroll: deque[np.ndarray] = deque()
    preroll_samples = 0
    active = False
    chunks: list[np.ndarray] = []
    total = 0
    quiet = 0
    active_samples = 0
    started_at = 0.0

    for block in mic.blocks:
        loud = rms(block) >= cfg.threshold

        if not active:
            if loud:
                active = True
                chunks = list(preroll) + [block]
                total = preroll_samples + len(block)
                quiet = 0
                active_samples = len(block)
                started_at = time.time() - (preroll_samples / sr)
                preroll.clear()
                preroll_samples = 0
            else:
                preroll.append(block)
                preroll_samples += len(block)
                while preroll and preroll_samples - len(preroll[0]) >= preroll_target:
                    preroll_samples -= len(preroll.popleft())
            continue

        chunks.append(block)
        total += len(block)
        if loud:
            quiet = 0
            active_samples += len(block)
        else:
            quiet += len(block)

        if quiet < hang_target and total < max_target:
            continue

        audio = np.concatenate(chunks)
        active = False
        chunks = []
        if active_samples < min_active_samples:
            continue  # a click, a door, a keystroke — not worth a file
        duration = len(audio) / sr
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))
        yield VoxEvent(
            audio=audio,
            samplerate=sr,
            started_at=started_at,
            duration=duration,
            active_seconds=active_samples / sr,
            peak=float(np.max(np.abs(audio))) if audio.size else 0.0,
            level=rms(audio),
        )


# ---------------------------------------------------------------- output


def save_recording(path: Path, audio: np.ndarray, samplerate: int = TARGET_SR) -> Path:
    """Resample to 48 kHz if needed and write 16-bit PCM WAV.

    Absolute level is preserved (no peak normalization, unlike `utils.save_wav`) —
    it is the thing the energy gate gets calibrated against.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = resample_to_target(audio, samplerate)
    clipped = np.clip(audio, -1.0, 1.0).astype(np.float32)
    sf.write(path.as_posix(), clipped, TARGET_SR, subtype="PCM_16")
    return path


def timestamped_name(prefix: str = "mic", when: float | None = None, suffix: str = ".wav") -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(when if when is not None else time.time()))
    return f"{prefix}_{stamp}{suffix}"
