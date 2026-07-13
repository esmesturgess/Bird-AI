from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
LOCAL_OUTPUT_OVERRIDE_ENV = "BIRD_TRANSLATION_ALLOW_LOCAL_OUTPUT"
GOOGLE_DRIVE_ROOT_ENV = "BIRD_TRANSLATION_GOOGLE_DRIVE_ROOT"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _truthy_env(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def google_drive_roots() -> list[Path]:
    configured = os.environ.get(GOOGLE_DRIVE_ROOT_ENV, "").strip()
    if configured:
        return [Path(configured).expanduser().resolve()]

    cloud_storage = Path.home() / "Library" / "CloudStorage"
    if not cloud_storage.exists():
        return []
    return sorted(path.resolve() for path in cloud_storage.glob("GoogleDrive-*") if path.is_dir())


def require_google_drive_output(path: Path, *, purpose: str = "output directory") -> Path:
    resolved = path.expanduser().resolve()
    if _truthy_env(os.environ.get(LOCAL_OUTPUT_OVERRIDE_ENV)):
        return resolved

    roots = google_drive_roots()
    if any(resolved == root or resolved.is_relative_to(root) for root in roots):
        return resolved

    roots_text = ", ".join(root.as_posix() for root in roots) or "no Google Drive roots found"
    raise ValueError(
        f"{purpose} must be inside Google Drive so it does not take up local project space. "
        f"Got: {resolved.as_posix()}. Google Drive roots: {roots_text}. "
        f"Set {GOOGLE_DRIVE_ROOT_ENV} if Drive moved, or set {LOCAL_OUTPUT_OVERRIDE_ENV}=1 "
        "only for an intentional local scratch run."
    )


def list_audio_files(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return text.strip("_") or "item"


def load_audio_mono(path: Path, sr: int) -> np.ndarray:
    y, _ = librosa.load(path.as_posix(), sr=sr, mono=True)
    return normalize_audio(y)


def normalize_audio(y: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)


def save_wav(path: Path, y: np.ndarray, sr: int) -> None:
    ensure_dir(path.parent)
    sf.write(path.as_posix(), y, sr)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = float(np.nanmean(values)) if values.size else 0.0
    std = float(np.nanstd(values)) if values.size else 0.0
    if std <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - mean) / std
