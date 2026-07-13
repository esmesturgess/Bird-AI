from __future__ import annotations

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .utils import ensure_dir, save_json, zscore


def _spectral_entropy(spec_power: np.ndarray) -> float:
    power = np.maximum(spec_power, 1e-12)
    p = power / np.sum(power, axis=0, keepdims=True)
    entropy = -np.sum(p * np.log(p), axis=0)
    norm = np.log(power.shape[0] + 1e-12)
    return float(np.mean(entropy / norm))


def _pitch_features(y: np.ndarray, sr: int, hop_length: int = 256) -> dict[str, float]:
    try:
        f0 = librosa.yin(
            y,
            fmin=400,
            fmax=min(8000, sr // 2 - 50),
            sr=sr,
            frame_length=1024,
            hop_length=hop_length,
        )
    except Exception:
        return {
            "pitch_mean_hz": np.nan,
            "pitch_std_hz": np.nan,
            "pitch_stability": 0.0,
            "pitch_voiced_fraction": 0.0,
        }
    valid = np.isfinite(f0) & (f0 > 0)
    voiced_frac = float(np.mean(valid)) if valid.size else 0.0
    if np.sum(valid) < 3:
        return {
            "pitch_mean_hz": np.nan,
            "pitch_std_hz": np.nan,
            "pitch_stability": 0.0,
            "pitch_voiced_fraction": voiced_frac,
        }
    f0v = f0[valid]
    mean = float(np.mean(f0v))
    std = float(np.std(f0v))
    cv = std / (mean + 1e-12)
    stability = float(1.0 / (1.0 + cv))
    return {
        "pitch_mean_hz": mean,
        "pitch_std_hz": std,
        "pitch_stability": stability,
        "pitch_voiced_fraction": voiced_frac,
    }


def _repetition_features(y: np.ndarray, sr: int, duration_s: float) -> dict[str, float]:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    if onset_env.size == 0:
        return {
            "repetition_rate_hz": 0.0,
            "ioi_mean_s": np.nan,
            "ioi_std_s": np.nan,
            "onset_count": 0,
        }
    threshold = float(np.mean(onset_env) + 0.5 * np.std(onset_env))
    min_dist = max(1, int(round(0.08 * sr / 256)))
    peaks, _ = find_peaks(onset_env, height=threshold, distance=min_dist)
    onset_times = librosa.frames_to_time(peaks, sr=sr, hop_length=256)
    ioi = np.diff(onset_times) if len(onset_times) > 1 else np.array([])
    rep_rate = float(len(peaks) / max(duration_s, 1e-6))
    return {
        "repetition_rate_hz": rep_rate,
        "ioi_mean_s": float(np.mean(ioi)) if ioi.size else np.nan,
        "ioi_std_s": float(np.std(ioi)) if ioi.size else np.nan,
        "onset_count": int(len(peaks)),
    }


def extract_features(y: np.ndarray, sr: int) -> dict[str, float]:
    duration_s = len(y) / sr if sr > 0 else 0.0
    if len(y) < 1024:
        return {
            "duration_s": duration_s,
            "rms": 0.0,
            "peak": 0.0,
            "spectral_centroid": 0.0,
            "spectral_bandwidth": 0.0,
            "spectral_rolloff": 0.0,
            "spectral_flatness": 1.0,
            "spectral_entropy": 0.0,
            "harmonic_percussive_ratio": 0.0,
            "tonality_proxy": 0.0,
            "repetition_rate_hz": 0.0,
            "ioi_mean_s": np.nan,
            "ioi_std_s": np.nan,
            "onset_count": 0,
            "pitch_mean_hz": np.nan,
            "pitch_std_hz": np.nan,
            "pitch_stability": 0.0,
            "pitch_voiced_fraction": 0.0,
        }

    rms = float(np.sqrt(np.mean(np.square(y)) + 1e-12))
    peak = float(np.max(np.abs(y)))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]

    spec = np.abs(librosa.stft(y=y, n_fft=1024, hop_length=256)) ** 2
    entropy = _spectral_entropy(spec)

    harmonic, percussive = librosa.effects.hpss(y)
    h_rms = float(np.sqrt(np.mean(np.square(harmonic)) + 1e-12))
    p_rms = float(np.sqrt(np.mean(np.square(percussive)) + 1e-12))
    hpr = h_rms / (p_rms + 1e-12)
    tonality_proxy = float((1.0 - np.mean(flatness)) * hpr)

    repetition = _repetition_features(y=y, sr=sr, duration_s=duration_s)
    pitch = _pitch_features(y=y, sr=sr)

    return {
        "duration_s": duration_s,
        "rms": rms,
        "peak": peak,
        "spectral_centroid": float(np.mean(centroid)),
        "spectral_bandwidth": float(np.mean(bandwidth)),
        "spectral_rolloff": float(np.mean(rolloff)),
        "spectral_flatness": float(np.mean(flatness)),
        "spectral_entropy": entropy,
        "harmonic_percussive_ratio": hpr,
        "tonality_proxy": tonality_proxy,
        **repetition,
        **pitch,
    }


def add_indices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in (
        "spectral_bandwidth",
        "spectral_flatness",
        "repetition_rate_hz",
        "rms",
        "tonality_proxy",
        "pitch_stability",
        "duration_s",
    ):
        if col not in out:
            out[col] = 0.0
    out["urgency_index"] = (
        zscore(out["spectral_bandwidth"].to_numpy())
        + zscore(out["spectral_flatness"].to_numpy())
        + zscore(out["repetition_rate_hz"].to_numpy())
        + zscore(out["rms"].to_numpy())
    )
    out["song_likeness"] = (
        zscore(out["tonality_proxy"].to_numpy())
        + zscore(np.nan_to_num(out["pitch_stability"].to_numpy(), nan=0.0))
        + zscore(out["duration_s"].to_numpy())
    )
    return out


def save_spectrogram_assets(
    y: np.ndarray,
    sr: int,
    clip_id: str,
    specs_dir: Path,
    force: bool = False,
) -> tuple[Path, Path]:
    ensure_dir(specs_dir)
    npy_path = specs_dir / f"{clip_id}.npy"
    png_path = specs_dir / f"{clip_id}.png"
    if npy_path.exists() and png_path.exists() and not force:
        return npy_path, png_path

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        fmin=200,
        fmax=min(12000, sr // 2),
        power=2.0,
    )
    mel_db = librosa.power_to_db(np.maximum(mel, 1e-12), ref=np.max)
    np.save(npy_path, mel_db.astype(np.float32))

    fig, ax = plt.subplots(figsize=(6, 3))
    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=256, x_axis="time", y_axis="mel", ax=ax
    )
    ax.set_title(f"Mel Spectrogram: {clip_id}")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return npy_path, png_path


def save_feature_json(path: Path, features: dict[str, float]) -> None:
    save_json(path, features)
