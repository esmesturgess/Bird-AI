from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from scipy.signal import medfilt


@dataclass(frozen=True)
class SegmentConfig:
    sr: int = 32000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    fmin: int = 200
    fmax: int = 12000
    seg_threshold_db: float = 8.0
    min_dur: float = 0.25
    max_dur: float = 4.0
    merge_gap: float = 0.25
    birdiness_min_hz: float = 700.0
    birdiness_ratio_min: float = 0.0
    rms_min: float = 0.003


def _find_regions(active: np.ndarray) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start = None
    for idx, val in enumerate(active):
        if val and start is None:
            start = idx
        if not val and start is not None:
            regions.append((start, idx))
            start = None
    if start is not None:
        regions.append((start, len(active)))
    return regions


def _merge_regions(
    regions: list[tuple[int, int]], max_gap_frames: int
) -> list[tuple[int, int]]:
    if not regions:
        return []
    merged: list[tuple[int, int]] = [regions[0]]
    for start, end in regions[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap_frames:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def _birdiness_ratio(
    y: np.ndarray, sr: int, split_hz: float, n_fft: int = 1024, hop_length: int = 256
) -> float:
    if len(y) < n_fft:
        return 0.0
    spec = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    high = np.sum(spec[freqs >= split_hz])
    total = np.sum(spec)
    if total <= 1e-12:
        return 0.0
    return float(high / total)


def segment_events(
    y: np.ndarray,
    cfg: SegmentConfig,
) -> list[tuple[int, int]]:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0,
    )
    mel_db = librosa.power_to_db(np.maximum(mel, 1e-12), ref=np.max)
    mel_freqs = librosa.mel_frequencies(n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax)
    # Use broad-band activity for detection so low-frequency calls (e.g., coos)
    # are still segmented. "Birdiness" remains an optional post-filter.
    band_mask = (mel_freqs >= cfg.fmin) & (mel_freqs <= cfg.fmax)
    if not np.any(band_mask):
        return []

    envelope = np.median(mel_db[band_mask], axis=0)
    kernel = 9 if len(envelope) >= 9 else max(1, len(envelope) | 1)
    smooth = medfilt(envelope, kernel_size=kernel)
    baseline = float(np.median(smooth))
    active = smooth > (baseline + cfg.seg_threshold_db)
    regions = _find_regions(active)

    max_gap_frames = max(1, int(round(cfg.merge_gap * cfg.sr / cfg.hop_length)))
    merged = _merge_regions(regions, max_gap_frames=max_gap_frames)

    events: list[tuple[int, int]] = []
    for s_frame, e_frame in merged:
        start = int(s_frame * cfg.hop_length)
        end = int(min(len(y), e_frame * cfg.hop_length))
        dur = (end - start) / cfg.sr
        if dur < cfg.min_dur:
            continue
        if dur > cfg.max_dur:
            continue
        clip = y[start:end]
        if np.sqrt(np.mean(np.square(clip)) + 1e-12) < cfg.rms_min:
            continue
        if cfg.birdiness_ratio_min > 0:
            birdiness = _birdiness_ratio(clip, cfg.sr, split_hz=cfg.birdiness_min_hz)
            if birdiness < cfg.birdiness_ratio_min:
                continue
        events.append((start, end))
    return events
