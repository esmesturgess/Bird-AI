from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

@dataclass(frozen=True)
class BirdNETConfig:
    binary: str = "birdnet-analyze"
    top_k: int = 5
    model_dir: Path | None = None
    timeout_s: int = 1800  # was 120s; too short once a full-recording-frontend batch's
    # combined audio duration grows large (confirmed 2026-08-12: 372-file/354-minute batch
    # silently timed out and returned empty results -- the timeout error was swallowed by
    # the caller). See also: pipeline.py now surfaces batch_err instead of hiding it.


def ensure_birdnet_available(binary: str) -> None:
    if shutil.which(binary):
        return
    raise RuntimeError(
        f"BirdNET binary '{binary}' was not found in PATH. "
        "Install BirdNET-Analyzer and verify it is callable from your shell."
    )


def get_birdnet_version(binary: str) -> str:
    for args in ([binary, "--version"], [binary, "version"]):
        try:
            proc = subprocess.run(args, capture_output=True, text=True, check=False, timeout=15)
        except Exception:
            continue
        if proc.returncode != 0:
            continue
        text = (proc.stdout or proc.stderr or "").strip()
        if text:
            return text.splitlines()[0].strip()
    return "unknown"


def _prepare_audio_for_birdnet(audio_path: Path, work_dir: Path, min_seconds: float = 3.0) -> Path:
    try:
        y, sr = sf.read(audio_path.as_posix(), always_2d=False)
    except Exception:
        return audio_path
    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = np.mean(y, axis=1)
    if not isinstance(y, np.ndarray) or sr <= 0:
        return audio_path
    target_len = int(round(min_seconds * sr))
    if len(y) >= target_len:
        return audio_path
    padded = np.zeros(target_len, dtype=np.float32)
    padded[: len(y)] = y.astype(np.float32)
    out_path = work_dir / f"{audio_path.stem}_padded.wav"
    sf.write(out_path.as_posix(), padded, sr)
    return out_path


def _command_attempts(input_path: Path, out_dir: Path, cfg: BirdNETConfig) -> list[list[str]]:
    # BirdNET-Analyzer 2.4+ CLI: birdnet-analyze INPUT --output ... --rtype csv
    modern = [
        cfg.binary,
        input_path.as_posix(),
        "--output",
        out_dir.as_posix(),
        "--rtype",
        "csv",
        "--top_n",
        str(cfg.top_k),
        "--min_conf",
        "0.0",
        "--combine_results",
    ]
    modern_no_combine = [
        cfg.binary,
        input_path.as_posix(),
        "--output",
        out_dir.as_posix(),
        "--rtype",
        "csv",
        "--top_n",
        str(cfg.top_k),
        "--min_conf",
        "0.0",
    ]
    legacy = [
        cfg.binary,
        "analyze",
        input_path.as_posix(),
        "--output",
        out_dir.as_posix(),
        "--rtype",
        "csv",
        "--top_n",
        str(cfg.top_k),
    ]
    return [modern, modern_no_combine, legacy]


def _recording_command_attempts(
    *,
    input_path: Path,
    out_dir: Path,
    cfg: BirdNETConfig,
    overlap: float,
    min_conf: float,
    merge_consecutive: int,
) -> list[list[str]]:
    modern = [
        cfg.binary,
        input_path.as_posix(),
        "--output",
        out_dir.as_posix(),
        "--rtype",
        "csv",
        "--top_n",
        str(cfg.top_k),
        "--min_conf",
        str(min_conf),
        "--overlap",
        str(overlap),
        "--merge_consecutive",
        str(merge_consecutive),
        "--combine_results",
    ]
    modern_no_combine = [
        cfg.binary,
        input_path.as_posix(),
        "--output",
        out_dir.as_posix(),
        "--rtype",
        "csv",
        "--top_n",
        str(cfg.top_k),
        "--min_conf",
        str(min_conf),
        "--overlap",
        str(overlap),
        "--merge_consecutive",
        str(merge_consecutive),
    ]
    legacy = [
        cfg.binary,
        "analyze",
        input_path.as_posix(),
        "--output",
        out_dir.as_posix(),
        "--rtype",
        "csv",
        "--top_n",
        str(cfg.top_k),
    ]
    return [modern, modern_no_combine, legacy]


def _extract_label(item: dict[str, Any]) -> str:
    normalized = {str(k).strip().lower().replace(" ", "_"): v for k, v in item.items()}
    for key in (
        "scientific_name",
        "common_name",
        "label",
        "species",
        "sci_name",
        "species_code",
    ):
        value = normalized.get(key)
        if value:
            return str(value)
    return "unknown"


def _extract_score(item: dict[str, Any]) -> float:
    normalized = {str(k).strip().lower().replace(" ", "_"): v for k, v in item.items()}
    for key in ("confidence", "score", "probability", "p"):
        if key in normalized:
            try:
                return float(normalized[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def _normalize_predictions(items: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    best_by_label: dict[str, float] = {}
    for item in items:
        label = _extract_label(item)
        score = _extract_score(item)
        if label not in best_by_label or score > best_by_label[label]:
            best_by_label[label] = score
    parsed = [{"label": label, "score": score} for label, score in best_by_label.items()]
    parsed = sorted(parsed, key=lambda x: x["score"], reverse=True)
    return parsed[:top_k]


def _load_from_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("predictions", "results", "detections"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        return [payload]
    return []


def _load_from_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_dict = dict(row)
            normalized_keys = {str(k).strip().lower().replace(" ", "_") for k in row_dict.keys()}
            # Keep only prediction-style CSV rows.
            has_prediction_keys = (
                "confidence" in normalized_keys
                and ("scientific_name" in normalized_keys or "common_name" in normalized_keys)
            )
            if not has_prediction_keys:
                continue
            rows.append(row_dict)
    return rows


def _collect_output_predictions(out_dir: Path) -> list[dict[str, Any]]:
    csv_files = sorted(out_dir.rglob("*.csv"))
    preferred = [
        p
        for p in csv_files
        if ".birdnet.results.csv" in p.name.lower() or "combinedtable" in p.name.lower()
    ]
    ordered = preferred + [p for p in csv_files if p not in preferred]
    for f in ordered:
        items = _load_from_csv(f)
        if items:
            return items
    json_files = sorted(out_dir.rglob("*.json"))
    for f in json_files:
        items = _load_from_json(f)
        if items:
            return items
    return []


def _extract_file_path(item: dict[str, Any]) -> str:
    normalized = {str(k).strip().lower().replace(" ", "_"): v for k, v in item.items()}
    for key in ("file_path", "file", "audio_file", "path"):
        value = normalized.get(key)
        if value:
            return str(value)
    return ""


def _extract_window_value(item: dict[str, Any], prefix: str) -> float | None:
    normalized = {str(k).strip().lower().replace(" ", "_"): v for k, v in item.items()}
    preferred_keys = [
        f"{prefix}_(s)",
        f"{prefix}_s",
        prefix,
    ]
    for key in preferred_keys:
        if key in normalized:
            try:
                return float(normalized[key])
            except (TypeError, ValueError):
                continue
    for key, value in normalized.items():
        if key.startswith(prefix):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _prepare_recordings_dir(
    audio_paths: list[Path],
    work_dir: Path,
) -> tuple[Path, dict[str, str]]:
    prepared_dir = work_dir / "prepared_recordings"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_to_original: dict[str, str] = {}
    for idx, src in enumerate(audio_paths):
        ext = src.suffix if src.suffix else ".wav"
        prepared_name = f"recording_{idx:05d}{ext}"
        prepared_path = prepared_dir / prepared_name
        try:
            if prepared_path.exists() or prepared_path.is_symlink():
                prepared_path.unlink()
            os.symlink(src.resolve().as_posix(), prepared_path.as_posix())
        except Exception:
            shutil.copy2(src, prepared_path)
        prepared_to_original[prepared_name] = src.resolve().as_posix()
    return prepared_dir, prepared_to_original


def infer_recordings_with_birdnet(
    *,
    audio_paths: list[Path],
    cfg: BirdNETConfig,
    overlap: float = 0.0,
    min_conf: float = 0.0,
    merge_consecutive: int = 0,
) -> tuple[dict[str, list[dict[str, Any]]], str | None]:
    ensure_birdnet_available(cfg.binary)
    targets = [path.resolve().as_posix() for path in audio_paths]
    if not targets:
        return {}, None

    with tempfile.TemporaryDirectory(prefix="birdnet_recordings_") as tmp:
        tmp_dir = Path(tmp)
        prepared_dir, prepared_to_original = _prepare_recordings_dir(audio_paths, tmp_dir)
        out_dir = tmp_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        last_error = "BirdNET produced no parseable full-recording output."
        for cmd in _recording_command_attempts(
            input_path=prepared_dir,
            out_dir=out_dir,
            cfg=cfg,
            overlap=overlap,
            min_conf=min_conf,
            merge_consecutive=merge_consecutive,
        ):
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=cfg.timeout_s,
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                continue

            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip()
                last_error = stderr.splitlines()[-1] if stderr else f"Command failed with return code {proc.returncode}"
                continue

            items = _collect_output_predictions(out_dir)
            grouped: dict[str, dict[tuple[float, float], list[dict[str, Any]]]] = {key: {} for key in targets}
            for item in items:
                raw_path = _extract_file_path(item)
                if not raw_path:
                    continue
                prepared_name = Path(raw_path).name
                original_path = prepared_to_original.get(prepared_name, raw_path)
                original_key = Path(original_path).resolve().as_posix()
                if original_key not in grouped:
                    continue
                start_s = _extract_window_value(item, "start")
                end_s = _extract_window_value(item, "end")
                if start_s is None or end_s is None:
                    continue
                window_key = (float(start_s), float(end_s))
                grouped[original_key].setdefault(window_key, []).append(item)

            out: dict[str, list[dict[str, Any]]] = {}
            for key in targets:
                windows = grouped.get(key, {})
                segments: list[dict[str, Any]] = []
                for (start_s, end_s), window_items in sorted(windows.items(), key=lambda item: item[0]):
                    segments.append(
                        {
                            "start_s": float(start_s),
                            "end_s": float(end_s),
                            "predictions": _normalize_predictions(window_items, top_k=cfg.top_k),
                        }
                    )
                out[key] = segments
            return out, None

        return {}, last_error


def _prepare_events_dir(
    events_dir: Path,
    event_wavs: list[str],
    work_dir: Path,
    min_seconds: float = 3.0,
) -> tuple[Path, dict[str, str]]:
    prepared_dir = work_dir / "prepared_events"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_to_original: dict[str, str] = {}
    for rel in event_wavs:
        src = events_dir / rel
        if not src.exists():
            continue
        prepared = _prepare_audio_for_birdnet(src, prepared_dir, min_seconds=min_seconds)
        if prepared == src:
            copied = prepared_dir / src.name
            shutil.copy2(src, copied)
            prepared = copied
        prepared_name = prepared.name
        prepared_to_original[prepared_name] = Path(rel).name
    return prepared_dir, prepared_to_original


def infer_events_with_birdnet(
    events_dir: Path,
    event_wavs: list[str],
    cfg: BirdNETConfig,
) -> tuple[dict[str, list[dict[str, float | str]]], str | None]:
    """
    Run BirdNET once over all event clips and return top-k predictions per clip filename.
    Returns mapping keyed by event wav basename.
    """
    ensure_birdnet_available(cfg.binary)
    target_keys = [Path(x).name for x in event_wavs]
    if not target_keys:
        return {}, None

    with tempfile.TemporaryDirectory(prefix="birdnet_batch_") as tmp:
        tmp_dir = Path(tmp)
        prepared_dir, prepared_to_original = _prepare_events_dir(
            events_dir=events_dir,
            event_wavs=event_wavs,
            work_dir=tmp_dir,
            min_seconds=3.0,
        )
        out_dir = tmp_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        last_error = "BirdNET produced no parseable output."
        for cmd in _command_attempts(input_path=prepared_dir, out_dir=out_dir, cfg=cfg):
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=cfg.timeout_s,
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                continue

            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip()
                last_error = stderr.splitlines()[-1] if stderr else f"Command failed with return code {proc.returncode}"
                continue

            # Successful run; empty outputs are allowed (no detections).
            items = _collect_output_predictions(out_dir)
            grouped: dict[str, list[dict[str, Any]]] = {k: [] for k in target_keys}
            for item in items:
                raw_path = _extract_file_path(item)
                if not raw_path:
                    continue
                prepared_name = Path(raw_path).name
                original_name = prepared_to_original.get(prepared_name, prepared_name)
                if original_name in grouped:
                    grouped[original_name].append(item)

            preds_map = {
                key: _normalize_predictions(grouped.get(key, []), top_k=cfg.top_k)
                for key in target_keys
            }
            return preds_map, None

        return {}, last_error


def infer_clip_with_birdnet(
    audio_path: Path,
    cfg: BirdNETConfig,
) -> tuple[list[dict[str, float | str]], str | None]:
    # Retained for compatibility; implemented via batch path with one clip.
    preds_map, err = infer_events_with_birdnet(
        events_dir=audio_path.parent,
        event_wavs=[audio_path.name],
        cfg=cfg,
    )
    return preds_map.get(audio_path.name, []), err
