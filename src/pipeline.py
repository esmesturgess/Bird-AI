from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from .artifacts import export_species_check_artifacts, export_training_artifacts
from .birdnet_infer import (
    BirdNETConfig,
    ensure_birdnet_available,
    get_birdnet_version,
    infer_events_with_birdnet,
    infer_recordings_with_birdnet,
)
from .cluster import representative_indices_by_cluster, run_dual_mode_clustering
from .dataset import (
    allowed_values,
    load_manifest_records,
    load_species_config,
    target_species_labels,
    validate_manifest,
)
from .embed import EmbeddingBackend, FallbackBackend, generate_embeddings
from .features import add_indices, extract_features, save_feature_json, save_spectrogram_assets
from .preprocess import preprocess_audio
from .report import (
    create_cluster_mean_spectrograms,
    create_distribution_plots,
    generate_html_report,
    write_cluster_playlists,
)
from .segment import SegmentConfig, segment_events
from .utils import ensure_dir, list_audio_files, load_audio_mono, require_google_drive_output, save_wav, slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bird vocalization vertical-slice pipeline")
    parser.add_argument("--input_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output_dir", type=Path, default=Path("data"))
    parser.add_argument("--species_config", type=Path, default=Path("config/v1_species.yaml"))
    parser.add_argument("--train_manifest", type=Path, default=Path("data/manifests/train_manifest.csv"))
    parser.add_argument("--val_manifest", type=Path, default=Path("data/manifests/val_manifest.csv"))
    parser.add_argument("--sr", type=int, default=32000)
    parser.add_argument("--min_dur", type=float, default=0.25)
    parser.add_argument("--max_dur", type=float, default=4.0)
    parser.add_argument("--seg_threshold_db", type=float, default=8.0)
    parser.add_argument("--merge_gap", type=float, default=0.25)
    parser.add_argument("--birdiness_min_hz", type=float, default=700.0)
    parser.add_argument("--birdiness_ratio_min", type=float, default=0.0)
    parser.add_argument("--seg_rms_min", type=float, default=0.003)
    parser.add_argument(
        "--max_events_per_recording",
        type=int,
        default=10,
        help="Maximum segmented events to keep per source recording. Use 0 to disable the cap.",
    )
    parser.add_argument(
        "--selection_pref_duration",
        type=float,
        default=0.0,
        help="Preferred event duration in seconds for duration-aware segment ranking. Use 0 to disable.",
    )
    parser.add_argument(
        "--selection_score_min",
        type=float,
        default=0.0,
        help="Absolute minimum selection quality score required to keep an event after ranking. Use 0 to disable.",
    )
    parser.add_argument(
        "--selection_relative_min",
        type=float,
        default=0.0,
        help="Minimum selection quality score as a fraction of the best event within a recording. Use 0 to disable.",
    )
    parser.add_argument(
        "--birdnet_context_seconds",
        type=float,
        default=3.0,
        help="Length of BirdNET inference clips built from the original recording around each event. Use 0 to pass raw event clips instead.",
    )
    parser.add_argument("--conf_thresh", type=float, default=0.25)
    parser.add_argument(
        "--species_frontend",
        type=str,
        choices=["segment_then_birdnet", "birdnet_windows"],
        default="segment_then_birdnet",
        help="Use the current custom segmenter before BirdNET, or let BirdNET scan full recordings in overlapping windows first.",
    )
    parser.add_argument(
        "--birdnet_window_overlap",
        type=float,
        default=1.5,
        help="Overlap in seconds for full-recording BirdNET window scans. BirdNET uses 3-second windows.",
    )
    parser.add_argument(
        "--birdnet_window_merge_consecutive",
        type=int,
        default=0,
        help="Passed through to BirdNET --merge_consecutive for full-recording window scans.",
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--birdnet_binary", type=str, default="birdnet-analyze")
    parser.add_argument("--birdnet_model_dir", type=Path, default=None)
    parser.add_argument("--min_species_samples", type=int, default=5)
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=0)
    parser.add_argument("--hdbscan_min_samples", type=int, default=0)
    parser.add_argument(
        "--use_umap",
        action="store_true",
        help="Use UMAP for 2D projection (can be unstable on some systems).",
    )
    parser.add_argument(
        "--embedding_backend",
        type=str,
        choices=["birdnet", "mel_time", "mel_stats", "avex_effnet_bio"],
        default="birdnet",
    )
    parser.add_argument("--embedding_mel_frames", type=int, default=96)
    parser.add_argument("--birdnet_embeddings_binary", type=str, default="birdnet-embeddings")
    parser.add_argument(
        "--embedding_fallback_backend",
        type=str,
        choices=["none", "mel_time", "mel_stats"],
        default="mel_time",
    )
    parser.add_argument(
        "--stop_after_species",
        action="store_true",
        help="Run segmentation and BirdNET predictions only, then stop before clustering.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute all outputs")
    return parser.parse_args()


def _paths(output_dir: Path) -> dict[str, Path]:
    return {
        "events": output_dir / "events",
        "preds": output_dir / "preds",
        "features": output_dir / "features",
        "specs": output_dir / "specs",
        "embeddings": output_dir / "embeddings",
        "clusters": output_dir / "clusters",
        "reports": output_dir / "reports",
        "manifests": output_dir / "manifests",
        "artifacts": output_dir / "artifacts",
    }


def _resolve_audio_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _normalize_source_set(values: pd.Series) -> set[str]:
    normalized: set[str] = set()
    for value in values.fillna("").astype(str):
        if not value:
            continue
        normalized.add(Path(value).resolve().as_posix())
    return normalized


def _stat_fingerprint(path_str: str) -> dict[str, object]:
    path = Path(path_str).resolve()
    stat = path.stat()
    return {
        "audio_path": path.as_posix(),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _segmentation_cache_meta(
    recordings_df: pd.DataFrame,
    cfg: SegmentConfig,
    max_events_per_recording: int,
    selection_pref_duration: float,
    selection_score_min: float,
    selection_relative_min: float,
) -> dict[str, object]:
    recordings: list[dict[str, object]] = []
    for audio_path in recordings_df["audio_path"].fillna("").astype(str):
        if not audio_path:
            continue
        recordings.append(_stat_fingerprint(audio_path))

    recordings.sort(key=lambda item: str(item["audio_path"]))
    return {
        "cache_version": 1,
        "segmentation": {
            "sr": int(cfg.sr),
            "min_dur": float(cfg.min_dur),
            "max_dur": float(cfg.max_dur),
            "seg_threshold_db": float(cfg.seg_threshold_db),
            "merge_gap": float(cfg.merge_gap),
            "birdiness_min_hz": float(cfg.birdiness_min_hz),
            "birdiness_ratio_min": float(cfg.birdiness_ratio_min),
            "rms_min": float(cfg.rms_min),
            "max_events_per_recording": int(max_events_per_recording),
            "selection_pref_duration": float(selection_pref_duration),
            "selection_score_min": float(selection_score_min),
            "selection_relative_min": float(selection_relative_min),
        },
        "recordings": recordings,
    }


def _events_cache_matches(
    manifest_path: Path,
    meta_path: Path,
    recordings_df: pd.DataFrame,
    cfg: SegmentConfig,
    max_events_per_recording: int,
    selection_pref_duration: float,
    selection_score_min: float,
    selection_relative_min: float,
) -> bool:
    if not manifest_path.exists() or not meta_path.exists():
        return False
    try:
        cached = pd.read_csv(manifest_path)
        cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if "clip_id" not in cached:
        return False
    return cached_meta == _segmentation_cache_meta(
        recordings_df=recordings_df,
        cfg=cfg,
        max_events_per_recording=max_events_per_recording,
        selection_pref_duration=selection_pref_duration,
        selection_score_min=selection_score_min,
        selection_relative_min=selection_relative_min,
    )


def _selection_duration_weight(duration_s: float, preferred_duration: float) -> float:
    if preferred_duration <= 0:
        return 1.0
    ratio = duration_s / max(preferred_duration, 1e-6)
    return float(np.clip(ratio, 0.35, 1.5))


def _birdnet_window_cache_meta(
    recordings_df: pd.DataFrame,
    cfg: BirdNETConfig,
    overlap: float,
    merge_consecutive: int,
    sr: int,
) -> dict[str, object]:
    recordings: list[dict[str, object]] = []
    for audio_path in recordings_df["audio_path"].fillna("").astype(str):
        if not audio_path:
            continue
        recordings.append(_stat_fingerprint(audio_path))
    recordings.sort(key=lambda item: str(item["audio_path"]))
    return {
        "cache_version": 1,
        "frontend": {
            "type": "birdnet_windows",
            "birdnet_binary": str(cfg.binary),
            "birdnet_model_dir": cfg.model_dir.as_posix() if cfg.model_dir else "",
            "top_k": int(cfg.top_k),
            "overlap": float(overlap),
            "merge_consecutive": int(merge_consecutive),
            "sample_rate": int(sr),
        },
        "recordings": recordings,
    }


def _birdnet_window_cache_matches(
    *,
    manifest_path: Path,
    preds_path: Path,
    meta_path: Path,
    recordings_df: pd.DataFrame,
    cfg: BirdNETConfig,
    overlap: float,
    merge_consecutive: int,
    sr: int,
) -> bool:
    if not manifest_path.exists() or not preds_path.exists() or not meta_path.exists():
        return False
    try:
        cached_events = pd.read_csv(manifest_path)
        cached_preds = pd.read_csv(preds_path)
        cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if "clip_id" not in cached_events or "clip_id" not in cached_preds:
        return False
    if set(cached_events["clip_id"]) != set(cached_preds["clip_id"]):
        return False
    return cached_meta == _birdnet_window_cache_meta(
        recordings_df=recordings_df,
        cfg=cfg,
        overlap=overlap,
        merge_consecutive=merge_consecutive,
        sr=sr,
    )


def _recordings_from_manifest(
    train_manifest_path: Path,
    species_config_path: Path,
) -> pd.DataFrame:
    config = load_species_config(species_config_path)
    validate_manifest(
        path=train_manifest_path,
        species_labels=target_species_labels(config),
        allowed_recording_sources=allowed_values(config, "allowed_recording_sources"),
        allowed_splits=allowed_values(config, "allowed_splits"),
        require_files=True,
    )
    rows = load_manifest_records(train_manifest_path)
    normalized_rows: list[dict[str, object]] = []
    for row in rows:
        audio_path = _resolve_audio_path(row["audio_path"], base_dir=Path.cwd())
        normalized_rows.append(
            {
                **row,
                "audio_path": audio_path.as_posix(),
                "audio_path_original": row["audio_path"],
                "selection_source": "train_manifest",
            }
        )
    return pd.DataFrame(normalized_rows)


def _recordings_from_input_dir(input_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for audio_path in list_audio_files(input_dir):
        if audio_path.name == ".DS_Store":
            continue
        rows.append(
            {
                "recording_id": slugify(audio_path.stem),
                "audio_path": audio_path.resolve().as_posix(),
                "audio_path_original": audio_path.resolve().as_posix(),
                "species_label": "",
                "species_common_name": "",
                "scientific_name": "",
                "species_source": "input_dir_scan",
                "recording_source": "unspecified",
                "split": "train",
                "notes": "",
                "selection_source": "input_dir_scan",
            }
        )
    return pd.DataFrame(rows)


def load_training_recordings(
    input_dir: Path,
    train_manifest_path: Path,
    species_config_path: Path,
) -> tuple[pd.DataFrame, str]:
    if train_manifest_path.exists():
        manifest_df = _recordings_from_manifest(
            train_manifest_path=train_manifest_path,
            species_config_path=species_config_path,
        )
        if not manifest_df.empty:
            return manifest_df, "train_manifest"
        print(f"Train manifest is empty at {train_manifest_path.as_posix()}; falling back to input_dir scan.")

    scanned_df = _recordings_from_input_dir(input_dir)
    if scanned_df.empty:
        raise FileNotFoundError(
            f"No recordings found. Populate {train_manifest_path.as_posix()} or add audio files under {input_dir.as_posix()}."
        )
    return scanned_df, "input_dir_scan"


def run_segmentation(
    recordings_df: pd.DataFrame,
    paths: dict[str, Path],
    cfg: SegmentConfig,
    max_events_per_recording: int,
    selection_pref_duration: float,
    selection_score_min: float,
    selection_relative_min: float,
    force: bool,
) -> pd.DataFrame:
    manifest_path = paths["manifests"] / "events_manifest.csv"
    meta_path = paths["manifests"] / "events_manifest.meta.json"
    # Segmentation is cached because it is one of the slowest repeatable steps and
    # its outputs only depend on the input recordings plus the selector settings.
    if not force and _events_cache_matches(
        manifest_path=manifest_path,
        meta_path=meta_path,
        recordings_df=recordings_df,
        cfg=cfg,
        max_events_per_recording=max_events_per_recording,
        selection_pref_duration=selection_pref_duration,
        selection_score_min=selection_score_min,
        selection_relative_min=selection_relative_min,
    ):
        print(f"Reusing cached segmented events from {manifest_path.as_posix()}")
        return pd.read_csv(manifest_path)

    rows: list[dict[str, object]] = []
    total_detected = 0
    total_kept = 0
    for _, recording in tqdm(recordings_df.iterrows(), total=len(recordings_df), desc="Segment recordings"):
        audio_path = Path(str(recording["audio_path"]))
        y = load_audio_mono(audio_path, sr=cfg.sr)
        y = preprocess_audio(y)
        events = segment_events(y, cfg)
        total_detected += len(events)

        event_candidates: list[dict[str, object]] = []
        for start, end in events:
            clip = y[start:end]
            duration_s = (end - start) / cfg.sr
            segment_rms = float(np.sqrt(np.mean(np.square(clip)) + 1e-12))
            keep_score = float(segment_rms * duration_s)
            duration_weight = _selection_duration_weight(duration_s, selection_pref_duration)
            selection_quality_score = float(keep_score * duration_weight)
            event_candidates.append(
                {
                    "start": start,
                    "end": end,
                    "duration_s": duration_s,
                    "segment_rms": segment_rms,
                    "segment_keep_score": keep_score,
                    "selection_duration_weight": duration_weight,
                    "selection_quality_score": selection_quality_score,
                }
            )

        # When a recording produces lots of segments, we keep the strongest ones
        # rather than treating every tiny fragment as equally useful training data.
        if max_events_per_recording > 0 and len(event_candidates) > max_events_per_recording:
            ranked = sorted(
                event_candidates,
                key=lambda item: (
                    float(item["selection_quality_score"]),
                    float(item["segment_keep_score"]),
                    float(item["segment_rms"]),
                    float(item["duration_s"]),
                ),
                reverse=True,
            )
            best_quality = float(ranked[0]["selection_quality_score"]) if ranked else 0.0
            quality_floor = float(selection_score_min)
            if selection_relative_min > 0 and best_quality > 0:
                quality_floor = max(quality_floor, best_quality * float(selection_relative_min))
            filtered_ranked = [item for item in ranked if float(item["selection_quality_score"]) >= quality_floor]
            if not filtered_ranked:
                filtered_ranked = ranked[:1]
            kept_lookup: dict[tuple[int, int], int] = {}
            for rank, item in enumerate(filtered_ranked[:max_events_per_recording], start=1):
                kept_lookup[(int(item["start"]), int(item["end"]))] = rank
            kept_events = [item for item in event_candidates if (int(item["start"]), int(item["end"])) in kept_lookup]
            for item in kept_events:
                item["event_rank_in_recording"] = kept_lookup[(int(item["start"]), int(item["end"]))]
            selected_events = sorted(kept_events, key=lambda item: int(item["start"]))
        else:
            quality_floor = float(selection_score_min)
            if selection_relative_min > 0 and event_candidates:
                best_quality = max(float(item["selection_quality_score"]) for item in event_candidates)
                if best_quality > 0:
                    quality_floor = max(quality_floor, best_quality * float(selection_relative_min))
            filtered_candidates = [
                item for item in event_candidates if float(item["selection_quality_score"]) >= quality_floor
            ]
            if event_candidates and not filtered_candidates:
                ranked = sorted(
                    event_candidates,
                    key=lambda event: (
                        float(event["selection_quality_score"]),
                        float(event["segment_keep_score"]),
                        float(event["segment_rms"]),
                        float(event["duration_s"]),
                    ),
                    reverse=True,
                )
                filtered_candidates = ranked[:1]
            selected_events = sorted(filtered_candidates, key=lambda item: int(item["start"]))
            for rank, item in enumerate(
                sorted(
                    selected_events,
                    key=lambda event: (
                        float(event["selection_quality_score"]),
                        float(event["segment_keep_score"]),
                        float(event["segment_rms"]),
                        float(event["duration_s"]),
                    ),
                    reverse=True,
                ),
                start=1,
            ):
                item["event_rank_in_recording"] = rank

        total_kept += len(selected_events)
        stem = slugify(audio_path.stem)
        for idx, item in enumerate(selected_events):
            start = int(item["start"])
            end = int(item["end"])
            clip_id = f"{stem}_evt_{idx:05d}"
            clip_rel = Path(f"{clip_id}.wav")
            clip_path = paths["events"] / clip_rel
            clip = y[start:end]
            save_wav(clip_path, clip, cfg.sr)
            rows.append(
                {
                    "clip_id": clip_id,
                    "recording_id": str(recording.get("recording_id", "")),
                    "source_file": audio_path.as_posix(),
                    "audio_path": audio_path.as_posix(),
                    "audio_path_original": str(recording.get("audio_path_original", audio_path.as_posix())),
                    "species_label": str(recording.get("species_label", "")),
                    "species_common_name": str(recording.get("species_common_name", "")),
                    "scientific_name": str(recording.get("scientific_name", "")),
                    "species_source": str(recording.get("species_source", "")),
                    "recording_source": str(recording.get("recording_source", "")),
                    "split": str(recording.get("split", "train")),
                    "notes": str(recording.get("notes", "")),
                    "selection_source": str(recording.get("selection_source", "")),
                    "event_wav": clip_rel.as_posix(),
                    "start_s": start / cfg.sr,
                    "end_s": end / cfg.sr,
                    "duration_s": float(item["duration_s"]),
                    "segment_rms": float(item["segment_rms"]),
                    "segment_keep_score": float(item["segment_keep_score"]),
                    "selection_duration_weight": float(item["selection_duration_weight"]),
                    "selection_quality_score": float(item["selection_quality_score"]),
                    "event_rank_in_recording": int(item["event_rank_in_recording"]),
                    "n_events_detected_in_recording": int(len(event_candidates)),
                    "n_events_kept_in_recording": int(len(selected_events)),
                }
            )
    out = pd.DataFrame(rows)
    ensure_dir(paths["manifests"])
    out.to_csv(manifest_path, index=False)
    meta_path.write_text(
        json.dumps(
            _segmentation_cache_meta(
                recordings_df=recordings_df,
                cfg=cfg,
                max_events_per_recording=max_events_per_recording,
                selection_pref_duration=selection_pref_duration,
                selection_score_min=selection_score_min,
                selection_relative_min=selection_relative_min,
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if max_events_per_recording > 0 and total_detected > total_kept:
        print(
            f"Segmentation cap kept {total_kept}/{total_detected} detected events "
            f"with max {max_events_per_recording} per recording."
        )
    return out


def run_features(
    events_df: pd.DataFrame,
    paths: dict[str, Path],
    sr: int,
    force: bool,
) -> pd.DataFrame:
    master_path = paths["features"] / "master_features.csv"
    if master_path.exists() and not force:
        return pd.read_csv(master_path)

    rows: list[dict[str, object]] = []
    ensure_dir(paths["features"])
    ensure_dir(paths["specs"])

    for _, row in tqdm(events_df.iterrows(), total=len(events_df), desc="Extract features"):
        clip_path = paths["events"] / row["event_wav"]
        y, _ = librosa.load(clip_path.as_posix(), sr=sr, mono=True)
        feat = extract_features(y, sr)
        npy_path, png_path = save_spectrogram_assets(
            y=y, sr=sr, clip_id=row["clip_id"], specs_dir=paths["specs"], force=force
        )
        feat_json_path = paths["features"] / f"{row['clip_id']}.json"
        merged = {
            **row.to_dict(),
            **feat,
            "spec_npy": npy_path.name,
            "spec_png": png_path.name,
            "feature_json": feat_json_path.name,
        }
        save_feature_json(feat_json_path, {k: v for k, v in merged.items() if k != "feature_json"})
        rows.append(merged)

    df = pd.DataFrame(rows)
    df = add_indices(df)
    df.to_csv(master_path, index=False)
    return df


def _context_window_samples(
    *,
    start_s: float,
    end_s: float,
    sr: int,
    total_samples: int,
    target_seconds: float,
) -> tuple[int, int, int]:
    target_len = max(1, int(round(target_seconds * sr)))
    start = max(0, int(round(start_s * sr)))
    end = min(total_samples, int(round(end_s * sr)))
    center = max(0, min(total_samples, int(round((start + end) / 2.0))))
    if total_samples <= target_len:
        return 0, total_samples, target_len
    win_start = center - target_len // 2
    win_end = win_start + target_len
    if win_start < 0:
        win_start = 0
        win_end = target_len
    if win_end > total_samples:
        win_end = total_samples
        win_start = max(0, win_end - target_len)
    return int(win_start), int(win_end), target_len


def _prepare_birdnet_context_clips(
    events_df: pd.DataFrame,
    clips_dir: Path,
    context_seconds: float,
    force: bool,
) -> list[str]:
    ensure_dir(clips_dir)
    clip_names = [Path(str(v)).name for v in events_df["event_wav"].tolist()]
    if context_seconds <= 0:
        return clip_names

    expected_paths = [clips_dir / name for name in clip_names]
    if expected_paths and all(path.exists() for path in expected_paths) and not force:
        return clip_names

    for source_file, group in events_df.groupby("source_file", sort=False):
        source_path = Path(str(source_file))
        y, sr = librosa.load(source_path.as_posix(), sr=None, mono=True)
        if sr <= 0:
            continue
        total_samples = int(len(y))
        for _, row in group.iterrows():
            clip_name = Path(str(row["event_wav"])).name
            clip_path = clips_dir / clip_name
            if clip_path.exists() and not force:
                continue
            win_start, win_end, target_len = _context_window_samples(
                start_s=float(row["start_s"]),
                end_s=float(row["end_s"]),
                sr=int(sr),
                total_samples=total_samples,
                target_seconds=float(context_seconds),
            )
            clip = y[win_start:win_end].astype(np.float32, copy=False)
            if len(clip) < target_len:
                padded = np.zeros(target_len, dtype=np.float32)
                padded[: len(clip)] = clip
                clip = padded
            save_wav(clip_path, clip, int(sr))
    return clip_names


def run_birdnet_predictions(
    events_df: pd.DataFrame,
    paths: dict[str, Path],
    cfg: BirdNETConfig,
    birdnet_context_seconds: float,
    force: bool,
) -> pd.DataFrame:
    master_path = paths["preds"] / "master_predictions.csv"
    clips_dir = paths["preds"] / "clips"
    if master_path.exists() and not force:
        cached = pd.read_csv(master_path)
        if "clip_id" in cached and set(cached["clip_id"]) == set(events_df["clip_id"]):
            print(f"Reusing cached BirdNET predictions from {master_path.as_posix()}")
            return cached

    ensure_birdnet_available(cfg.binary)
    birdnet_version = get_birdnet_version(cfg.binary)
    rows: list[dict[str, object]] = []
    ensure_dir(paths["preds"])
    ensure_dir(clips_dir)
    # BirdNET tends to behave better on short fixed-length clips with surrounding
    # context than on extremely tight event fragments, so we optionally rebuild
    # 3-second-ish inference clips from the original recording first.
    event_wavs = _prepare_birdnet_context_clips(
        events_df=events_df,
        clips_dir=clips_dir,
        context_seconds=birdnet_context_seconds,
        force=force,
    )
    preds_map, batch_err = infer_events_with_birdnet(
        events_dir=clips_dir if birdnet_context_seconds > 0 else paths["events"],
        event_wavs=event_wavs,
        cfg=cfg,
    )
    for _, row in tqdm(events_df.iterrows(), total=len(events_df), desc="BirdNET postprocess"):
        clip_id = str(row["clip_id"])
        clip_name = Path(str(row["event_wav"])).name
        preds = preds_map.get(clip_name, [])
        err = batch_err

        top1_label = preds[0]["label"] if preds else ""
        top1_score = float(preds[0]["score"]) if preds else np.nan
        topk_labels = [p["label"] for p in preds]
        topk_scores = [float(p["score"]) for p in preds]
        inference_ok = err is None

        record = {
            "clip_id": clip_id,
            "event_wav": row["event_wav"],
            "species_top1": top1_label,
            "species_top1_conf": top1_score,
            "species_bucket": "",
            "topk_labels_json": json.dumps(topk_labels),
            "topk_scores_json": json.dumps(topk_scores),
            "birdnet_version": birdnet_version,
            "inference_ok": inference_ok,
            "error_message": "" if inference_ok else (err or "Unknown BirdNET failure"),
        }
        rows.append(record)
        save_feature_json(
            clips_dir / f"{clip_id}.json",
            {
                "clip_id": clip_id,
                "top1": {
                    "label": top1_label,
                    "confidence": None if np.isnan(top1_score) else top1_score,
                },
                "topk": preds,
                "metadata": {
                    "birdnet_version": birdnet_version,
                    "binary": cfg.binary,
                    "top_k": cfg.top_k,
                    "model_dir": cfg.model_dir.as_posix() if cfg.model_dir else None,
                    "inference_ok": inference_ok,
                    "error_message": "" if inference_ok else (err or "Unknown BirdNET failure"),
                },
            },
        )

    out = pd.DataFrame(rows)
    out.to_csv(master_path, index=False)
    return out


def run_birdnet_window_frontend(
    *,
    recordings_df: pd.DataFrame,
    paths: dict[str, Path],
    cfg: BirdNETConfig,
    sr: int,
    overlap: float,
    merge_consecutive: int,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_path = paths["manifests"] / "events_manifest.csv"
    preds_path = paths["preds"] / "master_predictions.csv"
    meta_path = paths["manifests"] / "events_manifest.meta.json"
    ensure_dir(paths["manifests"])
    ensure_dir(paths["events"])
    ensure_dir(paths["preds"])

    if not force and _birdnet_window_cache_matches(
        manifest_path=manifest_path,
        preds_path=preds_path,
        meta_path=meta_path,
        recordings_df=recordings_df,
        cfg=cfg,
        overlap=overlap,
        merge_consecutive=merge_consecutive,
        sr=sr,
    ):
        print(f"Reusing cached BirdNET window detections from {manifest_path.as_posix()}")
        return pd.read_csv(manifest_path), pd.read_csv(preds_path)

    # This frontend skips our custom segmenter and lets BirdNET scan whole
    # recordings in overlapping windows. It is useful when BirdNET's own windowing
    # works better than our event segmentation for a species or call type.
    ensure_birdnet_available(cfg.binary)
    birdnet_version = get_birdnet_version(cfg.binary)
    audio_paths = [Path(str(v)).resolve() for v in recordings_df["audio_path"].fillna("").astype(str) if str(v)]
    detection_map, batch_err = infer_recordings_with_birdnet(
        audio_paths=audio_paths,
        cfg=cfg,
        overlap=overlap,
        min_conf=0.0,
        merge_consecutive=merge_consecutive,
    )
    if batch_err:
        print(f"WARNING: BirdNET full-recording frontend reported an error: {batch_err}")

    event_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    for _, recording in tqdm(recordings_df.iterrows(), total=len(recordings_df), desc="BirdNET window clips"):
        source_path = Path(str(recording["audio_path"])).resolve()
        windows = detection_map.get(source_path.as_posix(), [])
        if not windows:
            continue
        y = load_audio_mono(source_path, sr=sr)
        total_windows = len(windows)
        stem = slugify(source_path.stem)
        for idx, window in enumerate(windows):
            start_s = float(window.get("start_s", 0.0))
            end_s = float(window.get("end_s", 0.0))
            if end_s <= start_s:
                continue
            start = max(0, int(round(start_s * sr)))
            end = min(len(y), int(round(end_s * sr)))
            if end <= start:
                continue
            clip = y[start:end]
            duration_s = float((end - start) / sr)
            clip_id = f"{stem}_bn_{int(round(start_s * 1000)):08d}_{int(round(end_s * 1000)):08d}"
            event_wav = f"{clip_id}.wav"
            save_wav(paths["events"] / event_wav, clip, sr)

            preds = list(window.get("predictions", []))
            top1_label = preds[0]["label"] if preds else ""
            top1_score = float(preds[0]["score"]) if preds else np.nan
            topk_labels = [str(p["label"]) for p in preds]
            topk_scores = [float(p["score"]) for p in preds]
            segment_rms = float(np.sqrt(np.mean(np.square(clip)) + 1e-12))
            keep_score = float(segment_rms * duration_s)

            event_rows.append(
                {
                    "clip_id": clip_id,
                    "recording_id": str(recording.get("recording_id", "")),
                    "source_file": source_path.as_posix(),
                    "audio_path": source_path.as_posix(),
                    "audio_path_original": str(recording.get("audio_path_original", source_path.as_posix())),
                    "species_label": str(recording.get("species_label", "")),
                    "species_common_name": str(recording.get("species_common_name", "")),
                    "scientific_name": str(recording.get("scientific_name", "")),
                    "species_source": str(recording.get("species_source", "")),
                    "recording_source": str(recording.get("recording_source", "")),
                    "split": str(recording.get("split", "train")),
                    "notes": str(recording.get("notes", "")),
                    "selection_source": "birdnet_windows",
                    "event_wav": event_wav,
                    "start_s": float(start_s),
                    "end_s": float(end_s),
                    "duration_s": duration_s,
                    "segment_rms": segment_rms,
                    "segment_keep_score": keep_score,
                    "selection_duration_weight": 1.0,
                    "selection_quality_score": keep_score,
                    "event_rank_in_recording": int(idx + 1),
                    "n_events_detected_in_recording": int(total_windows),
                    "n_events_kept_in_recording": int(total_windows),
                }
            )
            pred_rows.append(
                {
                    "clip_id": clip_id,
                    "event_wav": event_wav,
                    "species_top1": top1_label,
                    "species_top1_conf": top1_score,
                    "species_bucket": "",
                    "topk_labels_json": json.dumps(topk_labels),
                    "topk_scores_json": json.dumps(topk_scores),
                    "birdnet_version": birdnet_version,
                    "inference_ok": batch_err is None,
                    "error_message": "" if batch_err is None else batch_err,
                }
            )

    events_df = pd.DataFrame(event_rows)
    preds_df = pd.DataFrame(pred_rows)
    events_df.to_csv(manifest_path, index=False)
    preds_df.to_csv(preds_path, index=False)
    meta_path.write_text(
        json.dumps(
            _birdnet_window_cache_meta(
                recordings_df=recordings_df,
                cfg=cfg,
                overlap=overlap,
                merge_consecutive=merge_consecutive,
                sr=sr,
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        f"BirdNET full-recording frontend kept {len(events_df)} windows from "
        f"{events_df['recording_id'].nunique() if not events_df.empty else 0}/{len(recordings_df)} recordings."
    )
    return events_df, preds_df


def apply_species_buckets(
    preds_df: pd.DataFrame,
    conf_thresh: float,
) -> pd.DataFrame:
    # Downstream stages want a simple accepted species label. Anything below the
    # confidence threshold becomes "unknown" so clustering is not trained on weak
    # or noisy species guesses by default.
    out = preds_df.copy()
    top1 = out["species_top1"].fillna("").astype(str)
    conf = pd.to_numeric(out["species_top1_conf"], errors="coerce")
    ok = out["inference_ok"].fillna(False).astype(bool)
    out["species_bucket"] = np.where(ok & (conf >= conf_thresh) & (top1 != ""), top1, "unknown")
    return out


def run_clustering(
    features_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    paths: dict[str, Path],
    sr: int,
    embedding_backend: EmbeddingBackend,
    embedding_fallback_backend: FallbackBackend,
    embedding_mel_frames: int,
    birdnet_embeddings_binary: str,
    min_species_samples: int,
    hdbscan_min_cluster_size: int | None,
    hdbscan_min_samples: int | None,
    use_umap: bool,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    if embedding_mel_frames < 8:
        raise ValueError("--embedding_mel_frames must be >= 8")
    emb_path = paths["embeddings"] / "embeddings.npy"
    # Embeddings are the main representation we cluster on. BirdNET embeddings are
    # the default because they inherit BirdNET's trained bird-audio representation.
    embeddings = generate_embeddings(
        clips_df=features_df,
        events_dir=paths["events"],
        sr=sr,
        out_path=emb_path,
        backend=embedding_backend,
        mel_frames=embedding_mel_frames,
        birdnet_embeddings_binary=birdnet_embeddings_binary,
        fallback_backend=embedding_fallback_backend,
        force=force,
    )
    merged = features_df.merge(
        preds_df[["clip_id", "species_bucket", "species_top1", "species_top1_conf"]],
        on="clip_id",
        how="left",
    )
    merged["species_bucket"] = merged["species_bucket"].fillna("unknown")
    clustered_df, master_clusters, pca_emb, umap_emb = run_dual_mode_clustering(
        features_df=merged,
        embeddings=embeddings,
        embeddings_dir=paths["embeddings"],
        clusters_dir=paths["clusters"],
        min_species_samples=min_species_samples,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_samples=hdbscan_min_samples,
        use_umap=use_umap,
        force=force,
    )
    clustered_path = paths["features"] / "master_features_clustered.csv"
    clustered_df.to_csv(clustered_path, index=False)
    master_clusters.to_csv(paths["clusters"] / "master_clusters.csv", index=False)
    return clustered_df, master_clusters, pca_emb, umap_emb


def run_reporting(
    *,
    clustered_df: pd.DataFrame,
    pca_emb: np.ndarray,
    dirs: dict[str, Path],
) -> Path:
    diversity_col = "recording_id" if "recording_id" in clustered_df.columns else None
    reps = representative_indices_by_cluster(
        clustered_df,
        pca_emb=pca_emb,
        per_cluster=10,
        diversity_col=diversity_col,
        max_per_diversity_value=2,
    )
    dist_plots = create_distribution_plots(clustered_df, out_dir=dirs["reports"])
    cluster_mean_specs = create_cluster_mean_spectrograms(
        clustered_df, specs_dir=dirs["specs"], reports_dir=dirs["reports"]
    )
    umap_image = dirs["embeddings"] / "umap_clusters_global.png"
    write_cluster_playlists(clustered_df, events_dir=dirs["events"], reports_dir=dirs["reports"])
    return generate_html_report(
        df=clustered_df,
        representatives=reps,
        reports_dir=dirs["reports"],
        events_dir=dirs["events"],
        specs_dir=dirs["specs"],
        umap_image=umap_image,
        dist_plots=dist_plots,
        cluster_mean_specs=cluster_mean_specs,
    )


def print_summary(
    recordings_df: pd.DataFrame,
    events_df: pd.DataFrame,
    clustered_df: pd.DataFrame,
    artifact_paths: dict[str, Path],
) -> None:
    if len(events_df) == 0:
        print("No events found.")
        return

    per_recording = events_df.groupby("source_file")["clip_id"].count().sort_values(ascending=False)
    noise_frac = float(np.mean(clustered_df["cluster_id"] == -1)) if len(clustered_df) else 0.0

    print("\nPipeline summary")
    print(f"- Recordings used: {len(recordings_df)}")
    print(f"- Total events: {len(events_df)}")
    print(f"- Avg duration (s): {events_df['duration_s'].mean():.3f}")
    print(f"- Noise fraction (cluster_id == -1): {noise_frac:.3f}")
    if "species_label" in recordings_df:
        species_counts = recordings_df["species_label"].fillna("").astype(str)
        species_counts = species_counts[species_counts != ""].value_counts().sort_index()
        if not species_counts.empty:
            print("- Training species counts:")
            for species, count in species_counts.items():
                print(f"  - {species}: {count}")
    print("- Events per recording:")
    for source, count in per_recording.items():
        print(f"  - {source}: {count}")

    print("\nCheckpoints")
    print(f"- Recordings snapshot: {artifact_paths['recordings_snapshot'].as_posix()}")
    print(f"- Cluster summary: {artifact_paths['cluster_summary'].as_posix()}")
    print(f"- Training run summary: {artifact_paths['training_summary'].as_posix()}")


def print_species_check_summary(
    recordings_df: pd.DataFrame,
    events_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    artifact_paths: dict[str, Path],
) -> None:
    print("\nSpecies-check summary")
    print(f"- Recordings used: {len(recordings_df)}")
    print(f"- Total events: {len(events_df)}")
    print(f"- Predictions generated: {len(preds_df)}")
    if not preds_df.empty:
        top_counts = preds_df["species_top1"].fillna("").astype(str)
        top_counts = top_counts[top_counts != ""].value_counts().head(5)
        if not top_counts.empty:
            print("- Top predicted species:")
            for label, count in top_counts.items():
                print(f"  - {label}: {count}")
    print("\nCheckpoints")
    print(f"- Recordings snapshot: {artifact_paths['recordings_snapshot'].as_posix()}")
    print(f"- Species prediction summary: {artifact_paths['species_summary'].as_posix()}")
    print(f"- Species run summary: {artifact_paths['species_run_summary'].as_posix()}")


def run_training_pipeline(
    *,
    args: argparse.Namespace,
    dirs: dict[str, Path],
    recordings_df: pd.DataFrame,
) -> dict[str, object]:
    birdnet_cfg = BirdNETConfig(
        binary=args.birdnet_binary,
        top_k=args.top_k,
        model_dir=args.birdnet_model_dir,
    )

    # The repo currently supports two species frontends:
    # 1. custom event segmentation followed by BirdNET on the resulting clips
    # 2. BirdNET scanning the full recording in overlapping windows first
    if args.species_frontend == "birdnet_windows":
        events_df, preds_df = run_birdnet_window_frontend(
            recordings_df=recordings_df,
            paths=dirs,
            cfg=birdnet_cfg,
            sr=args.sr,
            overlap=args.birdnet_window_overlap,
            merge_consecutive=args.birdnet_window_merge_consecutive,
            force=args.force,
        )
        if len(events_df) == 0:
            print("No BirdNET windows were generated from the full recordings.")
            return {"events_df": events_df}
    else:
        cfg = SegmentConfig(
            sr=args.sr,
            min_dur=args.min_dur,
            max_dur=args.max_dur,
            seg_threshold_db=args.seg_threshold_db,
            merge_gap=args.merge_gap,
            birdiness_min_hz=args.birdiness_min_hz,
            birdiness_ratio_min=args.birdiness_ratio_min,
            rms_min=args.seg_rms_min,
        )

        events_df = run_segmentation(
            recordings_df=recordings_df,
            paths=dirs,
            cfg=cfg,
            max_events_per_recording=args.max_events_per_recording,
            selection_pref_duration=args.selection_pref_duration,
            selection_score_min=args.selection_score_min,
            selection_relative_min=args.selection_relative_min,
            force=args.force,
        )
        if len(events_df) == 0:
            print("No events detected. Try lowering --seg_threshold_db or birdiness constraints in segment.py.")
            return {"events_df": events_df}

        preds_df = run_birdnet_predictions(
            events_df=events_df,
            paths=dirs,
            cfg=birdnet_cfg,
            birdnet_context_seconds=args.birdnet_context_seconds,
            force=args.force,
        )

    preds_df = apply_species_buckets(preds_df=preds_df, conf_thresh=args.conf_thresh)
    preds_df.to_csv(dirs["preds"] / "master_predictions.csv", index=False)
    ok_count = int(preds_df["inference_ok"].sum()) if "inference_ok" in preds_df else 0
    print(f"BirdNET predictions: {ok_count}/{len(preds_df)} clips successful")

    # Species-check mode is the place to stop when we only want to verify that the
    # BirdNET species stage is behaving sensibly before doing embeddings/clusters.
    if args.stop_after_species:
        artifact_paths = export_species_check_artifacts(
            artifacts_dir=dirs["artifacts"],
            recordings_df=recordings_df,
            events_df=events_df,
            preds_df=preds_df,
            species_config_path=args.species_config if args.species_config.exists() else None,
            train_manifest_path=args.train_manifest if args.train_manifest.exists() else None,
            val_manifest_path=args.val_manifest if args.val_manifest.exists() else None,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )
        print_species_check_summary(
            recordings_df=recordings_df,
            events_df=events_df,
            preds_df=preds_df,
            artifact_paths=artifact_paths,
        )
        print(f"Predictions CSV: {(dirs['preds'] / 'master_predictions.csv').as_posix()}")
        return {
            "recordings_df": recordings_df,
            "events_df": events_df,
            "preds_df": preds_df,
            "artifact_paths": artifact_paths,
        }

    # The full offline batch path continues into interpretable features, learned
    # embeddings, clustering, and an HTML/listening report for review.
    features_df = run_features(events_df=events_df, paths=dirs, sr=args.sr, force=args.force)
    clustered_df, master_clusters, pca_emb, _ = run_clustering(
        features_df=features_df,
        preds_df=preds_df,
        paths=dirs,
        sr=args.sr,
        embedding_backend=args.embedding_backend,
        embedding_fallback_backend=args.embedding_fallback_backend,
        embedding_mel_frames=args.embedding_mel_frames,
        birdnet_embeddings_binary=args.birdnet_embeddings_binary,
        min_species_samples=args.min_species_samples,
        hdbscan_min_cluster_size=(None if args.hdbscan_min_cluster_size <= 0 else args.hdbscan_min_cluster_size),
        hdbscan_min_samples=(None if args.hdbscan_min_samples <= 0 else args.hdbscan_min_samples),
        use_umap=args.use_umap,
        force=args.force,
    )
    report_path = run_reporting(clustered_df=clustered_df, pca_emb=pca_emb, dirs=dirs)
    artifact_paths = export_training_artifacts(
        artifacts_dir=dirs["artifacts"],
        recordings_df=recordings_df,
        events_df=events_df,
        preds_df=preds_df,
        clustered_df=clustered_df,
        master_clusters=master_clusters,
        species_config_path=args.species_config if args.species_config.exists() else None,
        train_manifest_path=args.train_manifest if args.train_manifest.exists() else None,
        val_manifest_path=args.val_manifest if args.val_manifest.exists() else None,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        report_path=report_path,
        embedding_meta_path=dirs["embeddings"] / "embedding_meta.json",
    )

    print_summary(
        recordings_df=recordings_df,
        events_df=events_df,
        clustered_df=clustered_df,
        artifact_paths=artifact_paths,
    )
    print(f"Predictions CSV: {(dirs['preds'] / 'master_predictions.csv').as_posix()}")
    print(f"Clusters CSV: {(dirs['clusters'] / 'master_clusters.csv').as_posix()}")
    print(f"Master CSV: {(dirs['features'] / 'master_features_clustered.csv').as_posix()}")
    print(f"Report: {report_path.as_posix()}")

    return {
        "recordings_df": recordings_df,
        "events_df": events_df,
        "preds_df": preds_df,
        "clustered_df": clustered_df,
        "master_clusters": master_clusters,
        "report_path": report_path,
        "artifact_paths": artifact_paths,
    }


def main() -> None:
    args = parse_args()
    args.output_dir = require_google_drive_output(args.output_dir, purpose="--output_dir")
    dirs = _paths(args.output_dir)
    for p in dirs.values():
        ensure_dir(p)

    recordings_df, recording_source = load_training_recordings(
        input_dir=args.input_dir,
        train_manifest_path=args.train_manifest,
        species_config_path=args.species_config,
    )
    print(f"Training source: {recording_source}")
    print(f"Recordings queued: {len(recordings_df)}")
    if "species_label" in recordings_df:
        species_counts = recordings_df["species_label"].fillna("").astype(str)
        species_counts = species_counts[species_counts != ""].value_counts().sort_index()
        if not species_counts.empty:
            print("Manifest species counts:")
            for species, count in species_counts.items():
                print(f"  - {species}: {count}")

    run_training_pipeline(args=args, dirs=dirs, recordings_df=recordings_df)


if __name__ == "__main__":
    main()
