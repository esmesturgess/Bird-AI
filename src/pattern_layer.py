from __future__ import annotations

"""Build within-species pattern-layer artifacts from a completed species run.

This module sits between:

1. a BirdNET species run
2. downstream within-species clustering / semantic labeling

Its two main jobs are:

- decide which species detections count as accepted units for one target species
- optionally merge nearby accepted events into phrase-level pooled units
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from .embed import generate_embeddings
from .utils import (
    ensure_dir,
    load_audio_mono,
    normalize_audio,
    require_google_drive_output,
    save_json,
    save_wav,
    slugify,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare accepted-events pattern-layer artifacts from a species run.")
    parser.add_argument("--species_run_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--target_species_bucket", type=str, required=True)
    parser.add_argument(
        "--accept_target_in_topk",
        action="store_true",
        help=(
            "Also accept an event if the target species appears anywhere in BirdNET top-k, "
            "even when it was not assigned as species_bucket."
        ),
    )
    parser.add_argument(
        "--target_species_topk_min_score",
        type=float,
        default=0.0,
        help=(
            "Minimum BirdNET top-k score required for target-species fallback acceptance. "
            "Only used with --accept_target_in_topk."
        ),
    )
    parser.add_argument(
        "--analysis_unit",
        type=str,
        choices=["event", "phrase_pool"],
        default="event",
        help="Keep event-level embeddings or pool accepted nearby events into phrase-level embeddings.",
    )
    parser.add_argument(
        "--embedding_clip_source",
        type=str,
        choices=["context", "event"],
        default="context",
        help="Use BirdNET context clips from preds/clips or raw event clips from events/ for embeddings.",
    )
    parser.add_argument(
        "--embedding_backend",
        type=str,
        choices=["birdnet", "mel_time", "mel_stats", "avex_effnet_bio"],
        default="birdnet",
    )
    parser.add_argument("--birdnet_embeddings_binary", type=str, default="birdnet-embeddings")
    parser.add_argument("--sr", type=int, default=32000)
    parser.add_argument("--embedding_mel_frames", type=int, default=96)
    parser.add_argument("--phrase_gap_seconds", type=float, default=0.8)
    parser.add_argument("--phrase_max_duration_seconds", type=float, default=6.0)
    parser.add_argument("--phrase_padding_seconds", type=float, default=0.4)
    parser.add_argument("--phrase_min_events", type=int, default=1)
    parser.add_argument(
        "--phrase_pooling",
        type=str,
        choices=["uniform", "duration", "confidence", "duration_confidence"],
        default="duration_confidence",
    )
    parser.add_argument(
        "--embedding_fallback_backend",
        type=str,
        choices=["none", "mel_time", "mel_stats"],
        default="mel_time",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_species_run(species_run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    events_path = species_run_dir / "manifests" / "events_manifest.csv"
    preds_path = species_run_dir / "preds" / "master_predictions.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"Events manifest not found: {events_path.as_posix()}")
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {preds_path.as_posix()}")
    return pd.read_csv(events_path), pd.read_csv(preds_path)


def _embedding_source_dir(species_run_dir: Path, clip_source: str) -> tuple[Path, str]:
    if clip_source == "context":
        return species_run_dir / "preds" / "clips", "birdnet_context_clip"
    return species_run_dir / "events", "event_clip"


def _compute_target_species_fallback(
    merged: pd.DataFrame,
    *,
    target_species_bucket: str,
    ok_mask: pd.Series,
    accept_target_in_topk: bool,
    target_species_topk_min_score: float,
) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    fallback_mask = pd.Series(False, index=merged.index)
    fallback_scores = np.full(len(merged), np.nan, dtype=np.float32)
    fallback_ranks = np.full(len(merged), -1, dtype=np.int32)
    if not accept_target_in_topk:
        return fallback_mask, fallback_scores, fallback_ranks

    # This softer acceptance path is mainly for cases like Wren where BirdNET may
    # mention the target species in top-k even when another label wins top-1.
    for idx, row in merged.iterrows():
        if not bool(ok_mask.iloc[idx]):
            continue
        try:
            labels = json.loads(str(row.get("topk_labels_json", "[]")))
            scores = json.loads(str(row.get("topk_scores_json", "[]")))
        except Exception:
            continue
        for rank, (label, score) in enumerate(zip(labels, scores), start=1):
            if str(label) != target_species_bucket:
                continue
            score_f = float(score)
            fallback_scores[idx] = score_f
            fallback_ranks[idx] = int(rank)
            if score_f >= float(target_species_topk_min_score):
                fallback_mask.iloc[idx] = True
            break
    return fallback_mask, fallback_scores, fallback_ranks


def _set_embedding_audio_paths(
    accepted: pd.DataFrame,
    *,
    species_run_dir: Path,
    embedding_clip_source: str,
) -> tuple[pd.DataFrame, Path]:
    embedding_dir, clip_kind = _embedding_source_dir(species_run_dir, embedding_clip_source)
    accepted = accepted.copy()
    accepted["embedding_clip_kind"] = clip_kind
    accepted["embedding_audio_path"] = accepted["event_wav"].map(
        lambda name: (embedding_dir / str(name)).resolve().as_posix()
    )
    accepted["pattern_species_slug"] = accepted["species_bucket"].fillna("").map(slugify)

    missing_embedding = [
        path for path in accepted["embedding_audio_path"].astype(str).tolist() if not Path(path).exists()
    ]
    if missing_embedding:
        preview = "\n".join(missing_embedding[:10])
        raise FileNotFoundError(
            f"Selected embedding clips are missing under {embedding_dir.as_posix()}:\n{preview}"
        )
    return accepted, embedding_dir


def _write_accepted_events_outputs(
    *,
    accepted: pd.DataFrame,
    species_run_dir: Path,
    manifests_dir: Path,
    target_species_bucket: str,
    embedding_clip_source: str,
    embedding_dir: Path,
    accept_target_in_topk: bool,
    target_species_topk_min_score: float,
) -> None:
    accepted.to_csv(manifests_dir / "accepted_events_manifest.csv", index=False)
    summary = {
        "species_run_dir": species_run_dir.resolve().as_posix(),
        "target_species_bucket": target_species_bucket,
        "embedding_clip_source": embedding_clip_source,
        "accept_target_in_topk": bool(accept_target_in_topk),
        "target_species_topk_min_score": float(target_species_topk_min_score),
        "embedding_dir": embedding_dir.resolve().as_posix(),
        "accepted_events": int(len(accepted)),
        "source_recordings_with_accepted_events": int(accepted["recording_id"].nunique()) if not accepted.empty else 0,
        "mean_species_confidence": float(accepted["species_top1_conf"].mean()) if not accepted.empty else 0.0,
        "mean_duration_s": float(accepted["duration_s"].mean()) if not accepted.empty else 0.0,
    }
    save_json(manifests_dir / "accepted_events_summary.json", summary)


def build_accepted_events_manifest(
    *,
    species_run_dir: Path,
    output_dir: Path,
    target_species_bucket: str,
    embedding_clip_source: str,
    accept_target_in_topk: bool,
    target_species_topk_min_score: float,
    force: bool,
) -> pd.DataFrame:
    manifests_dir = output_dir / "manifests"
    ensure_dir(manifests_dir)
    out_path = manifests_dir / "accepted_events_manifest.csv"
    if out_path.exists() and not force:
        return pd.read_csv(out_path)

    events_df, preds_df = _load_species_run(species_run_dir)
    merged = events_df.merge(preds_df, on=["clip_id", "event_wav"], how="left", validate="one_to_one")
    ok_mask = merged["inference_ok"].fillna(False).astype(bool)
    direct_mask = merged["species_bucket"].fillna("").eq(target_species_bucket) & ok_mask
    fallback_mask, fallback_scores, fallback_ranks = _compute_target_species_fallback(
        merged,
        target_species_bucket=target_species_bucket,
        ok_mask=ok_mask,
        accept_target_in_topk=accept_target_in_topk,
        target_species_topk_min_score=target_species_topk_min_score,
    )

    # Accepted units are the inputs to the within-species pattern layer. Everything
    # outside this mask is deliberately excluded from downstream clustering.
    accepted_mask = direct_mask | fallback_mask
    accepted = merged[accepted_mask].copy()
    accepted = accepted.sort_values(["source_file", "start_s", "clip_id"]).reset_index(drop=True)
    accepted["accepted_target_match_mode"] = np.where(
        accepted["species_bucket"].fillna("").eq(target_species_bucket),
        "species_bucket_top1",
        "target_in_topk_fallback",
    )
    accepted["target_species_topk_score"] = accepted["clip_id"].map(
        dict(zip(merged["clip_id"].astype(str), fallback_scores, strict=False))
    )
    accepted["target_species_topk_rank"] = accepted["clip_id"].map(
        dict(zip(merged["clip_id"].astype(str), fallback_ranks, strict=False))
    )
    accepted, embedding_dir = _set_embedding_audio_paths(
        accepted,
        species_run_dir=species_run_dir,
        embedding_clip_source=embedding_clip_source,
    )
    accepted["pattern_species_slug"] = slugify(target_species_bucket)
    _write_accepted_events_outputs(
        accepted=accepted,
        species_run_dir=species_run_dir,
        manifests_dir=manifests_dir,
        target_species_bucket=target_species_bucket,
        embedding_clip_source=embedding_clip_source,
        embedding_dir=embedding_dir,
        accept_target_in_topk=accept_target_in_topk,
        target_species_topk_min_score=target_species_topk_min_score,
    )
    return accepted


def generate_pattern_embeddings(
    *,
    accepted_df: pd.DataFrame,
    species_run_dir: Path,
    output_dir: Path,
    embedding_clip_source: str,
    embedding_backend: str,
    birdnet_embeddings_binary: str,
    sr: int,
    mel_frames: int,
    fallback_backend: str,
    force: bool,
    out_name: str,
) -> Path:
    embeddings_dir = output_dir / "embeddings"
    ensure_dir(embeddings_dir)
    out_path = embeddings_dir / out_name
    events_dir, clip_kind = _embedding_source_dir(species_run_dir, embedding_clip_source)
    _ = clip_kind
    generate_embeddings(
        clips_df=accepted_df,
        events_dir=events_dir,
        sr=sr,
        out_path=out_path,
        backend=embedding_backend,
        mel_frames=mel_frames,
        birdnet_embeddings_binary=birdnet_embeddings_binary,
        fallback_backend=fallback_backend,
        force=force,
    )
    return out_path


def _group_phrase_indices(
    accepted_df: pd.DataFrame,
    *,
    phrase_gap_seconds: float,
    phrase_max_duration_seconds: float,
    phrase_min_events: int,
) -> list[list[int]]:
    groups: list[list[int]] = []
    if accepted_df.empty:
        return groups

    sorted_df = accepted_df.sort_values(["recording_id", "start_s", "end_s", "clip_id"]).reset_index()
    current_indices: list[int] = []
    current_recording = ""
    current_phrase_start = 0.0
    current_phrase_end = 0.0

    for _, row in sorted_df.iterrows():
        original_idx = int(row["index"])
        recording_id = str(row["recording_id"])
        start_s = float(row["start_s"])
        end_s = float(row["end_s"])

        if not current_indices:
            current_indices = [original_idx]
            current_recording = recording_id
            current_phrase_start = start_s
            current_phrase_end = end_s
            continue

        gap_s = start_s - current_phrase_end
        candidate_duration_s = max(current_phrase_end, end_s) - current_phrase_start
        # Nearby accepted events from the same recording can be pooled into one
        # phrase unit so we do not treat tiny shifted slices of the same sound as
        # independent training examples.
        same_phrase = (
            recording_id == current_recording
            and gap_s <= float(phrase_gap_seconds)
            and candidate_duration_s <= float(phrase_max_duration_seconds)
        )
        if same_phrase:
            current_indices.append(original_idx)
            current_phrase_end = max(current_phrase_end, end_s)
            continue

        if len(current_indices) >= int(phrase_min_events):
            groups.append(current_indices)
        current_indices = [original_idx]
        current_recording = recording_id
        current_phrase_start = start_s
        current_phrase_end = end_s

    if current_indices and len(current_indices) >= int(phrase_min_events):
        groups.append(current_indices)
    return groups


def _phrase_weights(phrase_events: pd.DataFrame, mode: str) -> np.ndarray:
    if phrase_events.empty:
        return np.zeros((0,), dtype=np.float32)
    if mode == "uniform":
        weights = np.ones(len(phrase_events), dtype=np.float32)
    elif mode == "duration":
        weights = phrase_events["duration_s"].fillna(0.0).astype(float).to_numpy(dtype=np.float32)
    elif mode == "confidence":
        weights = phrase_events["species_top1_conf"].fillna(0.0).astype(float).to_numpy(dtype=np.float32)
    else:
        duration = phrase_events["duration_s"].fillna(0.0).astype(float).to_numpy(dtype=np.float32)
        conf = phrase_events["species_top1_conf"].fillna(0.0).astype(float).to_numpy(dtype=np.float32)
        weights = duration * np.clip(conf, 1e-3, None)
    total = float(weights.sum())
    if total <= 0:
        return np.full(len(phrase_events), 1.0 / max(1, len(phrase_events)), dtype=np.float32)
    return (weights / total).astype(np.float32)


def _reconstruct_phrase_clip_from_events(
    phrase_events: pd.DataFrame,
    *,
    sr: int,
    event_start_s: float,
    event_end_s: float,
) -> np.ndarray:
    target_samples = max(1, int(round((event_end_s - event_start_s) * sr)))
    acc = np.zeros(target_samples, dtype=np.float32)
    weights = np.zeros(target_samples, dtype=np.float32)
    missing_paths: list[str] = []

    for _, event in phrase_events.iterrows():
        event_path = Path(str(event["embedding_audio_path"]))
        if not event_path.exists():
            missing_paths.append(event_path.as_posix())
            continue
        y, _ = librosa.load(event_path.as_posix(), sr=sr, mono=True)
        y = y.astype(np.float32, copy=False)
        offset = int(round((float(event["start_s"]) - event_start_s) * sr))
        dst_start = max(0, offset)
        src_start = max(0, -offset)
        n = min(y.size - src_start, target_samples - dst_start)
        if n <= 0:
            continue
        acc[dst_start : dst_start + n] += y[src_start : src_start + n]
        weights[dst_start : dst_start + n] += 1.0

    used = weights > 0
    if not np.any(used):
        preview = "\n".join(missing_paths[:10])
        raise FileNotFoundError(
            "Could not reconstruct phrase clip because no source audio or event clips were available."
            + (f"\nMissing event clips:\n{preview}" if preview else "")
        )
    acc[used] = acc[used] / np.maximum(weights[used], 1.0)
    return normalize_audio(acc)


def _extract_phrase_clip(
    phrase_events: pd.DataFrame,
    *,
    sr: int,
    phrase_padding_seconds: float,
) -> tuple[np.ndarray, float, float, str]:
    event_start_s = float(phrase_events["start_s"].min())
    event_end_s = float(phrase_events["end_s"].max())
    source_path = Path(str(phrase_events.iloc[0]["source_file"]))

    if source_path.exists():
        y = load_audio_mono(source_path, sr=sr)
        clip_start_s = max(0.0, event_start_s - float(phrase_padding_seconds))
        clip_end_s = min(len(y) / sr, event_end_s + float(phrase_padding_seconds))
        if clip_end_s <= clip_start_s:
            clip_end_s = min(len(y) / sr, clip_start_s + max(0.25, event_end_s - event_start_s))
        start_idx = int(round(clip_start_s * sr))
        end_idx = int(round(clip_end_s * sr))
        return y[start_idx:end_idx], clip_start_s, clip_end_s, "source_file"

    clip = _reconstruct_phrase_clip_from_events(
        phrase_events,
        sr=sr,
        event_start_s=event_start_s,
        event_end_s=event_end_s,
    )
    return clip, event_start_s, event_end_s, "reconstructed_from_event_clips"


def build_phrase_pooled_artifacts(
    *,
    accepted_df: pd.DataFrame,
    event_embeddings: np.ndarray,
    output_dir: Path,
    sr: int,
    phrase_gap_seconds: float,
    phrase_max_duration_seconds: float,
    phrase_padding_seconds: float,
    phrase_min_events: int,
    phrase_pooling: str,
    force: bool,
) -> tuple[pd.DataFrame, Path]:
    manifests_dir = output_dir / "manifests"
    embeddings_dir = output_dir / "embeddings"
    phrases_dir = output_dir / "phrases"
    ensure_dir(manifests_dir)
    ensure_dir(embeddings_dir)
    ensure_dir(phrases_dir)

    manifest_path = manifests_dir / "accepted_events_manifest.csv"
    summary_path = manifests_dir / "accepted_events_summary.json"
    out_path = embeddings_dir / "accepted_embeddings.npy"
    meta_path = embeddings_dir / "embedding_meta.json"
    index_path = embeddings_dir / "embedding_index.csv"

    if (
        manifest_path.exists()
        and summary_path.exists()
        and out_path.exists()
        and meta_path.exists()
        and index_path.exists()
        and not force
    ):
        return pd.read_csv(manifest_path), out_path

    if len(accepted_df) != int(event_embeddings.shape[0]):
        raise ValueError(
            f"Accepted event rows ({len(accepted_df)}) do not match event embeddings ({event_embeddings.shape[0]})."
        )

    accepted_df = accepted_df.copy().reset_index(drop=True)
    accepted_df["event_embedding_row"] = np.arange(len(accepted_df))
    phrase_groups = _group_phrase_indices(
        accepted_df,
        phrase_gap_seconds=phrase_gap_seconds,
        phrase_max_duration_seconds=phrase_max_duration_seconds,
        phrase_min_events=phrase_min_events,
    )

    rows: list[dict[str, object]] = []
    pooled_embeddings: list[np.ndarray] = []
    for phrase_group in phrase_groups:
        phrase_events = accepted_df.iloc[phrase_group].copy().sort_values(["start_s", "end_s", "clip_id"]).reset_index(drop=True)
        first = phrase_events.iloc[0]
        event_start_s = float(phrase_events["start_s"].min())
        event_end_s = float(phrase_events["end_s"].max())
        clip, clip_start_s, clip_end_s, phrase_audio_source = _extract_phrase_clip(
            phrase_events,
            sr=sr,
            phrase_padding_seconds=phrase_padding_seconds,
        )

        phrase_idx_in_recording = len([row for row in rows if row["recording_id"] == str(first["recording_id"])])
        source_path = Path(str(first["source_file"]))
        clip_id = f"{slugify(source_path.stem)}_phrase_{phrase_idx_in_recording:05d}"
        event_wav = f"{clip_id}.wav"
        clip_path = phrases_dir / event_wav
        save_wav(clip_path, clip, sr)

        weights = _phrase_weights(phrase_events, phrase_pooling)
        # A phrase embedding is a weighted average of the source event embeddings.
        # This keeps BirdNET as the frozen encoder while giving us a more stable
        # phrase-level unit for clustering.
        pooled = np.average(
            event_embeddings[phrase_events["event_embedding_row"].to_numpy(dtype=int)],
            axis=0,
            weights=weights,
        ).astype(np.float32)
        pooled_embeddings.append(pooled)

        mean_conf = float(np.average(phrase_events["species_top1_conf"].fillna(0.0).astype(float), weights=weights))
        row = first.to_dict()
        row.update(
            {
                "analysis_unit": "phrase_pool",
                "clip_id": clip_id,
                "event_wav": event_wav,
                "embedding_audio_path": clip_path.resolve().as_posix(),
                "embedding_clip_kind": "phrase_pool_clip",
                "pooled_from_n_events": int(len(phrase_events)),
                "pooled_event_ids_json": json.dumps(phrase_events["clip_id"].astype(str).tolist()),
                "pooled_event_wavs_json": json.dumps(phrase_events["event_wav"].astype(str).tolist()),
                "pooled_weights_json": json.dumps([float(v) for v in weights.tolist()]),
                "pooled_source_event_embedding_rows_json": json.dumps(
                    [int(v) for v in phrase_events["event_embedding_row"].tolist()]
                ),
                "phrase_event_start_s": event_start_s,
                "phrase_event_end_s": event_end_s,
                "phrase_event_duration_s": float(event_end_s - event_start_s),
                "phrase_audio_source": phrase_audio_source,
                "start_s": clip_start_s,
                "end_s": clip_end_s,
                "duration_s": float(clip_end_s - clip_start_s),
                "species_top1_conf": mean_conf,
                "segment_rms": float(np.sqrt(np.mean(np.square(clip)) + 1e-12)) if clip.size else 0.0,
            }
        )
        rows.append(row)

    phrase_df = pd.DataFrame(rows)
    if phrase_df.empty:
        raise RuntimeError("Phrase pooling produced zero accepted phrase clips.")
    phrase_df = phrase_df.sort_values(["source_file", "start_s", "clip_id"]).reset_index(drop=True)
    phrase_df.to_csv(manifest_path, index=False)

    pooled_array = np.vstack(pooled_embeddings).astype(np.float32)
    np.save(out_path, pooled_array)
    pd.DataFrame({"clip_id": phrase_df["clip_id"].to_numpy(), "embedding_row": np.arange(len(phrase_df))}).to_csv(
        index_path, index=False
    )

    event_conf = accepted_df["species_top1_conf"].fillna(0.0).astype(float)
    event_embedding_dir = ""
    if not accepted_df.empty:
        event_embedding_dir = str(Path(str(accepted_df["embedding_audio_path"].iloc[0])).parent.resolve().as_posix())

    summary = {
        "analysis_unit": "phrase_pool",
        "accepted_events": int(len(phrase_df)),
        "source_events_used": int(len(accepted_df)),
        "source_recordings_with_accepted_events": int(phrase_df["recording_id"].nunique()),
        "mean_species_confidence": float(phrase_df["species_top1_conf"].mean()),
        "mean_duration_s": float(phrase_df["duration_s"].mean()),
        "mean_pooled_events_per_phrase": float(phrase_df["pooled_from_n_events"].mean()),
        "phrase_gap_seconds": float(phrase_gap_seconds),
        "phrase_max_duration_seconds": float(phrase_max_duration_seconds),
        "phrase_padding_seconds": float(phrase_padding_seconds),
        "phrase_min_events": int(phrase_min_events),
        "phrase_pooling": phrase_pooling,
        "embedding_dir": phrases_dir.resolve().as_posix(),
        "event_embedding_dir": event_embedding_dir,
        "event_mean_species_confidence": float(event_conf.mean()) if not accepted_df.empty else 0.0,
    }
    save_json(summary_path, summary)

    meta_payload = {
        "analysis_unit": "phrase_pool",
        "backend": "birdnet_pooled",
        "requested_backend": "birdnet_pooled",
        "embedding_dim": int(pooled_array.shape[1]) if pooled_array.ndim == 2 else 0,
        "n_clips": int(len(phrase_df)),
        "source_event_count": int(len(accepted_df)),
        "phrase_gap_seconds": float(phrase_gap_seconds),
        "phrase_max_duration_seconds": float(phrase_max_duration_seconds),
        "phrase_padding_seconds": float(phrase_padding_seconds),
        "phrase_min_events": int(phrase_min_events),
        "phrase_pooling": phrase_pooling,
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2, sort_keys=True), encoding="utf-8")
    return phrase_df, out_path


def main() -> None:
    args = parse_args()
    species_run_dir = args.species_run_dir.resolve()
    output_dir = require_google_drive_output(args.output_dir, purpose="--output_dir")
    ensure_dir(output_dir)

    accepted_df = build_accepted_events_manifest(
        species_run_dir=species_run_dir,
        output_dir=output_dir,
        target_species_bucket=args.target_species_bucket,
        embedding_clip_source=args.embedding_clip_source,
        accept_target_in_topk=bool(args.accept_target_in_topk),
        target_species_topk_min_score=float(args.target_species_topk_min_score),
        force=args.force,
    )
    if accepted_df.empty:
        raise RuntimeError(
            f"No accepted events found for species_bucket='{args.target_species_bucket}' in {species_run_dir.as_posix()}."
        )

    if args.analysis_unit == "event":
        embeddings_path = generate_pattern_embeddings(
            accepted_df=accepted_df,
            species_run_dir=species_run_dir,
            output_dir=output_dir,
            embedding_clip_source=args.embedding_clip_source,
            embedding_backend=args.embedding_backend,
            birdnet_embeddings_binary=args.birdnet_embeddings_binary,
            sr=args.sr,
            mel_frames=args.embedding_mel_frames,
            fallback_backend=args.embedding_fallback_backend,
            force=args.force,
            out_name="accepted_embeddings.npy",
        )
        accepted_units = accepted_df
    else:
        source_event_embeddings_path = generate_pattern_embeddings(
            accepted_df=accepted_df,
            species_run_dir=species_run_dir,
            output_dir=output_dir,
            embedding_clip_source=args.embedding_clip_source,
            embedding_backend=args.embedding_backend,
            birdnet_embeddings_binary=args.birdnet_embeddings_binary,
            sr=args.sr,
            mel_frames=args.embedding_mel_frames,
            fallback_backend=args.embedding_fallback_backend,
            force=args.force,
            out_name="accepted_event_embeddings.npy",
        )
        source_event_embeddings = np.load(source_event_embeddings_path)
        accepted_units, embeddings_path = build_phrase_pooled_artifacts(
            accepted_df=accepted_df,
            event_embeddings=source_event_embeddings,
            output_dir=output_dir,
            sr=args.sr,
            phrase_gap_seconds=args.phrase_gap_seconds,
            phrase_max_duration_seconds=args.phrase_max_duration_seconds,
            phrase_padding_seconds=args.phrase_padding_seconds,
            phrase_min_events=args.phrase_min_events,
            phrase_pooling=args.phrase_pooling,
            force=args.force,
        )

    print("Pattern-layer preparation complete")
    print(f"- Species run dir: {species_run_dir.as_posix()}")
    print(f"- Target species bucket: {args.target_species_bucket}")
    print(f"- Analysis unit: {args.analysis_unit}")
    print(f"- Accepted units: {len(accepted_units)}")
    print(f"- Source recordings with accepted units: {accepted_units['recording_id'].nunique()}")
    print(f"- Accepted events manifest: {(output_dir / 'manifests' / 'accepted_events_manifest.csv').as_posix()}")
    print(f"- Accepted events summary: {(output_dir / 'manifests' / 'accepted_events_summary.json').as_posix()}")
    print(f"- Embeddings array: {embeddings_path.as_posix()}")
    print(f"- Embedding meta: {(output_dir / 'embeddings' / 'embedding_meta.json').as_posix()}")
    print(f"- Embedding index: {(output_dir / 'embeddings' / 'embedding_index.csv').as_posix()}")


if __name__ == "__main__":
    main()
