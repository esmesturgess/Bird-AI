from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import ensure_dir


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _string_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df or df.empty:
        return {}
    series = df[column].fillna("").astype(str).str.strip()
    series = series[series != ""]
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


def build_cluster_training_summary(
    clustered_df: pd.DataFrame,
    master_clusters: pd.DataFrame,
) -> pd.DataFrame:
    if master_clusters.empty:
        return pd.DataFrame(
            columns=[
                "mode",
                "species_bucket",
                "cluster_id",
                "n_events",
                "n_recordings",
                "mean_cluster_prob",
                "mean_species_top1_conf",
                "mean_duration_s",
                "mean_urgency_index",
                "mean_song_likeness",
                "recording_sources",
                "splits",
            ]
        )

    merge_cols = ["clip_id", "source_file", "recording_source", "split", "species_top1_conf", "duration_s"]
    for optional in ("urgency_index", "song_likeness"):
        if optional in clustered_df.columns:
            merge_cols.append(optional)
    merged = master_clusters.merge(
        clustered_df[merge_cols].drop_duplicates("clip_id"),
        on="clip_id",
        how="left",
    )

    rows: list[dict[str, Any]] = []
    for (mode, species_bucket, cluster_id), group in merged.groupby(["mode", "species_bucket", "cluster_id"], dropna=False):
        recording_sources = sorted(
            {v for v in group.get("recording_source", pd.Series(dtype=str)).fillna("").astype(str) if v}
        )
        splits = sorted({v for v in group.get("split", pd.Series(dtype=str)).fillna("").astype(str) if v})
        rows.append(
            {
                "mode": str(mode),
                "species_bucket": str(species_bucket),
                "cluster_id": int(cluster_id),
                "n_events": int(len(group)),
                "n_recordings": int(group["source_file"].fillna("").astype(str).replace("", np.nan).nunique()),
                "mean_cluster_prob": float(pd.to_numeric(group["cluster_prob"], errors="coerce").fillna(0.0).mean()),
                "mean_species_top1_conf": float(
                    pd.to_numeric(group.get("species_top1_conf"), errors="coerce").fillna(0.0).mean()
                ),
                "mean_duration_s": float(pd.to_numeric(group.get("duration_s"), errors="coerce").fillna(0.0).mean()),
                "mean_urgency_index": float(
                    pd.to_numeric(group.get("urgency_index"), errors="coerce").fillna(0.0).mean()
                ),
                "mean_song_likeness": float(
                    pd.to_numeric(group.get("song_likeness"), errors="coerce").fillna(0.0).mean()
                ),
                "recording_sources": json.dumps(recording_sources),
                "splits": json.dumps(splits),
            }
        )
    return pd.DataFrame(rows).sort_values(["mode", "species_bucket", "cluster_id"]).reset_index(drop=True)


def export_training_artifacts(
    *,
    artifacts_dir: Path,
    recordings_df: pd.DataFrame,
    events_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    clustered_df: pd.DataFrame,
    master_clusters: pd.DataFrame,
    species_config_path: Path | None,
    train_manifest_path: Path | None,
    val_manifest_path: Path | None,
    input_dir: Path,
    output_dir: Path,
    report_path: Path,
    embedding_meta_path: Path,
) -> dict[str, Path]:
    ensure_dir(artifacts_dir)

    recordings_snapshot_path = artifacts_dir / "train_recordings_used.csv"
    recordings_df.to_csv(recordings_snapshot_path, index=False)

    cluster_summary = build_cluster_training_summary(
        clustered_df=clustered_df,
        master_clusters=master_clusters,
    )
    cluster_summary_path = artifacts_dir / "cluster_training_summary.csv"
    cluster_summary.to_csv(cluster_summary_path, index=False)

    summary_payload = {
        "species_config_path": species_config_path.as_posix() if species_config_path else "",
        "train_manifest_path": train_manifest_path.as_posix() if train_manifest_path else "",
        "val_manifest_path": val_manifest_path.as_posix() if val_manifest_path else "",
        "input_dir": input_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "n_recordings": int(len(recordings_df)),
        "n_events": int(len(events_df)),
        "n_predictions": int(len(preds_df)),
        "n_clustered_events": int(len(clustered_df)),
        "n_species_conditioned_clusters": int(
            cluster_summary[
                (cluster_summary["mode"] == "species_conditioned") & (cluster_summary["cluster_id"] != -1)
            ].shape[0]
        )
        if not cluster_summary.empty
        else 0,
        "n_global_clusters": int(
            cluster_summary[(cluster_summary["mode"] == "global") & (cluster_summary["cluster_id"] != -1)].shape[0]
        )
        if not cluster_summary.empty
        else 0,
        "recording_source_counts": _string_counts(recordings_df, "recording_source"),
        "split_counts": _string_counts(recordings_df, "split"),
        "species_label_counts": _string_counts(recordings_df, "species_label"),
        "species_bucket_counts": _string_counts(clustered_df, "species_bucket"),
        "checkpoints": {
            "recordings_snapshot_csv": recordings_snapshot_path.as_posix(),
            "cluster_training_summary_csv": cluster_summary_path.as_posix(),
            "embedding_meta_json": embedding_meta_path.as_posix(),
            "master_predictions_csv": (output_dir / "preds" / "master_predictions.csv").as_posix(),
            "master_features_clustered_csv": (output_dir / "features" / "master_features_clustered.csv").as_posix(),
            "master_clusters_csv": (output_dir / "clusters" / "master_clusters.csv").as_posix(),
            "report_html": report_path.as_posix(),
        },
    }

    summary_path = artifacts_dir / "training_run_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True, default=_json_ready),
        encoding="utf-8",
    )

    return {
        "recordings_snapshot": recordings_snapshot_path,
        "cluster_summary": cluster_summary_path,
        "training_summary": summary_path,
    }


def export_species_check_artifacts(
    *,
    artifacts_dir: Path,
    recordings_df: pd.DataFrame,
    events_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    species_config_path: Path | None,
    train_manifest_path: Path | None,
    val_manifest_path: Path | None,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    ensure_dir(artifacts_dir)

    recordings_snapshot_path = artifacts_dir / "train_recordings_used.csv"
    recordings_df.to_csv(recordings_snapshot_path, index=False)

    summary_rows: list[dict[str, Any]] = []
    if not preds_df.empty:
        merged = preds_df.merge(
            events_df[["clip_id", "source_file", "species_label", "recording_source", "split"]].drop_duplicates("clip_id"),
            on="clip_id",
            how="left",
        )
        for species_top1, group in merged.groupby("species_top1", dropna=False):
            summary_rows.append(
                {
                    "species_top1": str(species_top1),
                    "n_events": int(len(group)),
                    "mean_confidence": float(
                        pd.to_numeric(group["species_top1_conf"], errors="coerce").fillna(0.0).mean()
                    ),
                    "inference_ok_fraction": float(group["inference_ok"].fillna(False).astype(bool).mean()),
                    "n_recordings": int(group["source_file"].fillna("").astype(str).replace("", np.nan).nunique()),
                }
            )
    species_summary_df = pd.DataFrame(summary_rows).sort_values(
        ["n_events", "species_top1"], ascending=[False, True]
    ) if summary_rows else pd.DataFrame(
        columns=["species_top1", "n_events", "mean_confidence", "inference_ok_fraction", "n_recordings"]
    )
    species_summary_path = artifacts_dir / "species_prediction_summary.csv"
    species_summary_df.to_csv(species_summary_path, index=False)

    summary_payload = {
        "species_config_path": species_config_path.as_posix() if species_config_path else "",
        "train_manifest_path": train_manifest_path.as_posix() if train_manifest_path else "",
        "val_manifest_path": val_manifest_path.as_posix() if val_manifest_path else "",
        "input_dir": input_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "mode": "species_only",
        "n_recordings": int(len(recordings_df)),
        "n_events": int(len(events_df)),
        "n_predictions": int(len(preds_df)),
        "recording_source_counts": _string_counts(recordings_df, "recording_source"),
        "split_counts": _string_counts(recordings_df, "split"),
        "species_label_counts": _string_counts(recordings_df, "species_label"),
        "species_prediction_counts": _string_counts(preds_df, "species_top1"),
        "checkpoints": {
            "recordings_snapshot_csv": recordings_snapshot_path.as_posix(),
            "species_prediction_summary_csv": species_summary_path.as_posix(),
            "master_predictions_csv": (output_dir / "preds" / "master_predictions.csv").as_posix(),
            "events_manifest_csv": (output_dir / "manifests" / "events_manifest.csv").as_posix(),
        },
    }
    summary_path = artifacts_dir / "species_run_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True, default=_json_ready),
        encoding="utf-8",
    )

    return {
        "recordings_snapshot": recordings_snapshot_path,
        "species_summary": species_summary_path,
        "species_run_summary": summary_path,
    }
