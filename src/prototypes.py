from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import ensure_dir, save_json


def _unit_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("Expected a 2D embedding array.")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def _cluster_ids(df: pd.DataFrame, *, include_noise: bool = False) -> list[int]:
    ids = sorted(int(v) for v in df["cluster_id"].dropna().astype(int).unique().tolist())
    if include_noise:
        return ids
    return [cluster_id for cluster_id in ids if cluster_id != -1]


def _cluster_overview(df: pd.DataFrame, cluster_id: int) -> dict[str, Any]:
    subset = df[df["cluster_id"].astype(int) == int(cluster_id)]
    recording_counts = (
        subset["recording_id"].fillna("").astype(str).value_counts()
        if "recording_id" in subset.columns and not subset.empty
        else pd.Series(dtype=int)
    )
    dominant_recording_id = str(recording_counts.index[0]) if not recording_counts.empty else ""
    dominant_recording_share = float(recording_counts.iloc[0] / max(1, len(subset))) if not recording_counts.empty else 0.0

    return {
        "cluster_id": int(cluster_id),
        "n_units": int(len(subset)),
        "n_recordings": int(subset["recording_id"].nunique()) if "recording_id" in subset.columns else 0,
        "dominant_recording_id": dominant_recording_id,
        "dominant_recording_share": dominant_recording_share,
        "mean_duration_s": float(pd.to_numeric(subset.get("duration_s", pd.Series(dtype=float)), errors="coerce").mean())
        if not subset.empty and "duration_s" in subset.columns
        else 0.0,
        "mean_species_top1_conf": float(
            pd.to_numeric(subset.get("species_top1_conf", pd.Series(dtype=float)), errors="coerce").mean()
        )
        if not subset.empty and "species_top1_conf" in subset.columns
        else 0.0,
    }


def export_cluster_prototypes(
    *,
    clustered_df: pd.DataFrame,
    embeddings: np.ndarray,
    output_dir: Path,
    threshold_quantile: float = 0.95,
) -> dict[str, Path]:
    if len(clustered_df) != int(embeddings.shape[0]):
        raise ValueError("clustered_df rows must match embedding rows for prototype export.")

    models_dir = output_dir / "models"
    ensure_dir(models_dir)
    cluster_ids = _cluster_ids(clustered_df, include_noise=False)
    threshold_quantile = float(np.clip(threshold_quantile, 0.5, 1.0))

    centroid_rows: list[np.ndarray] = []
    centroid_unit_rows: list[np.ndarray] = []
    cosine_thresholds: list[float] = []
    l2_thresholds: list[float] = []
    cluster_payloads: list[dict[str, Any]] = []

    for prototype_row, cluster_id in enumerate(cluster_ids):
        idx = np.where(clustered_df["cluster_id"].astype(int).to_numpy() == int(cluster_id))[0]
        cluster_embeddings = embeddings[idx].astype(np.float32, copy=False)
        centroid = np.mean(cluster_embeddings, axis=0).astype(np.float32)
        centroid_unit = centroid / max(float(np.linalg.norm(centroid)), 1e-12)

        cluster_unit = _unit_rows(cluster_embeddings)
        cosine_distances = 1.0 - np.clip(cluster_unit @ centroid_unit, -1.0, 1.0)
        l2_distances = np.linalg.norm(cluster_embeddings - centroid.reshape(1, -1), axis=1)

        cosine_threshold = float(np.quantile(cosine_distances, threshold_quantile)) if len(cosine_distances) else 0.0
        l2_threshold = float(np.quantile(l2_distances, threshold_quantile)) if len(l2_distances) else 0.0

        centroid_rows.append(centroid)
        centroid_unit_rows.append(centroid_unit.astype(np.float32))
        cosine_thresholds.append(cosine_threshold)
        l2_thresholds.append(l2_threshold)

        subset = clustered_df.iloc[idx]
        exemplar_clip_ids = subset.sort_values("cluster_prob", ascending=False)["clip_id"].astype(str).head(8).tolist()
        cluster_payload = {
            **_cluster_overview(clustered_df, cluster_id),
            "prototype_row": int(prototype_row),
            "cosine_distance_threshold": cosine_threshold,
            "cosine_distance_p50": float(np.quantile(cosine_distances, 0.50)) if len(cosine_distances) else 0.0,
            "cosine_distance_p90": float(np.quantile(cosine_distances, 0.90)) if len(cosine_distances) else 0.0,
            "cosine_distance_max": float(np.max(cosine_distances)) if len(cosine_distances) else 0.0,
            "l2_distance_threshold": l2_threshold,
            "l2_distance_p50": float(np.quantile(l2_distances, 0.50)) if len(l2_distances) else 0.0,
            "l2_distance_p90": float(np.quantile(l2_distances, 0.90)) if len(l2_distances) else 0.0,
            "l2_distance_max": float(np.max(l2_distances)) if len(l2_distances) else 0.0,
            "exemplar_clip_ids": exemplar_clip_ids,
            "review_status": "unreviewed",
            "review_cluster_name": "",
            "review_notes": "",
        }
        cluster_payloads.append(cluster_payload)

    vectors_path = models_dir / "cluster_prototypes.npz"
    if centroid_rows:
        np.savez_compressed(
            vectors_path,
            cluster_ids=np.asarray(cluster_ids, dtype=np.int32),
            centroids=np.vstack(centroid_rows).astype(np.float32),
            centroids_unit=np.vstack(centroid_unit_rows).astype(np.float32),
            cosine_thresholds=np.asarray(cosine_thresholds, dtype=np.float32),
            l2_thresholds=np.asarray(l2_thresholds, dtype=np.float32),
        )
    else:
        np.savez_compressed(
            vectors_path,
            cluster_ids=np.zeros((0,), dtype=np.int32),
            centroids=np.zeros((0, embeddings.shape[1] if embeddings.ndim == 2 else 0), dtype=np.float32),
            centroids_unit=np.zeros((0, embeddings.shape[1] if embeddings.ndim == 2 else 0), dtype=np.float32),
            cosine_thresholds=np.zeros((0,), dtype=np.float32),
            l2_thresholds=np.zeros((0,), dtype=np.float32),
        )

    metadata_path = models_dir / "cluster_prototypes.json"
    save_json(
        metadata_path,
        {
            "artifact_type": "cluster_prototypes",
            "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            "distance_metric": "cosine_distance_to_l2_normalized_centroid",
            "threshold_quantile": threshold_quantile,
            "vectors_npz": vectors_path.name,
            "n_clusters": int(len(cluster_payloads)),
            "clusters": cluster_payloads,
        },
    )
    return {"prototype_vectors": vectors_path, "prototype_metadata": metadata_path}


def export_cluster_merge_suggestions(
    *,
    clustered_df: pd.DataFrame,
    embeddings: np.ndarray,
    output_dir: Path,
    suggestions_per_cluster: int = 3,
    max_pairs: int = 60,
) -> tuple[pd.DataFrame, Path]:
    if len(clustered_df) != int(embeddings.shape[0]):
        raise ValueError("clustered_df rows must match embedding rows for merge suggestions.")

    reports_dir = output_dir / "reports"
    ensure_dir(reports_dir)
    out_path = reports_dir / "cluster_merge_suggestions.csv"

    cluster_ids = _cluster_ids(clustered_df, include_noise=False)
    if len(cluster_ids) < 2:
        df = pd.DataFrame()
        df.to_csv(out_path, index=False)
        return df, out_path

    centroids: list[np.ndarray] = []
    overviews = {cluster_id: _cluster_overview(clustered_df, cluster_id) for cluster_id in cluster_ids}
    for cluster_id in cluster_ids:
        idx = np.where(clustered_df["cluster_id"].astype(int).to_numpy() == int(cluster_id))[0]
        centroids.append(np.mean(embeddings[idx].astype(np.float32, copy=False), axis=0).astype(np.float32))

    centroid_matrix = np.vstack(centroids).astype(np.float32)
    centroid_unit = _unit_rows(centroid_matrix)

    pair_rows: list[dict[str, Any]] = []
    for i, cluster_a in enumerate(cluster_ids):
        for j in range(i + 1, len(cluster_ids)):
            cluster_b = cluster_ids[j]
            cosine_similarity = float(np.clip(centroid_unit[i] @ centroid_unit[j], -1.0, 1.0))
            cosine_distance = float(1.0 - cosine_similarity)
            l2_distance = float(np.linalg.norm(centroid_matrix[i] - centroid_matrix[j]))
            overview_a = overviews[cluster_a]
            overview_b = overviews[cluster_b]
            pair_rows.append(
                {
                    "cluster_a": int(cluster_a),
                    "cluster_b": int(cluster_b),
                    "cosine_similarity": cosine_similarity,
                    "cosine_distance": cosine_distance,
                    "centroid_l2_distance": l2_distance,
                    "n_units_a": int(overview_a["n_units"]),
                    "n_units_b": int(overview_b["n_units"]),
                    "n_recordings_a": int(overview_a["n_recordings"]),
                    "n_recordings_b": int(overview_b["n_recordings"]),
                    "dominant_recording_share_a": float(overview_a["dominant_recording_share"]),
                    "dominant_recording_share_b": float(overview_b["dominant_recording_share"]),
                    "mean_duration_s_a": float(overview_a["mean_duration_s"]),
                    "mean_duration_s_b": float(overview_b["mean_duration_s"]),
                    "mean_species_top1_conf_a": float(overview_a["mean_species_top1_conf"]),
                    "mean_species_top1_conf_b": float(overview_b["mean_species_top1_conf"]),
                    "review_merge_yes_no": "",
                    "review_same_call_type_yes_no": "",
                    "review_notes": "",
                }
            )

    pair_df = pd.DataFrame(pair_rows).sort_values(["cosine_distance", "centroid_l2_distance"]).reset_index(drop=True)
    if suggestions_per_cluster > 0:
        keep_indices: set[int] = set()
        for cluster_id in cluster_ids:
            mask = (pair_df["cluster_a"] == cluster_id) | (pair_df["cluster_b"] == cluster_id)
            keep_indices.update(pair_df[mask].head(int(suggestions_per_cluster)).index.tolist())
        pair_df = pair_df.loc[sorted(keep_indices)].sort_values(["cosine_distance", "centroid_l2_distance"]).reset_index(drop=True)
    if max_pairs > 0:
        pair_df = pair_df.head(int(max_pairs)).copy()
    pair_df.insert(0, "suggestion_rank", np.arange(1, len(pair_df) + 1, dtype=int))
    pair_df.to_csv(out_path, index=False)
    return pair_df, out_path


def export_same_different_review_template(
    *,
    clustered_df: pd.DataFrame,
    representatives: dict[int, list[int]],
    merge_suggestions_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    reports_dir = output_dir / "reports"
    ensure_dir(reports_dir)
    out_path = reports_dir / "same_different_review_pairs.csv"

    rows: list[dict[str, Any]] = []
    for _, suggestion in merge_suggestions_df.iterrows():
        cluster_a = int(suggestion["cluster_a"])
        cluster_b = int(suggestion["cluster_b"])
        reps_a = representatives.get(cluster_a, [])
        reps_b = representatives.get(cluster_b, [])
        if not reps_a or not reps_b:
            continue
        row_a = clustered_df.iloc[int(reps_a[0])]
        row_b = clustered_df.iloc[int(reps_b[0])]
        rows.append(
            {
                "suggestion_rank": int(suggestion["suggestion_rank"]),
                "cluster_a": cluster_a,
                "cluster_b": cluster_b,
                "cosine_distance": float(suggestion["cosine_distance"]),
                "clip_id_a": str(row_a["clip_id"]),
                "clip_id_b": str(row_b["clip_id"]),
                "audio_path_a": str(row_a["embedding_audio_path"]),
                "audio_path_b": str(row_b["embedding_audio_path"]),
                "review_same_call_type_yes_no": "",
                "review_confidence_1to5": "",
                "review_notes": "",
            }
        )

    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path
