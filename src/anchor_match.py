from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .adapter_model import EmbeddingAdapter, require_torch
from .embed import generate_embeddings
from .utils import ensure_dir, require_google_drive_output, save_json, slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Embed a small set of reference audio clips and compare them against an existing "
            "within-species clustering run."
        )
    )
    parser.add_argument("--pattern_run_dir", type=Path, required=True)
    parser.add_argument("--cluster_csv", type=Path, required=True)
    parser.add_argument("--anchors_manifest_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--adapter_checkpoint", type=Path, default=None)
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape={x.shape}.")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (x / norms).astype(np.float32)


def _normalize_vector(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = float(np.linalg.norm(x))
    if norm <= 1e-12:
        return x.astype(np.float32)
    return (x / norm).astype(np.float32)


def _load_adapter(checkpoint_path: Path) -> tuple[dict[str, object], EmbeddingAdapter]:
    require_torch()
    import torch

    checkpoint = torch.load(checkpoint_path.as_posix(), map_location="cpu")
    model = EmbeddingAdapter(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        dropout=float(checkpoint.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return checkpoint, model


def _maybe_transform_embeddings(
    embeddings: np.ndarray,
    *,
    pattern_meta: dict[str, object],
    adapter_checkpoint: Path | None,
) -> tuple[np.ndarray, str]:
    backend = str(pattern_meta.get("backend", "")).strip()
    if backend != "birdnet_adapter":
        return embeddings.astype(np.float32), "identity"

    checkpoint_path = adapter_checkpoint
    if checkpoint_path is None:
        checkpoint_text = str(pattern_meta.get("adapter_checkpoint", "")).strip()
        checkpoint_path = Path(checkpoint_text).expanduser() if checkpoint_text else None
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(
            "Pattern embeddings are in adapter space, but no valid adapter checkpoint was provided "
            "or recorded in embedding_meta.json."
        )

    checkpoint, model = _load_adapter(checkpoint_path)
    if int(embeddings.shape[1]) != int(checkpoint["input_dim"]):
        raise ValueError(
            f"Adapter expects input_dim={checkpoint['input_dim']}, but anchor embeddings have dim={embeddings.shape[1]}."
        )

    require_torch()
    import torch

    with torch.no_grad():
        transformed = model(torch.from_numpy(embeddings.astype(np.float32))).cpu().numpy().astype(np.float32)
    return transformed, checkpoint_path.as_posix()


def _load_pattern_inputs(pattern_run_dir: Path, cluster_csv: Path) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    manifest_path = pattern_run_dir / "manifests" / "accepted_events_manifest.csv"
    emb_path = pattern_run_dir / "embeddings" / "accepted_embeddings.npy"
    meta_path = pattern_run_dir / "embeddings" / "embedding_meta.json"
    index_path = pattern_run_dir / "embeddings" / "embedding_index.csv"

    for path in (manifest_path, emb_path, meta_path, index_path, cluster_csv):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path.as_posix()}")

    manifest_df = pd.read_csv(manifest_path)
    emb = np.load(emb_path).astype(np.float32)
    meta = _load_json(meta_path)
    index_df = pd.read_csv(index_path)
    cluster_df = pd.read_csv(cluster_csv)

    merged = (
        index_df[["clip_id", "embedding_row"]]
        .merge(manifest_df, on="clip_id", how="left")
        .merge(cluster_df[["clip_id", "cluster_id"]].drop_duplicates("clip_id"), on="clip_id", how="inner")
    )
    merged["embedding_row"] = pd.to_numeric(merged["embedding_row"], errors="coerce").astype(int)
    if merged.empty:
        raise ValueError("No overlap between cluster CSV and embedding index.")
    if int(emb.shape[0]) <= int(merged["embedding_row"].max()):
        raise ValueError("Embedding rows in index exceed saved embedding matrix length.")
    return merged.reset_index(drop=True), emb, meta


def _load_anchor_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"anchor_id", "anchor_label", "audio_path"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Anchor manifest is missing required columns: {', '.join(missing)}")
    df = df.copy()
    df["anchor_id"] = df["anchor_id"].fillna("").astype(str).str.strip()
    df["anchor_label"] = df["anchor_label"].fillna("").astype(str).str.strip()
    df["audio_path"] = df["audio_path"].fillna("").astype(str).str.strip()
    df = df[(df["anchor_id"] != "") & (df["anchor_label"] != "") & (df["audio_path"] != "")]
    if df.empty:
        raise ValueError("Anchor manifest has no usable rows with anchor_id, anchor_label, and audio_path.")
    return df.reset_index(drop=True)


def _prepare_anchor_audio(anchor_df: pd.DataFrame, temp_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, row in anchor_df.iterrows():
        source_path = Path(str(row["audio_path"])).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"Anchor audio not found: {source_path.as_posix()}")
        suffix = source_path.suffix.lower() or ".wav"
        safe_name = f"{idx:03d}_{slugify(str(row['anchor_id']))}{suffix}"
        dest_path = temp_dir / safe_name
        shutil.copy2(source_path, dest_path)
        rows.append(
            {
                **row.to_dict(),
                "resolved_audio_path": source_path.as_posix(),
                "copied_audio_path": dest_path.as_posix(),
                "event_wav": safe_name,
                "clip_id": str(row["anchor_id"]),
            }
        )
    return pd.DataFrame(rows)


def _build_cluster_centroids(pattern_df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster_id, group in pattern_df.groupby("cluster_id", dropna=False):
        cluster_int = int(cluster_id)
        if cluster_int == -1:
            continue
        idxs = group["embedding_row"].astype(int).to_numpy()
        cluster_vectors = _normalize_rows(embeddings[idxs])
        centroid = _normalize_vector(cluster_vectors.mean(axis=0))
        recording_counts = (
            group["recording_id"].fillna("").astype(str).value_counts()
            if "recording_id" in group.columns
            else pd.Series(dtype=int)
        )
        rows.append(
            {
                "cluster_id": cluster_int,
                "n_units": int(len(group)),
                "n_recordings": int(group["recording_id"].nunique()) if "recording_id" in group.columns else 0,
                "dominant_recording_share": float(recording_counts.iloc[0] / max(1, len(group)))
                if not recording_counts.empty
                else 0.0,
                "centroid": centroid,
            }
        )
    if not rows:
        raise ValueError("No non-noise clusters found in cluster CSV.")
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def _cosine_scores(anchors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return _normalize_rows(anchors) @ _normalize_rows(centroids).T


def main() -> None:
    args = parse_args()
    pattern_run_dir = args.pattern_run_dir.resolve()
    cluster_csv = args.cluster_csv.resolve()
    anchors_manifest_csv = args.anchors_manifest_csv.resolve()
    output_dir = require_google_drive_output(args.output_dir, purpose="--output_dir")

    ensure_dir(output_dir)
    ensure_dir(output_dir / "reports")
    ensure_dir(output_dir / "anchors")

    pattern_df, pattern_embeddings, pattern_meta = _load_pattern_inputs(pattern_run_dir, cluster_csv)
    anchor_manifest = _load_anchor_manifest(anchors_manifest_csv)

    with tempfile.TemporaryDirectory(prefix="anchor_match_") as tmp:
        temp_dir = Path(tmp)
        prepared_anchor_df = _prepare_anchor_audio(anchor_manifest, temp_dir=temp_dir)

        clip_df = prepared_anchor_df[["clip_id", "event_wav"]].copy()
        anchor_embeddings_path = output_dir / "anchors" / "anchor_embeddings.npy"
        anchor_embeddings = generate_embeddings(
            clips_df=clip_df,
            events_dir=temp_dir,
            sr=int(args.sample_rate),
            out_path=anchor_embeddings_path,
            backend="birdnet",
            fallback_backend="none",
            force=args.force,
        )

    adapter_checkpoint_path = args.adapter_checkpoint.expanduser().resolve() if args.adapter_checkpoint else None
    anchor_embeddings, adapter_used = _maybe_transform_embeddings(
        anchor_embeddings,
        pattern_meta=pattern_meta,
        adapter_checkpoint=adapter_checkpoint_path,
    )
    np.save(anchor_embeddings_path, anchor_embeddings.astype(np.float32))

    centroids_df = _build_cluster_centroids(pattern_df=pattern_df, embeddings=pattern_embeddings)
    centroid_matrix = np.vstack(centroids_df["centroid"].to_list()).astype(np.float32)
    score_matrix = _cosine_scores(anchor_embeddings, centroid_matrix)

    score_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    for anchor_idx, anchor_row in prepared_anchor_df.reset_index(drop=True).iterrows():
        scores = score_matrix[anchor_idx]
        order = np.argsort(scores)[::-1]
        top_order = order[: max(1, int(args.top_k))]
        for rank, cluster_pos in enumerate(top_order, start=1):
            cluster_meta = centroids_df.iloc[int(cluster_pos)]
            score_rows.append(
                {
                    "anchor_id": str(anchor_row["anchor_id"]),
                    "anchor_label": str(anchor_row["anchor_label"]),
                    "rank": int(rank),
                    "cluster_id": int(cluster_meta["cluster_id"]),
                    "cosine_similarity": float(scores[int(cluster_pos)]),
                    "n_units": int(cluster_meta["n_units"]),
                    "n_recordings": int(cluster_meta["n_recordings"]),
                    "dominant_recording_share": float(cluster_meta["dominant_recording_share"]),
                    "resolved_audio_path": str(anchor_row["resolved_audio_path"]),
                    "source_url": str(anchor_row.get("source_url", "")),
                    "notes": str(anchor_row.get("notes", "")),
                }
            )
        best_cluster_meta = centroids_df.iloc[int(order[0])]
        best_rows.append(
            {
                "anchor_id": str(anchor_row["anchor_id"]),
                "anchor_label": str(anchor_row["anchor_label"]),
                "best_cluster_id": int(best_cluster_meta["cluster_id"]),
                "best_cluster_similarity": float(scores[int(order[0])]),
                "best_cluster_n_units": int(best_cluster_meta["n_units"]),
                "best_cluster_n_recordings": int(best_cluster_meta["n_recordings"]),
                "best_cluster_dominant_recording_share": float(best_cluster_meta["dominant_recording_share"]),
            }
        )

    score_df = pd.DataFrame(score_rows)
    best_df = pd.DataFrame(best_rows)

    label_summary = (
        best_df.groupby(["anchor_label", "best_cluster_id"])
        .size()
        .reset_index(name="n_anchors")
        .sort_values(["anchor_label", "n_anchors", "best_cluster_id"], ascending=[True, False, True])
    )

    cluster_label_scores = (
        score_df.groupby(["cluster_id", "anchor_label"])
        .agg(
            mean_similarity=("cosine_similarity", "mean"),
            max_similarity=("cosine_similarity", "max"),
            n_anchor_examples=("anchor_id", "nunique"),
        )
        .reset_index()
        .sort_values(["cluster_id", "mean_similarity", "max_similarity"], ascending=[True, False, False])
    )

    prepared_anchor_df.to_csv(output_dir / "anchors" / "anchor_manifest_resolved.csv", index=False)
    centroids_df.drop(columns=["centroid"]).to_csv(output_dir / "reports" / "cluster_centroid_summary.csv", index=False)
    score_df.to_csv(output_dir / "reports" / "anchor_to_cluster_scores.csv", index=False)
    best_df.to_csv(output_dir / "reports" / "anchor_best_cluster_matches.csv", index=False)
    label_summary.to_csv(output_dir / "reports" / "anchor_label_cluster_summary.csv", index=False)
    cluster_label_scores.to_csv(output_dir / "reports" / "cluster_anchor_label_scores.csv", index=False)

    save_json(
        output_dir / "reports" / "anchor_match_summary.json",
        {
            "pattern_run_dir": pattern_run_dir.as_posix(),
            "cluster_csv": cluster_csv.as_posix(),
            "anchors_manifest_csv": anchors_manifest_csv.as_posix(),
            "n_anchors": int(len(prepared_anchor_df)),
            "n_anchor_labels": int(prepared_anchor_df["anchor_label"].nunique()),
            "n_clusters_scored": int(len(centroids_df)),
            "pattern_embedding_backend": str(pattern_meta.get("backend", "")),
            "adapter_used_for_anchors": adapter_used,
            "top_k": int(args.top_k),
        },
    )

    print(f"Wrote anchor-to-cluster scores to {(output_dir / 'reports' / 'anchor_to_cluster_scores.csv').as_posix()}")
    print(f"Wrote best anchor matches to {(output_dir / 'reports' / 'anchor_best_cluster_matches.csv').as_posix()}")


if __name__ == "__main__":
    main()
