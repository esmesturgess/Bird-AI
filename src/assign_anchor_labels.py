At from __future__ import annotations

"""Assign semantic anchor labels such as song/call/alarm to pattern-layer clips.

This module takes a completed pattern-layer run and a reviewed anchor manifest,
embeds the anchor clips in the same space, and assigns each pattern clip to the
nearest anchor family. It can optionally smooth assignments at the cluster level.
"""

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


CLUSTER_SMOOTHING_COLUMNS = [
    "cluster_id",
    "n_clips",
    "cluster_top_anchor_label",
    "cluster_top_similarity",
    "cluster_second_anchor_label",
    "cluster_second_similarity",
    "cluster_similarity_margin",
    "used_cluster_label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign every clip in a pattern-layer run to the nearest anchor label "
            "such as song/call/alarm/juvenile_begging."
        )
    )
    parser.add_argument("--pattern_run_dir", type=Path, required=True)
    parser.add_argument("--cluster_csv", type=Path, default=None)
    parser.add_argument("--anchors_manifest_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--adapter_checkpoint", type=Path, default=None)
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--disable_cluster_smoothing", action="store_true")
    parser.add_argument("--cluster_margin_threshold", type=float, default=0.03)
    parser.add_argument("--copy_audio", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
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


def _maybe_transform_anchor_embeddings(
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


def _load_pattern_inputs(pattern_run_dir: Path, cluster_csv: Path | None) -> tuple[pd.DataFrame, np.ndarray, dict[str, object], dict[str, object]]:
    manifest_path = pattern_run_dir / "manifests" / "accepted_events_manifest.csv"
    summary_path = pattern_run_dir / "manifests" / "accepted_events_summary.json"
    emb_path = pattern_run_dir / "embeddings" / "accepted_embeddings.npy"
    meta_path = pattern_run_dir / "embeddings" / "embedding_meta.json"

    for path in (manifest_path, summary_path, emb_path, meta_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path.as_posix()}")

    manifest_df = pd.read_csv(manifest_path)
    emb = np.load(emb_path).astype(np.float32)
    summary = _load_json(summary_path)
    meta = _load_json(meta_path)
    if len(manifest_df) != emb.shape[0]:
        raise ValueError("Manifest rows do not match embedding rows.")

    if cluster_csv is not None:
        cluster_df = pd.read_csv(cluster_csv)
        manifest_df = manifest_df.merge(
            cluster_df[["clip_id", "cluster_id"]].drop_duplicates("clip_id"),
            on="clip_id",
            how="left",
        )
    else:
        manifest_df["cluster_id"] = -999

    return manifest_df.reset_index(drop=True), emb, summary, meta


def _resolve_audio_path(row: pd.Series, summary: dict[str, object]) -> Path:
    clip_name = Path(str(row["event_wav"])).name
    embedding_dir = Path(str(summary.get("embedding_dir", ""))).expanduser()
    candidate = embedding_dir / clip_name
    if candidate.exists():
        return candidate
    row_path = Path(str(row.get("embedding_audio_path", ""))).expanduser()
    if row_path.exists():
        return row_path
    raise FileNotFoundError(f"Audio clip not found for {clip_name}")


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


def _build_label_prototypes(anchor_df: pd.DataFrame, anchor_embeddings: np.ndarray) -> tuple[list[str], np.ndarray]:
    label_rows: list[str] = []
    centroid_rows: list[np.ndarray] = []
    normalized = _normalize_rows(anchor_embeddings)
    anchor_df = anchor_df.reset_index(drop=True)
    # Each semantic family is represented by a centroid over its reviewed anchor
    # clips. Later we compare every pattern clip against these family prototypes.
    for label, group in anchor_df.groupby("anchor_label", sort=True):
        idxs = group.index.to_numpy(dtype=int)
        centroid = _normalize_vector(normalized[idxs].mean(axis=0))
        label_rows.append(str(label))
        centroid_rows.append(centroid)
    if not centroid_rows:
        raise ValueError("No anchor label prototypes could be built.")
    return label_rows, np.vstack(centroid_rows).astype(np.float32)


def _copy_assigned_audio(assignments_df: pd.DataFrame, output_dir: Path) -> None:
    audio_dir = output_dir / "assigned_audio"
    ensure_dir(audio_dir)
    for _, row in assignments_df.iterrows():
        label = str(row["final_anchor_label"])
        target_dir = audio_dir / label
        ensure_dir(target_dir)
        source_audio = Path(str(row["resolved_audio_path"]))
        cluster_id = int(row["cluster_id"])
        prefix = f"cluster_{cluster_id:02d}" if cluster_id >= 0 else "cluster_noise"
        filename = (
            f"{prefix}__score_{float(row['best_similarity']):.3f}"
            f"__margin_{float(row['similarity_margin']):.3f}__{source_audio.name}"
        )
        shutil.copy2(source_audio, target_dir / filename)


def _empty_cluster_smoothing_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CLUSTER_SMOOTHING_COLUMNS)


def _build_other_candidate_report(
    assignments_df: pd.DataFrame,
    *,
    cluster_smoothing_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    vote_by_cluster = (
        cluster_smoothing_df.set_index("cluster_id").to_dict(orient="index")
        if not cluster_smoothing_df.empty
        else {}
    )
    rows: list[dict[str, object]] = []
    total = int(len(assignments_df))
    for cluster_id, group in assignments_df.groupby("cluster_id", dropna=False):
        cluster_int = int(cluster_id)
        n_clips = int(len(group))
        n_recordings = int(group["recording_id"].nunique()) if "recording_id" in group.columns else 0
        label_counts = group["final_anchor_label"].fillna("").astype(str).value_counts()
        dominant_label = str(label_counts.index[0]) if not label_counts.empty else ""
        dominant_share = float(label_counts.iloc[0] / max(1, n_clips)) if not label_counts.empty else 0.0
        vote = vote_by_cluster.get(cluster_int, {})
        reasons: list[str] = []
        if cluster_int == -1:
            if n_clips >= max(5, int(round(0.10 * total))) and n_recordings >= 4:
                reasons.append("large_multi_recording_noise_bucket")
        else:
            if not bool(vote.get("used_cluster_label", True)) and n_clips >= 4:
                reasons.append("ambiguous_cluster_vote")
            if dominant_share < 0.75 and n_clips >= 4:
                reasons.append("mixed_label_distribution")
        if not reasons:
            continue
        rows.append(
            {
                "cluster_id": cluster_int,
                "n_clips": n_clips,
                "n_recordings": n_recordings,
                "dominant_label": dominant_label,
                "dominant_label_share": dominant_share,
                "cluster_top_anchor_label": str(vote.get("cluster_top_anchor_label", "")),
                "cluster_second_anchor_label": str(vote.get("cluster_second_anchor_label", "")),
                "cluster_similarity_margin": float(vote.get("cluster_similarity_margin", float("nan"))),
                "reasons": ";".join(reasons),
            }
        )
    report_df = pd.DataFrame(rows).sort_values(["cluster_id"]).reset_index(drop=True) if rows else pd.DataFrame(
        columns=[
            "cluster_id",
            "n_clips",
            "n_recordings",
            "dominant_label",
            "dominant_label_share",
            "cluster_top_anchor_label",
            "cluster_second_anchor_label",
            "cluster_similarity_margin",
            "reasons",
        ]
    )
    summary = {
        "n_candidate_clusters": int(len(report_df)),
        "n_candidate_clips": int(report_df["n_clips"].sum()) if not report_df.empty else 0,
        "candidate_clip_fraction": float(report_df["n_clips"].sum() / max(1, total)) if not report_df.empty else 0.0,
    }
    return report_df, summary


def _apply_cluster_smoothing(
    assignments_df: pd.DataFrame,
    *,
    label_names: list[str],
    scores: np.ndarray,
    margin_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = assignments_df.copy()
    out["cluster_top_anchor_label"] = ""
    out["cluster_top_similarity"] = np.nan
    out["cluster_second_anchor_label"] = ""
    out["cluster_second_similarity"] = np.nan
    out["cluster_similarity_margin"] = np.nan
    out["final_anchor_label"] = out["assigned_anchor_label"]
    out["assignment_strategy"] = "clip_nearest"

    cluster_rows: list[dict[str, object]] = []
    for cluster_id, group in out.groupby("cluster_id", dropna=False):
        cluster_int = int(cluster_id)
        if cluster_int == -1 or len(group) == 0:
            continue
        idxs = group.index.to_numpy(dtype=int)
        cluster_scores = scores[idxs]
        mean_scores = np.mean(cluster_scores, axis=0)
        order = np.argsort(mean_scores)[::-1]
        best_idx = int(order[0])
        second_idx = int(order[1]) if len(order) > 1 else best_idx
        best_label = label_names[best_idx]
        second_label = label_names[second_idx]
        best_score = float(mean_scores[best_idx])
        second_score = float(mean_scores[second_idx])
        margin = best_score - second_score
        # If a discovered cluster strongly prefers one anchor family, we let the
        # whole cluster vote together instead of trusting every clip independently.
        use_cluster_label = margin >= float(margin_threshold)

        out.loc[idxs, "cluster_top_anchor_label"] = best_label
        out.loc[idxs, "cluster_top_similarity"] = best_score
        out.loc[idxs, "cluster_second_anchor_label"] = second_label
        out.loc[idxs, "cluster_second_similarity"] = second_score
        out.loc[idxs, "cluster_similarity_margin"] = margin
        if use_cluster_label:
            out.loc[idxs, "final_anchor_label"] = best_label
            out.loc[idxs, "assignment_strategy"] = "cluster_smoothed"

        cluster_rows.append(
            {
                "cluster_id": cluster_int,
                "n_clips": int(len(group)),
                "cluster_top_anchor_label": best_label,
                "cluster_top_similarity": best_score,
                "cluster_second_anchor_label": second_label,
                "cluster_second_similarity": second_score,
                "cluster_similarity_margin": margin,
                "used_cluster_label": bool(use_cluster_label),
            }
        )

    cluster_summary = pd.DataFrame(cluster_rows).sort_values("cluster_id").reset_index(drop=True) if cluster_rows else pd.DataFrame(
        columns=[
            "cluster_id",
            "n_clips",
            "cluster_top_anchor_label",
            "cluster_top_similarity",
            "cluster_second_anchor_label",
            "cluster_second_similarity",
            "cluster_similarity_margin",
            "used_cluster_label",
        ]
    )
    return out, cluster_summary


def _assign_nearest_labels(
    *,
    pattern_df: pd.DataFrame,
    scores: np.ndarray,
    label_names: list[str],
    summary: dict[str, object],
) -> pd.DataFrame:
    order = np.argsort(scores, axis=1)[:, ::-1]
    assigned_labels: list[str] = []
    second_labels: list[str] = []
    best_scores: list[float] = []
    second_scores: list[float] = []
    margins: list[float] = []
    audio_paths: list[str] = []

    for i in range(scores.shape[0]):
        best_idx = int(order[i, 0])
        second_idx = int(order[i, 1]) if scores.shape[1] > 1 else best_idx
        best_score = float(scores[i, best_idx])
        second_score = float(scores[i, second_idx])
        assigned_labels.append(label_names[best_idx])
        second_labels.append(label_names[second_idx])
        best_scores.append(best_score)
        second_scores.append(second_score)
        margins.append(best_score - second_score)
        audio_paths.append(_resolve_audio_path(pattern_df.iloc[i], summary).as_posix())

    assignments_df = pattern_df.copy()
    assignments_df["resolved_audio_path"] = audio_paths
    assignments_df["assigned_anchor_label"] = assigned_labels
    assignments_df["second_anchor_label"] = second_labels
    assignments_df["best_similarity"] = best_scores
    assignments_df["second_similarity"] = second_scores
    assignments_df["similarity_margin"] = margins
    return assignments_df


def _write_assignment_reports(
    *,
    output_dir: Path,
    assignments_df: pd.DataFrame,
    cluster_smoothing_df: pd.DataFrame,
    prepared_anchor_df: pd.DataFrame,
) -> dict[str, object]:
    reports_dir = output_dir / "reports"
    assignments_csv = reports_dir / "all_clip_anchor_label_assignments.csv"
    assignments_df.to_csv(assignments_csv, index=False)

    label_summary = (
        assignments_df.groupby("final_anchor_label")
        .agg(
            n_clips=("clip_id", "count"),
            n_recordings=("recording_id", "nunique"),
            mean_best_similarity=("best_similarity", "mean"),
            mean_similarity_margin=("similarity_margin", "mean"),
        )
        .reset_index()
        .sort_values("final_anchor_label")
    )
    label_summary.to_csv(reports_dir / "anchor_label_summary.csv", index=False)

    cluster_label_summary = (
        assignments_df.groupby(["cluster_id", "final_anchor_label"])
        .size()
        .reset_index(name="n_clips")
        .sort_values(["cluster_id", "n_clips", "final_anchor_label"], ascending=[True, False, True])
    )
    cluster_label_summary.to_csv(reports_dir / "cluster_by_anchor_label_summary.csv", index=False)
    cluster_smoothing_df.to_csv(reports_dir / "cluster_anchor_label_votes.csv", index=False)

    other_candidates_df, other_candidate_summary = _build_other_candidate_report(
        assignments_df,
        cluster_smoothing_df=cluster_smoothing_df,
    )
    other_candidates_df.to_csv(reports_dir / "other_candidate_clusters.csv", index=False)
    prepared_anchor_df.to_csv(reports_dir / "resolved_anchor_manifest.csv", index=False)

    return {
        "assignments_csv": assignments_csv,
        "other_candidate_summary": other_candidate_summary,
    }


def main() -> None:
    args = parse_args()
    pattern_run_dir = args.pattern_run_dir.resolve()
    cluster_csv = args.cluster_csv.resolve() if args.cluster_csv else None
    anchors_manifest_csv = args.anchors_manifest_csv.resolve()
    output_dir = require_google_drive_output(args.output_dir, purpose="--output_dir")

    ensure_dir(output_dir)
    ensure_dir(output_dir / "reports")

    pattern_df, pattern_embeddings, summary, pattern_meta = _load_pattern_inputs(pattern_run_dir, cluster_csv)
    anchor_manifest = _load_anchor_manifest(anchors_manifest_csv)

    with tempfile.TemporaryDirectory(prefix="anchor_assign_") as tmp:
        temp_dir = Path(tmp)
        prepared_anchor_df = _prepare_anchor_audio(anchor_manifest, temp_dir=temp_dir)
        clip_df = prepared_anchor_df[["clip_id", "event_wav"]].copy()
        anchor_embeddings_path = output_dir / "reports" / "anchor_embeddings.npy"
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
    anchor_embeddings, adapter_used = _maybe_transform_anchor_embeddings(
        anchor_embeddings,
        pattern_meta=pattern_meta,
        adapter_checkpoint=adapter_checkpoint_path,
    )
    np.save(anchor_embeddings_path, anchor_embeddings.astype(np.float32))

    label_names, label_prototypes = _build_label_prototypes(prepared_anchor_df, anchor_embeddings)
    clip_embeddings = _normalize_rows(pattern_embeddings)
    # Cosine similarity in normalized embedding space is the main matching score
    # between pattern clips and semantic anchor families.
    scores = clip_embeddings @ _normalize_rows(label_prototypes).T
    assignments_df = _assign_nearest_labels(
        pattern_df=pattern_df,
        scores=scores,
        label_names=label_names,
        summary=summary,
    )

    if args.disable_cluster_smoothing:
        assignments_df["cluster_top_anchor_label"] = ""
        assignments_df["cluster_top_similarity"] = np.nan
        assignments_df["cluster_second_anchor_label"] = ""
        assignments_df["cluster_second_similarity"] = np.nan
        assignments_df["cluster_similarity_margin"] = np.nan
        assignments_df["final_anchor_label"] = assignments_df["assigned_anchor_label"]
        assignments_df["assignment_strategy"] = "clip_nearest"
        cluster_smoothing_df = _empty_cluster_smoothing_df()
    else:
        assignments_df, cluster_smoothing_df = _apply_cluster_smoothing(
            assignments_df,
            label_names=label_names,
            scores=scores,
            margin_threshold=float(args.cluster_margin_threshold),
        )

    report_outputs = _write_assignment_reports(
        output_dir=output_dir,
        assignments_df=assignments_df,
        cluster_smoothing_df=cluster_smoothing_df,
        prepared_anchor_df=prepared_anchor_df,
    )

    if args.copy_audio:
        _copy_assigned_audio(assignments_df, output_dir=output_dir)

    save_json(
        output_dir / "reports" / "anchor_label_assignment_summary.json",
        {
            "pattern_run_dir": pattern_run_dir.as_posix(),
            "cluster_csv": cluster_csv.as_posix() if cluster_csv else "",
            "anchors_manifest_csv": anchors_manifest_csv.as_posix(),
            "n_clips": int(len(assignments_df)),
            "n_anchor_labels": int(len(label_names)),
            "anchor_labels": label_names,
            "pattern_embedding_backend": str(pattern_meta.get("backend", "")),
            "adapter_used_for_anchors": adapter_used,
            "cluster_smoothing": not bool(args.disable_cluster_smoothing),
            "cluster_margin_threshold": float(args.cluster_margin_threshold),
            "other_candidate_summary": report_outputs["other_candidate_summary"],
            "copy_audio": bool(args.copy_audio),
        },
    )

    print(f"Wrote assignments to {report_outputs['assignments_csv'].as_posix()}")
    if args.copy_audio:
        print(f"Wrote assigned audio folders to {(output_dir / 'assigned_audio').as_posix()}")


if __name__ == "__main__":
    main()
