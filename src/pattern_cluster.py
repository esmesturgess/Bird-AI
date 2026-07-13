from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from .cluster import _cluster_single_set, _save_cluster_scatter, representative_indices_by_cluster
from .features import add_indices, extract_features, save_feature_json, save_spectrogram_assets
from .prototypes import (
    export_cluster_merge_suggestions,
    export_cluster_prototypes,
    export_same_different_review_template,
)
from .report import (
    build_cluster_overview_rows,
    create_cluster_feature_heatmap,
    create_cluster_mean_spectrograms,
    create_cluster_quality_plot,
    create_cluster_recording_heatmap,
    create_cluster_source_balance_plot,
    create_distribution_plots,
    export_cluster_review_sheets,
    export_review_bundle,
    generate_html_report,
    write_cluster_playlists,
)
from .utils import ensure_dir, require_google_drive_output, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster accepted within-species pattern embeddings and export a review report.")
    parser.add_argument("--pattern_run_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sr", type=int, default=32000)
    parser.add_argument("--min_cluster_size", type=int, default=4)
    parser.add_argument("--min_samples", type=int, default=2)
    parser.add_argument("--use_umap", action="store_true")
    parser.add_argument("--representatives_per_cluster", type=int, default=8)
    parser.add_argument("--representative_max_per_recording", type=int, default=2)
    parser.add_argument(
        "--prototype_threshold_quantile",
        type=float,
        default=0.95,
        help="Quantile of within-cluster distance used as the future-assignment prototype threshold.",
    )
    parser.add_argument(
        "--merge_suggestions_per_cluster",
        type=int,
        default=3,
        help="How many nearest centroid-neighbour merge candidates to keep per cluster. Use 0 to keep all pairs.",
    )
    parser.add_argument(
        "--merge_suggestions_max_pairs",
        type=int,
        default=60,
        help="Maximum number of merge suggestion rows to export. Use 0 to keep all selected pairs.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_pattern_inputs(pattern_run_dir: Path) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    manifest_path = pattern_run_dir / "manifests" / "accepted_events_manifest.csv"
    emb_path = pattern_run_dir / "embeddings" / "accepted_embeddings.npy"
    meta_path = pattern_run_dir / "embeddings" / "embedding_meta.json"
    summary_path = pattern_run_dir / "manifests" / "accepted_events_summary.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Accepted manifest not found: {manifest_path.as_posix()}")
    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding array not found: {emb_path.as_posix()}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Embedding meta not found: {meta_path.as_posix()}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Accepted events summary not found: {summary_path.as_posix()}")

    manifest = pd.read_csv(manifest_path)
    embeddings = np.load(emb_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if len(manifest) != embeddings.shape[0]:
        raise ValueError(
            f"Accepted manifest rows ({len(manifest)}) do not match embedding rows ({embeddings.shape[0]})."
        )

    return manifest, embeddings, {"embedding_meta": meta, "accepted_summary": summary}


def _compute_review_features(
    manifest_df: pd.DataFrame,
    audio_dir: Path,
    output_dir: Path,
    sr: int,
    force: bool,
) -> pd.DataFrame:
    features_dir = output_dir / "features"
    specs_dir = output_dir / "specs"
    ensure_dir(features_dir)
    ensure_dir(specs_dir)
    master_path = features_dir / "accepted_features.csv"
    if master_path.exists() and not force:
        return pd.read_csv(master_path)

    rows: list[dict[str, object]] = []
    for _, row in manifest_df.iterrows():
        clip_name = Path(str(row["event_wav"])).name
        clip_path = audio_dir / clip_name
        if not clip_path.exists():
            clip_path = Path(str(row["embedding_audio_path"]))
        if not clip_path.exists():
            raise FileNotFoundError(f"Audio clip not found for review: {clip_name}")

        y, _ = librosa.load(clip_path.as_posix(), sr=sr, mono=True)
        feat = extract_features(y, sr)
        context_duration_s = float(feat.pop("duration_s", np.nan))
        npy_path, png_path = save_spectrogram_assets(
            y=y,
            sr=sr,
            clip_id=str(row["clip_id"]),
            specs_dir=specs_dir,
            force=force,
        )
        merged = {
            **row.to_dict(),
            **feat,
            "context_duration_s": context_duration_s,
            "spec_npy": npy_path.name,
            "spec_png": png_path.name,
        }
        feat_json_path = features_dir / f"{row['clip_id']}.json"
        save_feature_json(feat_json_path, merged)
        merged["feature_json"] = feat_json_path.name
        rows.append(merged)

    df = pd.DataFrame(rows)
    df = add_indices(df)
    df.to_csv(master_path, index=False)
    return df


def _build_cluster_summary(df: pd.DataFrame) -> dict[str, object]:
    counts = Counter(int(v) for v in df["cluster_id"].to_numpy())
    overview_rows = {int(row["cluster_id"]): row for row in build_cluster_overview_rows(df)}
    cluster_rows = []
    for cluster_id, count in sorted(counts.items(), key=lambda item: item[0]):
        subset = df[df["cluster_id"] == cluster_id]
        overview = overview_rows.get(int(cluster_id), {})
        cluster_rows.append(
            {
                "cluster_id": int(cluster_id),
                "n_events": int(count),
                "n_recordings": int(overview.get("n_recordings", 0)),
                "dominant_recording_id": str(overview.get("dominant_recording_id", "")),
                "dominant_recording_share": float(overview.get("dominant_recording_share", 0.0)),
                "mean_cluster_prob": float(subset["cluster_prob"].mean()) if not subset.empty else 0.0,
                "mean_duration_s": float(subset["duration_s"].mean()) if not subset.empty else 0.0,
                "mean_species_top1_conf": float(subset["species_top1_conf"].mean()) if "species_top1_conf" in subset else 0.0,
                "mean_repetition_rate_hz": float(subset["repetition_rate_hz"].mean()) if not subset.empty else 0.0,
                "mean_song_likeness": float(subset["song_likeness"].mean()) if not subset.empty else 0.0,
                "mean_urgency_index": float(subset["urgency_index"].mean()) if not subset.empty else 0.0,
            }
        )
    return {
        "n_events": int(len(df)),
        "n_clusters_excluding_noise": int(sum(1 for cid in counts if cid != -1)),
        "n_noise_events": int(counts.get(-1, 0)),
        "cluster_rows": cluster_rows,
    }


def _export_representatives(
    df: pd.DataFrame,
    representatives: dict[int, list[int]],
    output_dir: Path,
) -> Path:
    rows: list[dict[str, object]] = []
    for cluster_id, idx_list in sorted(representatives.items(), key=lambda item: item[0]):
        for exemplar_rank, row_idx in enumerate(idx_list, start=1):
            row = df.iloc[row_idx]
            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "exemplar_rank": int(exemplar_rank),
                    "clip_id": str(row["clip_id"]),
                    "event_wav": str(row["event_wav"]),
                    "cluster_prob": float(row["cluster_prob"]),
                    "duration_s": float(row["duration_s"]),
                    "species_top1_conf": float(row["species_top1_conf"]),
                    "embedding_audio_path": str(row["embedding_audio_path"]),
                }
            )
    out_path = output_dir / "reports" / "cluster_representatives.csv"
    ensure_dir(out_path.parent)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def main() -> None:
    args = parse_args()
    pattern_run_dir = args.pattern_run_dir.resolve()
    output_dir = require_google_drive_output(args.output_dir, purpose="--output_dir")
    ensure_dir(output_dir)
    ensure_dir(output_dir / "clusters")
    ensure_dir(output_dir / "reports")

    accepted_df, embeddings, meta = _load_pattern_inputs(pattern_run_dir)
    accepted_summary = meta["accepted_summary"]
    analysis_unit = str(accepted_summary.get("analysis_unit", "event"))
    audio_dir = Path(str(accepted_summary["embedding_dir"])).resolve()

    features_df = _compute_review_features(
        manifest_df=accepted_df,
        audio_dir=audio_dir,
        output_dir=output_dir,
        sr=args.sr,
        force=args.force,
    )

    labels, probs, pca_emb, umap_emb = _cluster_single_set(
        embeddings=embeddings,
        out_prefix=output_dir / "clusters" / "wren_patterns",
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        use_umap=args.use_umap,
        force=args.force,
    )

    clustered_df = features_df.copy()
    clustered_df["cluster_id"] = labels.astype(int)
    clustered_df["cluster_prob"] = probs.astype(float)
    clustered_df.to_csv(output_dir / "clusters" / "accepted_features_clustered.csv", index=False)

    umap_image = output_dir / "clusters" / "umap_patterns.png"
    _save_cluster_scatter(clustered_df, umap_emb, umap_image)

    representatives = representative_indices_by_cluster(
        clustered_df,
        pca_emb=pca_emb,
        cluster_col="cluster_id",
        per_cluster=args.representatives_per_cluster,
        diversity_col="recording_id",
        max_per_diversity_value=args.representative_max_per_recording,
    )
    reps_path = _export_representatives(clustered_df, representatives, output_dir)
    prototype_paths = export_cluster_prototypes(
        clustered_df=clustered_df,
        embeddings=embeddings,
        output_dir=output_dir,
        threshold_quantile=args.prototype_threshold_quantile,
    )
    merge_suggestions_df, merge_suggestions_path = export_cluster_merge_suggestions(
        clustered_df=clustered_df,
        embeddings=embeddings,
        output_dir=output_dir,
        suggestions_per_cluster=args.merge_suggestions_per_cluster,
        max_pairs=args.merge_suggestions_max_pairs,
    )
    same_different_path = export_same_different_review_template(
        clustered_df=clustered_df,
        representatives=representatives,
        merge_suggestions_df=merge_suggestions_df,
        output_dir=output_dir,
    )

    cluster_mean_specs = create_cluster_mean_spectrograms(
        clustered_df,
        specs_dir=output_dir / "specs",
        reports_dir=output_dir / "reports",
    )
    dist_plots = create_distribution_plots(clustered_df, output_dir / "reports")
    overview_rows = build_cluster_overview_rows(clustered_df)
    analysis_plots: list[dict[str, str]] = []
    recording_heatmap = create_cluster_recording_heatmap(clustered_df, output_dir / "reports")
    if recording_heatmap is not None:
        analysis_plots.append(
            {
                "title": "Cluster vs Recording Heatmap",
                "description": "Shows whether a cluster is spread across many recordings or dominated by one source file.",
                "path": recording_heatmap.resolve().as_posix(),
            }
        )
    source_balance = create_cluster_source_balance_plot(clustered_df, output_dir / "reports")
    if source_balance is not None:
        analysis_plots.append(
            {
                "title": "Cluster Source Balance",
                "description": "Compares how many recordings feed each cluster and how much one dominant recording controls it.",
                "path": source_balance.resolve().as_posix(),
            }
        )
    feature_heatmap = create_cluster_feature_heatmap(clustered_df, output_dir / "reports")
    if feature_heatmap is not None:
        analysis_plots.append(
            {
                "title": "Cluster Feature Profiles",
                "description": "Z-scored cluster means across the main interpretable audio features, useful for spotting whether clusters differ in a stable way.",
                "path": feature_heatmap.resolve().as_posix(),
            }
        )
    quality_plot = create_cluster_quality_plot(clustered_df, output_dir / "reports")
    if quality_plot is not None:
        analysis_plots.append(
            {
                "title": "Confidence and Duration by Cluster",
                "description": "Helps check whether a cluster is made of weak fragments or clear BirdNET-friendly events.",
                "path": quality_plot.resolve().as_posix(),
            }
        )
    write_cluster_playlists(clustered_df, events_dir=audio_dir, reports_dir=output_dir / "reports")
    review_sheet_paths = export_cluster_review_sheets(
        clustered_df,
        representatives=representatives,
        reports_dir=output_dir / "reports",
        specs_dir=output_dir / "specs",
    )
    review_bundle_dir = export_review_bundle(
        clustered_df,
        representatives=representatives,
        reports_dir=output_dir / "reports",
        specs_dir=output_dir / "specs",
    )
    report_path = generate_html_report(
        df=clustered_df,
        representatives=representatives,
        reports_dir=output_dir / "reports",
        events_dir=audio_dir,
        specs_dir=output_dir / "specs",
        umap_image=umap_image,
        dist_plots=dist_plots,
        cluster_mean_specs=cluster_mean_specs,
        analysis_plots=analysis_plots,
        review_sheet_paths=review_sheet_paths,
        cluster_overview_rows=overview_rows,
    )

    summary = _build_cluster_summary(clustered_df)
    summary.update(
        {
            "pattern_run_dir": pattern_run_dir.as_posix(),
            "analysis_unit": analysis_unit,
            "audio_dir": audio_dir.as_posix(),
            "embedding_backend": str(meta["embedding_meta"].get("backend", "")),
            "embedding_dim": int(meta["embedding_meta"].get("embedding_dim", 0)),
            "min_cluster_size": int(args.min_cluster_size),
            "min_samples": int(args.min_samples),
            "representatives_per_cluster": int(args.representatives_per_cluster),
            "prototype_threshold_quantile": float(args.prototype_threshold_quantile),
            "prototype_metadata": prototype_paths["prototype_metadata"].as_posix(),
            "prototype_vectors": prototype_paths["prototype_vectors"].as_posix(),
            "cluster_merge_suggestions": merge_suggestions_path.as_posix(),
            "same_different_review_pairs": same_different_path.as_posix(),
        }
    )
    save_json(output_dir / "reports" / "cluster_summary.json", summary)

    print("Pattern clustering complete")
    print(f"- Pattern run dir: {pattern_run_dir.as_posix()}")
    print(f"- Analysis unit: {analysis_unit}")
    print(f"- Accepted units: {len(clustered_df)}")
    print(f"- Clusters excluding noise: {summary['n_clusters_excluding_noise']}")
    print(f"- Noise events: {summary['n_noise_events']}")
    print(f"- Clustered features: {(output_dir / 'clusters' / 'accepted_features_clustered.csv').as_posix()}")
    print(f"- Cluster summary: {(output_dir / 'reports' / 'cluster_summary.json').as_posix()}")
    print(f"- Cluster representatives: {reps_path.as_posix()}")
    print(f"- Cluster prototypes: {prototype_paths['prototype_metadata'].as_posix()}")
    print(f"- Merge suggestions: {merge_suggestions_path.as_posix()}")
    print(f"- Same/different review pairs: {same_different_path.as_posix()}")
    print(f"- Review bundle: {review_bundle_dir.as_posix()}")
    print(f"- HTML report: {report_path.as_posix()}")


if __name__ == "__main__":
    main()
