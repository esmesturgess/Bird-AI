from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .cluster import _cluster_single_set, representative_indices_by_cluster
from .utils import ensure_dir, require_google_drive_output, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast cluster review export with representative audio clips, without heavy feature/spec generation."
    )
    parser.add_argument("--pattern_run_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--baseline_cluster_csv", type=Path, default=None)
    parser.add_argument("--min_cluster_size", type=int, default=4)
    parser.add_argument("--min_samples", type=int, default=2)
    parser.add_argument("--representatives_per_cluster", type=int, default=6)
    parser.add_argument("--representative_max_per_recording", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_inputs(pattern_run_dir: Path) -> tuple[pd.DataFrame, np.ndarray, dict[str, object], dict[str, object]]:
    manifest_path = pattern_run_dir / "manifests" / "accepted_events_manifest.csv"
    summary_path = pattern_run_dir / "manifests" / "accepted_events_summary.json"
    emb_path = pattern_run_dir / "embeddings" / "accepted_embeddings.npy"
    meta_path = pattern_run_dir / "embeddings" / "embedding_meta.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path.as_posix()}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path.as_posix()}")
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_path.as_posix()}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Embedding meta not found: {meta_path.as_posix()}")
    manifest = pd.read_csv(manifest_path)
    emb = np.load(emb_path).astype(np.float32)
    if len(manifest) != emb.shape[0]:
        raise ValueError("Manifest rows do not match embedding rows.")
    return manifest, emb, _read_json(summary_path), _read_json(meta_path)


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


def _cluster_summary(df: pd.DataFrame) -> dict[str, object]:
    counts = Counter(int(v) for v in df["cluster_id"].astype(int).tolist())
    cluster_rows: list[dict[str, object]] = []
    for cluster_id in sorted(counts):
        subset = df[df["cluster_id"].astype(int) == int(cluster_id)]
        recording_counts = subset["recording_id"].fillna("").astype(str).value_counts() if "recording_id" in subset.columns else pd.Series(dtype=int)
        cluster_rows.append(
            {
                "cluster_id": int(cluster_id),
                "n_units": int(len(subset)),
                "n_recordings": int(subset["recording_id"].nunique()) if "recording_id" in subset.columns else 0,
                "dominant_recording_share": float(recording_counts.iloc[0] / max(1, len(subset))) if not recording_counts.empty else 0.0,
            }
        )
    return {
        "n_units": int(len(df)),
        "n_noise_units": int(counts.get(-1, 0)),
        "n_clusters_excluding_noise": int(sum(1 for cid in counts if cid != -1)),
        "cluster_rows": cluster_rows,
    }


def _export_representatives(
    *,
    df: pd.DataFrame,
    reps: dict[int, list[int]],
    summary: dict[str, object],
    output_dir: Path,
) -> Path:
    bundle_dir = output_dir / "review_bundle"
    ensure_dir(bundle_dir)
    rows: list[dict[str, object]] = []
    for cluster_id, idxs in sorted(reps.items(), key=lambda item: item[0]):
        cluster_dir = bundle_dir / f"cluster_{cluster_id:02d}" if cluster_id >= 0 else bundle_dir / "cluster_noise"
        ensure_dir(cluster_dir)
        for rank, row_idx in enumerate(idxs, start=1):
            row = df.iloc[int(row_idx)]
            audio_path = _resolve_audio_path(row, summary)
            out_name = f"{rank:02d}__{Path(audio_path).name}"
            shutil.copy2(audio_path, cluster_dir / out_name)
            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "rank": int(rank),
                    "clip_id": str(row["clip_id"]),
                    "recording_id": str(row.get("recording_id", "")),
                    "audio_path": audio_path.as_posix(),
                    "copied_audio": (cluster_dir / out_name).as_posix(),
                }
            )
    out_csv = output_dir / "reports" / "cluster_representatives.csv"
    ensure_dir(out_csv.parent)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def _load_baseline_map(path: Path | None) -> dict[str, int]:
    if path is None or not path.exists():
        return {}
    mapping: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            clip_id = str(row.get("clip_id", "")).strip()
            if not clip_id:
                continue
            mapping[clip_id] = int(float(row.get("cluster_id", "0")))
    return mapping


def main() -> None:
    args = parse_args()
    pattern_run_dir = args.pattern_run_dir.resolve()
    output_dir = require_google_drive_output(args.output_dir, purpose="--output_dir")
    ensure_dir(output_dir)
    ensure_dir(output_dir / "clusters")
    ensure_dir(output_dir / "reports")

    manifest, emb, summary, emb_meta = _load_inputs(pattern_run_dir)
    labels, probs, pca_emb, _ = _cluster_single_set(
        embeddings=emb,
        out_prefix=output_dir / "clusters" / "adapter_quick",
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        use_umap=False,
        force=args.force,
    )

    clustered_df = manifest.copy()
    clustered_df["cluster_id"] = labels.astype(int)
    clustered_df["cluster_prob"] = probs.astype(float)

    baseline_map = _load_baseline_map(args.baseline_cluster_csv.resolve() if args.baseline_cluster_csv else None)
    if baseline_map:
        clustered_df["baseline_cluster_id"] = clustered_df["clip_id"].map(lambda clip_id: baseline_map.get(str(clip_id), -999))

    clustered_path = output_dir / "clusters" / "accepted_features_clustered.csv"
    clustered_df.to_csv(clustered_path, index=False)

    reps = representative_indices_by_cluster(
        clustered_df,
        pca_emb=pca_emb,
        cluster_col="cluster_id",
        per_cluster=args.representatives_per_cluster,
        diversity_col="recording_id",
        max_per_diversity_value=args.representative_max_per_recording,
    )
    reps_csv = _export_representatives(df=clustered_df, reps=reps, summary=summary, output_dir=output_dir)

    if baseline_map:
        crosswalk = (
            clustered_df.groupby(["baseline_cluster_id", "cluster_id"])
            .size()
            .reset_index(name="n_units")
            .sort_values(["baseline_cluster_id", "cluster_id"])
        )
        crosswalk.to_csv(output_dir / "reports" / "baseline_vs_adapted_cluster_crosswalk.csv", index=False)

    save_json(
        output_dir / "reports" / "cluster_summary.json",
        {
            **_cluster_summary(clustered_df),
            "pattern_run_dir": pattern_run_dir.as_posix(),
            "embedding_backend": str(emb_meta.get("backend", "")),
            "representatives_csv": reps_csv.as_posix(),
        },
    )
    print(f"Wrote clustered assignments to {clustered_path.as_posix()}")
    print(f"Wrote representatives to {reps_csv.as_posix()}")


if __name__ == "__main__":
    main()
