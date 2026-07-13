from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import ensure_dir, require_google_drive_output, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Suggest nearest-prototype candidates for HDBSCAN noise units and export a listening bundle."
    )
    parser.add_argument("--cluster_run_dir", type=Path, required=True)
    parser.add_argument("--pattern_run_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--threshold_multiplier",
        type=float,
        default=1.1,
        help="Suggest a candidate if best prototype distance is within this multiple of that cluster's prototype threshold.",
    )
    parser.add_argument(
        "--max_candidate_distance",
        type=float,
        default=0.0,
        help=(
            "Optional absolute cosine-distance limit for candidate suggestions. "
            "If >0, a noise clip can be suggested when it passes either this limit or the prototype threshold limit."
        ),
    )
    parser.add_argument(
        "--min_margin_to_second",
        type=float,
        default=0.05,
        help="Suggest only if second-best prototype distance minus best distance is at least this value.",
    )
    parser.add_argument(
        "--reference_examples_per_cluster",
        type=int,
        default=3,
        help="Reference examples copied into each review folder from the existing non-noise clusters.",
    )
    parser.add_argument(
        "--review_max_per_recording",
        type=int,
        default=1,
        help="Maximum review examples to copy from the same source recording within each cluster folder. Use 0 for no cap.",
    )
    parser.add_argument(
        "--include_unassigned_noise",
        action="store_true",
        help="Also copy rejected noise candidates into an unassigned_noise folder.",
    )
    parser.add_argument(
        "--apply_assignments",
        action="store_true",
        help=(
            "Actually promote accepted candidates into soft_cluster_id. "
            "Leave unset for the safer candidate-review-only mode."
        ),
    )
    return parser.parse_args()


def _unit_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def _safe_name(text: str) -> str:
    keep = []
    for char in str(text):
        keep.append(char if char.isalnum() or char in {"-", "_", "."} else "_")
    return "".join(keep).strip("_") or "item"


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def _diverse_rows_by_recording(df: pd.DataFrame, *, max_per_recording: int) -> pd.DataFrame:
    if max_per_recording <= 0 or "recording_id" not in df.columns:
        return df
    seen: dict[str, int] = {}
    keep_indices: list[Any] = []
    for idx, row in df.iterrows():
        recording_id = str(row.get("recording_id", ""))
        count = seen.get(recording_id, 0)
        if count >= max_per_recording:
            continue
        seen[recording_id] = count + 1
        keep_indices.append(idx)
    return df.loc[keep_indices]


def _audio_path(row: pd.Series, *, pattern_run_dir: Path) -> Path | None:
    event_wav = Path(str(row.get("event_wav", ""))).name
    candidates = []
    if event_wav:
        candidates.append(pattern_run_dir / "phrases" / event_wav)
    raw_path = str(row.get("embedding_audio_path", "") or "")
    if raw_path:
        candidates.append(Path(raw_path))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _spec_path(row: pd.Series, *, cluster_run_dir: Path) -> Path | None:
    spec_png = Path(str(row.get("spec_png", ""))).name
    if not spec_png:
        return None
    candidate = cluster_run_dir / "specs" / spec_png
    return candidate if candidate.exists() else None


def _load_prototype_thresholds(metadata_path: Path) -> dict[int, float]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        int(cluster["cluster_id"]): float(cluster["cosine_distance_threshold"])
        for cluster in payload.get("clusters", [])
    }


def _candidate_rows(
    *,
    clustered_df: pd.DataFrame,
    embeddings: np.ndarray,
    prototype_npz: Path,
    prototype_metadata: Path,
    threshold_multiplier: float,
    max_candidate_distance: float,
    min_margin_to_second: float,
) -> pd.DataFrame:
    prototypes = np.load(prototype_npz)
    prototype_cluster_ids = prototypes["cluster_ids"].astype(int)
    centroid_unit = prototypes["centroids_unit"].astype(np.float32)
    thresholds = _load_prototype_thresholds(prototype_metadata)

    if len(clustered_df) != int(embeddings.shape[0]):
        raise ValueError("Clustered rows must match embedding rows.")
    if len(prototype_cluster_ids) == 0:
        raise ValueError("No non-noise prototypes found.")

    embedding_unit = _unit_rows(embeddings)
    distances = 1.0 - np.clip(embedding_unit @ centroid_unit.T, -1.0, 1.0)
    noise_indices = np.where(clustered_df["cluster_id"].astype(int).to_numpy() == -1)[0]

    rows: list[dict[str, Any]] = []
    for row_idx in noise_indices:
        row = clustered_df.iloc[int(row_idx)]
        order = np.argsort(distances[row_idx])
        best_pos = int(order[0])
        second_pos = int(order[1]) if len(order) > 1 else best_pos
        best_cluster = int(prototype_cluster_ids[best_pos])
        second_cluster = int(prototype_cluster_ids[second_pos])
        best_distance = float(distances[row_idx, best_pos])
        second_distance = float(distances[row_idx, second_pos])
        margin = float(second_distance - best_distance)
        threshold = float(thresholds.get(best_cluster, 0.0))
        threshold_limit = float(threshold * threshold_multiplier)
        passes_threshold_limit = bool(best_distance <= threshold_limit)
        passes_absolute_limit = bool(max_candidate_distance > 0 and best_distance <= float(max_candidate_distance))
        accepted = bool((passes_threshold_limit or passes_absolute_limit) and margin >= min_margin_to_second)
        accepted_by = ""
        if accepted and passes_threshold_limit:
            accepted_by = "prototype_threshold"
        elif accepted and passes_absolute_limit:
            accepted_by = "absolute_distance"
        rows.append(
            {
                "row_index": int(row_idx),
                "clip_id": str(row["clip_id"]),
                "recording_id": str(row.get("recording_id", "")),
                "original_cluster_id": -1,
                "soft_assigned_cluster_id": best_cluster if accepted else -1,
                "candidate_cluster_id": best_cluster,
                "candidate_distance": best_distance,
                "candidate_threshold": threshold,
                "candidate_threshold_limit": threshold_limit,
                "threshold_multiplier": float(threshold_multiplier),
                "max_candidate_distance": float(max_candidate_distance),
                "distance_to_threshold_ratio": best_distance / threshold if threshold > 0 else np.inf,
                "second_best_cluster_id": second_cluster,
                "second_best_distance": second_distance,
                "margin_to_second": margin,
                "accepted_by": accepted_by,
                "soft_assignment_status": "soft_assigned" if accepted else "unassigned_noise",
                "duration_s": float(row.get("duration_s", 0.0)),
                "species_top1_conf": float(row.get("species_top1_conf", 0.0)),
                "event_wav": str(row.get("event_wav", "")),
                "spec_png": str(row.get("spec_png", "")),
                "review_assignment_makes_sense_yes_no": "",
                "review_notes": "",
            }
        )
    return pd.DataFrame(rows).sort_values(["soft_assignment_status", "candidate_cluster_id", "candidate_distance"])


def _export_review_bundle(
    *,
    clustered_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    cluster_run_dir: Path,
    pattern_run_dir: Path,
    output_dir: Path,
    reference_examples_per_cluster: int,
    review_max_per_recording: int,
    include_unassigned_noise: bool,
) -> Path:
    bundle_dir = output_dir / "review_bundle_soft_assigned"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    ensure_dir(bundle_dir)

    copied_rows: list[dict[str, Any]] = []
    assigned_clusters = {
        int(value)
        for value in candidates_df.loc[
            candidates_df["soft_assignment_status"] == "soft_assigned", "soft_assigned_cluster_id"
        ].unique()
        if int(value) != -1
    }
    original_clusters = {
        int(value)
        for value in clustered_df["cluster_id"].dropna().astype(int).unique().tolist()
        if int(value) != -1
    }
    review_clusters = sorted(original_clusters | assigned_clusters)

    for cluster_id in review_clusters:
        cluster_dir = bundle_dir / f"cluster_{cluster_id}"
        ensure_dir(cluster_dir)
        reference_dir = cluster_dir / "reference_examples"
        assigned_dir = cluster_dir / "candidate_new_clips"
        ensure_dir(reference_dir)
        ensure_dir(assigned_dir)
        refs = (
            clustered_df[clustered_df["cluster_id"].astype(int) == int(cluster_id)]
            .sort_values("cluster_prob", ascending=False)
        )
        refs = _diverse_rows_by_recording(refs, max_per_recording=review_max_per_recording).head(
            max(0, int(reference_examples_per_cluster))
        )
        for ref_rank, (_, ref_row) in enumerate(refs.iterrows(), start=1):
            clip_id = _safe_name(ref_row["clip_id"])
            audio = _audio_path(ref_row, pattern_run_dir=pattern_run_dir)
            spec = _spec_path(ref_row, cluster_run_dir=cluster_run_dir)
            prefix = f"REFERENCE_CLUSTER_{cluster_id}__{ref_rank:02d}__{clip_id}"
            if audio is not None:
                _copy_if_exists(audio, reference_dir / f"{prefix}.wav")
            if spec is not None:
                _copy_if_exists(spec, reference_dir / f"{prefix}.png")

        assigned = candidates_df[
            (candidates_df["soft_assignment_status"] == "soft_assigned")
            & (candidates_df["soft_assigned_cluster_id"].astype(int) == int(cluster_id))
        ].sort_values("candidate_distance")
        assigned = _diverse_rows_by_recording(assigned, max_per_recording=review_max_per_recording)
        for rank, candidate in enumerate(assigned.itertuples(index=False), start=1):
            source_row = clustered_df.iloc[int(candidate.row_index)]
            clip_id = _safe_name(candidate.clip_id)
            audio = _audio_path(source_row, pattern_run_dir=pattern_run_dir)
            spec = _spec_path(source_row, cluster_run_dir=cluster_run_dir)
            prefix = (
                f"CANDIDATE_FOR_CLUSTER_{cluster_id}__{rank:02d}"
                f"__dist_{float(candidate.candidate_distance):.3f}"
                f"__margin_{float(candidate.margin_to_second):.3f}"
                f"__{clip_id}"
            )
            copied_audio = False
            copied_spec = False
            if audio is not None:
                copied_audio = _copy_if_exists(audio, assigned_dir / f"{prefix}.wav")
            if spec is not None:
                copied_spec = _copy_if_exists(spec, assigned_dir / f"{prefix}.png")
            copied_rows.append(
                {
                    "clip_id": candidate.clip_id,
                    "candidate_cluster_id": int(cluster_id),
                    "candidate_distance": float(candidate.candidate_distance),
                    "margin_to_second": float(candidate.margin_to_second),
                    "copied_audio": bool(copied_audio),
                    "copied_spectrogram": bool(copied_spec),
                    "review_folder": assigned_dir.as_posix(),
                    "review_prefix": prefix,
                }
            )

    if include_unassigned_noise:
        unassigned_dir = bundle_dir / "unassigned_noise"
        ensure_dir(unassigned_dir)
        unassigned = candidates_df[candidates_df["soft_assignment_status"] == "unassigned_noise"].sort_values(
            "candidate_distance"
        )
        for rank, candidate in enumerate(unassigned.itertuples(index=False), start=1):
            source_row = clustered_df.iloc[int(candidate.row_index)]
            clip_id = _safe_name(candidate.clip_id)
            audio = _audio_path(source_row, pattern_run_dir=pattern_run_dir)
            spec = _spec_path(source_row, cluster_run_dir=cluster_run_dir)
            prefix = (
                f"NOT_ASSIGNED__nearest_cluster_{int(candidate.candidate_cluster_id)}__{rank:02d}"
                f"__dist_{float(candidate.candidate_distance):.3f}"
                f"__margin_{float(candidate.margin_to_second):.3f}"
                f"__{clip_id}"
            )
            if audio is not None:
                _copy_if_exists(audio, unassigned_dir / f"{prefix}.wav")
            if spec is not None:
                _copy_if_exists(spec, unassigned_dir / f"{prefix}.png")

    pd.DataFrame(copied_rows).to_csv(output_dir / "soft_assigned_review_bundle_index.csv", index=False)
    (bundle_dir / "README.md").write_text(
        "\n".join(
            [
                "# Prototype-candidate review bundle",
                "",
                "This bundle tests whether HDBSCAN noise phrases might fit existing cluster prototypes.",
                "These are candidate suggestions only. Do not treat them as accepted cluster membership until they have been reviewed by listening.",
                "",
                "Inside each `cluster_*` folder:",
                "- `reference_examples/` contains existing examples from that cluster.",
                "- `candidate_new_clips/` contains clips that were originally HDBSCAN noise (`cluster_id = -1`) and are candidate additions.",
                "- Each subfolder is capped to one clip per source recording by default.",
                "",
                "Listen to the reference clips first, then the candidate clips. If a candidate sounds wrong for that cluster, mark it in `soft_assignment_candidates.csv`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return bundle_dir


def main() -> None:
    args = parse_args()
    cluster_run_dir = args.cluster_run_dir.resolve()
    pattern_run_dir = args.pattern_run_dir.resolve()
    output_dir = require_google_drive_output(args.output_dir, purpose="--output_dir")
    ensure_dir(output_dir)

    clustered_csv = cluster_run_dir / "clusters" / "accepted_features_clustered.csv"
    embeddings_npy = pattern_run_dir / "embeddings" / "accepted_embeddings.npy"
    prototype_npz = cluster_run_dir / "models" / "cluster_prototypes.npz"
    prototype_metadata = cluster_run_dir / "models" / "cluster_prototypes.json"

    clustered_df = pd.read_csv(clustered_csv)
    embeddings = np.load(embeddings_npy).astype(np.float32)
    candidates_df = _candidate_rows(
        clustered_df=clustered_df,
        embeddings=embeddings,
        prototype_npz=prototype_npz,
        prototype_metadata=prototype_metadata,
        threshold_multiplier=args.threshold_multiplier,
        max_candidate_distance=args.max_candidate_distance,
        min_margin_to_second=args.min_margin_to_second,
    )
    candidates_df["review_status"] = "candidate_only_unreviewed"
    candidates_path = output_dir / "soft_assignment_candidates.csv"
    candidates_df.to_csv(candidates_path, index=False)

    soft_df = clustered_df.copy()
    soft_df["original_cluster_id"] = soft_df["cluster_id"].astype(int)
    soft_df["soft_cluster_id"] = soft_df["original_cluster_id"]
    soft_df["soft_assignment_status"] = np.where(
        soft_df["original_cluster_id"].eq(-1), "unassigned_noise", "original_cluster"
    )
    soft_df["soft_assignment_distance"] = np.nan
    soft_df["soft_assignment_margin_to_second"] = np.nan
    soft_df["candidate_cluster_id"] = -1
    soft_df["candidate_assignment_status"] = "not_a_candidate"
    soft_df["candidate_assignment_note"] = "Existing HDBSCAN result retained."

    for candidate in candidates_df.itertuples(index=False):
        row_index = int(candidate.row_index)
        if str(candidate.soft_assignment_status) != "soft_assigned":
            continue
        soft_df.loc[row_index, "candidate_cluster_id"] = int(candidate.soft_assigned_cluster_id)
        soft_df.loc[row_index, "candidate_assignment_status"] = "candidate_only_unreviewed"
        soft_df.loc[
            row_index,
            "candidate_assignment_note",
        ] = "Nearest-prototype suggestion only; HDBSCAN noise is retained unless --apply_assignments is set."
        if args.apply_assignments:
            soft_df.loc[row_index, "soft_cluster_id"] = int(candidate.soft_assigned_cluster_id)
            soft_df.loc[row_index, "soft_assignment_status"] = "soft_assigned_from_noise"
        soft_df.loc[row_index, "soft_assignment_distance"] = float(candidate.candidate_distance)
        soft_df.loc[row_index, "soft_assignment_margin_to_second"] = float(candidate.margin_to_second)

    soft_features_path = output_dir / (
        "accepted_features_soft_assigned.csv" if args.apply_assignments else "candidate_features_with_nearest_prototype.csv"
    )
    soft_df.to_csv(soft_features_path, index=False)

    bundle_dir = _export_review_bundle(
        clustered_df=clustered_df,
        candidates_df=candidates_df,
        cluster_run_dir=cluster_run_dir,
        pattern_run_dir=pattern_run_dir,
        output_dir=output_dir,
        reference_examples_per_cluster=args.reference_examples_per_cluster,
        review_max_per_recording=args.review_max_per_recording,
        include_unassigned_noise=args.include_unassigned_noise,
    )

    assigned = candidates_df[candidates_df["soft_assignment_status"] == "soft_assigned"]
    summary_by_cluster = (
        assigned.groupby("soft_assigned_cluster_id")
        .agg(
            n_soft_assigned=("clip_id", "count"),
            mean_distance=("candidate_distance", "mean"),
            mean_margin=("margin_to_second", "mean"),
        )
        .reset_index()
        .sort_values("soft_assigned_cluster_id")
    )
    summary_by_cluster_path = output_dir / "soft_assignment_summary_by_cluster.csv"
    summary_by_cluster.to_csv(summary_by_cluster_path, index=False)

    summary_path = output_dir / "soft_assignment_summary.json"
    save_json(
        summary_path,
        {
            "artifact_type": "prototype_soft_assignment",
            "cluster_run_dir": cluster_run_dir.as_posix(),
            "pattern_run_dir": pattern_run_dir.as_posix(),
            "threshold_multiplier": float(args.threshold_multiplier),
            "max_candidate_distance": float(args.max_candidate_distance),
            "min_margin_to_second": float(args.min_margin_to_second),
            "review_max_per_recording": int(args.review_max_per_recording),
            "n_noise_candidates": int(len(candidates_df)),
            "n_soft_assigned": int(len(assigned)),
            "n_remaining_noise": int(len(candidates_df) - len(assigned)),
            "assignments_applied": bool(args.apply_assignments),
            "safety_note": (
                "Prototype nearest-neighbour results are candidate suggestions only unless --apply_assignments is set. "
                "Use listening review before treating them as call-pattern labels."
            ),
            "soft_assignment_candidates_csv": candidates_path.as_posix(),
            "candidate_features_csv": soft_features_path.as_posix(),
            "soft_assignment_summary_by_cluster_csv": summary_by_cluster_path.as_posix(),
            "review_bundle": bundle_dir.as_posix(),
        },
    )

    print("Prototype candidate review complete")
    print(f"- Noise candidates: {len(candidates_df)}")
    print(f"- Candidate suggestions: {len(assigned)}")
    print(f"- Remaining unassigned noise candidates: {len(candidates_df) - len(assigned)}")
    print(f"- Assignments applied: {bool(args.apply_assignments)}")
    print(f"- Candidates: {candidates_path.as_posix()}")
    print(f"- Candidate features: {soft_features_path.as_posix()}")
    print(f"- Summary by cluster: {summary_by_cluster_path.as_posix()}")
    print(f"- Review bundle: {bundle_dir.as_posix()}")


if __name__ == "__main__":
    main()
