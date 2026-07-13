from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from .utils import ensure_dir, require_google_drive_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a per-species semantic library from assigned clips plus reference anchors. "
            "This keeps labels such as juvenile_begging available even when no observed clips "
            "were assigned to that label in the current corpus."
        )
    )
    parser.add_argument("--assignments_csv", type=Path, required=True)
    parser.add_argument("--anchors_manifest_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--label_map_csv", type=Path, default=None)
    return parser.parse_args()


def _load_label_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    df = pd.read_csv(path)
    required = {"source_label", "library_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Label map CSV missing required columns: {', '.join(missing)}")
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        src = str(row["source_label"]).strip()
        dst = str(row["library_label"]).strip()
        if src and dst:
            out[src] = dst
    return out


def _apply_label_map(series: pd.Series, label_map: dict[str, str]) -> pd.Series:
    values = series.fillna("").astype(str).str.strip()
    if not label_map:
        return values
    return values.map(lambda x: label_map.get(x, x))


def _safe_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if not dst.exists():
        shutil.copy2(src, dst)


def _copy_observed_audio(assignments_df: pd.DataFrame, output_dir: Path) -> None:
    root = output_dir / "semantic_clusters"
    for _, row in assignments_df.iterrows():
        label = str(row["library_label"]).strip()
        if not label:
            continue
        src = Path(str(row["resolved_audio_path"])).expanduser()
        if not src.exists():
            continue
        cluster_id = int(row["cluster_id"]) if pd.notna(row.get("cluster_id")) else -1
        prefix = f"cluster_{cluster_id:02d}" if cluster_id >= 0 else "cluster_noise"
        score = float(row.get("best_similarity", 0.0))
        margin = float(row.get("similarity_margin", 0.0))
        out_name = f"{prefix}__score_{score:.3f}__margin_{margin:.3f}__{src.name}"
        dst = root / label / "observed_clips" / out_name
        _safe_copy(src, dst)


def _copy_reference_audio(anchor_df: pd.DataFrame, output_dir: Path) -> None:
    root = output_dir / "semantic_clusters"
    for _, row in anchor_df.iterrows():
        label = str(row["library_label"]).strip()
        if not label:
            continue
        src = Path(str(row["audio_path"])).expanduser()
        if not src.exists():
            continue
        anchor_id = str(row["anchor_id"]).strip()
        out_name = f"reference_anchor__{anchor_id}__{src.name}"
        dst = root / label / "reference_support" / out_name
        _safe_copy(src, dst)


def main() -> None:
    args = parse_args()
    output_dir = require_google_drive_output(args.output_dir, purpose="--output_dir")
    ensure_dir(output_dir)
    ensure_dir(output_dir / "reports")

    assignments_df = pd.read_csv(args.assignments_csv.resolve())
    anchors_df = pd.read_csv(args.anchors_manifest_csv.resolve())
    label_map = _load_label_map(args.label_map_csv.resolve() if args.label_map_csv else None)

    if "final_anchor_label" not in assignments_df.columns:
        raise ValueError("Assignments CSV must contain final_anchor_label.")
    required_anchor_cols = {"anchor_id", "anchor_label", "audio_path"}
    missing_anchor_cols = sorted(required_anchor_cols - set(anchors_df.columns))
    if missing_anchor_cols:
        raise ValueError(
            f"Anchors manifest CSV missing required columns: {', '.join(missing_anchor_cols)}"
        )

    assignments_df = assignments_df.copy()
    assignments_df["library_label"] = _apply_label_map(assignments_df["final_anchor_label"], label_map)
    anchors_df = anchors_df.copy()
    anchors_df["library_label"] = _apply_label_map(anchors_df["anchor_label"], label_map)

    _copy_observed_audio(assignments_df, output_dir=output_dir)
    _copy_reference_audio(anchors_df, output_dir=output_dir)

    observed_summary = (
        assignments_df.groupby("library_label")
        .agg(
            n_observed_clips=("clip_id", "count"),
            n_observed_recordings=("recording_id", "nunique"),
        )
        .reset_index()
    )
    reference_summary = (
        anchors_df.groupby("library_label")
        .agg(n_reference_anchors=("anchor_id", "count"))
        .reset_index()
    )
    summary = observed_summary.merge(reference_summary, on="library_label", how="outer").fillna(0)
    summary["n_observed_clips"] = summary["n_observed_clips"].astype(int)
    summary["n_observed_recordings"] = summary["n_observed_recordings"].astype(int)
    summary["n_reference_anchors"] = summary["n_reference_anchors"].astype(int)
    summary["support_mode"] = summary.apply(
        lambda row: (
            "observed_and_reference"
            if row["n_observed_clips"] > 0 and row["n_reference_anchors"] > 0
            else "reference_only"
            if row["n_observed_clips"] == 0 and row["n_reference_anchors"] > 0
            else "observed_only"
        ),
        axis=1,
    )
    summary = summary.sort_values("library_label").reset_index(drop=True)

    assignments_df.to_csv(output_dir / "reports" / "library_assignments.csv", index=False)
    anchors_df.to_csv(output_dir / "reports" / "library_reference_manifest.csv", index=False)
    summary.to_csv(output_dir / "reports" / "semantic_library_summary.csv", index=False)

    print(f"Wrote semantic library to {output_dir.as_posix()}")


if __name__ == "__main__":
    main()
