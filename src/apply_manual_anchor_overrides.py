from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .utils import ensure_dir, require_google_drive_output, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply manual cluster-level semantic label overrides to an existing "
            "anchor-label assignment export."
        )
    )
    parser.add_argument("--assignments_csv", type=Path, required=True)
    parser.add_argument("--overrides_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--copy_audio", action="store_true")
    return parser.parse_args()


def _load_overrides(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"cluster_id", "manual_anchor_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Override CSV missing required columns: {', '.join(missing)}")
    df = df.copy()
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce").astype("Int64")
    df["manual_anchor_label"] = df["manual_anchor_label"].fillna("").astype(str).str.strip()
    df = df[df["cluster_id"].notna() & (df["manual_anchor_label"] != "")]
    if df.empty:
        raise ValueError("Override CSV has no usable rows.")
    return df.reset_index(drop=True)


def _copy_audio(df: pd.DataFrame, output_dir: Path) -> None:
    audio_root = output_dir / "assigned_audio"
    ensure_dir(audio_root)
    for _, row in df.iterrows():
        label = str(row["final_anchor_label"])
        target_dir = audio_root / label
        ensure_dir(target_dir)
        src = Path(str(row["resolved_audio_path"]))
        if not src.exists():
            continue
        cluster_id = int(row["cluster_id"])
        prefix = f"cluster_{cluster_id:02d}" if cluster_id >= 0 else "cluster_noise"
        out_name = (
            f"{prefix}__score_{float(row['best_similarity']):.3f}"
            f"__margin_{float(row['similarity_margin']):.3f}__{src.name}"
        )
        target = target_dir / out_name
        if not target.exists():
            target.write_bytes(src.read_bytes())


def main() -> None:
    args = parse_args()
    assignments_csv = args.assignments_csv.resolve()
    overrides_csv = args.overrides_csv.resolve()
    output_dir = require_google_drive_output(args.output_dir, purpose="--output_dir")

    assignments_df = pd.read_csv(assignments_csv)
    overrides_df = _load_overrides(overrides_csv)
    override_map = {
        int(row["cluster_id"]): str(row["manual_anchor_label"])
        for _, row in overrides_df.iterrows()
    }

    out_df = assignments_df.copy()
    if "final_anchor_label" not in out_df.columns:
        if "assigned_anchor_label" not in out_df.columns:
            raise ValueError("Assignments CSV must contain final_anchor_label or assigned_anchor_label.")
        out_df["final_anchor_label"] = out_df["assigned_anchor_label"]
    if "assignment_strategy" not in out_df.columns:
        out_df["assignment_strategy"] = "existing_assignment"

    mask = out_df["cluster_id"].astype(int).isin(list(override_map))
    out_df.loc[mask, "final_anchor_label"] = out_df.loc[mask, "cluster_id"].astype(int).map(override_map)
    out_df.loc[mask, "assignment_strategy"] = "manual_cluster_override"

    ensure_dir(output_dir / "reports")
    out_csv = output_dir / "reports" / "all_clip_anchor_label_assignments.csv"
    out_df.to_csv(out_csv, index=False)
    overrides_df.to_csv(output_dir / "reports" / "manual_cluster_overrides.csv", index=False)

    label_summary = (
        out_df.groupby("final_anchor_label")
        .agg(
            n_clips=("clip_id", "count"),
            n_recordings=("recording_id", "nunique"),
            mean_best_similarity=("best_similarity", "mean"),
            mean_similarity_margin=("similarity_margin", "mean"),
        )
        .reset_index()
        .sort_values("final_anchor_label")
    )
    label_summary.to_csv(output_dir / "reports" / "anchor_label_summary.csv", index=False)

    cluster_label_summary = (
        out_df.groupby(["cluster_id", "final_anchor_label"])
        .size()
        .reset_index(name="n_clips")
        .sort_values(["cluster_id", "n_clips", "final_anchor_label"], ascending=[True, False, True])
    )
    cluster_label_summary.to_csv(output_dir / "reports" / "cluster_by_anchor_label_summary.csv", index=False)

    save_json(
        output_dir / "reports" / "manual_override_summary.json",
        {
            "assignments_csv": assignments_csv.as_posix(),
            "overrides_csv": overrides_csv.as_posix(),
            "n_clips": int(len(out_df)),
            "n_overridden_clusters": int(len(override_map)),
            "copy_audio": bool(args.copy_audio),
        },
    )

    if args.copy_audio:
        _copy_audio(out_df, output_dir=output_dir)

    print(f"Wrote manual-override assignments to {out_csv.as_posix()}")


if __name__ == "__main__":
    main()
