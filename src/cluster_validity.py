from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import ensure_dir, require_google_drive_output, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mark within-species clusters as robust or recording-biased and export a filtered review bundle."
    )
    parser.add_argument("--cluster_run_dir", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Drive folder for the filtered validity review outputs.",
    )
    parser.add_argument(
        "--min_recordings",
        type=int,
        default=4,
        help="Minimum different source recordings required for a cluster to be treated as robust.",
    )
    parser.add_argument(
        "--min_units",
        type=int,
        default=8,
        help="Minimum phrase units required for a cluster to be treated as robust.",
    )
    parser.add_argument(
        "--max_dominant_recording_share",
        type=float,
        default=0.5,
        help="Maximum share allowed from the dominant recording for a robust cluster.",
    )
    parser.add_argument("--examples_per_cluster", type=int, default=8)
    parser.add_argument("--max_per_recording", type=int, default=2)
    parser.add_argument("--include_noise_examples", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _safe_name(value: Any) -> str:
    keep = []
    for char in str(value):
        keep.append(char if char.isalnum() or char in {"-", "_", "."} else "_")
    return "".join(keep).strip("_") or "item"


def _prepare_output_dir(output_dir: Path, *, force: bool) -> None:
    marker = output_dir / ".cluster_validity_output"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force or not marker.exists():
            raise FileExistsError(
                f"Output dir already exists and is not empty: {output_dir.as_posix()}. "
                "Choose a new folder, or rerun with --force if this folder was created by cluster_validity."
            )
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    ensure_dir(output_dir)
    marker.write_text("cluster_validity\n", encoding="utf-8")


def _audio_path(row: pd.Series) -> Path | None:
    for col in ("embedding_audio_path", "audio_path"):
        value = str(row.get(col, "") or "")
        if not value:
            continue
        candidate = Path(value)
        if candidate.exists():
            return candidate
    return None


def _spec_path(row: pd.Series, *, cluster_run_dir: Path) -> Path | None:
    spec_name = Path(str(row.get("spec_png", "") or "")).name
    if not spec_name:
        return None
    candidate = cluster_run_dir / "specs" / spec_name
    return candidate if candidate.exists() else None


def _status_for_cluster(
    *,
    cluster_id: int,
    n_units: int,
    n_recordings: int,
    dominant_share: float,
    min_recordings: int,
    min_units: int,
    max_dominant_recording_share: float,
) -> str:
    if int(cluster_id) == -1:
        return "noise_unresolved"
    if n_units < int(min_units):
        return "candidate_too_few_units"
    if n_recordings < int(min_recordings):
        return "candidate_recording_biased"
    if dominant_share > float(max_dominant_recording_share):
        return "candidate_dominant_recording"
    return "valid_for_review"


def _cluster_summary(
    df: pd.DataFrame,
    *,
    min_recordings: int,
    min_units: int,
    max_dominant_recording_share: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cluster_id in sorted(int(value) for value in df["cluster_id"].dropna().astype(int).unique()):
        subset = df[df["cluster_id"].astype(int) == int(cluster_id)].copy()
        recording_counts = (
            subset["recording_id"].fillna("").astype(str).value_counts()
            if "recording_id" in subset.columns and not subset.empty
            else pd.Series(dtype=int)
        )
        dominant_recording_id = str(recording_counts.index[0]) if not recording_counts.empty else ""
        dominant_share = float(recording_counts.iloc[0] / max(1, len(subset))) if not recording_counts.empty else 0.0
        n_units = int(len(subset))
        n_recordings = int(subset["recording_id"].nunique()) if "recording_id" in subset.columns else 0
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_validity_status": _status_for_cluster(
                    cluster_id=int(cluster_id),
                    n_units=n_units,
                    n_recordings=n_recordings,
                    dominant_share=dominant_share,
                    min_recordings=min_recordings,
                    min_units=min_units,
                    max_dominant_recording_share=max_dominant_recording_share,
                ),
                "n_units": n_units,
                "n_recordings": n_recordings,
                "dominant_recording_id": dominant_recording_id,
                "dominant_recording_share": dominant_share,
                "mean_species_top1_conf": float(pd.to_numeric(subset.get("species_top1_conf"), errors="coerce").mean()),
                "mean_duration_s": float(pd.to_numeric(subset.get("duration_s"), errors="coerce").mean()),
                "review_keep_yes_no": "",
                "review_call_type_name": "",
                "review_notes": "",
            }
        )
    return pd.DataFrame(rows).sort_values(["cluster_validity_status", "cluster_id"]).reset_index(drop=True)


def _choose_examples(df: pd.DataFrame, *, count: int, max_per_recording: int) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ("cluster_prob", "species_top1_conf"):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out = out.sort_values(["cluster_prob", "species_top1_conf", "clip_id"], ascending=[False, False, True])

    if max_per_recording <= 0 or "recording_id" not in out.columns:
        return out.head(max(0, int(count)))

    selected: list[Any] = []
    seen: Counter[str] = Counter()
    for idx, row in out.iterrows():
        recording_id = str(row.get("recording_id", "")) or "__missing__"
        if seen[recording_id] >= max_per_recording:
            continue
        selected.append(idx)
        seen[recording_id] += 1
        if len(selected) >= count:
            break

    if len(selected) < count:
        selected_set = set(selected)
        for idx, _ in out.iterrows():
            if idx in selected_set:
                continue
            selected.append(idx)
            if len(selected) >= count:
                break

    return out.loc[selected]


def _copy_examples(
    *,
    examples: pd.DataFrame,
    cluster_id: int,
    status: str,
    destination: Path,
    cluster_run_dir: Path,
) -> list[dict[str, Any]]:
    ensure_dir(destination)
    rows: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(examples.iterrows(), start=1):
        clip_id = _safe_name(row.get("clip_id", f"clip_{rank}"))
        recording_id = _safe_name(row.get("recording_id", "recording_unknown"))
        prefix = f"{rank:02d}__cluster_{cluster_id}__rec_{recording_id}__{clip_id}"
        audio = _audio_path(row)
        spec = _spec_path(row, cluster_run_dir=cluster_run_dir)
        copied_audio = ""
        copied_spec = ""
        if audio is not None:
            audio_dst = destination / f"{prefix}.wav"
            shutil.copy2(audio, audio_dst)
            copied_audio = audio_dst.as_posix()
        if spec is not None:
            spec_dst = destination / f"{prefix}.png"
            shutil.copy2(spec, spec_dst)
            copied_spec = spec_dst.as_posix()
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_validity_status": status,
                "rank": int(rank),
                "clip_id": str(row.get("clip_id", "")),
                "recording_id": str(row.get("recording_id", "")),
                "copied_audio_path": copied_audio,
                "copied_spec_path": copied_spec,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    cluster_run_dir = args.cluster_run_dir.resolve()
    output_dir = require_google_drive_output(args.output_dir, purpose="--output_dir")
    _prepare_output_dir(output_dir, force=bool(args.force))

    clustered_csv = cluster_run_dir / "clusters" / "accepted_features_clustered.csv"
    if not clustered_csv.exists():
        raise FileNotFoundError(f"Clustered features CSV not found: {clustered_csv.as_posix()}")

    clustered_df = pd.read_csv(clustered_csv)
    summary_df = _cluster_summary(
        clustered_df,
        min_recordings=int(args.min_recordings),
        min_units=int(args.min_units),
        max_dominant_recording_share=float(args.max_dominant_recording_share),
    )
    status_by_cluster = dict(zip(summary_df["cluster_id"].astype(int), summary_df["cluster_validity_status"]))
    clustered_df["cluster_validity_status"] = clustered_df["cluster_id"].astype(int).map(status_by_cluster)

    summary_path = output_dir / "cluster_validity_summary.csv"
    annotated_path = output_dir / "accepted_features_clustered_with_validity.csv"
    summary_df.to_csv(summary_path, index=False)
    clustered_df.to_csv(annotated_path, index=False)

    bundle_dir = output_dir / "review_bundle_validity_filtered"
    copied_rows: list[dict[str, Any]] = []
    for _, summary in summary_df.iterrows():
        cluster_id = int(summary["cluster_id"])
        status = str(summary["cluster_validity_status"])
        if status == "noise_unresolved" and not args.include_noise_examples:
            continue
        status_dir = bundle_dir / status
        cluster_dir = status_dir / f"cluster_{cluster_id}"
        examples = _choose_examples(
            clustered_df[clustered_df["cluster_id"].astype(int) == cluster_id],
            count=int(args.examples_per_cluster),
            max_per_recording=int(args.max_per_recording),
        )
        copied_rows.extend(
            _copy_examples(
                examples=examples,
                cluster_id=cluster_id,
                status=status,
                destination=cluster_dir,
                cluster_run_dir=cluster_run_dir,
            )
        )

    copied_index_path = output_dir / "validity_review_copied_files.csv"
    pd.DataFrame(copied_rows).to_csv(copied_index_path, index=False)

    counts = summary_df["cluster_validity_status"].value_counts().sort_index().to_dict()
    guide_path = output_dir / "README.md"
    guide_path.write_text(
        "\n".join(
            [
                "# Cluster Validity Review",
                "",
                "This filters machine clusters before interpretation.",
                "",
                f"`valid_for_review` requires at least {int(args.min_units)} phrase units, "
                f"at least {int(args.min_recordings)} different source recordings, and a dominant-recording share "
                f"no greater than {float(args.max_dominant_recording_share):.2f}.",
                "",
                "Start with `review_bundle_validity_filtered/valid_for_review/`.",
                "Treat candidate folders as recording-biased or under-supported unless listening strongly suggests otherwise.",
                "Noise remains unresolved and is not forced into a call type.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    json_path = output_dir / "cluster_validity_summary.json"
    save_json(
        json_path,
        {
            "artifact_type": "cluster_validity_review",
            "cluster_run_dir": cluster_run_dir.as_posix(),
            "output_dir": output_dir.as_posix(),
            "min_recordings": int(args.min_recordings),
            "min_units": int(args.min_units),
            "max_dominant_recording_share": float(args.max_dominant_recording_share),
            "status_counts": counts,
            "cluster_validity_summary_csv": summary_path.as_posix(),
            "annotated_features_csv": annotated_path.as_posix(),
            "review_bundle": bundle_dir.as_posix(),
            "copied_files_csv": copied_index_path.as_posix(),
            "guide": guide_path.as_posix(),
        },
    )

    print("Cluster validity review complete")
    print(f"- Status counts: {counts}")
    print(f"- Summary: {summary_path.as_posix()}")
    print(f"- Annotated features: {annotated_path.as_posix()}")
    print(f"- Review bundle: {bundle_dir.as_posix()}")


if __name__ == "__main__":
    main()
