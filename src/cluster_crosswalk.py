from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import ensure_dir, require_google_drive_output, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare an older window/event clustering run with a newer phrase-pooled clustering run."
    )
    parser.add_argument("--old_clustered_csv", type=Path, required=True)
    parser.add_argument("--new_clustered_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def _json_list(value: Any) -> list[Any]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON list, got: {text[:80]}")
    return parsed


def _json_dict(counter: Counter[int]) -> str:
    return json.dumps({str(k): int(v) for k, v in sorted(counter.items())})


def _as_int(value: Any, default: int = -1) -> int:
    if value is None or pd.isna(value):
        return default
    return int(value)


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    return float(value)


def _as_str(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _write_matrix_heatmap(matrix_df: pd.DataFrame, out_path: Path) -> Path | None:
    if matrix_df.empty:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    old_ids = sorted(int(value) for value in matrix_df["old_event_cluster_id"].unique())
    new_ids = sorted(int(value) for value in matrix_df["new_phrase_cluster_id"].unique())
    old_lookup = {cluster_id: idx for idx, cluster_id in enumerate(old_ids)}
    new_lookup = {cluster_id: idx for idx, cluster_id in enumerate(new_ids)}
    values = np.zeros((len(old_ids), len(new_ids)), dtype=float)

    for _, row in matrix_df.iterrows():
        old_idx = old_lookup[int(row["old_event_cluster_id"])]
        new_idx = new_lookup[int(row["new_phrase_cluster_id"])]
        values[old_idx, new_idx] = float(row["n_source_windows"])

    fig_w = max(6.0, 0.9 * len(new_ids) + 2.5)
    fig_h = max(5.0, 0.45 * len(old_ids) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    image = ax.imshow(values, aspect="auto", cmap="YlGnBu")
    ax.set_title("Old window clusters -> new phrase clusters")
    ax.set_xlabel("New phrase cluster")
    ax.set_ylabel("Old event/window cluster")
    ax.set_xticks(range(len(new_ids)), labels=[str(value) for value in new_ids])
    ax.set_yticks(range(len(old_ids)), labels=[str(value) for value in old_ids])
    ax.tick_params(axis="x", rotation=45)

    max_value = float(values.max()) if values.size else 0.0
    for old_idx in range(values.shape[0]):
        for new_idx in range(values.shape[1]):
            value = int(values[old_idx, new_idx])
            if value <= 0:
                continue
            color = "white" if max_value and value > max_value * 0.55 else "black"
            ax.text(new_idx, old_idx, str(value), ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(image, ax=ax, label="Source 3s windows")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def build_crosswalk(
    *,
    old_clustered_csv: Path,
    new_clustered_csv: Path,
    output_dir: Path,
) -> dict[str, Path | None]:
    old_df = pd.read_csv(old_clustered_csv)
    new_df = pd.read_csv(new_clustered_csv)

    required_old = {"clip_id", "cluster_id"}
    required_new = {"clip_id", "cluster_id", "pooled_event_ids_json"}
    missing_old = sorted(required_old - set(old_df.columns))
    missing_new = sorted(required_new - set(new_df.columns))
    if missing_old:
        raise ValueError(f"Old clustered CSV is missing columns: {missing_old}")
    if missing_new:
        raise ValueError(f"New clustered CSV is missing columns: {missing_new}")

    ensure_dir(output_dir)
    old_cluster_by_clip = {
        str(row["clip_id"]): int(row["cluster_id"])
        for _, row in old_df[["clip_id", "cluster_id"]].iterrows()
    }

    crosswalk_rows: list[dict[str, Any]] = []
    new_cluster_mix: dict[int, Counter[int]] = defaultdict(Counter)
    old_cluster_fate: dict[int, Counter[int]] = defaultdict(Counter)
    matrix_window_counts: Counter[tuple[int, int]] = Counter()
    matrix_phrase_ids: dict[tuple[int, int], set[str]] = defaultdict(set)

    for _, row in new_df.iterrows():
        phrase_id = _as_str(row["clip_id"])
        new_cluster_id = _as_int(row["cluster_id"])
        old_event_ids = [str(value) for value in _json_list(row["pooled_event_ids_json"])]
        old_event_clusters: list[int] = []
        missing_old_event_ids: list[str] = []

        for old_event_id in old_event_ids:
            old_cluster_id = old_cluster_by_clip.get(old_event_id)
            if old_cluster_id is None:
                missing_old_event_ids.append(old_event_id)
                continue
            old_event_clusters.append(int(old_cluster_id))
            new_cluster_mix[new_cluster_id][int(old_cluster_id)] += 1
            old_cluster_fate[int(old_cluster_id)][new_cluster_id] += 1
            matrix_window_counts[(int(old_cluster_id), new_cluster_id)] += 1
            matrix_phrase_ids[(int(old_cluster_id), new_cluster_id)].add(phrase_id)

        counts = Counter(old_event_clusters)
        dominant_old_cluster = ""
        dominant_share = 0.0
        if counts:
            dominant_old_cluster, dominant_count = counts.most_common(1)[0]
            dominant_share = float(dominant_count / sum(counts.values()))

        crosswalk_rows.append(
            {
                "phrase_clip_id": phrase_id,
                "new_phrase_cluster_id": new_cluster_id,
                "new_cluster_prob": _as_float(row.get("cluster_prob")),
                "recording_id": _as_str(row.get("recording_id")),
                "start_s": _as_float(row.get("start_s")),
                "end_s": _as_float(row.get("end_s")),
                "duration_s": _as_float(row.get("duration_s")),
                "pooled_from_n_events": int(len(old_event_ids)),
                "mapped_old_event_count": int(len(old_event_clusters)),
                "missing_old_event_count": int(len(missing_old_event_ids)),
                "old_event_ids_json": json.dumps(old_event_ids),
                "old_event_clusters_json": json.dumps(old_event_clusters),
                "old_event_cluster_counts_json": _json_dict(counts),
                "dominant_old_event_cluster": dominant_old_cluster,
                "dominant_old_event_cluster_share": dominant_share,
                "missing_old_event_ids_json": json.dumps(missing_old_event_ids),
                "phrase_audio_path": _as_str(row.get("embedding_audio_path")),
                "review_old_to_new_feels_right_yes_no": "",
                "review_notes": "",
            }
        )

    crosswalk_df = pd.DataFrame(crosswalk_rows)
    crosswalk_path = output_dir / "old_to_phrase_cluster_crosswalk.csv"
    crosswalk_df.to_csv(crosswalk_path, index=False)

    matrix_rows = []
    for (old_cluster_id, new_cluster_id), n_source_windows in sorted(matrix_window_counts.items()):
        phrase_ids = sorted(matrix_phrase_ids[(old_cluster_id, new_cluster_id)])
        matrix_rows.append(
            {
                "old_event_cluster_id": int(old_cluster_id),
                "new_phrase_cluster_id": int(new_cluster_id),
                "n_source_windows": int(n_source_windows),
                "n_phrase_units": int(len(phrase_ids)),
                "example_phrase_ids_json": json.dumps(phrase_ids[:8]),
                "review_notes": "",
            }
        )
    matrix_columns = [
        "old_event_cluster_id",
        "new_phrase_cluster_id",
        "n_source_windows",
        "n_phrase_units",
        "example_phrase_ids_json",
        "review_notes",
    ]
    matrix_df = pd.DataFrame(matrix_rows, columns=matrix_columns)
    if not matrix_df.empty:
        matrix_df = matrix_df.sort_values(
            ["new_phrase_cluster_id", "n_source_windows", "old_event_cluster_id"],
            ascending=[True, False, True],
        )
    matrix_path = output_dir / "old_cluster_to_new_phrase_cluster_matrix.csv"
    matrix_df.to_csv(matrix_path, index=False)
    heatmap_path = _write_matrix_heatmap(matrix_df, output_dir / "old_to_phrase_cluster_heatmap.png")

    new_mix_rows = []
    for new_cluster_id, counts in sorted(new_cluster_mix.items()):
        total = sum(counts.values())
        dominant = counts.most_common(1)[0][0] if counts else ""
        dominant_share = float(counts[dominant] / total) if counts else 0.0
        subset = crosswalk_df[crosswalk_df["new_phrase_cluster_id"] == new_cluster_id]
        new_mix_rows.append(
            {
                "new_phrase_cluster_id": int(new_cluster_id),
                "n_phrase_units": int(len(subset)),
                "n_recordings": int(subset["recording_id"].nunique()) if "recording_id" in subset else 0,
                "n_source_windows": int(total),
                "old_event_cluster_counts_json": _json_dict(counts),
                "dominant_old_event_cluster": dominant,
                "dominant_old_event_cluster_share": dominant_share,
                "review_cluster_feels_coherent_yes_no": "",
                "review_notes": "",
            }
        )
    new_mix_df = pd.DataFrame(new_mix_rows).sort_values("new_phrase_cluster_id")
    new_mix_path = output_dir / "new_phrase_cluster_to_old_cluster_mix.csv"
    new_mix_df.to_csv(new_mix_path, index=False)

    old_fate_rows = []
    for old_cluster_id, counts in sorted(old_cluster_fate.items()):
        total = sum(counts.values())
        dominant = counts.most_common(1)[0][0] if counts else ""
        dominant_share = float(counts[dominant] / total) if counts else 0.0
        old_fate_rows.append(
            {
                "old_event_cluster_id": int(old_cluster_id),
                "n_source_windows": int(total),
                "new_phrase_cluster_counts_json": _json_dict(counts),
                "dominant_new_phrase_cluster": dominant,
                "dominant_new_phrase_cluster_share": dominant_share,
                "review_notes": "",
            }
        )
    old_fate_df = pd.DataFrame(old_fate_rows).sort_values("old_event_cluster_id")
    old_fate_path = output_dir / "old_event_cluster_to_new_phrase_cluster_fate.csv"
    old_fate_df.to_csv(old_fate_path, index=False)

    summary_path = output_dir / "old_to_phrase_cluster_crosswalk_summary.json"
    save_json(
        summary_path,
        {
            "artifact_type": "old_to_phrase_cluster_crosswalk",
            "old_clustered_csv": old_clustered_csv.resolve().as_posix(),
            "new_clustered_csv": new_clustered_csv.resolve().as_posix(),
            "n_old_event_rows": int(len(old_df)),
            "n_new_phrase_rows": int(len(new_df)),
            "n_crosswalk_rows": int(len(crosswalk_df)),
            "n_unmapped_old_event_references": int(crosswalk_df["missing_old_event_count"].sum()),
            "crosswalk_csv": crosswalk_path.resolve().as_posix(),
            "matrix_csv": matrix_path.resolve().as_posix(),
            "heatmap_png": heatmap_path.resolve().as_posix() if heatmap_path is not None else "",
            "new_mix_csv": new_mix_path.resolve().as_posix(),
            "old_fate_csv": old_fate_path.resolve().as_posix(),
        },
    )

    guide_path = output_dir / "old_to_phrase_cluster_crosswalk_guide.md"
    guide_path.write_text(
        "\n".join(
            [
                "# Old-to-phrase cluster crosswalk",
                "",
                "This report compares the older 3-second BirdNET-window clustering to the newer phrase-pooled clustering.",
                "",
                "The old clusters were not manually merged. Adjacent BirdNET-positive windows from the same recording were first pooled into longer phrase units, then HDBSCAN reclustered those phrase embeddings.",
                "",
                "Files:",
                "- `old_to_phrase_cluster_crosswalk.csv`: one row per new phrase unit, with the old window IDs and old cluster IDs that fed into it.",
                "- `new_phrase_cluster_to_old_cluster_mix.csv`: one row per new phrase cluster, showing which old clusters it is made from.",
                "- `old_event_cluster_to_new_phrase_cluster_fate.csv`: one row per old cluster, showing where its source windows ended up.",
                "- `old_cluster_to_new_phrase_cluster_matrix.csv`: long-form matrix of old cluster to new cluster overlap.",
                "- `old_to_phrase_cluster_heatmap.png`: visual matrix of the same overlap.",
                "",
                "Review columns are left blank so Esme can mark whether the automatic regrouping sounds right.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "crosswalk": crosswalk_path,
        "matrix": matrix_path,
        "heatmap": heatmap_path,
        "new_mix": new_mix_path,
        "old_fate": old_fate_path,
        "summary": summary_path,
        "guide": guide_path,
    }


def main() -> None:
    args = parse_args()
    output_dir = require_google_drive_output(args.output_dir, purpose="--output_dir")
    paths = build_crosswalk(
        old_clustered_csv=args.old_clustered_csv,
        new_clustered_csv=args.new_clustered_csv,
        output_dir=output_dir,
    )
    print("Old-to-phrase crosswalk complete")
    for label, path in paths.items():
        if path is not None:
            print(f"- {label}: {path.resolve().as_posix()}")


if __name__ == "__main__":
    main()
