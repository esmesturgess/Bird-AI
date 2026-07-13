from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path

from .utils import ensure_dir, save_json


HARDCODED_PATTERN_RUNS = {
    "blackbird_pattern_clusters_aug_uk_birdnet_windows_v1": Path("data/runs/blackbird_pattern_layer_aug_uk_birdnet_windows_v1"),
    "house_sparrow_pattern_clusters_aug_uk_birdnet_windows_v1": Path("data/runs/house_sparrow_pattern_layer_aug_uk_birdnet_windows_v1"),
    "wood_pigeon_pattern_clusters_aug_uk_birdnet_windows_v1": Path("data/runs/wood_pigeon_pattern_layer_aug_uk_birdnet_windows_v1"),
    "wren_pattern_clusters_aug90_uk_v1": Path("data/runs/wren_pattern_layer_aug90_uk_birdnet_v1"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand manual cluster-level review labels into clip-level training pairs for adapter training."
    )
    parser.add_argument(
        "--manual_pair_csv",
        type=Path,
        default=Path("data/reviews/manual_cluster_pair_labels_v1.csv"),
    )
    parser.add_argument(
        "--manual_group_csv",
        type=Path,
        default=Path("data/reviews/manual_cluster_group_map_v1.csv"),
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("data/reviews/clip_pair_labels_v1.csv"),
    )
    parser.add_argument(
        "--summary_json",
        type=Path,
        default=Path("data/reviews/clip_pair_labels_v1_summary.json"),
    )
    parser.add_argument(
        "--max_pairs_per_cluster_pair",
        type=int,
        default=24,
        help="Maximum sampled clip pairs exported for each reviewed cross-cluster pair. Use 0 for all.",
    )
    parser.add_argument(
        "--max_within_cluster_pairs",
        type=int,
        default=24,
        help="Maximum sampled positive pairs exported from inside one reviewed cluster. Use 0 for all.",
    )
    parser.add_argument(
        "--include_unsure",
        action="store_true",
        help="Keep rows labelled unsure instead of skipping them.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path.as_posix()}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _resolve_cluster_run_dir(row: dict[str, str]) -> Path:
    source_run_dir = Path(str(row.get("source_run_dir", "")).strip()).expanduser()
    if source_run_dir.exists():
        return source_run_dir.resolve()

    source_run_name = str(row.get("source_run_name", "")).strip()
    local_candidate = Path("data/runs") / source_run_name
    if local_candidate.exists():
        return local_candidate.resolve()

    raise FileNotFoundError(
        f"Could not resolve cluster run dir for {source_run_name!r}. "
        f"Tried source_run_dir={source_run_dir.as_posix()} and local fallback={local_candidate.as_posix()}."
    )


def _resolve_pattern_run_dir(*, source_run_name: str, species_label: str) -> Path:
    hardcoded = HARDCODED_PATTERN_RUNS.get(source_run_name)
    if hardcoded is not None and hardcoded.exists():
        return hardcoded.resolve()

    local_guess = Path("data/runs") / source_run_name.replace("_pattern_clusters_", "_pattern_layer_")
    if local_guess.exists():
        return local_guess.resolve()

    species_guess = sorted(Path("data/runs").glob(f"{species_label}_pattern_layer*"))
    if len(species_guess) == 1:
        return species_guess[0].resolve()

    raise FileNotFoundError(
        f"Could not resolve a local pattern-layer run for source_run_name={source_run_name!r}, "
        f"species_label={species_label!r}."
    )


def _load_embedding_index(pattern_run_dir: Path) -> dict[str, int]:
    path = pattern_run_dir / "embeddings" / "embedding_index.csv"
    if not path.exists():
        raise FileNotFoundError(f"Embedding index not found: {path.as_posix()}")
    mapping: dict[str, int] = {}
    for row in _read_csv_rows(path):
        clip_id = str(row.get("clip_id", "")).strip()
        if not clip_id:
            continue
        mapping[clip_id] = int(float(row.get("embedding_row", "0")))
    return mapping


def _load_cluster_rows(cluster_run_dir: Path, embedding_index: dict[str, int]) -> dict[int, list[dict[str, object]]]:
    path = cluster_run_dir / "clusters" / "accepted_features_clustered.csv"
    if not path.exists():
        raise FileNotFoundError(f"Clustered features CSV not found: {path.as_posix()}")

    by_cluster: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in _read_csv_rows(path):
        clip_id = str(row.get("clip_id", "")).strip()
        cluster_id_text = str(row.get("cluster_id", "")).strip()
        if not clip_id or not cluster_id_text:
            continue
        if clip_id not in embedding_index:
            continue
        cluster_id = int(float(cluster_id_text))
        by_cluster[cluster_id].append(
            {
                "clip_id": clip_id,
                "embedding_row": int(embedding_index[clip_id]),
                "embedding_audio_path": str(row.get("embedding_audio_path", "")).strip(),
                "audio_path": str(row.get("audio_path", "")).strip(),
                "recording_id": str(row.get("recording_id", "")).strip(),
                "source_file": str(row.get("source_file", "")).strip(),
                "cluster_id": cluster_id,
            }
        )
    return by_cluster


def _sample_pairs(
    rows_a: list[dict[str, object]],
    rows_b: list[dict[str, object]],
    *,
    max_pairs: int,
    rng: random.Random,
    same_cluster: bool,
) -> list[tuple[dict[str, object], dict[str, object]]]:
    if same_cluster:
        candidates = [(rows_a[i], rows_a[j]) for i, j in combinations(range(len(rows_a)), 2)]
    else:
        candidates = [
            (row_a, row_b)
            for row_a, row_b in product(rows_a, rows_b)
            if str(row_a["clip_id"]) != str(row_b["clip_id"])
        ]

    if max_pairs > 0 and len(candidates) > max_pairs:
        return rng.sample(candidates, max_pairs)
    return candidates


def _build_pair_record(
    *,
    species_label: str,
    source_run_name: str,
    source_run_dir: str,
    cluster_run_dir: Path,
    pattern_run_dir: Path,
    cluster_a: int,
    cluster_b: int,
    pair_source: str,
    review_label: str,
    review_confidence: str,
    review_notes: str,
    row_a: dict[str, object],
    row_b: dict[str, object],
) -> dict[str, object]:
    return {
        "species_label": species_label,
        "review_label": review_label,
        "pair_source": pair_source,
        "source_run_name": source_run_name,
        "source_run_dir": source_run_dir,
        "cluster_run_dir": cluster_run_dir.as_posix(),
        "pattern_run_dir": pattern_run_dir.as_posix(),
        "cluster_id_a": int(cluster_a),
        "cluster_id_b": int(cluster_b),
        "clip_id_a": str(row_a["clip_id"]),
        "clip_id_b": str(row_b["clip_id"]),
        "embedding_row_a": int(row_a["embedding_row"]),
        "embedding_row_b": int(row_b["embedding_row"]),
        "recording_id_a": str(row_a.get("recording_id", "")),
        "recording_id_b": str(row_b.get("recording_id", "")),
        "audio_path_a": str(row_a.get("embedding_audio_path") or row_a.get("audio_path") or row_a.get("source_file") or ""),
        "audio_path_b": str(row_b.get("embedding_audio_path") or row_b.get("audio_path") or row_b.get("source_file") or ""),
        "review_confidence": str(review_confidence or ""),
        "review_notes": review_notes,
    }


def build_clip_pair_rows(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    pair_rows = _read_csv_rows(args.manual_pair_csv)
    group_rows = _read_csv_rows(args.manual_group_csv)
    rng = random.Random(int(args.seed))

    cluster_cache: dict[str, dict[int, list[dict[str, object]]]] = {}
    pattern_dir_cache: dict[str, Path] = {}

    def load_cluster_cache(source_run_name: str, species_label: str, source_run_dir: str) -> tuple[Path, Path, dict[int, list[dict[str, object]]]]:
        key = f"{species_label}::{source_run_name}"
        cluster_run_dir = _resolve_cluster_run_dir(
            {
                "source_run_name": source_run_name,
                "source_run_dir": source_run_dir,
            }
        )
        if key not in pattern_dir_cache:
            pattern_dir_cache[key] = _resolve_pattern_run_dir(
                source_run_name=source_run_name,
                species_label=species_label,
            )
        if key not in cluster_cache:
            cluster_cache[key] = _load_cluster_rows(
                cluster_run_dir=cluster_run_dir,
                embedding_index=_load_embedding_index(pattern_dir_cache[key]),
            )
        return cluster_run_dir, pattern_dir_cache[key], cluster_cache[key]

    records: list[dict[str, object]] = []

    for row in pair_rows:
        label = str(row.get("review_label", "")).strip().lower()
        if label == "unsure" and not args.include_unsure:
            continue
        if label not in {"same", "different", "unsure"}:
            continue

        species_label = str(row.get("species_label", "")).strip()
        source_run_name = str(row.get("source_run_name", "")).strip()
        source_run_dir = str(row.get("source_run_dir", "")).strip()
        cluster_a = int(float(row.get("cluster_a", "0")))
        cluster_b = int(float(row.get("cluster_b", "0")))
        cluster_run_dir, pattern_run_dir, cached_rows = load_cluster_cache(source_run_name, species_label, source_run_dir)
        rows_a = cached_rows.get(cluster_a, [])
        rows_b = cached_rows.get(cluster_b, [])
        if not rows_a or not rows_b:
            continue

        sampled = _sample_pairs(
            rows_a,
            rows_b,
            max_pairs=int(args.max_pairs_per_cluster_pair),
            rng=rng,
            same_cluster=(cluster_a == cluster_b),
        )
        for row_a, row_b in sampled:
            records.append(
                _build_pair_record(
                    species_label=species_label,
                    source_run_name=source_run_name,
                    source_run_dir=source_run_dir,
                    cluster_run_dir=cluster_run_dir,
                    pattern_run_dir=pattern_run_dir,
                    cluster_a=cluster_a,
                    cluster_b=cluster_b,
                    pair_source="manual_cross_cluster_review",
                    review_label=label,
                    review_confidence=str(row.get("review_confidence", "")).strip(),
                    review_notes=str(row.get("review_notes", "")).strip(),
                    row_a=row_a,
                    row_b=row_b,
                )
            )

    for row in group_rows:
        status = str(row.get("group_status", "")).strip().lower()
        if status != "confirmed":
            continue
        species_label = str(row.get("species_label", "")).strip()
        source_run_name = str(row.get("source_run_name", "")).strip()
        source_run_dir = str(row.get("source_run_dir", "")).strip()
        cluster_id = int(float(row.get("original_cluster_id", "0")))
        cluster_run_dir, pattern_run_dir, cached_rows = load_cluster_cache(source_run_name, species_label, source_run_dir)
        cluster_members = cached_rows.get(cluster_id, [])
        if len(cluster_members) < 2:
            continue

        sampled = _sample_pairs(
            cluster_members,
            cluster_members,
            max_pairs=int(args.max_within_cluster_pairs),
            rng=rng,
            same_cluster=True,
        )
        for row_a, row_b in sampled:
            records.append(
                _build_pair_record(
                    species_label=species_label,
                    source_run_name=source_run_name,
                    source_run_dir=source_run_dir,
                    cluster_run_dir=cluster_run_dir,
                    pattern_run_dir=pattern_run_dir,
                    cluster_a=cluster_id,
                    cluster_b=cluster_id,
                    pair_source="within_cluster_positive",
                    review_label="same",
                    review_confidence=str(row.get("review_confidence", "")).strip(),
                    review_notes=str(row.get("review_notes", "")).strip(),
                    row_a=row_a,
                    row_b=row_b,
                )
            )

    records.sort(
        key=lambda item: (
            str(item["species_label"]),
            str(item["review_label"]),
            str(item["source_run_name"]),
            int(item["cluster_id_a"]),
            int(item["cluster_id_b"]),
            str(item["clip_id_a"]),
            str(item["clip_id_b"]),
        )
    )
    summary = {
        "n_clip_pairs": int(len(records)),
        "n_species": int(len({str(row["species_label"]) for row in records})),
        "label_counts": {
            label: int(sum(1 for row in records if str(row["review_label"]) == label))
            for label in sorted({str(row["review_label"]) for row in records})
        },
        "pair_source_counts": {
            pair_source: int(sum(1 for row in records if str(row["pair_source"]) == pair_source))
            for pair_source in sorted({str(row["pair_source"]) for row in records})
        },
    }
    return records, summary


def main() -> None:
    args = parse_args()
    rows, summary = build_clip_pair_rows(args)

    ensure_dir(args.output_csv.parent)
    fieldnames = [
        "species_label",
        "review_label",
        "pair_source",
        "source_run_name",
        "source_run_dir",
        "cluster_run_dir",
        "pattern_run_dir",
        "cluster_id_a",
        "cluster_id_b",
        "clip_id_a",
        "clip_id_b",
        "embedding_row_a",
        "embedding_row_b",
        "recording_id_a",
        "recording_id_b",
        "audio_path_a",
        "audio_path_b",
        "review_confidence",
        "review_notes",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    save_json(args.summary_json, summary)
    print(f"Wrote {len(rows)} clip-level pair rows to {args.output_csv.as_posix()}")
    print(f"Summary JSON: {args.summary_json.as_posix()}")


if __name__ == "__main__":
    main()
