from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


REQUIRED_MANIFEST_COLUMNS = (
    "recording_id",
    "audio_path",
    "species_label",
    "species_common_name",
    "scientific_name",
    "species_source",
    "recording_source",
    "split",
    "notes",
)


def load_species_config(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} must contain JSON-compatible YAML for Phase 1 tooling."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a top-level object.")
    return payload


def target_species_labels(config: dict[str, object]) -> set[str]:
    raw_species = config.get("target_species", [])
    if not isinstance(raw_species, list):
        raise ValueError("Species config must contain a list at 'target_species'.")

    labels: set[str] = set()
    for item in raw_species:
        if not isinstance(item, dict):
            raise ValueError("Each target species entry must be an object.")
        label = str(item.get("label", "")).strip()
        if not label:
            raise ValueError("Each target species entry must include a non-empty 'label'.")
        labels.add(label)
    return labels


def allowed_values(config: dict[str, object], key: str) -> set[str]:
    raw = config.get(key, [])
    if not isinstance(raw, list):
        raise ValueError(f"Species config field '{key}' must be a list.")
    return {str(value).strip() for value in raw if str(value).strip()}


def _expected_split_from_name(path: Path) -> str | None:
    mapping = {
        "train_manifest.csv": "train",
        "val_manifest.csv": "val",
        "test_field_manifest.csv": "test_field",
    }
    return mapping.get(path.name)


def _load_manifest_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows: list[dict[str, str]] = []
        for row in reader:
            normalized = {key: (value or "").strip() for key, value in row.items() if key is not None}
            rows.append(normalized)
    return rows, [str(name).strip() for name in fieldnames]


def load_manifest_records(path: Path) -> list[dict[str, str]]:
    rows, fieldnames = _load_manifest_rows(path)
    missing = [col for col in REQUIRED_MANIFEST_COLUMNS if col not in fieldnames]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    return rows


def validate_manifest(
    path: Path,
    species_labels: set[str],
    allowed_recording_sources: set[str],
    allowed_splits: set[str],
    require_files: bool = False,
) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    rows, fieldnames = _load_manifest_rows(path)
    missing = [col for col in REQUIRED_MANIFEST_COLUMNS if col not in fieldnames]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    expected_split = _expected_split_from_name(path)

    invalid_species = sorted({row["species_label"] for row in rows if row["species_label"] and row["species_label"] not in species_labels})
    if invalid_species:
        raise ValueError(f"{path} contains unknown species labels: {', '.join(invalid_species)}")

    invalid_sources = sorted(
        {row["recording_source"] for row in rows if row["recording_source"] and row["recording_source"] not in allowed_recording_sources}
    )
    if invalid_sources:
        raise ValueError(f"{path} contains invalid recording_source values: {', '.join(invalid_sources)}")

    invalid_splits = sorted({row["split"] for row in rows if row["split"] and row["split"] not in allowed_splits})
    if invalid_splits:
        raise ValueError(f"{path} contains invalid split values: {', '.join(invalid_splits)}")

    if expected_split is not None:
        mismatched = [row for row in rows if row["split"] and row["split"] != expected_split]
        if mismatched:
            raise ValueError(
                f"{path} contains rows whose split does not match the file name expectation '{expected_split}'."
            )

    required_non_empty = (
        "recording_id",
        "audio_path",
        "species_label",
        "species_common_name",
        "scientific_name",
        "species_source",
        "recording_source",
        "split",
    )
    if rows:
        for col in required_non_empty:
            empty_rows = [idx for idx, row in enumerate(rows, start=2) if row.get(col, "") == ""]
            if empty_rows:
                raise ValueError(f"{path} has empty values in required column '{col}'.")

    missing_files: list[str] = []
    if require_files and rows:
        for row in rows:
            value = row["audio_path"]
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            if not candidate.exists():
                missing_files.append(value)

    species_counts = dict(sorted(Counter(row["species_label"] for row in rows if row["species_label"]).items()))
    source_counts = dict(sorted(Counter(row["recording_source"] for row in rows if row["recording_source"]).items()))
    split_counts = dict(sorted(Counter(row["split"] for row in rows if row["split"]).items()))

    return {
        "path": path.as_posix(),
        "rows": int(len(rows)),
        "expected_split": expected_split,
        "species_counts": species_counts,
        "recording_source_counts": source_counts,
        "split_counts": split_counts,
        "missing_files": missing_files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate V1 split manifests.")
    parser.add_argument(
        "--species-config",
        type=Path,
        default=Path("config/v1_species.yaml"),
        help="Path to the V1 species scope config.",
    )
    parser.add_argument(
        "--manifests",
        type=Path,
        nargs="*",
        default=[
            Path("data/manifests/train_manifest.csv"),
            Path("data/manifests/val_manifest.csv"),
            Path("data/manifests/test_field_manifest.csv"),
        ],
        help="Manifest CSV files to validate.",
    )
    parser.add_argument(
        "--require-files",
        action="store_true",
        help="Also verify that every audio_path exists on disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_species_config(args.species_config)
    species_labels = target_species_labels(config)
    allowed_recording_sources = allowed_values(config, "allowed_recording_sources")
    allowed_splits = allowed_values(config, "allowed_splits")

    print(f"Species scope: {', '.join(sorted(species_labels))}")
    for manifest_path in args.manifests:
        summary = validate_manifest(
            path=manifest_path,
            species_labels=species_labels,
            allowed_recording_sources=allowed_recording_sources,
            allowed_splits=allowed_splits,
            require_files=args.require_files,
        )
        print(f"\n{summary['path']}")
        print(f"  rows: {summary['rows']}")
        print(f"  expected_split: {summary['expected_split']}")
        print(f"  species_counts: {summary['species_counts']}")
        print(f"  recording_source_counts: {summary['recording_source_counts']}")
        print(f"  split_counts: {summary['split_counts']}")
        if args.require_files:
            print(f"  missing_files: {len(summary['missing_files'])}")
            for item in summary["missing_files"][:10]:
                print(f"    - {item}")


if __name__ == "__main__":
    main()
