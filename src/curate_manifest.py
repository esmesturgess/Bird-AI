from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .utils import AUDIO_EXTENSIONS, slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create simple train/val manifests from a species folder.")
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--train_manifest", type=Path, default=Path("data/manifests/train_manifest.csv"))
    parser.add_argument("--val_manifest", type=Path, default=Path("data/manifests/val_manifest.csv"))
    parser.add_argument("--species_label", type=str, required=True)
    parser.add_argument("--species_common_name", type=str, required=True)
    parser.add_argument("--scientific_name", type=str, required=True)
    parser.add_argument("--species_source", type=str, default="folder_curation")
    parser.add_argument("--recording_source", type=str, default="web")
    parser.add_argument("--val_count", type=int, default=10)
    return parser.parse_args()


def list_audio_files(source_dir: Path) -> list[Path]:
    return sorted(
        p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def build_rows(
    audio_files: list[Path],
    *,
    species_label: str,
    species_common_name: str,
    scientific_name: str,
    species_source: str,
    recording_source: str,
    split: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for audio_path in audio_files:
        rows.append(
            {
                "recording_id": slugify(audio_path.stem),
                "audio_path": audio_path.as_posix(),
                "species_label": species_label,
                "species_common_name": species_common_name,
                "scientific_name": scientific_name,
                "species_source": species_source,
                "recording_source": recording_source,
                "split": split,
                "notes": "",
            }
        )
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "recording_id",
        "audio_path",
        "species_label",
        "species_common_name",
        "scientific_name",
        "species_source",
        "recording_source",
        "split",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {source_dir}")

    audio_files = list_audio_files(source_dir)
    if not audio_files:
        raise FileNotFoundError(f"No supported audio files found under {source_dir}")

    val_count = max(0, min(args.val_count, len(audio_files)))
    val_files = audio_files[-val_count:] if val_count else []
    train_files = audio_files[:-val_count] if val_count else audio_files

    train_rows = build_rows(
        train_files,
        species_label=args.species_label,
        species_common_name=args.species_common_name,
        scientific_name=args.scientific_name,
        species_source=args.species_source,
        recording_source=args.recording_source,
        split="train",
    )
    val_rows = build_rows(
        val_files,
        species_label=args.species_label,
        species_common_name=args.species_common_name,
        scientific_name=args.scientific_name,
        species_source=args.species_source,
        recording_source=args.recording_source,
        split="val",
    )

    write_manifest(args.train_manifest, train_rows)
    write_manifest(args.val_manifest, val_rows)

    print(f"Source dir: {source_dir.as_posix()}")
    print(f"Audio files found: {len(audio_files)}")
    print(f"Train rows written: {len(train_rows)} -> {args.train_manifest.as_posix()}")
    print(f"Val rows written: {len(val_rows)} -> {args.val_manifest.as_posix()}")
    if train_rows:
        print(f"First train file: {train_rows[0]['audio_path']}")
    if val_rows:
        print(f"First val file: {val_rows[0]['audio_path']}")


if __name__ == "__main__":
    main()
