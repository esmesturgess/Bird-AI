from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .utils import AUDIO_EXTENSIONS, require_google_drive_output, slugify


LABEL_ALIASES = {
    "song": "song",
    "subsong": "song",
    "call": "call",
    "alarm": "alarm",
    "juvenile_begging": "juvenile_begging",
    "juvenile": "juvenile_begging",
    "begging": "juvenile_begging",
    "baby": "juvenile_begging",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an anchor manifest CSV from a folder of reference audio clips. "
            "Labels are inferred from parent folder names or filename prefixes."
        )
    )
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    return parser.parse_args()


def _normalize_label(text: str) -> str:
    key = slugify(text).lower()
    return LABEL_ALIASES.get(key, "")


def _infer_label(path: Path) -> str:
    parent_label = _normalize_label(path.parent.name)
    if parent_label:
        return parent_label

    stem = path.stem.strip()
    prefix = stem.split("__", 1)[0] if "__" in stem else stem.split("_", 1)[0]
    return _normalize_label(prefix)


def _iter_audio_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def main() -> None:
    args = parse_args()
    audio_dir = args.audio_dir.expanduser().resolve()
    output_csv = require_google_drive_output(args.output_csv, purpose="--output_csv")

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir.as_posix()}")

    rows: list[dict[str, str]] = []
    for path in _iter_audio_files(audio_dir):
        label = _infer_label(path)
        if not label:
            continue
        rel = path.relative_to(audio_dir).as_posix()
        rows.append(
            {
                "anchor_id": slugify(path.stem),
                "anchor_label": label,
                "source_url": "",
                "notes": f"Imported from {rel}",
                "audio_path": path.as_posix(),
            }
        )

    if not rows:
        raise ValueError(
            "No labeled anchor audio files found. Put files into folders like "
            "`song`, `call`, `alarm`, or `juvenile_begging`, or prefix filenames with those labels."
        )

    df = pd.DataFrame(rows).sort_values(["anchor_label", "anchor_id"]).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df)} anchor rows to {output_csv.as_posix()}")


if __name__ == "__main__":
    main()
