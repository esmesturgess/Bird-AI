from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from .utils import ensure_dir, require_google_drive_output, save_json, slugify


USER_AGENT = "BirdTranslationAI/0.1 (+https://xeno-canto.org/)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reviewed reference-audio bundle from Xeno-canto metadata pulls and a simple selection CSV."
        )
    )
    parser.add_argument("--metadata_root_dir", type=Path, required=True)
    parser.add_argument("--selection_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--summary_name", type=str, default="reference_shortlist.csv")
    return parser.parse_args()


def _load_selection(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"category", "xc_id", "why_selected"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Selection CSV missing required columns: {', '.join(missing)}")
    df = df.copy()
    df["category"] = df["category"].fillna("").astype(str).str.strip()
    df["xc_id"] = df["xc_id"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    df["why_selected"] = df["why_selected"].fillna("").astype(str).str.strip()
    df = df[(df["category"] != "") & (df["xc_id"] != "")]
    if df.empty:
        raise ValueError("Selection CSV has no usable rows.")
    return df.reset_index(drop=True)


def _load_metadata(root: Path) -> pd.DataFrame:
    paths = sorted(root.rglob("xc_recordings_metadata.csv"))
    if not paths:
        raise FileNotFoundError(f"No xc_recordings_metadata.csv files found under {root.as_posix()}")
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        df["query_source"] = path.parent.name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["xc_id"] = out["xc_id"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    out = out.sort_values(["xc_id", "query_source"]).drop_duplicates("xc_id", keep="first")
    return out.reset_index(drop=True)


def _download(url: str, out_path: Path) -> str:
    if out_path.exists():
        return "existing"
    ensure_dir(out_path.parent)
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=60) as response, out_path.open("wb") as f:
        f.write(response.read())
    return "downloaded"


def main() -> None:
    args = parse_args()
    metadata_root_dir = args.metadata_root_dir.expanduser().resolve()
    selection_csv = args.selection_csv.expanduser().resolve()
    output_dir = require_google_drive_output(args.output_dir, purpose="--output_dir")

    selection_df = _load_selection(selection_csv)
    metadata_df = _load_metadata(metadata_root_dir)
    by_id = {str(row["xc_id"]): row for _, row in metadata_df.iterrows()}

    ensure_dir(output_dir)
    rows: list[dict[str, object]] = []
    for _, sel in selection_df.iterrows():
        xc_id = str(sel["xc_id"])
        category = str(sel["category"])
        why_selected = str(sel["why_selected"])
        meta = by_id.get(xc_id)
        if meta is None:
            rows.append(
                {
                    "category": category,
                    "xc_id": xc_id,
                    "status": "missing_metadata",
                    "why_selected": why_selected,
                }
            )
            continue
        file_url = str(meta.get("file_url", "")).strip()
        if not file_url:
            rows.append(
                {
                    "category": category,
                    "xc_id": xc_id,
                    "status": "missing_file_url",
                    "why_selected": why_selected,
                }
            )
            continue
        ext = Path(file_url).suffix or ".mp3"
        filename = f"XC{xc_id}_{slugify(category)}_{slugify(str(meta.get('recordist', '')))}{ext}"
        out_path = output_dir / category / filename
        status = _download(file_url, out_path)
        rows.append(
            {
                "category": category,
                "xc_id": xc_id,
                "status": status,
                "audio_path": out_path.as_posix(),
                "query_source": str(meta.get("query_source", "")),
                "country": str(meta.get("country", "")),
                "sound_type": str(meta.get("sound_type", "")),
                "stage": str(meta.get("stage", "")),
                "quality": str(meta.get("quality", "")),
                "length_seconds": float(meta.get("length_seconds", 0.0) or 0.0),
                "recordist": str(meta.get("recordist", "")),
                "license_url": str(meta.get("license_url", "")),
                "recording_url": str(meta.get("recording_url", "")),
                "remarks": str(meta.get("remarks", "")),
                "why_selected": why_selected,
            }
        )

    summary_path = output_dir / str(args.summary_name)
    summary_df = pd.DataFrame(rows).sort_values(["category", "xc_id"]).reset_index(drop=True)
    summary_df.to_csv(summary_path, index=False)

    readme_lines = [
        "# Reference shortlist",
        "",
        f"Built from metadata root: `{metadata_root_dir.as_posix()}`",
        f"Selection CSV: `{selection_csv.as_posix()}`",
        "",
        "See the summary CSV for descriptions, categories, and license information.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    save_json(
        output_dir / "curation_summary.json",
        {
            "metadata_root_dir": metadata_root_dir.as_posix(),
            "selection_csv": selection_csv.as_posix(),
            "summary_csv": summary_path.as_posix(),
            "n_selected": int(len(selection_df)),
            "n_output_rows": int(len(summary_df)),
        },
    )
    print(f"Wrote review set to {output_dir.as_posix()}")
    print(f"Wrote summary CSV to {summary_path.as_posix()}")


if __name__ == "__main__":
    main()
