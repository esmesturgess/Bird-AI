from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .dataset import load_species_config
from .utils import AUDIO_EXTENSIONS, ensure_dir, require_google_drive_output, slugify


API_ENDPOINT = "https://xeno-canto.org/api/3/recordings"
USER_AGENT = "BirdTranslationAI/0.1 (+https://xeno-canto.org/)"
QUALITY_ORDER = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}


@dataclass(frozen=True)
class SpeciesTarget:
    label: str
    common_name: str
    scientific_name: str
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch bird recordings from the xeno-canto v3 API and write metadata plus local audio files."
    )
    parser.add_argument("--species", action="append", default=[], help="Scientific species name. Repeat for multiple species.")
    parser.add_argument(
        "--species_csv",
        type=Path,
        help="CSV with at least a 'scientific_name' column and optional 'label', 'common_name', 'notes'.",
    )
    parser.add_argument(
        "--species_config",
        type=Path,
        help="JSON-compatible YAML config with a 'target_species' list, like config/v1_species.yaml.",
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory where species folders and metadata files will be written.")
    parser.add_argument("--api_key", type=str, default=os.environ.get("XC_API_KEY", ""), help="xeno-canto API key. Defaults to XC_API_KEY.")
    parser.add_argument("--group", type=str, default="birds", help="xeno-canto grp tag, default: birds.")
    parser.add_argument("--country", action="append", default=[], help="Optional cnt tag value. Repeat for multiple countries.")
    parser.add_argument("--area", action="append", default=[], help="Optional area tag value. Repeat for multiple areas.")
    parser.add_argument("--sound_type", action="append", default=[], help="Optional type tag value. Repeat for multiple sound types.")
    parser.add_argument(
        "--quality_tag",
        type=str,
        default="",
        help='Advanced xeno-canto q tag value, for example A or >C. Applied in the API query.',
    )
    parser.add_argument(
        "--min_quality",
        type=str,
        default="",
        help="Post-filter on returned quality letter. Example: C keeps A/B/C. Leave empty to disable.",
    )
    parser.add_argument(
        "--extra_tag",
        action="append",
        default=[],
        help='Additional raw xeno-canto query tag, for example \'len:"<20"\' or \'playback:no\'. Repeat as needed.',
    )
    parser.add_argument("--per_page", type=int, default=100, help="xeno-canto results per page, between 50 and 500.")
    parser.add_argument("--max_pages", type=int, default=0, help="Maximum pages to fetch per species. 0 means fetch all available pages.")
    parser.add_argument(
        "--max_recordings_per_species",
        type=int,
        default=0,
        help="Stop after this many kept recordings per species. 0 means no cap.",
    )
    parser.add_argument(
        "--min_length_seconds",
        type=float,
        default=0.0,
        help="Discard recordings shorter than this length after metadata retrieval.",
    )
    parser.add_argument(
        "--max_length_seconds",
        type=float,
        default=0.0,
        help="Discard recordings longer than this length after metadata retrieval. 0 disables the cap.",
    )
    parser.add_argument("--allow_nd", action="store_true", help="Include no-derivatives licenses. By default they are excluded.")
    parser.add_argument("--prefer_shorter", action="store_true", help="Sort kept candidates by shorter duration before applying caps.")
    parser.add_argument(
        "--max_per_recordist",
        type=int,
        default=0,
        help="Soft cap on how many kept recordings can come from the same recordist per species. 0 disables the cap.",
    )
    parser.add_argument(
        "--skip_existing_dir",
        type=Path,
        action="append",
        default=[],
        help="Directory to scan for existing XC ids so they are not downloaded again. Repeat as needed.",
    )
    parser.add_argument(
        "--no_species_subdir",
        action="store_true",
        help="Write audio files directly into output_dir instead of output_dir/species_label.",
    )
    parser.add_argument("--metadata_only", action="store_true", help="Fetch metadata only and skip audio downloads.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing audio files and metadata outputs.")
    parser.add_argument("--sleep_seconds", type=float, default=0.5, help="Pause between page requests.")
    parser.add_argument("--request_timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    parser.add_argument("--split", type=str, default="train", help="Split value to write into the generated manifest rows.")
    return parser.parse_args()


def _load_species_targets(args: argparse.Namespace) -> list[SpeciesTarget]:
    targets: list[SpeciesTarget] = []

    if args.species_config:
        config = load_species_config(args.species_config)
        for item in config.get("target_species", []):
            if not isinstance(item, dict):
                continue
            scientific_name = str(item.get("scientific_name", "")).strip()
            if not scientific_name:
                continue
            label = str(item.get("label", "")).strip() or slugify(scientific_name).lower()
            common_name = str(item.get("common_name", "")).strip()
            notes = str(item.get("notes", "")).strip()
            targets.append(
                SpeciesTarget(
                    label=label,
                    common_name=common_name,
                    scientific_name=scientific_name,
                    notes=notes,
                )
            )

    if args.species_csv:
        with args.species_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scientific_name = str(row.get("scientific_name", "")).strip()
                if not scientific_name:
                    continue
                label = str(row.get("label", "")).strip() or slugify(scientific_name).lower()
                common_name = str(row.get("common_name", "")).strip()
                notes = str(row.get("notes", "")).strip()
                targets.append(
                    SpeciesTarget(
                        label=label,
                        common_name=common_name,
                        scientific_name=scientific_name,
                        notes=notes,
                    )
                )

    for scientific_name in args.species:
        scientific_name = str(scientific_name).strip()
        if not scientific_name:
            continue
        targets.append(
            SpeciesTarget(
                label=slugify(scientific_name).lower(),
                common_name="",
                scientific_name=scientific_name,
                notes="",
            )
        )

    deduped: dict[str, SpeciesTarget] = {}
    for target in targets:
        key = target.scientific_name.casefold()
        if key not in deduped:
            deduped[key] = target
    out = list(deduped.values())
    if not out:
        raise ValueError("Provide at least one species via --species, --species_csv, or --species_config.")
    return out


def _xc_value(value: str, *, force_quote: bool = False) -> str:
    value = str(value).strip()
    if not value:
        return value
    if value.startswith('"') and value.endswith('"'):
        return value
    needs_quote = force_quote or any(ch in value for ch in [' ', '"', "<", ">", "="])
    if not needs_quote:
        return value
    escaped = value.replace('"', '\\"')
    return f'"{escaped}"'


def _build_query(target: SpeciesTarget, args: argparse.Namespace) -> str:
    parts = [f"sp:{_xc_value(target.scientific_name, force_quote=True)}", f"grp:{_xc_value(args.group)}"]
    for country in args.country:
        parts.append(f"cnt:{_xc_value(country)}")
    for area in args.area:
        parts.append(f"area:{_xc_value(area)}")
    for sound_type in args.sound_type:
        parts.append(f"type:{_xc_value(sound_type)}")
    if args.quality_tag:
        parts.append(f"q:{_xc_value(args.quality_tag)}")
    for tag in args.extra_tag:
        tag = str(tag).strip()
        if tag:
            parts.append(tag)
    return " ".join(parts)


def _fetch_json(url: str, *, timeout: float) -> dict[str, Any]:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP error from xeno-canto: {exc.code} {payload}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach xeno-canto: {exc.reason}") from exc


def _fetch_recordings_page(
    *,
    query: str,
    api_key: str,
    page: int,
    per_page: int,
    timeout: float,
) -> dict[str, Any]:
    params = {
        "query": query,
        "page": page,
        "per_page": per_page,
        "key": api_key,
    }
    url = f"{API_ENDPOINT}?{urlencode(params)}"
    payload = _fetch_json(url, timeout=timeout)
    error = payload.get("error")
    if error:
        if isinstance(error, dict):
            message = str(error.get("message", "xeno-canto API error"))
            code = str(error.get("code", "unknown_error"))
            raise RuntimeError(f"xeno-canto API error ({code}): {message}")
        message = str(payload.get("message", "xeno-canto API error"))
        raise RuntimeError(f"xeno-canto API error ({error}): {message}")
    return payload


def _normalize_url(url: str) -> str:
    url = str(url or "").strip()
    if not url:
        return ""
    if url.startswith("//"):
        return f"https:{url}"
    if url.startswith("/"):
        return f"https://xeno-canto.org{url}"
    return url


def _parse_length_seconds(value: str) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    parts = text.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            minutes, seconds = parts
            return (60.0 * float(minutes)) + float(seconds)
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return (3600.0 * float(hours)) + (60.0 * float(minutes)) + float(seconds)
    except ValueError:
        return 0.0
    return 0.0


def _quality_passes(value: str, min_quality: str) -> bool:
    quality = str(value or "").strip().upper()
    threshold = str(min_quality or "").strip().upper()
    if not threshold:
        return True
    if quality not in QUALITY_ORDER or threshold not in QUALITY_ORDER:
        return True
    return QUALITY_ORDER[quality] >= QUALITY_ORDER[threshold]


def _license_is_allowed(license_url: str, allow_nd: bool) -> bool:
    license_text = str(license_url or "").lower()
    if not allow_nd:
        marker = "/licenses/"
        if marker in license_text:
            slug = license_text.split(marker, 1)[1].split("/", 1)[0]
            parts = [part for part in slug.split("-") if part]
            if "nd" in parts:
                return False
    return True


def _same_species(record: dict[str, Any], target: SpeciesTarget) -> bool:
    gen = str(record.get("gen", "")).strip()
    sp = str(record.get("sp", "")).strip()
    scientific_name = f"{gen} {sp}".strip()
    return scientific_name.casefold() == target.scientific_name.casefold()


def _species_dir(output_dir: Path, target: SpeciesTarget) -> Path:
    return output_dir / target.label


def _extract_xc_id(text: str) -> str:
    match = re.search(r"\bXC(\d+)\b", str(text), flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1)


def _collect_existing_xc_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        candidate_dir = path.expanduser().resolve()
        if not candidate_dir.exists():
            continue
        for file_path in candidate_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in AUDIO_EXTENSIONS and file_path.suffix.lower() != ".csv":
                continue
            xc_id = _extract_xc_id(file_path.name)
            if xc_id:
                ids.add(xc_id)
    return ids


def _download_audio(
    *,
    url: str,
    out_path: Path,
    timeout: float,
    overwrite: bool,
) -> str:
    if out_path.exists() and not overwrite:
        return "existing"

    ensure_dir(out_path.parent)
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=timeout) as response, out_path.open("wb") as f:
            shutil.copyfileobj(response, f)
    except HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Audio download failed: {exc.code} {payload}") from exc
    except URLError as exc:
        raise RuntimeError(f"Audio download failed: {exc.reason}") from exc
    return "downloaded"


def _recording_output_path(
    output_dir: Path,
    target: SpeciesTarget,
    record: dict[str, Any],
    *,
    no_species_subdir: bool,
) -> Path:
    raw_file_name = str(record.get("file-name", "")).strip()
    suffix = Path(raw_file_name).suffix or ".mp3"
    xc_id = str(record.get("id", "")).strip()
    stem_parts = [f"XC{xc_id}" if xc_id else "XC", slugify(target.common_name or target.scientific_name)]
    stem = "_".join(part for part in stem_parts if part)
    target_dir = output_dir if no_species_subdir else _species_dir(output_dir, target)
    return target_dir / f"{stem}{suffix}"


def _manifest_row(
    *,
    audio_path: Path,
    target: SpeciesTarget,
    record: dict[str, Any],
    split: str,
) -> dict[str, str]:
    xc_id = str(record.get("id", "")).strip()
    notes = {
        "xc_id": xc_id,
        "quality": str(record.get("q", "")).strip(),
        "type": str(record.get("type", "")).strip(),
        "license": _normalize_url(str(record.get("lic", "")).strip()),
        "country": str(record.get("cnt", "")).strip(),
        "recordist": str(record.get("rec", "")).strip(),
    }
    return {
        "recording_id": f"XC{xc_id}" if xc_id else slugify(audio_path.stem),
        "audio_path": audio_path.as_posix(),
        "species_label": target.label,
        "species_common_name": target.common_name or str(record.get("en", "")).strip(),
        "scientific_name": target.scientific_name,
        "species_source": "xeno_canto_api",
        "recording_source": "web",
        "split": split,
        "notes": json.dumps(notes, separators=(",", ":"), ensure_ascii=True),
    }


def _record_metadata(
    *,
    target: SpeciesTarget,
    record: dict[str, Any],
    query: str,
    local_audio_path: str,
    download_status: str,
) -> dict[str, Any]:
    return {
        "species_label": target.label,
        "species_common_name": target.common_name or str(record.get("en", "")).strip(),
        "scientific_name": target.scientific_name,
        "xc_query": query,
        "xc_id": str(record.get("id", "")).strip(),
        "gen": str(record.get("gen", "")).strip(),
        "sp": str(record.get("sp", "")).strip(),
        "ssp": str(record.get("ssp", "")).strip(),
        "group": str(record.get("grp", "")).strip(),
        "english_name": str(record.get("en", "")).strip(),
        "recordist": str(record.get("rec", "")).strip(),
        "country": str(record.get("cnt", "")).strip(),
        "location": str(record.get("loc", "")).strip(),
        "latitude": str(record.get("lat", "")).strip(),
        "longitude": str(record.get("lon", "")).strip(),
        "sound_type": str(record.get("type", "")).strip(),
        "sex": str(record.get("sex", "")).strip(),
        "stage": str(record.get("stage", "")).strip(),
        "method": str(record.get("method", "")).strip(),
        "recording_url": _normalize_url(str(record.get("url", "")).strip()),
        "file_url": _normalize_url(str(record.get("file", "")).strip()),
        "file_name": str(record.get("file-name", "")).strip(),
        "license_url": _normalize_url(str(record.get("lic", "")).strip()),
        "quality": str(record.get("q", "")).strip(),
        "length": str(record.get("length", "")).strip(),
        "length_seconds": _parse_length_seconds(str(record.get("length", "")).strip()),
        "time": str(record.get("time", "")).strip(),
        "date": str(record.get("date", "")).strip(),
        "uploaded": str(record.get("uploaded", "")).strip(),
        "sample_rate": str(record.get("smp", "")).strip(),
        "auto_recording": str(record.get("auto", "")).strip(),
        "playback_used": str(record.get("playback-used", "")).strip(),
        "animal_seen": str(record.get("animal-seen", "")).strip(),
        "remarks": str(record.get("rmk", "")).strip(),
        "background_species_json": json.dumps(record.get("also", []), ensure_ascii=True),
        "local_audio_path": local_audio_path,
        "download_status": download_status,
        "target_notes": target.notes,
    }


def _apply_recordist_cap(records: list[dict[str, Any]], max_per_recordist: int) -> list[dict[str, Any]]:
    if max_per_recordist <= 0:
        return records
    out: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for record in records:
        recordist = str(record.get("rec", "")).strip() or "__missing__"
        if counts[recordist] >= max_per_recordist:
            continue
        out.append(record)
        counts[recordist] += 1
    return out


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_dir(path.parent)
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
    targets = _load_species_targets(args)
    api_key = str(args.api_key or "").strip()
    if not api_key:
        raise ValueError("xeno-canto API key required. Pass --api_key or set XC_API_KEY.")

    per_page = max(50, min(int(args.per_page), 500))
    min_quality = str(args.min_quality or "").strip().upper()
    output_dir = require_google_drive_output(args.output_dir, purpose="--output_dir")
    ensure_dir(output_dir)
    existing_xc_ids = _collect_existing_xc_ids(list(args.skip_existing_dir))

    metadata_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, str]] = []
    per_species_summary: list[dict[str, Any]] = []

    for target in targets:
        query = _build_query(target, args)
        kept_for_species = 0
        pages_fetched = 0
        download_statuses: Counter[str] = Counter()
        filtered_counts: Counter[str] = Counter()
        candidate_records: list[dict[str, Any]] = []

        page = 1
        total_pages = 1
        while True:
            payload = _fetch_recordings_page(
                query=query,
                api_key=api_key,
                page=page,
                per_page=per_page,
                timeout=args.request_timeout,
            )
            pages_fetched += 1
            total_pages = int(payload.get("numPages", 1) or 1)
            recordings = payload.get("recordings", []) or []

            for record in recordings:
                if not isinstance(record, dict):
                    filtered_counts["invalid_record"] += 1
                    continue
                if not _same_species(record, target):
                    filtered_counts["species_mismatch"] += 1
                    continue
                if not _license_is_allowed(str(record.get("lic", "")), args.allow_nd):
                    filtered_counts["license_nd"] += 1
                    continue
                if not _quality_passes(str(record.get("q", "")), min_quality):
                    filtered_counts["quality_below_threshold"] += 1
                    continue

                length_seconds = _parse_length_seconds(str(record.get("length", "")))
                if args.min_length_seconds > 0 and length_seconds < args.min_length_seconds:
                    filtered_counts["too_short"] += 1
                    continue
                if args.max_length_seconds > 0 and length_seconds > args.max_length_seconds:
                    filtered_counts["too_long"] += 1
                    continue
                xc_id = str(record.get("id", "")).strip()
                if xc_id and xc_id in existing_xc_ids:
                    filtered_counts["already_present"] += 1
                    continue

                audio_url = _normalize_url(str(record.get("file", "")).strip())
                if not audio_url and not args.metadata_only:
                    filtered_counts["missing_audio_url"] += 1
                    continue
                candidate_records.append(record)

            if args.max_pages > 0 and page >= args.max_pages:
                break
            if page >= total_pages:
                break
            page += 1
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

        if args.prefer_shorter:
            candidate_records.sort(
                key=lambda record: (
                    -QUALITY_ORDER.get(str(record.get("q", "")).strip().upper(), -1),
                    _parse_length_seconds(str(record.get("length", "")).strip()),
                    str(record.get("id", "")).strip(),
                )
            )

        candidate_records = _apply_recordist_cap(candidate_records, args.max_per_recordist)

        if args.max_recordings_per_species > 0:
            candidate_records = candidate_records[: args.max_recordings_per_species]

        for record in candidate_records:
            local_audio_path = ""
            download_status = "metadata_only"
            out_path = _recording_output_path(
                output_dir,
                target,
                record,
                no_species_subdir=bool(args.no_species_subdir),
            )
            if not args.metadata_only:
                download_status = _download_audio(
                    url=_normalize_url(str(record.get("file", "")).strip()),
                    out_path=out_path,
                    timeout=args.request_timeout,
                    overwrite=args.overwrite,
                )
                download_statuses[download_status] += 1
                local_audio_path = out_path.as_posix()
                manifest_rows.append(
                    _manifest_row(
                        audio_path=out_path,
                        target=target,
                        record=record,
                        split=args.split,
                    )
                )

            metadata_rows.append(
                _record_metadata(
                    target=target,
                    record=record,
                    query=query,
                    local_audio_path=local_audio_path,
                    download_status=download_status,
                )
            )
            xc_id = str(record.get("id", "")).strip()
            if xc_id:
                existing_xc_ids.add(xc_id)
            kept_for_species += 1

        per_species_summary.append(
            {
                "species_label": target.label,
                "scientific_name": target.scientific_name,
                "query": query,
                "pages_fetched": pages_fetched,
                "candidate_records": len(candidate_records),
                "kept_recordings": kept_for_species,
                "download_status_counts": dict(sorted(download_statuses.items())),
                "filtered_counts": dict(sorted(filtered_counts.items())),
            }
        )

    metadata_path = output_dir / "xc_recordings_metadata.csv"
    summary_path = output_dir / "xc_fetch_summary.json"
    manifest_path = output_dir / "xc_recordings_manifest.csv"

    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    if manifest_rows:
        _write_manifest(manifest_path, manifest_rows)

    summary_payload = {
        "api_endpoint": API_ENDPOINT,
        "output_dir": output_dir.as_posix(),
        "metadata_only": bool(args.metadata_only),
        "n_species_targets": len(targets),
        "n_metadata_rows": len(metadata_rows),
        "n_manifest_rows": len(manifest_rows),
        "species": per_species_summary,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    print("xeno-canto fetch complete")
    print(f"- Output dir: {output_dir.as_posix()}")
    print(f"- Metadata rows: {len(metadata_rows)} -> {metadata_path.as_posix()}")
    if manifest_rows:
        print(f"- Manifest rows: {len(manifest_rows)} -> {manifest_path.as_posix()}")
    else:
        print("- Manifest rows: 0 (metadata-only mode or no successful downloads)")
    print(f"- Summary: {summary_path.as_posix()}")
    for item in per_species_summary:
        print(
            f"- {item['species_label']}: kept {item['kept_recordings']} recordings "
            f"across {item['pages_fetched']} page(s)"
        )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc))
