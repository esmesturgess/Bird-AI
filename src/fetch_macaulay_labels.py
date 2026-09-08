"""Fetch Macaulay Library audio + free-text/structured metadata for a species,
and deduce a call-type label (song / call / alarm / juvenile) per clip by combining:

  - structured `tags` (Macaulay's "Sounds" field)
  - free-text `mediaNotes` (recordist's own description) via keyword match
  - structured `ageSex` juvenile counts (direct evidence of a juvenile bird)

Priority: an explicit notes keyword for alarm/juvenile OVERRIDES a generic song/call tag
(the recordist's specific note is more precise than the broad default tag). Clips with no
matching signal are EXCLUDED, not forced into a category. Clips with conflicting signals
(e.g. both juvenile and alarm evidence, or both song and call tags with no override) are
marked AMBIGUOUS for human review rather than guessed.

Audio is downloaded ONLY for assigned + ambiguous clips (not excluded ones), to avoid
wasting bandwidth/disk on clips we won't use.

    ./.venv/bin/python -m src.fetch_macaulay_labels \
        --taxon_code eurbla --region_code GB \
        --output_dir data/raw/blackbird_macaulay_uk_v1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

SEARCH_API = "https://search.macaulaylibrary.org/api/v2/search"
AUDIO_URL = "https://cdn.download.ams.birds.cornell.edu/api/v1/asset/{asset_id}/audio"
USER_AGENT = "BirdTranslationAI/0.1 (non-commercial research; +https://github.com/esmesturgess/Bird-AI)"

# dawn_song/flight_song excluded per user. `duet` also excluded (2026-09-03): a duet contains
# two birds answering each other, so the clip carries two different vocal types at once and
# cannot take a single label — consistent with the XC side in build_species_dataset.py.
SONG_TAGS = {"song", "courtship_display_or_copulation"}
CALL_TAGS = {"call", "flight_call"}
ALARM_RE = re.compile(r"\balarm\b|\bagitated\b|\bdistress\b|\bmobbing\b|\bscold", re.I)
JUVENILE_RE = re.compile(r"\bjuv(enile)?\b|\bfledgling\b|\bnestling\b|\bbeg(ging)?\b", re.I)
SONG_RE = re.compile(r"\bsong\b|\bsinging\b", re.I)
CALL_RE = re.compile(r"\bcalling\b|\bcalls?\b", re.I)

MIN_FREE_BYTES = 500 * 1024 * 1024  # stop downloading if free disk drops below this


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--taxon_code", required=True, help="Macaulay/eBird species code, e.g. eurbla")
    p.add_argument("--region_code", default=None, help="e.g. GB. Omit for worldwide.")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--api_key", default=os.environ.get("MACAULAY_API_KEY", ""))
    p.add_argument("--max_pages", type=int, default=30, help="Metadata pages of 100.")
    p.add_argument("--download_audio", action="store_true", default=True)
    p.add_argument("--no_download_audio", dest="download_audio", action="store_false")
    p.add_argument("--only_categories", type=str, default=None,
                    help="Comma-separated subset to download, e.g. 'juvenile'. Metadata is still "
                         "fetched/classified for everything; this only limits audio downloads.")
    p.add_argument("--skip_asset_ids_csv", type=Path, default=None,
                    help="CSV with an 'assetId' column of IDs to skip downloading (already have them).")
    p.add_argument("--max_per_category", type=int, default=None,
                    help="Cap downloads per category. Rows are already rating-sorted, so this "
                         "naturally keeps the highest-quality clips first.")
    p.add_argument("--sleep_seconds", type=float, default=0.3)
    return p.parse_args()


def _get_json(url: str, key: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"X-eBirdApiToken": key, "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def fetch_all_metadata(taxon_code: str, region_code: str | None, key: str, max_pages: int) -> list[dict]:
    assets: list[dict] = []
    cursor = None
    for _ in range(max_pages):
        params = {"taxonCode": taxon_code, "mediaType": "audio", "count": "100", "sort": "rating_rank_desc"}
        if region_code:
            params["regionCode"] = region_code
        if cursor:
            params["initialCursorMark"] = cursor
        page = _get_json(f"{SEARCH_API}?{urllib.parse.urlencode(params)}", key)
        if not isinstance(page, list) or not page:
            break
        assets.extend(page)
        cursor = page[-1].get("cursorMark")
        if not cursor:
            break
        time.sleep(0.2)
    return assets


def classify(asset: dict) -> tuple[str, str]:
    tags = set(asset.get("tags") or [])
    notes = str(asset.get("mediaNotes") or "")
    age_sex = asset.get("ageSex") or {}

    juv_structured = any(int(v or 0) > 0 for k, v in age_sex.items() if "juvenile" in k.lower())
    juv_kw = bool(JUVENILE_RE.search(notes))
    alarm_kw = bool(ALARM_RE.search(notes))

    juvenile_signal = juv_structured or juv_kw
    alarm_signal = alarm_kw

    song_tag = bool(tags & SONG_TAGS)
    call_tag = bool(tags & CALL_TAGS)
    song_kw = bool(SONG_RE.search(notes)) and not song_tag
    call_kw = bool(CALL_RE.search(notes)) and not call_tag
    song_signal = song_tag or song_kw
    call_signal = call_tag or call_kw

    if juvenile_signal and alarm_signal:
        return "ambiguous", f"conflict: juvenile signal (structured={juv_structured}, kw={juv_kw}) AND alarm keyword"
    if juvenile_signal:
        reason = "ageSex juvenile count > 0" if juv_structured else f"notes keyword match: {notes!r}"
        return "juvenile", reason
    if alarm_signal:
        return "alarm", f"notes keyword match: {notes!r}"
    if song_signal and call_signal:
        return "ambiguous", f"conflict: both song signal (tag={song_tag}) and call signal (tag={call_tag}), no override"
    if song_signal:
        return "song", f"tag match: {sorted(tags & SONG_TAGS)}" if song_tag else f"notes keyword: {notes!r}"
    if call_signal:
        return "call", f"tag match: {sorted(tags & CALL_TAGS)}" if call_tag else f"notes keyword: {notes!r}"
    return "excluded", f"no matching signal (tags={sorted(tags)}, notes={notes!r})"


def free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return usage.free


def download_audio(asset_id: str, out_path: Path, key: str) -> str:
    if out_path.exists():
        return "already_present"
    url = AUDIO_URL.format(asset_id=asset_id)
    req = urllib.request.Request(url, headers={"X-eBirdApiToken": key, "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        out_path.write_bytes(data)
        return "downloaded"
    except Exception as e:  # noqa: BLE001
        return f"error:{e}"


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("No API key. Pass --api_key or set MACAULAY_API_KEY.")

    out_dir = args.output_dir
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)

    print(f"Fetching metadata: taxon={args.taxon_code} region={args.region_code or 'ALL'} ...")
    assets = fetch_all_metadata(args.taxon_code, args.region_code, args.api_key, args.max_pages)
    print(f"  {len(assets)} audio assets found.")

    rows = []
    for a in assets:
        category, reason = classify(a)
        loc = a.get("location") or {}
        rows.append({
            "assetId": a.get("assetId"),
            "category": category,
            "reason": reason,
            "tags": ";".join(a.get("tags") or []),
            "mediaNotes": a.get("mediaNotes") or "",
            "ageSex": json.dumps(a.get("ageSex") or {}),
            "rating": a.get("rating"),
            "licenseId": a.get("licenseId"),
            "countryCode": loc.get("countryCode"),
            "locality": loc.get("name"),
            "obsDate": a.get("obsDtDisplay"),
            "userDisplayName": a.get("userDisplayName"),
            "restricted": a.get("restricted"),
            "audio_url": AUDIO_URL.format(asset_id=a.get("assetId")),
            "local_audio_path": "",
        })
    df = pd.DataFrame(rows)

    counts = df["category"].value_counts()
    print("\n=== category breakdown ===")
    for cat in ["song", "call", "alarm", "juvenile", "ambiguous", "excluded"]:
        print(f"  {cat:10s} {counts.get(cat, 0)}")

    if args.download_audio:
        wanted = df["category"] != "excluded"
        if args.only_categories:
            keep = {c.strip() for c in args.only_categories.split(",")}
            wanted = df["category"].isin(keep)
            print(f"Restricting downloads to categories: {sorted(keep)}")
        to_download = df[wanted].copy()
        if args.skip_asset_ids_csv and args.skip_asset_ids_csv.exists():
            known = set(pd.read_csv(args.skip_asset_ids_csv)["assetId"])
            before = len(to_download)
            to_download = to_download[~to_download["assetId"].isin(known)]
            print(f"Skipping {before - len(to_download)} already-known assetIds from {args.skip_asset_ids_csv}")
        if args.max_per_category:
            to_download = to_download.groupby("category", group_keys=False).head(args.max_per_category)
            print(f"Capped to {args.max_per_category} per category -> {len(to_download)} total")
        print(f"\nDownloading audio for {len(to_download)} clips...")
        statuses = []
        for i, row in to_download.iterrows():
            free = free_bytes(out_dir)
            if free < MIN_FREE_BYTES:
                print(f"  STOPPING: free disk {free/1e6:.0f}MB below safety threshold "
                      f"({MIN_FREE_BYTES/1e6:.0f}MB). Downloaded {len(statuses)}/{len(to_download)} so far.")
                break
            fname = f"{row['category']}_{row['assetId']}.mp3"
            out_path = out_dir / "audio" / fname
            status = download_audio(str(row["assetId"]), out_path, args.api_key)
            if status in ("downloaded", "already_present"):
                df.loc[i, "local_audio_path"] = str(out_path)
            statuses.append(status)
            if len(statuses) % 50 == 0:
                print(f"  ... {len(statuses)}/{len(to_download)} "
                      f"(free disk: {free_bytes(out_dir)/1e9:.2f}GB)")
            time.sleep(args.sleep_seconds)
        print(f"download statuses: {pd.Series(statuses).value_counts().to_dict()}")

    manifest_path = out_dir / "macaulay_label_manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"\nWrote manifest: {manifest_path}  ({len(df)} rows)")
    print(f"Audio dir size: ", end="")
    os.system(f"du -sh '{out_dir}/audio' 2>/dev/null")


if __name__ == "__main__":
    main()
