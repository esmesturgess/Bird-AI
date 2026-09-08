"""Merge Xeno-canto + Macaulay fetches for one species into a single pipeline-ready
dataset: `pipeline_manifest.csv` (absolute audio paths) + `call_type_ground_truth.csv`
(kept SEPARATE so the labels survive downstream processing), plus a REVIEW CSV listing
every clip whose label is not clean enough to trust blindly.

Why the review step exists (Wren post-mortem, 2026-08-14): the earlier XC path labelled a
recording purely by which `type:` query returned it, so a recording tagged
`"call, song"` or `"dawn song, song"` silently entered the `song` bucket. Here the label is
re-derived from the RAW `sound_type` + `stage` metadata, and anything with signals from more
than one category -- or from an excluded category like dawn song -- is flagged rather than
guessed.

Taxonomy (matches the Blackbird/Macaulay convention):
  song      <- song, subsong, stereotyped/nocturnal song   (dawn song + flight song EXCLUDED)
  call      <- call, flight call, contact call
  alarm     <- alarm call, agitation, scold, distress, mobbing
  juvenile  <- begging call, or stage == juvenile
Priority when several fire: juvenile > alarm > (song XOR call; both = ambiguous).

    ./.venv/bin/python -m src.build_species_dataset \
        --species_slug robin --species_label robin \
        --scientific_name "Erithacus rubecula" --common_name "European Robin" \
        --xc_dir data/raw/robin_xc_eu_v1_song --xc_dir data/raw/robin_xc_eu_v1_call \
        --xc_dir data/raw/robin_xc_eu_v1_alarm --xc_dir data/raw/robin_xc_eu_v1_juv_begging \
        --xc_dir data/raw/robin_xc_eu_v1_juv_stage \
        --macaulay_dir data/raw/robin_macaulay_eu_v1 \
        --output_dir data/raw/robin_combined_v1
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from .utils import ensure_dir

# --- token -> category, applied to comma-separated xeno-canto `sound_type` values ---
# `duet` is EXCLUDED: a duet recording contains two birds answering each other, so it carries
# both a song and a call voice in one clip and cannot be given a single label. This matters
# most for Tawny Owl, where the classic "twit-twoo" is a female kewick answered by a male
# hoot and ~13% of recordings are duets (user decision 2026-09-03, matching how Robin's
# song+call conflicts were handled).
EXCLUDE_RE = re.compile(r"dawn song|flight song|duet", re.I)
# `hoot` is the owl equivalent of territorial song; `keewick`/`kewick` is the owl contact call.
SONG_RE = re.compile(r"^\s*(sub)?song\s*$|stereotyped song|nocturnal song|hoot", re.I)
CALL_RE = re.compile(r"^\s*call\s*$|flight call|contact call|kee?wick", re.I)
ALARM_RE = re.compile(r"alarm|agitation|aggression|scold|distress|mobbing", re.I)
JUV_RE = re.compile(r"begging", re.I)


def classify_xc(sound_type: str, stage: str) -> tuple[str, str]:
    """Return (category, reason). category may be 'ambiguous' or 'excluded'."""
    raw = str(sound_type or "")
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    stage_juv = "juvenile" in str(stage or "").lower()

    if EXCLUDE_RE.search(raw):
        return "excluded", f"excluded sound type (dawn/flight song) in {raw!r}"

    sig = set()
    for t in tokens:
        if JUV_RE.search(t):
            sig.add("juvenile")
        elif ALARM_RE.search(t):
            sig.add("alarm")
        elif SONG_RE.search(t):
            sig.add("song")
        elif CALL_RE.search(t):
            sig.add("call")
    if stage_juv:
        sig.add("juvenile")

    if not sig:
        return "excluded", f"no recognised category signal in sound_type={raw!r} stage={stage!r}"

    # priority: juvenile > alarm > song/call, but record when we overrode something
    if "juvenile" in sig:
        others = sig - {"juvenile"}
        if others:
            return "juvenile", f"juvenile wins over {sorted(others)}; raw={raw!r} stage={stage!r}"
        return "juvenile", f"raw={raw!r} stage={stage!r}"
    if "alarm" in sig:
        others = sig - {"alarm"}
        if others:
            return "alarm", f"alarm wins over {sorted(others)}; raw={raw!r}"
        return "alarm", f"raw={raw!r}"
    if sig == {"song", "call"}:
        return "ambiguous", f"both song and call signals, no override; raw={raw!r}"
    return sig.pop(), f"raw={raw!r}"


def needs_review(category: str, reason: str) -> bool:
    """Flag anything not cleanly derived from a single unambiguous signal."""
    return category == "ambiguous" or "wins over" in reason


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--species_slug", required=True)
    p.add_argument("--species_label", required=True)
    p.add_argument("--scientific_name", required=True)
    p.add_argument("--common_name", required=True)
    p.add_argument("--xc_dir", type=Path, action="append", default=[])
    p.add_argument("--macaulay_dir", type=Path, action="append", default=[])
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--exclude_ids_csv", type=Path, default=None,
                    help="CSV with a 'recording_id' column of clips a human review rejected "
                         "(e.g. mixed song+alarm content). Applied BEFORE the balance cap so a "
                         "dropped clip is replaced by the next-best one rather than lost.")
    p.add_argument("--max_per_category", type=int, default=None,
                    help="Cap recordings per category for BALANCE (the alarm-crowding lesson: a "
                         "category with far more data sharpens its centroid and steals its "
                         "neighbours' test points). Higher-quality recordings are kept first.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# Quality is expressed differently per source: xeno-canto uses letters A(best)..E, Macaulay
# uses numeric ratings 1..5 (5 best). Both are normalised to the SAME 0.0(best)..1.0(worst)
# scale so a cap draws fairly from both. An earlier version returned raw XC ranks (0..4) but
# NEGATED Macaulay ratings (5 -> -5.0), so every Macaulay clip sorted ahead of every XC clip
# and `--max_per_category` silently produced single-source categories -- which confounded
# category with recording source in the Robin dataset (see CLAUDE.md 2026-08-31).
_XC_QUALITY_RANK = {"A": 0.0, "B": 0.25, "C": 0.5, "D": 0.75, "E": 1.0}


def _quality_sort_key(value) -> float:
    s = str(value).strip().upper()
    if s in _XC_QUALITY_RANK:
        return _XC_QUALITY_RANK[s]
    try:
        rating = float(s)  # Macaulay 1..5, higher is better
    except ValueError:
        return 2.0  # unknown quality sorts after everything scored
    return min(max((5.0 - rating) / 4.0, 0.0), 1.0)


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    review: list[dict] = []

    # ---------------- Xeno-canto ----------------
    for d in args.xc_dir:
        meta_path = d / "xc_recordings_metadata.csv"
        if not meta_path.exists():
            print(f"  ! skipping {d} (no xc_recordings_metadata.csv)")
            continue
        df = pd.read_csv(meta_path)
        for _, r in df.iterrows():
            audio = str(r.get("local_audio_path") or "").strip()
            if not audio or not Path(audio).exists():
                continue
            cat, reason = classify_xc(r.get("sound_type", ""), r.get("stage", ""))
            rec_id = f"XC{r['xc_id']}_{args.scientific_name.replace(' ', '_')}"
            entry = {
                "recording_id": rec_id, "source": "xeno_canto", "category": cat,
                "reason": reason, "audio_path": str(Path(audio).resolve()),
                "raw_sound_type": r.get("sound_type", ""), "raw_stage": r.get("stage", ""),
                "remarks": str(r.get("remarks", ""))[:300], "quality": r.get("quality", ""),
                "recordist": r.get("recordist", ""),
            }
            rows.append(entry)
            if needs_review(cat, reason):
                review.append(entry)

    # ---------------- Macaulay ----------------
    for d in args.macaulay_dir:
        man_path = d / "macaulay_label_manifest.csv"
        if not man_path.exists():
            print(f"  ! skipping {d} (no macaulay_label_manifest.csv)")
            continue
        df = pd.read_csv(man_path)
        for _, r in df.iterrows():
            audio = str(r.get("local_audio_path") or "").strip()
            if not audio or audio == "nan" or not Path(audio).exists():
                continue
            cat = str(r.get("category", ""))
            rec_id = f"ML{r['assetId']}_{args.scientific_name.replace(' ', '_')}"
            entry = {
                "recording_id": rec_id, "source": "macaulay", "category": cat,
                "reason": str(r.get("reason", ""))[:300],
                "audio_path": str(Path(audio).resolve()),
                "raw_sound_type": r.get("tags", ""), "raw_stage": "",
                "remarks": str(r.get("mediaNotes", ""))[:300], "quality": r.get("rating", ""),
                "recordist": r.get("userDisplayName", ""),
            }
            rows.append(entry)
            if cat == "ambiguous":
                review.append(entry)

    if not rows:
        raise SystemExit("No audio found — did the fetches complete?")

    all_df = pd.DataFrame(rows).drop_duplicates(subset="recording_id", keep="first")
    print(f"\ntotal clips with audio on disk: {len(all_df)}")
    print("category breakdown (before dropping non-usable):")
    print(all_df["category"].value_counts().to_string())

    usable = all_df[all_df["category"].isin(["song", "call", "alarm", "juvenile"])].copy()
    print(f"\nusable (4 real categories): {len(usable)}")
    print(usable["category"].value_counts().to_string())

    if args.exclude_ids_csv and args.exclude_ids_csv.exists():
        drop = set(pd.read_csv(args.exclude_ids_csv)["recording_id"])
        hit = usable["recording_id"].isin(drop)
        print(f"\nhuman review exclusions from {args.exclude_ids_csv.name}: dropping {int(hit.sum())} "
              f"of {len(drop)} listed")
        usable = usable[~hit]
        print(usable["category"].value_counts().to_string())

    if args.max_per_category:
        before = len(usable)
        usable["_q"] = usable["quality"].map(_quality_sort_key)
        usable = (usable.sort_values("_q", kind="stable")
                        .groupby("category", group_keys=False)
                        .head(args.max_per_category)
                        .drop(columns="_q"))
        print(f"\ncapped to {args.max_per_category}/category for balance: {before} -> {len(usable)}")
        print(usable["category"].value_counts().to_string())

    out = args.output_dir
    ensure_dir(out)

    manifest = pd.DataFrame({
        "recording_id": usable["recording_id"],
        "audio_path": usable["audio_path"],
        "species_label": args.species_label,
        "species_common_name": args.common_name,
        "scientific_name": args.scientific_name,
        "species_source": usable["source"],
        "recording_source": "web",
        "split": "train",
        "notes": "call_type=" + usable["category"],
    })
    manifest.to_csv(out / "pipeline_manifest.csv", index=False)
    usable[["recording_id", "category"]].rename(
        columns={"category": "call_type_ground_truth"}
    ).to_csv(out / "call_type_ground_truth.csv", index=False)

    review_df = pd.DataFrame(review)
    if not review_df.empty:
        review_df = review_df.drop_duplicates(subset="recording_id")
        review_df.insert(0, "manual_label_override", "")
        review_df.to_csv(out / "labels_needing_review.csv", index=False)

    print(f"\nWrote: {out/'pipeline_manifest.csv'}  ({len(manifest)} rows)")
    print(f"Wrote: {out/'call_type_ground_truth.csv'}")
    if not review_df.empty:
        print(f"Wrote: {out/'labels_needing_review.csv'}  ({len(review_df)} rows NEED A SECOND LOOK)")
        print("  breakdown of flagged rows:")
        print("  " + review_df["category"].value_counts().to_string().replace("\n", "\n  "))
    else:
        print("No labels flagged for review.")


if __name__ == "__main__":
    main()
