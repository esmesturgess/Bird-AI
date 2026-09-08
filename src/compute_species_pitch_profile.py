"""Compute a single species-level pitch profile (median fundamental frequency +
semitone offset from a reference pitch) from a sample of that species' own raw audio.

This is the offline half of "shift the response sound's pitch to roughly match how
high/low this species' voice is" (see CLAUDE.md, pitch-shift-by-species prototype).
One number per species, applied uniformly to whichever response track gets triggered
-- not a per-call-type profile, matching the "wood pigeon would lower ALL the songs"
framing the feature was requested with.

Pitch is estimated per clip with librosa.pyin (robust to unvoiced/noisy sections,
picks a fundamental within a bird-appropriate frequency band), median-pooled per
clip, then median-pooled ACROSS CATEGORIES (not across clips) so a category with far
more available audio (e.g. song) doesn't dominate the species estimate.

Shifts are reported relative to a REFERENCE SPECIES, not a fixed musical pitch (like
440 Hz) -- bird vocalizations sit ~2 octaves above typical music, so an absolute
reference produces unusably extreme shifts. The first species run becomes the
reference (0 semitones by definition, `data/species_pitch_profiles/_reference.json`
is written); every later species is shifted relative to it, which also directly
matches the original ask ("wood pigeon would lower the pitch" -- lower relative to
what's already there, i.e. Blackbird).

    ./.venv/bin/python -m src.compute_species_pitch_profile \
        --species_slug blackbird \
        --audio_glob "data/raw/blackbird_macaulay*/**/*.mp3" \
        --audio_glob "data/raw/blackbird_xc_eu_v1_alarm/**/*.mp3" \
        --audio_glob "data/raw/blackbird_xc_eu_v1_juv_*/**/*.mp3" \
        --output_dir data/species_pitch_profiles
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import librosa
import numpy as np

from .utils import ensure_dir, save_json

FMIN_HZ = 500.0   # bird vocalizations sit well above the human voice range
FMAX_HZ = 8000.0

CATEGORY_RE = re.compile(r"(song|call|alarm|juvenile)", re.I)


def infer_category(path: Path) -> str | None:
    m = CATEGORY_RE.search(path.name) or CATEGORY_RE.search(str(path.parent))
    if not m:
        return None
    cat = m.group(1).lower()
    if "juv" in str(path).lower():
        return "juvenile"
    return cat


def clip_median_f0(path: Path) -> float | None:
    try:
        y, sr = librosa.load(path.as_posix(), sr=22050, mono=True)
    except Exception:
        return None
    if y.size < sr * 0.2:
        return None
    f0, voiced_flag, _ = librosa.pyin(y, fmin=FMIN_HZ, fmax=FMAX_HZ, sr=sr)
    voiced = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
    voiced = voiced[~np.isnan(voiced)]
    if voiced.size < 5:
        return None
    return float(np.median(voiced))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--species_slug", required=True, help="e.g. blackbird -- used as the output filename.")
    p.add_argument("--audio_glob", action="append", required=True,
                    help="Repeatable glob pattern (recursive, use **) for raw audio to sample from.")
    p.add_argument("--max_per_category", type=int, default=20)
    p.add_argument("--output_dir", type=Path, default=Path("data/species_pitch_profiles"))
    p.add_argument("--reference_hz", type=float, default=None,
                    help="Override the reference pitch explicitly. Default: read "
                         "_reference.json in --output_dir, or become the reference "
                         "(0 semitones) if none exists yet.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    paths: list[Path] = []
    for pattern in args.audio_glob:
        paths.extend(Path(p) for p in glob.glob(pattern, recursive=True))
    paths = sorted(set(paths))
    by_cat: dict[str, list[Path]] = {}
    for p in paths:
        cat = infer_category(p)
        if cat:
            by_cat.setdefault(cat, []).append(p)

    if not by_cat and paths:
        print("no song/call/alarm/juvenile labels found in filenames -- pooling all "
              "clips into a single 'all' group (species-level estimate only).")
        by_cat["all"] = paths

    print(f"found {len(paths)} candidate files -> categories: "
          f"{ {k: len(v) for k, v in by_cat.items()} }")

    category_medians: dict[str, float] = {}
    category_n_clips: dict[str, int] = {}
    for cat, files in sorted(by_cat.items()):
        sample = list(rng.choice(files, size=min(args.max_per_category, len(files)), replace=False))
        f0s = []
        for f in sample:
            f0 = clip_median_f0(f)
            if f0 is not None:
                f0s.append(f0)
        if not f0s:
            print(f"  {cat}: no usable pitch estimates, skipping")
            continue
        category_medians[cat] = float(np.median(f0s))
        category_n_clips[cat] = len(f0s)
        print(f"  {cat}: median F0 = {category_medians[cat]:.1f} Hz over {len(f0s)}/{len(sample)} usable clips")

    if not category_medians:
        raise SystemExit("No category yielded a usable pitch estimate.")

    species_f0 = float(np.median(list(category_medians.values())))

    ensure_dir(args.output_dir)
    reference_path = args.output_dir / "_reference.json"
    is_reference_species = False
    if args.reference_hz is not None:
        reference_hz = args.reference_hz
        reference_species_slug = None
    elif reference_path.exists():
        ref = json.loads(reference_path.read_text())
        reference_hz = ref["reference_hz"]
        reference_species_slug = ref["reference_species_slug"]
    else:
        reference_hz = species_f0
        reference_species_slug = args.species_slug
        is_reference_species = True
        save_json(reference_path, {"reference_species_slug": args.species_slug, "reference_hz": reference_hz})
        print(f"\nNo reference species set yet -- '{args.species_slug}' becomes the reference "
              f"({reference_hz:.1f} Hz = 0 semitones). Future species will be shifted relative to it.")

    semitone_shift = 12.0 * np.log2(species_f0 / reference_hz)

    profile = {
        "species_slug": args.species_slug,
        "species_median_f0_hz": species_f0,
        "reference_hz": reference_hz,
        "reference_species_slug": reference_species_slug,
        "is_reference_species": is_reference_species,
        "semitone_shift_vs_reference": semitone_shift,
        "category_median_f0_hz": category_medians,
        "category_n_clips_used": category_n_clips,
        "note": "semitone_shift_vs_reference is ONE species-level number (median across "
                "categories, not clips) meant to be applied uniformly to whichever response "
                "track this species triggers. Reference is another SPECIES (see "
                "reference_species_slug), not a fixed musical pitch -- see module docstring.",
    }
    out_path = args.output_dir / f"{args.species_slug}.json"
    save_json(out_path, profile)
    print(f"\nspecies median F0: {species_f0:.1f} Hz  ->  semitone shift vs reference species "
          f"'{reference_species_slug}' ({reference_hz:.1f} Hz): {semitone_shift:+.2f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
