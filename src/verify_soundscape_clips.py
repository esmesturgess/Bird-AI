"""Find clips the classifier actually gets right, to build the soundscape from.

Each candidate is trimmed to the segment length the installation will use, classified,
and scored against its known label. Only agreeing clips are kept.
"""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, librosa, soundfile as sf

sys.path.insert(0, "/Users/esmesturgess/Bird Translation AI")
from src.classify_clip import Classifier

ROOT = Path("/Users/esmesturgess/Bird Translation AI")
OUT = ROOT / "data/soundscapes/candidates"
OUT.mkdir(parents=True, exist_ok=True)
SEG = 25.0                        # user set 2026-09-08: more headroom for live classification
PER_CELL = 6                      # candidates to try per species x vocalisation

def candidates():
    rows = []
    # blackbird — freshly fetched, category from directory
    for cat, d in [("song","bb_demo_song"),("call","bb_demo_call"),
                   ("alarm","bb_demo_alarm"),("juvenile","bb_demo_juv")]:
        for f in sorted((ROOT/"data/raw"/d).rglob("*.mp3"))[:PER_CELL]:
            rows.append(("blackbird", cat, f))
    # tawny owl — category from the fetch directory
    for cat, d in [("song","owl_xc_eu_v1_song"),("call","owl_xc_eu_v1_call"),
                   ("alarm","owl_xc_eu_v1_alarm"),("juvenile","owl_xc_eu_v1_juv_begging")]:
        for f in sorted((ROOT/"data/raw"/d).rglob("*.mp3"))[:PER_CELL]:
            rows.append(("tawny_owl", cat, f))
    # robin — labels live in the Macaulay manifest
    man = pd.read_csv(ROOT/"data/raw/robin_macaulay_eu_v1/macaulay_label_manifest.csv")
    man = man[man.local_audio_path.notna()]
    for cat in ["song","call","alarm","juvenile"]:
        sub = man[man.category == cat].head(PER_CELL)
        for p in sub.local_audio_path:
            if Path(p).exists():
                rows.append(("robin", cat, Path(p)))
    return rows

cands = candidates()
print(f"{len(cands)} candidates", flush=True)
clf = Classifier(ROOT/"models", birdnet_binary=str(ROOT/".venv/bin/birdnet-analyze"))
print("classifier ready", flush=True)

results = []
t0 = time.time()
for i, (species, cat, src) in enumerate(cands, 1):
    try:
        y = librosa.load(str(src), sr=16000, mono=True, duration=SEG)[0]
        if len(y) < 16000 * 3:
            continue
        y = y / (np.abs(y).max() + 1e-9) * 0.9
        seg = OUT / f"{species}_{cat}_{src.stem}.wav"
        sf.write(seg, y, 16000)
        r = clf.classify(seg)
        ok = (r.species == species) and (r.vocalisation == cat)
        results.append({"species": species, "expected": cat, "file": seg.name,
                        "got_species": r.species, "got_vocalisation": r.vocalisation,
                        "species_conf": round(r.species_confidence, 3),
                        "match_conf": round(r.vocalisation_confidence, 3), "correct": ok})
        if not ok:
            seg.unlink(missing_ok=True)
        print(f"  [{i}/{len(cands)}] {species:10s} {cat:9s} -> "
              f"{str(r.species):10s} {str(r.vocalisation):9s} {'OK' if ok else 'x'}", flush=True)
    except Exception as e:
        print(f"  [{i}/{len(cands)}] {species} {cat} FAILED {type(e).__name__}: {e}", flush=True)

df = pd.DataFrame(results)
df.to_csv(ROOT/"data/soundscapes/verification.csv", index=False)
print(f"\ndone in {(time.time()-t0)/60:.1f} min")
print(f"correct: {int(df.correct.sum())}/{len(df)}")
print(pd.crosstab(df.species, df.correct).to_string())
print("\nverified clips per species x vocalisation:")
print(df[df.correct].groupby(["species","expected"]).size().to_string())
