# CLAUDE.md — Bird Translation AI

Persistent working memory for this repo. Read this first when re-entering the project,
then `PROJECT_REBRIEF.md` and `REPO_ZONES.md` for full architectural context.
Keep this file updated as work progresses — it's the handoff document between sessions.

## What this is

Non-commercial art/research pipeline: listen to bird sounds -> identify species via
BirdNET -> discover within-species vocal patterns (clustering) -> assign patterns to
rough semantic categories (`song`, `call`, `alarm`, `juvenile_begging`) -> eventually
trigger music live/pseudo-live from `species + semantic label`.

Architecture:
`audio -> event/window segmentation -> BirdNET species detection -> phrase pooling ->
BirdNET embeddings -> HDBSCAN clusters -> semantic anchor assignment -> reviewed
semantic library -> future live/pseudo-live music trigger`

**Key decision (do not relitigate without reason):** no live reclustering. Offline we
build reviewed semantic prototypes per species. Live/pseudo-live inference should just
detect species, extract phrase/window embeddings, and assign to the nearest *reviewed*
semantic prototype, falling back to `unknown` on low confidence.

## Main code files

- `src/pipeline.py` — original all-in-one batch pipeline (segmentation, BirdNET, features,
  embeddings, clustering, reporting). Still explains the artifact layout.
- `src/pattern_layer.py` — bridge from a completed species run to within-species units.
  Decides accepted events, does phrase pooling, produces phrase-level embeddings.
- `src/assign_anchor_labels.py` — semantic layer. Compares pattern clips to reviewed
  anchor/reference clips, assigns nearest label, optionally cluster-smooths. `--copy_audio`
  writes `assigned_audio/<label>/` folders.
- `src/embed.py` — embedding backends (BirdNET embeddings is default/primary).
- `src/birdnet_infer.py` — BirdNET wrapper (species front-end, frozen/pretrained).
- `src/export_semantic_library.py` — exports the smaller per-species semantic library
  closer to deployment shape.
- Also relevant: `src/pattern_cluster.py` (HDBSCAN clustering), `src/anchor_match.py`
  (anchor-to-cluster scoring), `src/light_cluster_review.py` (fast review bundle w/
  representative clips), `src/prototype_assign.py` (nearest-prototype suggestions for
  noise units, writes `review_bundle_soft_assigned/`).

See `REPO_ZONES.md` for the full offline-vs-future-live-runtime file split.

## Google Drive roots

Root: `~/Library/CloudStorage/GoogleDrive-esmesturgessdurden@gmail.com/My Drive/BirdTranslationAI`

Key subfolders:
- `runs/species_decision`
- `runs/pattern_layers`
- `runs/within_species_clusters/01_phrase_pool_cluster_runs`
- `runs/semantic_assignments`
- `reference_audio`
- `raw_xeno_canto_blue_tit_v1`

## Species status

| Species | Status |
|---|---|
| **Blackbird** | Strongest semantic species so far. Has reference-label assignment outputs and manual review history. Treat as the clearest reference case, not the only training source. |
| **Wren** | Harder/ambiguous. `call` and `alarm` blur together; juvenile references exist. Improved via larger reference set + target-species top-k fallback acceptance, but still less stable than Blackbird. |
| **Wood Pigeon** | Phrase-pooled run exists. Semantic v1 and v2 exist with labels `song_call` + `alarm` (merged song/call for this species). Mostly `song_call`, a few `alarm` candidates. v2 candidate-clip fraction (ambiguous margin) ~0.59. |
| **Blue Tit** | Most recent work (see below). Full phrase-pool + semantic v1 run exists but needs by-ear review before trusting labels — very high ambiguous-margin fraction (~0.91). |

## Blue Tit — current state (most recent work before handoff)

Ran the same phrase-pool process as Wood Pigeon/Blackbird via
`scripts/run_blue_tit_phrase_pool.sh` (fetch xeno-canto -> BirdNET species-decision ->
`pattern_layer` phrase-pool -> `pattern_cluster` HDBSCAN). **Note: this script stops after
clustering — it does not call `assign_anchor_labels.py`.** The semantic v1 run was
invoked separately/manually.

Inputs:
- `raw_xeno_canto_blue_tit_v1`
- `runs/pattern_layers/blue_tit_pattern_layer_phrasepool_birdnet_windows_v1`
- `runs/within_species_clusters/01_phrase_pool_cluster_runs/blue_tit_pattern_clusters_phrasepool_birdnet_windows_v1`

Phrase-pool summary: 244 phrase units, 68 recordings with accepted events, 563 source
events used, 14 non-noise clusters, 129 noise events.

Xeno-canto label mix (raw metadata, not our clusters): 35 call, 31 song, 6 alarm call,
2 begging call, 1 "alarm call, call", 1 "call, song".

Semantic assignment v1: `runs/semantic_assignments/blue_tit_phrasepool_semantic_v1`
(labels: `song`, `call`, `alarm`, `juvenile_begging`; cluster smoothing on, margin
threshold 0.03, `copy_audio: false`).

Summary: alarm 86 clips/30 recordings, call 65 clips/26 recordings, juvenile_begging 13
clips/4 recordings, song 80 clips/27 recordings.

**Concern (confirmed real via `other_candidate_summary` in the run's
`anchor_label_assignment_summary.json`):** `candidate_clip_fraction` is ~0.906 (221/244
clips, 13/14 clusters flagged as ambiguous-margin candidates) — vs. Wood Pigeon v2's
~0.59. Blue Tit semantic labels are **not yet trustworthy** without by-ear review.

**Confirmed gap:** `blue_tit_phrasepool_semantic_v1/` only has a `reports/` folder — no
`assigned_audio/` and no `review_bundle_v1/`-style folder (Wood Pigeon has both, at
`.../wood_pigeon_phrasepool_songcall_alarm_v1/assigned_audio/` and `.../review_bundle_v1/`
with `song_call_examples/`, `alarm_candidates/`, `reference_support/`,
`review_bundle_index.csv`). So the review/export step for Blue Tit genuinely never ran
(not just "check if it finished" — it didn't start).

**Open mystery, unresolved:** the exact script/command that produced Wood Pigeon's
`review_bundle_v1`/`review_bundle_v2` folder shape (`song_call_examples/`,
`alarm_candidates/`, `reference_support/`, `review_bundle_index.csv`) is not obviously
in current `src/`. Checked and ruled out:
- `assign_anchor_labels.py --copy_audio` → writes `assigned_audio/<label>/`, not this shape.
- `prototype_assign.py` → writes `review_bundle_soft_assigned/` +
  `soft_assigned_review_bundle_index.csv`, close in spirit but different names/labels.
- Both Wood Pigeon runs' own `anchor_label_assignment_summary.json` say `copy_audio: false`,
  yet `assigned_audio/` exists on disk anyway — suggests a rerun or manual step happened
  after that summary was written, or the folder was populated by something outside
  `assign_anchor_labels.py` entirely.
- `colab/make_bundle.sh` is unrelated (packages the repo for Colab upload).

Before repeating the Wood Pigeon review-bundle step for Blue Tit, either find the actual
generating script/command (check Drive for a stray script/README next to the wood_pigeon
runs, or ask the user) or write a small new step — don't guess and recreate blind.

## Next actions (as of 2026-07-13 handoff)

1. Resolve the review-bundle mystery above (or ask the user directly — they may just
   remember the command).
2. Build/run whatever review-bundle step is decided on for
   `blue_tit_phrasepool_semantic_v1`, prioritizing the 13 ambiguous clusters
   (~90% of clips) for by-ear review before trusting labels.
3. Only after by-ear review should Blue Tit semantic labels be treated as reliable
   enough to feed the shared cross-species semantic layer alongside Blackbird/Wren/Wood
   Pigeon.

## Working notes

- This directory is not a git repo (`ls .git` → none as of 2026-07-13). No version
  control currently in place for this project.
- Don't feed whole audio datasets into context — reference Drive paths and run
  summary/report CSVs/JSONs instead.
- `PROJECT_REBRIEF.md` and `REPO_ZONES.md` are curated docs kept up to date by hand;
  update them (not just this file) if the architecture or species status materially
  changes.
