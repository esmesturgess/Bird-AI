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

## Adapter / "our AI" experiments (2026-07-14) — READ before more adapter work

Decided to pursue **contrastive adapter + prototypes** (NOT a softmax classifier) as the
"train our own AI on top of frozen BirdNET" layer — it keeps the nearest-prototype live
design, gives unknown-fallback via distance, and extends to new species by adding
prototypes. The machinery already exists: `src/adapter_model.py` (`EmbeddingAdapter` MLP
1024->256->128 + `PairContrastiveLoss`), `src/train_adapter.py` (trains on
`clip_pair_labels_v1.csv` same/different rows), `assign_anchor_labels.py --adapter_checkpoint`.
All runs so far used `adapter=identity` (off).

**Key fix identified:** the adapter's training pairs must be defined by SEMANTIC LABEL
(same label = positive, diff = negative), NOT the acoustic "canonical group" map. Measured
on Blackbird: the existing `manual_cluster_group_map`/`clip_pair_labels` "same" pairs
bridge different semantic labels 70% (cluster-level) / 35% (clip-level) of the time — i.e.
training on them pulls alarm toward song. `review_pairs.py` should regenerate pairs from
the per-cluster semantic labels instead.

**Blackbird validation result (the reason to test before building):** built an honest
eval (5-fold by RECORDING, baseline raw-space vs adapter, nearest-prototype accuracy +
overfit check). Findings:
- Only 121 of 181 blackbird clips are labelled (representatives only); 29 recordings; 3
  imbalanced classes (song 73 / alarm 34 / call 14; majority floor 0.60).
- Adapter OVERFITS hard: train acc 1.000, test acc ~0.55 (below majority). Regularizing
  barely helped. 121 clips is far too little to train even the small MLP.
- Raw BirdNET baseline is AT the majority floor (0.601) on held-out recordings — raw space
  barely separates within-species function (confirms the "clustering optimizes wrong axis"
  concern empirically, for generalization).
- Conclusion: cannot validate the adapter per-species; the binding constraint is DATA, not
  model choice. "Blackbird is strongest" was an IN-SAMPLE judgment; it does not generalize
  to held-out recordings in this test.

**Getting more data — what was tried and ruled out (2026-07-14):**
- Cross-species pooling is blocked by two things: (a) each species was labelled with a
  DIFFERENT scheme — blackbird {song,call,alarm}, wren {song,call} (alarm merged), wood
  pigeon {song_call,alarm} (song+call merged), blue_tit {song,call,alarm,juvenile_begging}
  — so pooling needs a shared-taxonomy decision first (this IS the "is alarm the same
  across species" question); (b) full clip->cluster data + wood_pigeon/house_sparrow labels
  live on the online-only Google Drive that keeps timing out.
- Xeno-canto recordist `type` tags as bulk labels: DE-RISKED AND FAILED. Queried XC v3 for
  the 46 blackbird recordings; XC per-recording tags agree only 24% with manual per-clip
  labels (BELOW 3-class chance). Reasons: per-recording granularity (one tag for a whole
  recording whose phrases you labelled individually) + different vocabulary (flight call,
  nocturnal flight call, dawn song, subsong). NOT usable as training truth. Do not retry
  naively.
- Remaining realistic data levers: full-cluster label expansion (all clips in reviewed
  clusters, not just representatives) + data augmentation (pitch/time/noise — cheap
  multiplier, reuses trusted labels) + more human review (only reliable source of new good
  labels). All need the Drive files available offline.

**Embedding-space diagnostics + lever tests (2026-07-14, Blackbird, 121 cluster-derived
labels):**
- Probed the raw BirdNET embedding space: it is NOT collapsed (72/1024 dims for 90%
  variance) — but it is **dominated by recording/individual identity, not call function**:
  nearest-neighbour shares recording 0.92 (chance 0.03), silhouette by recording 0.42 vs by
  label 0.115; linear probe for function ~0.60–0.63 ≈ majority floor. So within-species
  variation IS retained, just the wrong kind (identity, not function). **Replicated on the
  fresh 329-unit blackbird set (2026-07-14): NN-same-recording 0.94 vs 0.02 chance, silhouette
  by recording 0.31 — recording-dominance is a stable property, and 3x more raw data did NOT
  shift the axis toward function (108/1024 dims for 90%; HDBSCAN gave 1 giant cluster + small
  satellites).** Confirms: more raw volume alone won't help; need recording-invariance + labels.
- Tried two fixes, BOTH FAILED to beat the 0.603 majority floor on held-out recordings:
  (1) cross-recording contrastive pairs (same label / different recording) → 0.55;
  (2) adding recording-invariant hand-crafted features (urgency_index, song_likeness, etc.)
  → features-only 0.35, embedding+features 0.59, fused prototypes hurt. Six conditions total,
  all at/below floor.
- **Key caveat: these Blackbird labels are cluster-derived from the OLD windows run whose
  grouping was 70% internally inconsistent — so the evaluation target itself is unreliable.
  Cannot separate "representation lacks signal" from "labels are noise."** The experiment is
  data/label-limited and INCONCLUSIVE on the representational question.
- **Do NOT re-run these modeling experiments until there are clean, trustworthy labels + more
  data.** They will keep returning "at the floor." Going back layers in BirdNET is NOT the fix
  (earlier layers retain MORE identity/nuisance, not more function). Scratchpad has the
  reusable probes/harness (`embedding_probe.py`, `cross_recording_exp.py`, `features_exp.py`).
  aug_uk features CSV was pulled from Drive to `scratchpad/bb_auguk_features.csv`.

**Strategic stance (2026-07-14):** the generalising AI is genuinely data-constrained.
Protect the INSTALLATION (precomputed, human-reviewed labels — needs no generalisation) as
the deliverable for the fixed deadline; pursue the adapter as a BOUNDED experiment on
expanded+augmented existing labels, with tempered expectations. The honest eval harness
(train/test by recording, baseline-vs-adapter, overfit check) lives in the session
scratchpad and should be promoted into a proper `src/` script + committed.

**Xeno-canto API key:** user shared it in chat; stored OUTSIDE the repo at
`scratchpad/xc.env`; `.env`/`*.env` are gitignored. Recommend the user ROTATE the key
since it appeared in chat history. Long-term home: shell profile or gitignored `.env`.

## Live deployment architecture (decided 2026-07-14)

**Key decision: no microphone in the live installation.** Originally planned as a mic
hidden in a nest-cone prop capturing gallery audio acoustically (source speaker -> open
air -> mic -> detection -> response speaker). Dropped in favor of a fully digital path:
the Pi plays a curated source bird-sound clip directly to Speaker A, and simultaneously
runs BirdNET species detection + embedding + nearest-semantic-prototype matching on that
clip's own clean in-memory audio buffer (no capture step at all) -> triggers Speaker B
with the matching response clip.

**Why:** eliminates gallery-noise SNR problems, removes the need for live denoising
(was being considered specifically to handle noisy mic capture), and removes the
feedback-loop risk between the two speakers (mic hearing its own triggered response and
misfiring) — all in one move, since there's no acoustic capture loop to have those
problems in the first place. The nest cone, if kept visually, is now purely sculptural;
no mic/audio-interface hardware needed in the live device.

**How to apply:** any future `live_audio.py`/`live_species.py` runtime work (see
`REPO_ZONES.md`'s Pi runtime sketch) should be built around "play known clip + classify
own buffer," not "capture unknown ambient audio." Don't reintroduce mic-capture code
paths without a documented reason to revisit this decision.

## Live hardware choice (revised 2026-07-14)

**CURRENT DECISION: buying a used x86 mini PC (Intel NUC-class, refurbished, Linux Mint
pre-installed) — NOT the Raspberry Pi, despite the Pi analysis below.** Why the reversal:
x86_64 removes even the residual ARM dependency risk entirely (every wheel is standard),
and since the live device is moving toward *precomputed labels + playback lookup* (not
live inference), raw speed stopped mattering — so a cheap older x86 box is fine. The old
i3-5xxx NUCs are the slowest option on paper but adequate for playback + occasional
lookup. Get one that includes a power adapter and (ideally) ships with Linux, not Windows.
The ARM investigation below is kept as the record of *why the Pi would also have worked* —
don't redo it.

---

**Earlier reasoning (Raspberry Pi 4/5, 8GB, 64-bit Raspberry Pi OS), not NVIDIA Jetson:**
- The live workload is CPU-cheap and sporadic (short BirdNET inference + cosine nearest-
  prototype match), so GPU acceleration (Jetson's main selling point) buys nothing —
  BirdNET's TFLite core is designed to run on Pi-class CPUs.
- Jetson inherits ARM's dependency-wheel risk *plus* NVIDIA's own JetPack/CUDA-version
  lock-in on top (can't just `pip install` standard TensorFlow/PyTorch) — strictly worse
  than plain Pi for this project's needs.
- Empirically tested (2026-07-14, via `pip download --platform` cross-arch wheel checks,
  no hardware needed) that the previously-worried-about ARM dependency risk is real but
  narrow and now resolved: `torch`, `perch-hoplite`, `ml-collections` all have genuine
  prebuilt `manylinux...aarch64` wheels for Python 3.12. `birdnet-analyzer[embeddings]`
  2.0.0–2.2.0 pinned an old `tensorflow==2.15.1` with zero Python 3.12 support anywhere
  (not an ARM issue, a Python-version issue) — but current `birdnet-analyzer` 2.4.0 (what's
  actually installed locally) loosened this to `tensorflow>=2.20`, and `tensorflow==2.20.0`
  has a confirmed real Linux aarch64 Python-3.12 wheel. Net: no known ARM install blocker
  remains for the live dependency stack.
- hdbscan (the one package most likely to need compiling on ARM) doesn't matter for the
  live device at all — clustering is offline-only per the "no live reclustering" decision.
- Still TODO: real Pi hardware for actual runtime speed/latency, thermal/soak testing —
  wheel availability was confirmed, but on-device performance was not.

**DONE (2026-08-09):** trimmed `requirements-deploy.txt` created (inference + playback
only; excludes `umap-learn`, `hdbscan`, `matplotlib`, `jinja2`). Plus `scripts/setup_nuc.sh`
(one-shot x86 Linux setup: apt packages incl. `espeak-ng`, venv, deploy deps,
`birdnet-analyzer[embeddings]`, sanity checks) and `scripts/README_NUC.md` (step-by-step to
run the `infer_clip.py --speak` demo on the NUC). Not yet run on real NUC hardware.

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

**Review bundle — RESOLVED (2026-07-14, by Codex):** Codex built the Blue Tit review
bundle as an *index-only* bundle (Drive audio-copy was too slow), at
`.../blue_tit_phrasepool_semantic_v1/review_links_v1/` with `blue_tit_review_index_v1.csv`
+ `README.md`. It points at existing audio instead of duplicating it. Contents: 8 assigned
examples each for alarm/call/song, 4 for juvenile_begging, 14 cluster examples (one per
non-noise cluster), all reference anchors (6 alarm, 7 call, 2 juvenile, 5 song). This is
the current review mechanism; the old "how was Wood Pigeon's bundle made" mystery is moot.

**Key finding from cluster votes (`cluster_anchor_label_votes.csv`):** `alarm` is the top
or runner-up label in 12 of 14 clusters — it's entangled with nearly everything. Confusion
splits into `call↔alarm` (clusters 0,4,5,8,10,13) and `song↔alarm` (3,6,9,11,12); cluster 5
is a dead tie (margin 0.00002). Only clusters 2 and 7 don't involve alarm. Strong signal
that either the `alarm` reference anchors are too diffuse/central, or `call`+`alarm` should
merge (as Wood Pigeon merged song+call). By-ear review decides.

**Decision sheet prepared:** `data/reviews/blue_tit_cluster_semantic_labels_v1.csv` (git-
tracked, follows the blackbird/wren convention, consumable by `apply_manual_anchor_overrides.py`).
Pre-filled per cluster with algo top/second label, margin, size, and a "listen_for" hint,
sorted smallest-margin-first. `manual_anchor_label` column is BLANK for the user to fill
after listening. NOTE: the user has NOT yet done the by-ear review (as of 2026-07-14).

## Project direction (set 2026-07-14) — two parallel tracks

Timeline: **fixed exhibition date, build in parallel.** Adapter committed, but only AFTER
Blue Tit review. Blue Tit review still to do (user only — requires ears).

**Track A — the "our AI" research spine (critical path):**
1. User does Blue Tit by-ear review via `review_links_v1`, fills
   `data/reviews/blue_tit_cluster_semantic_labels_v1.csv`, and decides taxonomy:
   keep 4-way (song/call/alarm/juvenile_begging) vs merge (likely `song`/`call_alarm`/
   `juvenile_begging`). Apply via `apply_manual_anchor_overrides.py`.
2. Lock academic dataset: train/val/test splits **by `recording_id`, never by clip**
   (70/15/15), tracked by species/label/recordist/country; separate field/live split.
   Flag rare labels (juvenile_begging ~2-4 recordings) as insufficient to evaluate yet.
   Existing `data/manifests/` already carry per-recording split columns for wood_pigeon/wren.
3. Train the adapter — the currently-UNUSED metric-learning layer that makes this "our AI"
   rather than "BirdNET + spreadsheets": `review_pairs.py` → `train_adapter.py` →
   `transform_embeddings.py` (see `src/adapter_model.py`). All runs so far use
   `"adapter_used_for_anchors": "identity"` (raw BirdNET space, adapter off). Turning it on
   is the deepest fix for the ambiguity AND the legitimacy story AND what gives train/val/
   test meaning. Re-run semantic assignment *through* the adapter; measure ambiguous-margin
   drop on validation.
4. Honest evaluation on the locked test split (unseen recordings).

**Track B — the installation (runs alongside, non-blocking):**
Build the thin live runtime (`live_*.py` from `REPO_ZONES.md`, none exist yet) +
`requirements-deploy.txt`, curate prompt→response clip library, wire two speakers
(x86 box's 3.5mm jack → L=Speaker A, R=Speaker B works; no mic), soak-test. Open sub-
decision: precompute labels offline (device just plays + looks up — simplest/robust) vs
live inference. Leaning precompute. `requirements-deploy.txt` is decision-independent and
a safe first Track B task.

## Live-runtime seed (2026-08-09)

`scripts/infer_clip.py` — the first end-to-end runtime seed (the `live_*.py` REPO_ZONES
sketch didn't exist before this). Single audio clip -> BirdNET species (aggregated over
3s windows, best score per species) -> friendly name -> optional **spoken output**
(`--speak`: macOS `say` / Linux `espeak-ng`; confidence gate speaks "Unknown" below
`--min_conf`). Milestone 0 (species) WORKS: on a clean blackbird clip
(`XC128838_Turdus_merula.mp3`) it detects `Turdus merula` @ 0.995 and speaks "Blackbird".
Runs identically on the x86 Intel NUC. Lesson from testing: feed CLEAN/curated clips —
raw multi-bird XC recordings give noisy per-clip species (one topped out as Chiffchaff);
matches the installation design (play a known curated clip).
**Milestone 1 (call-type) NOT wired yet:** phrase builder already supports
`"{species}. {call_type}"`; needs reviewed blackbird anchor clips/manifest (on Drive) +
a nearest-prototype call in the `# milestone 1 hook`. Output would SAY a call-type but it
would be UNRELIABLE (the unsolved label problem). Milestone 2 = full install loop
(play->classify->trigger 2nd speaker), not built.

## Working notes

- Git repo initialized 2026-07-13 (root commit `f439c9d`, branch `master`). `.gitignore`
  excludes `.venv/`, `__pycache__/`, and generated pipeline artifacts under `data/`
  (`raw/`, `events/`, `preds/`, `features/`, `specs/`, `embeddings/`, `clusters/`,
  `reports/`, `runs/`) per `STORAGE_POLICY.md` — those live on Google Drive, not in git.
  Curated hand-edited files (`data/manifests/`, `data/anchors/`, `data/reviews/`) ARE
  tracked. Remote: `origin` = `git@github.com:esmesturgess/Bird-AI.git` (SSH; pushed
  2026-07-14). `.env`/`*.env` also gitignored (local secrets, e.g. `XC_API_KEY`).
- **Results storage — planned change (decided 2026-07-14):** results are currently flat
  files (CSV for tabular results/manifests/labels/margins, JSON for run summaries/metadata,
  NPY for embeddings + projections, PNG/WAV/HTML for review). No results DB today. The one
  SQLite touchpoint is transient: `embed.py`'s BirdNET backend uses `perch-hoplite`'s
  sqlite+usearch vector store in a temp dir during embedding extraction, then reads out to
  `.npy`. **Decision: once results are trustworthy, move tabular results + per-run evaluation
  metrics into SQLite** (queryable index for cross-run comparison — "did accuracy improve",
  "all alarm clips with margin < 0.05"). Keep the big 1024-D embedding matrices as `.npy`
  (or a vector table) — don't store dense float blobs in SQLite. SQLite = the metrics/results
  index; `.npy` = the vectors it points to.
- **New local blackbird audio + run (2026-07-14):** 120 fresh UK `Turdus merula` recordings
  fetched from Xeno-canto to `data/raw/blackbird_xeno_canto_v1/` (LOCAL, gitignored under
  `data/raw/`, ~309MB) — deduped against all 108 known blackbird XC IDs (verified 0 overlap).
  Includes `xc_recordings_metadata.csv` (recordist type tags), `..._manifest_fixed.csv`
  (species_label corrected to `blackbird`). **PROCESSED through the pipeline** (local,
  gitignored, BIRD_TRANSLATION_ALLOW_LOCAL_OUTPUT=1):
  - `data/runs/blackbird_xeno_canto_v1_species/` — BirdNET species decision (1887 windows,
    118/120 recordings; Turdus merula top-predicted 933).
  - `data/runs/blackbird_xeno_canto_v1_pattern_layer/` — phrase_pool, birdnet embeddings →
    **329 phrase-unit embeddings (329×1024), 53 recordings** with accepted blackbird events.
  Only 53/120 recordings yielded units (strict top-bucket filter); rerun pattern_layer with
  `--accept_target_in_topk` for higher yield/lower purity. All UNLABELLED — ready for
  clustering / unlabelled-data levers; still needs by-ear review to become training labels.
- Don't feed whole audio datasets into context — reference Drive paths and run
  summary/report CSVs/JSONs instead.
- `PROJECT_REBRIEF.md` and `REPO_ZONES.md` are curated docs kept up to date by hand;
  update them (not just this file) if the architecture or species status materially
  changes.
