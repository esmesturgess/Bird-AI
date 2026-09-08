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

## Macaulay Library label acquisition (2026-08-12)

New source script: `src/fetch_macaulay_labels.py` — queries the Macaulay Library search
API (media.ebird.org's backend), deduces a call-type label per clip from THREE combined
signals (structured `tags`, free-text `mediaNotes` keyword match, structured `ageSex`
juvenile counts), downloads audio only for assigned/usable clips, writes a full manifest
CSV. Key finding that motivated this: `alarm` and `juvenile` NEVER appear as structured
tags on Macaulay (0/1000 in an earlier sample) but DO appear in free-text notes — e.g.
tag=`call` but notes=`"quiet alarm notes from perched male"`. Priority rule: explicit
notes keyword for alarm/juvenile OVERRIDES a generic song/call tag. Taxonomy: **song**
(absorbs `courtship_display_or_copulation`/`duet`, explicitly NOT `dawn_song`/`flight_song`
per user), **call**, **alarm**, **juvenile** — matches existing cross-species convention.
Audio download URL pattern: `https://cdn.download.ams.birds.cornell.edu/api/v1/asset/{id}/audio`.
eBird/Macaulay terms: non-commercial research/education use OK (matches this project);
attribution required; per-asset `licenseId` tracked in the manifest.

**UK blackbird batch (2026-08-12) — RESULT:** 1053 total UK blackbird audio assets ->
song 200, call 113, alarm 33, juvenile 3, excluded 704 (4 originally-ambiguous
song/call-tag-conflict clips were explicitly told by user to be dropped, folded into
excluded). Audio downloaded for the 349 usable clips only (~600MB) to
`data/raw/blackbird_macaulay_uk_v1/audio/` (LOCAL, gitignored). Manifest:
`data/raw/blackbird_macaulay_uk_v1/macaulay_label_manifest.csv` (all 1053 rows incl.
excluded, with reasoning). Human-readable checklist: `.../label_review.md`.
**As found, juvenile=3 is thin (expected) — user already agreed Europe-wide search is the
planned next step specifically to grow juvenile coverage, once UK is reviewed.**

**Known weaknesses in the heuristic, flagged during review, not yet fixed in code:**
1. Notes can describe general/background context (not the specific clip) — e.g. a clip
   tagged `song` got flagged ambiguous because notes mentioned local blackbirds "giving
   calls during the day" as general context, not describing this clip.
2. Song/call tag conflicts don't currently consult notes to break the tie (only
   alarm/juvenile use notes as an override) — e.g. asset 418004001 had tags [call,
   courtship_display_or_copulation] (-> ambiguous) but notes said "A female call" which
   would have resolved it to `call`.
3. Some `alarm` assignments are LOW CONFIDENCE and were flagged during review, NOT yet
   corrected in the manifest — user has not yet responded on these:
   - assetId 652268113: recordist's own notes say "Unsure if this is a continuation of
     initial alarm vs pre-roost calls" — overconfident as pure alarm.
   - assetId 320026421 ("Song & alarm call") and 243704901 ("Alarm from male bird...
     Short snatch of song in middle"): MIXED content, not pure alarm — if used as
     whole-clip training examples they'd mislabel the song portion.
   TODO before trusting the alarm set for training: resolve these 3, and reconsider
   whether "notes expresses uncertainty" (words like "unsure", "possibly", slash
   constructions like "pre-roost/alarm") should auto-route to ambiguous in the heuristic.

**User review resolved (2026-08-12):** all 4 originally-ambiguous song/call-tag-conflict
clips excluded (user: "just ignore"). The 3 flagged low-confidence/mixed-content alarm
clips (652268113, 320026421, 243704901) also excluded on user confirmation. Final UK
counts: **song 200, call 113, alarm 30, juvenile 3** (346 usable / 1053 screened).

**Europe juvenile expansion (2026-08-12):** ran the same fetch with `--region_code eu`
(confirmed this genuinely spans Europe, not just GB again) and `--only_categories
juvenile --skip_asset_ids_csv <uk manifest>` (new CLI flags added to
`fetch_macaulay_labels.py` for this) to avoid re-pulling song/call/alarm already covered.
3000 EU assets screened (hit the 30-page cap — more likely available for a future pull),
yielding song 667/call 422/alarm 77/**juvenile 28**/ambiguous 39/excluded 1767 EU-wide;
only the 26 NEW juvenile clips downloaded (2 already known from UK overlap). **Juvenile
total is now 3 (UK) + 26 (EU) = 29** — up from the original 3, genuinely usable as a small
category now. Output: `data/raw/blackbird_macaulay_eu_v1/` (LOCAL, gitignored, 55MB).

**Combined + pipeline-ready:** merged UK (song/call/alarm/juvenile) + EU (juvenile only)
into `data/raw/blackbird_macaulay_combined_v1/` — **372 clips total** (song 200, call 113,
alarm 30, juvenile 29), all audio verified present on disk. Built:
- `pipeline_manifest.csv` — standard project manifest format (recording_id=`ML{assetId}_
  Turdus_merula`, species_label=`blackbird`, `split=train` placeholder — see note below).
- `call_type_ground_truth.csv` — clean separate lookup (recording_id -> independent
  call-type label), kept SEPARATE from the pipeline manifest specifically so it survives
  regardless of downstream processing and stays available for evaluation.

**IMPORTANT OPEN DECISION, not yet resolved, revisit before adapter training:** this is
the first data in the whole project with an INDEPENDENT call-type label (not derived from
the user's own clustering/review judgment) — its biggest value may be as a genuine
HELD-OUT TEST set to finally check whether function is recoverable from BirdNET
embeddings, rather than folding it into training. Currently defaulted to `split=train` in
the manifest (harmless for species-decision/embeddings, which don't read split at all —
only matters once train/val/test partitioning happens for adapter work). Decide train vs.
test allocation before that point, not before running the pipeline.

**Pipeline run + real bugs found and FIXED (2026-08-12):** first two attempts at
species-decision on the 372-clip combined manifest failed with "kept 0 windows from
0/372 recordings" and NO error shown. Root-caused (don't repeat this debugging):
1. First manifest used RELATIVE audio_path values; the successful earlier Xeno-canto
   batch used absolute paths. Fixed to absolute — but this alone did NOT fix it (retry
   still failed identically), so it was a real but incomplete diagnosis.
2. Real root cause: `run_birdnet_window_frontend` (`src/pipeline.py`) calls
   `infer_recordings_with_birdnet` ONCE for the WHOLE batch (all files copied into one
   temp dir, one `birdnet-analyze` subprocess call), not per-file. `BirdNETConfig.
   timeout_s` was hardcoded to **120 seconds** with no CLI override. Verified directly
   (bypassing pipeline.py): 5 files worked fine (259 windows, fast); the full batch's
   combined audio was too much to finish in 120s, silently hit `subprocess.
   TimeoutExpired`, which was caught and returned as `batch_err` — but the caller in
   pipeline.py never checked/printed `batch_err`, so the error was completely swallowed
   and just looked like "0 recordings accepted."
3. User correctly suspected the clips themselves were too long/numerous, independent of
   the timeout bug — checked and confirmed: 372 clips totalled **354 minutes** combined
   (mean 57s, but 25 clips over 3 min, max 613s/10min). This is real excess load, not
   just an artifact of an overly strict timeout.

**Fixes applied (all committed to src/, not just this run):**
- `src/birdnet_infer.py`: `BirdNETConfig.timeout_s` raised 120 -> **1800**.
- `src/pipeline.py`: `run_birdnet_window_frontend` now prints a WARNING if `batch_err` is
  set, instead of silently proceeding — this class of failure can no longer hide silently
  for any future run/species.
- Trimmed the 76 clips over 90s to a 90s cap (`data/raw/blackbird_macaulay_combined_v1/
  audio_trimmed/`, ~92MB, originals untouched) — cuts total batch duration 354->251 min
  (29%) while leaving 80% of clips (296/372) completely untouched. `pipeline_manifest.csv`
  audio_path updated to point at trimmed copies where applicable.
- **Lesson for future large fetches (Macaulay or otherwise): check total combined audio
  duration BEFORE running the full-recording BirdNET frontend, not after a silent
  failure.** Xeno-canto batches so far have stayed naturally short; Macaulay recordings
  can run to 10+ minutes and this will recur if not checked.

**RESULT (2026-08-12), all 3 fixes confirmed working:** species-decision kept 9997
windows from **372/372 recordings (100%**, vs 0/372 before the fixes). Pattern layer:
**301/372 recordings (81%) had accepted blackbird units** — notably higher yield than the
plain Xeno-canto blackbird batch (53/120 = 44%), plausibly because Macaulay's curated
observations give BirdNET a cleaner signal than raw field recordings. Final: **1855
phrase-unit embeddings, shape (1855, 1024)**, same BirdNET embedding space as everything
else in the project. Disk used: 1.8GB (species) + 679MB (pattern_layer); 2.5GB free
remaining afterward. Outputs: `data/runs/blackbird_macaulay_v1_species/` ->
`data/runs/blackbird_macaulay_v1_pattern_layer/`, LOCAL, gitignored,
BIRD_TRANSLATION_ALLOW_LOCAL_OUTPUT=1.

**This is now the first opportunity in the project to test the adapter/raw-space
representational question against truly INDEPENDENT ground truth** (not derived from the
user's own clustering/review — see the 2026-07-14 Blackbird overfitting + recording-
dominance findings above, which were all measured against shaky cluster-derived labels).

**Join done, CRITICAL finding — acceptance yield was wildly uneven across categories
(2026-08-12):** joined phrase units to `call_type_ground_truth.csv` cleanly (0 unmatched).
BirdNET's species-decision/pattern-layer acceptance filter did NOT reject clips evenly:
song lost only 8% of recordings, call 27%, alarm 13% — but **juvenile lost 72%** (29 raw
clips -> only 8 accepted recordings). Diagnosed why (checked `species_top1`/confidence for
the 21 rejected juvenile recordings): NOT confident misclassification as one specific
species — top1 predictions were scattered across a dozen+ unrelated species with mean
confidence only 0.197. Reads as genuinely low species-diagnosticity of juvenile begging
calls to BirdNET (plausibly trained mostly on adult vocalizations), not a pipeline bug.
`--accept_target_in_topk` would only partially rescue this (blackbird appeared in top-k
for just 10/21 rejected recordings, in 4.5% of windows) and was rejected as an option
because it would apply a laxer/inconsistent acceptance standard only to juvenile,
confounding any later comparison across categories.

**Went back to Europe pool for more raw juvenile (user's choice) — result: EXHAUSTED the
available pool, not a partial fix.** Deep-scanned EU region with `--max_pages 150`: found
**5476 total EU blackbird audio assets — this is the FULL pool** (scan ended naturally,
not page-capped), containing juvenile=43 total (vs 28 found in the initial 3000-asset
partial scan). Of those 43, only 14 were new (29 already known from earlier passes).
Processed the 14 new raw clips through species-decision+pattern_layer: only **4 survived**
acceptance (28.6% yield — matches the diagnosed rate almost exactly, good sanity check).
**Juvenile final: 8 -> 12 distinct recordings (up from initial 8, but this is now the
ceiling from Macaulay/EU — there is no more raw juvenile-tagged blackbird audio available
in this pool to fetch).** If more juvenile is wanted beyond 12 recordings, the only
remaining levers are: expand geography beyond Europe (dilutes with potentially different
regional dialects), or `--accept_target_in_topk` (rejected above for the confound reason).

**FINAL merged dataset:** `data/runs/blackbird_macaulay_final_pattern_layer/` (embeddings
+ manifest concatenated from the two runs) — **1869 phrase-unit embeddings (1869x1024)**.

| category | distinct recordings | phrase units |
|---|---|---|
| song     | 184 | 1401 |
| call     |  83 |  243 |
| alarm    |  26 |  137 |
| juvenile |  12 |   88 |

`call_type_ground_truth.csv` updated with the 4 new juvenile recording_ids. Disk after
all this work: 2.5GB free.

## FIRST REAL RESULT: function IS recoverable from BirdNET embeddings (2026-08-12)

Split-strategy decision resolved: used `StratifiedGroupKFold` (sklearn), n_splits=6 —
ONE joint set of folds across all 4 categories together (not separate CV per category;
the classification task is inherently joint since a test clip is compared against
prototypes from ALL categories at once), grouped by `recording_id` (no leakage),
stratified so every fold gets a proportional slice of every category including the
smallest (juvenile=12 recordings -> 2 held out/fold). Rationale over a fixed dedicated
test set: the 2026-07-14 experiment already showed single-split accuracy is unstable at
this data scale (0.29-0.92 swing across folds) — CV's averaging directly addresses that,
and is cheap regardless of k here since only the small adapter (frozen BirdNET, ~300K-
param MLP) gets retrained per fold, not BirdNET itself.

**First run (naive random pair sampling, same method as 2026-07-14) — MISLEADING
result, don't trust the aggregate number:**
```
              raw     adapter   (majority floor 0.750)
overall      0.560     0.723
song         0.575     0.869
call         0.580     0.309  <- WORSE than raw
alarm        0.445     0.277  <- WORSE than raw
juvenile     0.432     0.250  <- WORSE than raw
```
Root cause diagnosed: `pairs_random` samples uniformly, and song=75% of data means
random pairs are ~56% likely to both be song by chance — the adapter mostly learned
"song vs not-song" and got WORSE at the minority categories despite the aggregate number
looking better. Classic imbalanced-training-signal trap; caught only because per-category
breakdown was checked, not just the pooled number.

**Fix: balanced pair sampling** (positives split EVENLY across the 4 classes, not
proportional to size; negatives split evenly across all 6 class-PAIR combinations, not
letting song-X dominate). Only this one variable changed; same folds/architecture/epochs.
**Result — genuine improvement, verified per-category:**
```
              raw     adapter(naive)   adapter(balanced)
overall      0.560       0.723              0.822   <- now clearly beats majority floor
song         0.575       0.869              0.896
call         0.580       0.309              0.650   <- fixed, genuinely beats raw now
alarm        0.445       0.277              0.635   <- fixed, genuinely beats raw now
juvenile     0.432       0.250              0.420   <- back to ~raw, NOT improved
```
Per-fold accuracy also much more stable (0.78-0.87 range vs the earlier 0.53-0.81 swing).

**HONEST CONCLUSION:** this is the first real evidence in the project that call-function
IS recoverable from BirdNET's embedding space via a trained adapter — for song, call, and
alarm specifically. **Juvenile is NOT fixed by this** (stuck at ~raw baseline) — balanced
sampling solves a class-imbalance-in-training-signal problem, but juvenile's limitation is
genuine data scarcity (12 recordings, already established as the ceiling from Macaulay/
EU), which no pair-sampling trick can manufacture around. This is a real, substantive,
defensible answer — first time in the project this question has been tested against truly
independent (non-self-referential) ground truth. Eval script (with the balanced-pairs fix)
lives in the session scratchpad (`macaulay_eval_balanced.py`); should be promoted into a
proper committed `src/` script.

## Data expansion round 2: call/alarm from EU pool (2026-08-12)

User chose to pull already-classified EU call/alarm data (found during the earlier
juvenile deep-scan but never downloaded) before trying new modeling ideas. Added
`--max_per_category` flag to `fetch_macaulay_labels.py` (rating-sorted, so a cap keeps
highest-quality clips first) — needed to avoid grabbing all 623 available new call clips
at once given disk constraints. **IMPORTANT bug caught before running: the "known
assetIds" skip-list must be built from assetIds with AUDIO ACTUALLY DOWNLOADED, not just
assetIds that were CLASSIFIED in a metadata-only scan** — the deep EU scan classified all
5476 assets but only downloaded juvenile; an early draft of the skip-list wrongly included
all 5476 as "known," which would have skipped every single call/alarm download. Fixed
before running (`data/raw/blackbird_macaulay_known_assetIds.csv` now built from
`local_audio_path.notna()`, not full manifest rows).

Downloaded 384 new clips (300 call capped from 623 available, 84 alarm uncapped). Applied
the 90s duration cap proactively this time (31/384 clips trimmed, 246.6->208.1 min) —
learned from the earlier timeout bug, didn't repeat it. Pipeline ran clean on the first
try: 384/384 recordings kept windows, 275/384 (72%) accepted, yielding 1390 new phrase
units (call 1139/214 recordings, alarm 251/61 recordings — much higher yield than
juvenile's 28%, consistent with call/alarm being more species-diagnostic to BirdNET).

Merged into `data/runs/blackbird_macaulay_final_pattern_layer/` (same location, updated
in place) and `data/raw/blackbird_macaulay_combined_v1/call_type_ground_truth.csv`.

**FINAL dataset after both expansion rounds — 3259 phrase-unit embeddings, 580 recordings:**
| category | recordings | phrase units |
|---|---|---|
| song     | 184 | 1401 |
| call     | 297 | 1382 |  (was 83/243)
| alarm    |  87 |  388 |  (was 26/137)
| juvenile |  12 |   88 |  (unchanged -- confirmed ceiling, see above)

**Disk crisis hit critical (560MB free) after this run — RESOLVED.** Found ~3.7GB of
redundant intermediate `data/runs/*_species/` directories (segmented event WAVs from
BirdNET species-decision stage, across 4 separate processing runs) — safe to delete since
final embeddings were already extracted from them and merged; regenerable from
`data/raw/` + pipeline scripts if ever needed again. User confirmed before deleting (asked
first given the volume). Freed to 4.2GB. Verified final embeddings (3259x1024) and ground
truth (651 rows) intact after cleanup.

**NOT YET DONE:** re-run the balanced-pairs adapter evaluation (StratifiedGroupKFold,
n_splits=6 or reconsider given call/alarm are now much bigger — could support more folds)
against this larger 3259-unit dataset. Given call/alarm nearly quadrupled, expect
meaningfully different (likely improved) per-category numbers vs the 2026-08-12 first
result (call 0.650, alarm 0.635 with the smaller 1869-unit dataset) — should re-check
before drawing conclusions, don't assume the old numbers still hold.

## Re-evaluation on full expanded dataset + juvenile-weighting experiment (2026-08-13)

**Re-ran the balanced-pairs evaluation on the full 3259-unit dataset (StratifiedGroupKFold,
n_splits=6, unchanged from before -- isolating dataset-size as the only new variable):**
```
              raw (1869-unit -> 3259-unit)   adapter (1869-unit -> 3259-unit)
song          0.575 -> 0.814                 0.896 -> 0.814
call          0.580 -> 0.541                 0.650 -> 0.653
alarm         0.445 -> 0.381                 0.635 -> 0.552
juvenile      0.432 -> 0.102 (!)             0.420 -> 0.375
overall       0.560 -> 0.627                 0.822 -> 0.703   (majority floor now 0.430, was 0.750)
```
**Mixed result, real not illusory (checked full precision, not just 3-decimal rounding --
initial "song raw==adapter" tie was 1140 vs 1141/1401, not a bug, just close rounding).**
Growing call/alarm was a genuine win for THOSE categories (both still clear their raw
baseline with the adapter) but came at a real cost to juvenile: raw-space juvenile
accuracy collapsed (0.432->0.102, 38/88->9/88 correct) even though juvenile's own data
didn't change. Likely explanation: call/alarm centroids are now estimated from far more
data (sharper, more accurate), and if juvenile begging calls sit acoustically closer to
call/alarm than to song in this embedding space (plausible), sharper competing centroids
pull juvenile test points away from its own still-small, noisier centroid. The adapter is
more robust to this than raw space (explicit juvenile-vs-alarm/call negative pairs during
training) but doesn't fully protect it.

**Tried: 3x loss-weighting on any pair involving juvenile (isolated single-variable
change, same folds/architecture otherwise) -- NEGATIVE RESULT, reverted.**
```
              unweighted adapter   3x juvenile-weighted adapter
song          0.814                0.797  (worse)
call          0.653                0.349  (much worse -- now below raw's 0.541)
alarm         0.552                0.536  (worse)
juvenile      0.375 (33/88)        0.375 (33/88)  -- EXACTLY unchanged, not just similar
overall       0.703                0.565  (worse, now below raw baseline)
```
Juvenile's accuracy was IDENTICAL (same 33/88 correct, not approximately similar) while
everything else degraded -- diluted training signal for the other classes with zero
juvenile benefit. Conclusion: juvenile's bottleneck is genuinely INFORMATION-LIMITED (only
~73 unique training examples/fold), not attention/weighting-limited -- reweighting
controls optimization effort, not available information, and can't manufacture what isn't
there. **Do not retry loss-reweighting as a juvenile fix.** Reverted to the unweighted
balanced-pairs version as the current best/working result (song 0.814, call 0.653, alarm
0.552, juvenile 0.375, overall 0.703 on the 3259-unit dataset).

**Current honest status:** juvenile is very likely near its ceiling given data
constraints already established (Macaulay/Europe exhausted at 12 recordings). Worth
treating as demonstration-only / lower-confidence category going forward rather than
continuing to chase fixes for it specifically, UNLESS more juvenile data becomes available
via a genuinely new source (e.g. worldwide search, discussed and deferred earlier with
named tradeoffs).

## SupCon loss — NEW BEST RESULT (2026-08-13)

Implemented proper Supervised Contrastive Loss (Khosla et al. 2020, "SupCon-Out" variant)
to replace the pairwise `PairContrastiveLoss`. Key mechanical difference: trains on
STRATIFIED BATCHES (equal count per class, all 4 classes every batch, same "don't let
majority dominate" principle as balanced-pairs but applied at the batch level) rather than
isolated pairs; each anchor's loss is a softmax over similarity to ALL other same/
different-label items in its batch simultaneously (temperature-scaled), not one pair at a
time -- generally more sample-efficient, directly relevant given data scarcity. Only the
training method changed; same folds (StratifiedGroupKFold, n_splits=6, same seed), same
`EmbeddingAdapter` architecture, same eval methodology. Implementation + batch training
loop in `macaulay_eval_v4_supcon.py` (scratchpad) -- should be promoted into `src/`
(e.g. add `SupConLoss` alongside `PairContrastiveLoss` in `adapter_model.py`, add a
`--loss supcon` mode to `train_adapter.py`).

**Result — beats the balanced-pairs adapter on EVERY category, not just aggregate
(checked specifically, since that was the earlier naive-pairs trap):**
```
              raw       balanced-pairs adapter   SupCon adapter
song          0.814     0.814                    0.819
call          0.541     0.653                    0.774   <- big improvement
alarm         0.381     0.552                    0.593   <- improvement
juvenile      0.102     0.375                    0.420   <- improvement, best result yet
overall       0.627     0.703                    0.762
```
Juvenile improved WITHOUT any special weighting (recall 3x loss-weighting was tried and
failed, exact-same 33/88 correct) -- directly validates the sample-efficiency rationale
for choosing SupCon: every anchor gets signal from all same/different-class batch members
at once, not just one random pair, so scarce classes benefit without needing to be
artificially boosted. Per-fold stable (0.74-0.81 range, no outliers).

**THIS IS NOW THE BEST/CURRENT WORKING RESULT for the Blackbird adapter.** Hyperparameters
used: iterations=400, batch_per_class=16 (batch size 64), temperature=0.1, same adapter
architecture (1024->256->128, dropout 0.10) -- these were NOT tuned/searched, just
reasonable SupCon defaults; a hyperparameter sweep (temperature, batch size, iterations)
is a legitimate next step but untried.

## Classifier-head vs SupCon+prototype — DONE, validates the architecture choice (2026-08-13)

Never-before-measured comparison: same-capacity backbone (1024->256->128), same folds,
same stratified-batch training, only the objective differs -- SupCon+nearest-prototype
vs. a plain softmax/cross-entropy classifier head. Script:
`macaulay_eval_v5_classifier_compare.py` (scratchpad).
```
              raw      SupCon+prototype   classifier-head
song          0.814    0.817              0.809
call          0.541    0.788              0.755
alarm         0.381    0.559              0.590   <- classifier slightly ahead
juvenile      0.102    0.489              0.295   <- SupCon dramatically ahead (+19pt)
overall       0.627    0.765              0.746
```
**Result: SupCon+prototype wins overall, and specifically wins big on juvenile (the
hardest, scarcest category) while being roughly a wash on the others (classifier even
edges ahead slightly on alarm).** Matches the theory: cross-entropy is more data-hungry,
so the classifier suffers most exactly where data is scarcest, while SupCon's batch-wide
comparison is more sample-efficient. **This validates the earlier architectural decision
(prototype-matching over a classifier, for unknown-fallback + extensibility) as NOT a
tradeoff against accuracy — for the hardest class it's also the more accurate choice, and
a wash elsewhere.** Given rare/data-scarce categories are a recurring pattern in this
project (not just Blackbird juvenile), this is a real, evidenced confirmation, not just an
architectural preference anymore.

## SupCon hyperparameter tuning — nested CV, humbling but important result (2026-08-13)

User correctly caught a methodology gap before this ran: tuning hyperparameters against
the SAME folds used for reporting results risks an optimistic/leaky number (standard
train/val/test practice) — I had glossed over this. Fix: proper NESTED cross-validation
(`macaulay_eval_v6_nested_sweep.py`, scratchpad) — for each of the 6 outer folds, split
ITS OWN training portion into an inner train/val slice (StratifiedGroupKFold n_splits=4),
search a grid (temperature in {0.05,0.1,0.2} x batch_per_class in {8,16,32}, 9 combos) by
inner-validation score ONLY, then retrain fresh on the full outer-train with the winning
combo and check the outer-test fold for the first and only time. No fixed validation
carve-out (would have hurt juvenile's already-thin 12 recordings); nesting reuses data
across folds instead. Chose different hyperparameters per outer fold (temp/batch varied:
(0.1,16),(0.2,16),(0.05,32)x3,(0.2,32)) -- no dominant single winner, fairly flat grid.

**RESULT: honestly-validated number is LOWER than the earlier informal one, not higher --
this is the important, humbling finding, not a failure:**
```
              raw      SupCon (informal defaults)   SupCon (properly nested-tuned)
song          0.814    0.817                         0.779   down
call          0.541    0.774                         0.792   ~same
alarm         0.381    0.593                         0.567   down
juvenile      0.102    0.420                         0.341   down
overall       0.627    0.762                         0.747   down
```
The earlier "untuned defaults" result (0.762/0.420 juvenile) was itself informally chosen
and checked against the same folds being reported -- exactly the leakage risk the user
flagged. Doing this properly didn't find a better configuration (the grid has no clear
winner, chosen hyperparams vary fold-to-fold) -- it corrected an overoptimistic estimate.
**UPDATE: 0.747 overall / song 0.779 / call 0.792 / alarm 0.567 / juvenile 0.341 is now
the trustworthy number to use going forward, superseding the earlier 0.762/0.420.** Still
clearly beats raw space and the balanced-pairs adapter on call/alarm/juvenile; call is
essentially unchanged; song and juvenile take the biggest (still real, not dramatic) hits
from the more honest evaluation.

**Lesson for future work in this project, not just this instance:** ANY informal
hyperparameter/configuration choice checked against the same folds used for reporting
carries this same leakage risk, even without an explicit grid search -- eyeballing "this
setting looks good" on your test folds is a soft form of the same problem. Nested CV (or
a genuinely separate validation set where data allows) should be the default for any
future tuning decision in this project, not an optional extra step.

## Xeno-canto expansion (2026-08-13/14) — juvenile ceiling broken, but a new confound found

**Correction to an earlier session error:** XC data was previously called "unlabelled" --
WRONG. XC has a `sound_type` field (recordist tags, same paradigm as Macaulay's `tags`)
plus `remarks` free text. The earlier "XC tags only 24% agreed with our labels" finding
(2026-07-27) was XC vs our *cluster*-derived labels, which we've since established are the
unreliable ones -- so that finding argues against the cluster labels, not against XC.
XC is a legitimate second independent label source, same trust level as Macaulay.

**Key config fact, don't lose again:** the actual key in `.env` is named
`xenocanto-api-key`, NOT `XC_API_KEY` as assumed earlier -- read whichever name is
present, don't just look for `XC_API_KEY`. Also: `fetch_xeno_canto.py` enforces Google
Drive output by default (same as before) -- must set
`BIRD_TRANSLATION_ALLOW_LOCAL_OUTPUT=1`. **Local output for transient training data is
the correct choice here** (user confirmed): fetch raw audio locally (gitignored) ->
extract embeddings via the pipeline -> delete the bulky raw audio afterward, keeping only
the tiny embeddings + labels as the permanent asset. Don't default to Drive for this kind
of disposable intermediate data.

**Scope decision:** user chose "juvenile + alarm only" from XC (not song/call, already
well-covered by Macaulay), Europe only (not worldwide), dawn song excluded from the song
rule (consistent with the Macaulay taxonomy). XC European availability check (metadata-
only, cheap): song 4603, call 2511, alarm call 1582, begging call 144, juvenile-stage 265.
Fetched: alarm capped at 400, ALL begging-call (86) + ALL juvenile-stage (108) --unique
juvenile after dedup 145 (12 recordings tagged both alarm+juvenile, resolved to juvenile
by the same priority rule as Macaulay). Applied 90s cap (only 1 clip needed it, XC fetch
already length-capped at fetch time).

**Pipeline result:** 492/501 recordings kept windows; 330 accepted units -> **alarm 291
recordings/897 units, juvenile 39 recordings/92 units** (juvenile acceptance rate 36%,
actually HIGHER than Macaulay's 28%).

**FULL COMBINED DATASET (Macaulay + XC Europe), `data/runs/blackbird_all_labelled_v1/`:**
```
          recordings   phrase units   (started the whole project at)
song      184           1401           184
call      297           1382            83
alarm     378           1285            26
juvenile   51            180            12
```
4248 total embeddings, 910 distinct recordings. **Juvenile ceiling broken: 12 -> 51
recordings (4x), no longer the desperate outlier.**

**Re-ran the honest SupCon evaluation on the full set -- IMPORTANT, non-obvious result,
overall accuracy DROPPED (0.762 informal / 0.747 nested-tuned -> 0.645), and NOT purely
a measurement artifact:**
```
              before (3259 units, alarm small)   now (4248 units, alarm large)
song          0.819                               0.752
call          0.774                               0.614   <- real drop
alarm         0.593                               0.605   <- small gain
juvenile      0.420                               0.350   <- dropped despite 4x more data
overall       0.762                               0.645   (majority floor 0.43->0.33, dataset now balanced)
macro-avg     ~0.65 (est.)                         0.580
```
Confirmed via CONFUSION MATRIX this is a real phenomenon, not noise: **44% of true
juvenile clips get predicted as ALARM**; call<->alarm also cross-confuse (23%/29%); song
stays clean (only ~13% confused with alarm). **Mechanism: juvenile, call, and alarm are
short, acoustically similar "call-like" sounds occupying overlapping BirdNET embedding
space, while song sits clearly apart. Growing alarm 26->378 recordings sharpened alarm's
presence in that crowded region and pulled probability mass away from juvenile AND call,
even though juvenile itself also grew 4x.** More data was not uniformly helpful --
growing one confusable category can actively hurt its neighbors.

**STATUS: NOT YET DECIDED which dataset/result to treat as current best** -- the smaller
3259-unit (Macaulay-only) SupCon result (juvenile 0.420) or the larger 4248-unit
(Macaulay+XC) result (juvenile 0.350, but 51 real recordings instead of 12, i.e. a more
trustworthy/less-noisy estimate of a lower number, vs a noisier estimate of a higher
number). Also not yet tried: does capping/subsampling alarm back down (e.g. to ~100-150
recordings, closer to call's scale) recover call/juvenile accuracy while keeping alarm's
gain? This would directly test the "too much alarm crowds out its neighbors" hypothesis
and is the natural next experiment before deciding.

## Blackbird LOCKED, Wren data fetched but PAUSED for architecture sweep (2026-08-14)

**Blackbird dataset decision, locked in:** the alarm-capped dataset
(`data/runs/blackbird_alarm_capped_v1/`, alarm subsampled 378->112 recordings) is the
FINAL dataset -- confirmed by direct experiment that capping recovers call (0.614->0.669)
and juvenile (0.350->0.417) substantially at a real but acceptable cost to alarm itself
(0.605->0.452), and that alarm barely benefited from having 378 vs 26 recordings in the
first place (0.605 vs 0.593) -- alarm was never really data-starved, it was crowding its
neighbors. This is now the canonical Blackbird dataset for training the final checkpoint.

**Wren replication started, then deliberately PAUSED before evaluation (user's call,
good instinct) -- do NOT skip this reasoning if resuming:** user pointed out we were about
to replicate the full recipe onto Wren using an adapter ARCHITECTURE (`1024->256->128`)
that has never itself been tested/tuned -- every experiment this whole session validated
the TRAINING RECIPE (loss function: naive pairs -> balanced pairs -> SupCon; prototype vs
classifier head) but the MLP shape itself has been constant throibkughout, inherited
unchanged from before this project's careful work started. Correct sequencing: settle
architecture on Blackbird (mature dataset, best evaluation harness) BEFORE spending eval
effort on Wren with what might be a suboptimal shape we'd have to redo anyway.

**Wren raw data already fetched and safely sitting, NOT YET processed through BirdNET
pipeline, NOT YET used for anything -- resume from here once architecture is settled:**
- Species code verified via API (don't guess again): Macaulay/eBird taxon code for
  Eurasian Wren is `winwre4` (NOT `eurwre1`, which is an Iceland SUBSPECIES that
  `reportAs`s to winwre4 -- same taxonomic-split trap as before, caught by verifying
  via the taxonomy API rather than guessing).
- Scope decisions (both user-confirmed, informed by the Blackbird alarm-crowding lesson):
  (1) balanced caps from the start (song/call/alarm capped ~200 each, juvenile uncapped)
  rather than maximize-then-discover-crowding; (2) keep call and alarm as SEPARATE
  categories and let the data decide whether Wren's known pre-existing "call/alarm blur
  together" issue (flagged in this project's docs before any of this session's work)
  actually shows up in the honest evaluation, rather than pre-emptively merging them.
- XC Europe fetched: song 200 (capped, of 3670 available), call 200 (capped, of 1513),
  alarm 200 (capped, of 596), juvenile 91+107=~190ish before dedup (begging_call + stage,
  ALL available, not capped). Output dirs: `data/raw/wren_xc_eu_v1_{song,call,alarm,
  juv_begging,juv_stage}/`.
- Macaulay Europe fetched: song/call/alarm capped 200 each (426 downloaded, 2 harmless
  network timeouts) -> call 347/alarm 28/juvenile(from the same batch, uncapped
  separately) 35. Note alarm came back thin (28) on Macaulay specifically -- consistent
  with alarm never being a structured Macaulay tag, needs the notes/remarks mining which
  is inherently sparser than XC's explicit `type:"alarm call"` filter. Output dirs:
  `data/raw/wren_macaulay_eu_v1{,_juv}/`.
- Disk after all Wren fetches: 4.7GB free.
- **NOT YET DONE for Wren:** labelling via the classify() heuristic, duration cap, BirdNET
  species-decision + pattern_layer, merging Macaulay+XC, honest SupCon evaluation. All
  deliberately deferred until the architecture sweep below concludes.

**Architecture sweep result (2026-08-14) -- DONE, real modest improvement found,
architecture updated:** tested hidden_dim x output_dim grid ({128,256,512} x {64,128,256})
via proper nested CV (same leakage-avoidance discipline as the temperature/batch-size
sweep), temperature/batch_per_class held fixed at 0.1/16 (already established as having
no clear better option). Unlike the temperature/batch-size sweep (no clear winner, tuned
result was actually WORSE than defaults), this sweep found a REAL, CONSISTENT signal:
```
                    old default (hidden_dim=256, output_dim=128)   architecture-tuned
song                0.799                                          0.792   ~same
call                0.669                                          0.715   real improvement
alarm               0.452                                          0.423   slight dip
juvenile            0.417                                          0.417   unchanged
overall             0.683                                          0.695   modest net gain
```
Chosen architecture across the 6 outer folds: `(512,128)` won 3/6, `(256,128)` 2/6, one
outlier -- consistent preference for WIDER hidden_dim (512 over 256), output_dim=128
unchanged as the sweet spot. **UPDATED DEFAULT: hidden_dim=512, output_dim=128** (was
256/128) for all future Blackbird (and provisionally Wren, pending its own validation)
adapter training -- update `train_supcon_adapter.py`'s default and any future eval
scripts to match.

## Blackbird final checkpoint — TRAINED, DONE (2026-08-14)

`data/runs/blackbird_supcon_adapter_v1/` — `adapter.pt` (SupCon, hidden_dim=512,
output_dim=128, matches the validated architecture sweep), `prototypes.npy` (4x128, one
per class), `prototype_labels.json`, `adapter_meta.json` (full provenance: locked
alarm-capped dataset, hyperparameters, class counts, train-set sanity accuracy 0.933 --
NOT the generalization estimate, that's the nested-CV 0.695 overall / song 0.792 / call
0.715 / alarm 0.423 / juvenile 0.417 reported above). Verified: checkpoint loads cleanly
via the standard `EmbeddingAdapter` loader, format-compatible with
`transform_embeddings.py --adapter_checkpoint` / `assign_anchor_labels.py`. **This is the
first genuinely deployable, non-scratchpad model artifact this whole project has
produced.** Trained on 100% of the locked dataset (no held-out split -- standard practice
once CV has already validated generalization).

**Blackbird is now DONE for this pass** -- dataset locked, architecture validated,
checkpoint trained and verified compatible. Next: resume Wren (data already fetched,
paused for the architecture sweep above -- run through labelling -> BirdNET pipeline ->
merge -> same honest SupCon eval, now with the settled hidden_dim=512/output_dim=128
recipe rather than the old untested default).

## Species selection is now PREDICTABLE: the song-vs-rest separability screen (2026-08-29)

**Why Blackbird succeeded and Wren failed — measured, not guessed.** Computed cosine
similarity between call-type centroids in raw BirdNET space (lower = more distinct):

| metric | Blackbird (0.695) | Wren (0.423, below floor) |
|---|---|---|
| silhouette by call-type | **+0.027** | **-0.024** |
| song vs rest | **0.9515** | **0.9963** |
| alarm vs call | 0.9935 | 0.9957 |

Blackbird's win came almost ENTIRELY from song standing apart (0.9515) while alarm/call/
juvenile were piled together (0.989-0.994) -- which maps exactly onto its per-category
scores (song 0.792, call 0.715, but alarm 0.423, juvenile 0.417). Wren has no such gap:
its song is as close to its calls as the calls are to each other. Biologically right --
Wren's song is an explosive trill/rattle burst and its alarm is a hard rattling churr, same
acoustic territory; Blackbird's song is long, low, fluty, tonal vs short sharp calls.

**USE THIS AS A CHEAP PRE-SCREEN before committing to any new species**: fetch a modest
sample, run species-decision + embeddings only, compute song-vs-rest. Under ~0.96 = proceed;
~0.996 = walk away. Costs ~20 min instead of a multi-hour full pipeline; would have caught
Wren before a day was spent on it.

**Candidate ranking for classic British birds** (XC Europe counts, checked 2026-08-29):
Robin (song 4595/call 2489/alarm 447/begging 62) and Song Thrush (4737/555/293/21, same
genus + song architecture as Blackbird) rank highest; Blackcap strong. AVOID Great Tit
despite the best data (3899/4840/669/242) -- its song is a simple repeated two-note whistle
(call-like) plus a famously huge heterogeneous call repertoire. AVOID Wood Pigeon: cooing-
only repertoire, and confirmed empirically -- Macaulay EU returned **0 alarm, 0 juvenile**
across 1503 assets; XC has only 19 alarm / 4 begging. The old `song_call` merge for Wood
Pigeon was probably detecting this same problem.

## ROBIN — trained, WORKS, and the inverse of Blackbird (2026-08-29)

**First successful replication of the Blackbird recipe on a second species.** Species code
`eurrob1` (verified via API). Dataset `data/raw/robin_combined_v1/` (Macaulay EU + XC EU),
final embeddings `data/runs/robin_final_pattern_layer/` -- **2980 phrase units, 450
recordings**. Eval: `data/runs/robin_supcon_eval_v1.json`.

Dataset is the best-balanced in the project: song 150 / call 150 / alarm 150 / juvenile 90
recordings (vs Blackbird's 184/297/112/51).

```
              Robin    Blackbird   Wren
overall       0.584    0.695       0.423
maj. floor    0.419    ~0.43       0.468
MARGIN        +0.165   +0.265      -0.045  (Wren FAILED)
song          0.646    0.792       0.354
call          0.487    0.715       0.614
alarm         0.646 <- 0.423       0.605   BEST ALARM IN THE PROJECT (+22pt vs Blackbird)
juvenile      0.412    0.417       0.350
```

**Robin is the INVERSE of Blackbird**: Blackbird nails song/call and fails alarm/juvenile;
Robin does the opposite. Predicted in advance by the separability screen and confirmed:
- Robin alarm-vs-call 0.9748 (vs Blackbird's 0.9935) -> alarm genuinely recoverable.
- Robin song-vs-call **0.9948** (Wren territory) -> song<->call is the dominant error
  (33% of calls predicted song, 24% of songs predicted call). Robin's overall score is
  dragged down by this one pair, not by general inseparability.
- Prediction was partly wrong on juvenile: expected a gain, got 0.412 vs Blackbird's 0.417
  (i.e. unchanged), despite juvenile-vs-call separability looking better (0.9761 vs 0.9896).

**ANOMALY worth investigating, not yet explained: on alarm, RAW space (0.708) BEATS the
adapter (0.646).** The adapter is actively costing accuracy on Robin's strongest category.
Raw overall is 0.478 vs adapter 0.584, so the adapter helps on net -- but something about
SupCon training is degrading an already-well-separated class. Do not assume the adapter is
strictly better per-category.

**Human label review WAS done for Robin** (unlike Wren) -- `data/reviews/
robin_label_exclusions_v1.csv` (git-tracked, with per-clip reasoning): 11 clips tagged both
`alarm call` and `song` dropped (mixed content would train song passages as alarm -- same
call the user made on Blackbird assets 320026421/243704901), plus 4 juvenile clips that were
subsong/singing rather than begging. 110 song+call conflicts left excluded.

**New reusable code:** `src/build_species_dataset.py` merges XC+Macaulay for any species,
re-derives labels from RAW `sound_type`+`stage` (NOT from which `type:` query returned the
clip -- that was the Wren bug), applies the juvenile>alarm>song/call priority rule, excludes
dawn/flight song, supports `--max_per_category` balance caps (quality-sorted) and
`--exclude_ids_csv`, and writes `labels_needing_review.csv` flagging every conflict/override.
`src/evaluate_supcon_adapter.py` is the committed nested-CV harness (was scratchpad-only).

## Disk: the pipeline needs ~2x the SOURCE AUDIO DURATION in scratch (2026-08-29)

Ran the machine to 0 bytes free twice in one session; at 100% full the shell can't even
write its own temp files and `echo` fails. Root cause of the underestimate:
**`--birdnet_window_overlap 1.5` on 3s windows writes every second of source audio into ~2
overlapping UNCOMPRESSED WAVs**, so the `events/` dir is far larger than the input. 383 min
of Robin audio needed ~4-5GB, not the ~2.6GB predicted from Wren's GB-per-minute ratio.

**Fix, now the standard approach: BATCHED processing** (`scratchpad/pipeline_robin_batched.sh`,
should be promoted to `src/`). Process ~100 recordings per chunk, harvest ONLY the embeddings
(a few MB), delete the bulky species/pattern intermediates before the next chunk, then
concatenate with embedding row-pointers re-based. Peak usage <900MB instead of 5GB. Each
chunk is resumable. This is also how Blackbird's final dataset was assembled, so it's
precedented, not a hack.

Other lessons: the 90s duration cap must be applied to MACAULAY too (only XC had it at fetch
time -- Robin had clips up to 794s); write trimmed clips as MONO (a stereo uncompressed trim
of 57 clips burned 900MB); and never blind-retry on `ENOSPC` (a 3x retry loop hammered a
full disk). A disk watchdog that kills the pipeline below 400MB free is worth keeping.

## BREAKTHROUGH: the embedding backend was the bottleneck — AVES2 >> BirdNET (2026-08-31)

**The single most important finding in the project so far.** BirdNET is trained to answer
"what species is this?", so it is REWARDED for making a robin's song and a robin's call look
identical (both -> "robin"). We were asking it for exactly the information its objective
tells it to discard. Swapping the frozen feature extractor fixes what no amount of adapter
training could.

Model: **`esp_aves2_sl_beats_bio`** (Earth Species AVES2, BEATs backbone, self-supervised +
bio post-training, 768-d, `pip install avex`, ~400MB, CC-BY-NC-SA = non-commercial, fits this
project). Loads via `from avex import load_model; load_model(name, device="cpu",
return_features_only=True)`; mean-pool over time. Needs 16kHz mono — use `librosa.load`,
NOT `torchaudio.load` (it now requires `torchcodec`, not installed). NOTE: the classic
`BirdAVES-biox-*` HF repos NO LONGER EXIST; superseded by the `esp-aves2-*` collection.

Head-to-head on **identical Robin clips** (matched by clip_id, same nested StratifiedGroupKFold
by recording_id, same SupCon recipe, only the input embedding differs):
```
category     BirdNET     AVES2     delta
song           0.530     0.983    +0.453   <- song/call merge SOLVED
call           0.493     0.582    +0.089
juvenile       0.417     0.399    -0.018
alarm          0.630     0.472    -0.157   <- REGRESSION, see below
overall        0.529     0.634    +0.105
```
Separability, song-vs-call: BirdNET **0.9939** -> AVES2 **0.6999**. Silhouette by call-type:
-0.0165 -> **+0.0755** (BirdNET's was NEGATIVE, i.e. worse than random). In AVES2 raw space
song scores 0.995 with NO adapter at all.

**Shortcut checks — both ruled out, do not redo:**
- Duration: within a single duration band (all 6.0s units) song-vs-call is 0.6807, vs 0.6999
  overall — unchanged. Duration alone predicts song-vs-call at 0.597 (floor 0.500). The
  separation is acoustic.
- Source: song and call are BOTH 100% Macaulay, so the song-vs-call comparison (the headline)
  is same-source and clean.

**The alarm regression is real but the baseline was suspect**: BirdNET's alarm 0.630 is
cross-source (alarm ~90% XC vs song/call 100% Macaulay) and BirdNET raw beat its own adapter
on alarm (0.722 vs 0.630) — classic shortcut signature. AVES2's 0.472 may simply be the more
honest number. Cannot resolve until the source confound is fixed. What IS real in AVES2:
song separates from everything (0.70-0.75) while alarm/call/juvenile stay crowded
(0.965-0.983) — i.e. AVES2 nails song-vs-call-like but does not solve call-type-within-calls.

**Implications before adopting wholesale:** everything downstream is built on BirdNET 1024-d
(prototypes, the trained Blackbird checkpoint, `infer_clip.py`) — switching means re-embedding
every species. AVES2 is NOT a species classifier, so the live runtime would run BirdNET for
species ID + AVES2 for call-type = two models on the device. Also untested on Blackbird: its
song/call already worked in BirdNET, so AVES2 may or may not add there.

## Robin dataset has a SOURCE CONFOUND — a bug in build_species_dataset.py (2026-08-31)

`_quality_sort_key` negates Macaulay's numeric ratings (5 -> -5.0) while XC letters map
A -> 0.0, so **every Macaulay clip sorts before every XC clip** and `--max_per_category`
silently took 150 Macaulay song + 150 Macaulay call and ZERO XC. With Macaulay having almost
no Robin alarm/juvenile, categories became confounded with recording source:
song/call 100% Macaulay, alarm/juvenile ~90% XC. The space organises by SOURCE
(silhouette +0.046) better than by CALL-TYPE (-0.019), and the pattern is 6/6 perfect:
every same-source pair merges (0.991-0.995), every cross-source pair separates (0.960-0.976).

**Consequence: Robin's alarm 0.646 reported earlier is NOT trustworthy** — likely partly
"recognise an XC recording". Robin's song-vs-call merge IS real (same-source, honest).
**Blackbird was checked and SURVIVES**: same source imbalance (song/call 100% Macaulay) but
NO source-separation pattern — its same-source song-vs-call is 0.9512, well separated, so its
result is biological.

**FIXED (2026-08-31).** `_quality_sort_key` now maps BOTH sources onto one 0.0(best)..1.0(worst)
scale (XC A-E -> 0/.25/.5/.75/1; Macaulay 1-5 -> (5-r)/4), so caps draw fairly from both.
Also rebuilt Robin as **XC-ONLY**: `data/raw/robin_xconly_v1/` +
`data/runs/robin_xconly_pattern_layer/` — **3021 units, 481 recordings, verified 0% Macaulay**
(song 1289 / alarm 854 / call 695 / juvenile 183). Macaulay Robin audio retained and now free
to serve as a genuine held-out test set.

## CLEAN (source-confound-free) Robin result — the alarm regression was the confound (2026-08-31)

Re-ran the BirdNET-vs-AVES2 head-to-head on the XC-only Robin set, where NEITHER encoder can
exploit a source shortcut:
```
category     BN raw  BN adapt   AV raw  AV adapt    delta
song          0.903     0.865    0.992     0.987    +0.122
call          0.465     0.485    0.652     0.633    +0.147
alarm         0.503     0.640    0.423     0.638    -0.002   <- regression GONE (was -0.157)
juvenile      0.453     0.452    0.296     0.378    -0.074
overall       0.601     0.636    0.637     0.703    +0.067
```
song-vs-call separability BirdNET 0.9352 -> AVES2 0.7058; silhouette +0.038 -> +0.090.

**Confirms the hypothesis: BirdNET's apparent alarm advantage was recording-source leakage,
not real.** On clean data alarm is a dead tie (0.640 vs 0.638). The tell had been that BirdNET
RAW beat its own trained adapter on alarm (0.722 vs 0.630) in the confounded set — that
inversion also disappears here (raw 0.503 vs adapter 0.640, the normal direction).

**AVES2 vs BirdNET across all three comparisons run so far:**
```
                Blackbird   Robin(confounded)   Robin(CLEAN XC-only)
song              +0.235          +0.453              +0.122
call              +0.100          +0.089              +0.147
juvenile          +0.161          -0.018              -0.074
alarm             -0.060          -0.157              -0.002
overall           +0.101          +0.105              +0.067
```
**song and call improve in ALL THREE; overall improves in ALL THREE.** juvenile is the only
genuinely mixed category (+0.161 Blackbird, -0.074 clean Robin) — note Robin juvenile is the
smallest class (183 units) so that estimate is the noisiest. Alarm, once the confound is
removed, is a wash.

**RECOMMENDATION: adopt AVES2 (`esp_aves2_sl_beats_bio`) as the default call-type encoder.**
Still to do before that is real: re-embed Blackbird's locked dataset and retrain its
deployable checkpoint on AVES2; decide the two-model story for the live device (BirdNET for
species ID + AVES2 for call-type); and re-check juvenile on a species where it isn't the
smallest class.

## "Is it actually better than guessing?" — YES, 30.8 sigma. And two cheap wins found (2026-09-01)

User challenged whether the model was really beating chance. Measured properly on clean
XC-only Robin (AVES2, 4-class):
```
uniform random over 4 classes      0.250
always-predict-majority            0.289
PERMUTATION NULL (labels shuffled BY RECORDING, same pipeline, 5 runs)   0.307 +- 0.013
ACTUAL                             0.703      -> 30.8 sigma above null
```
**Not chance. But the user's instinct was half-right**: the headline is carried by song being
easy. Per-class F1 was song 0.968 / call 0.641 / alarm 0.603 / juvenile 0.439. Strip song out
and the remaining 3-way alarm/call/juvenile problem scores only ~0.59. Confusion is
**symmetric** between alarm and call (101 vs 109 errors) — the signature of genuinely
overlapping categories rather than model failure.

**TWO CHEAP WINS, both replicated on Robin AND Blackbird (~+0.10 each species):**

1. **Decide per RECORDING, not per phrase unit.** ~~Robin 0.703 -> 0.745 on its own.~~
   **CORRECTED 2026-09-01: that gain was mostly a DENOMINATOR ARTIFACT, not real.** Per-unit
   accuracy averages over units, per-recording averages over recordings; recordings with many
   units are weighted differently, so the two numbers are not directly comparable. Measured
   properly (per-recording accuracy, varying how many units are aggregated):
   ```
   AVES2 only              1 unit 0.740 | 3 units 0.738 | all 0.740   <- aggregation adds NOTHING
   AVES2 + BirdNET concat  1 unit 0.771 | 3 units 0.799 | all 0.802   <- adds +0.031, saturates by 3
   ```
   Units within one recording are highly correlated (same bird, same session), so averaging
   them adds little independent evidence. Do NOT claim aggregation as a win for AVES2 alone.
2. **CONCATENATE AVES2 + BirdNET embeddings** (both L2-normalised, 768+1024=1792-d). The two
   encoders are genuinely COMPLEMENTARY — BirdNET is better on juvenile, AVES2 far better on
   song/call. Beats either alone.

```
                                    Robin(clean)   Blackbird
AVES2 alone, per-unit                   0.703        0.709
+ concat BirdNET + per-recording        0.802        0.819
```
Robin per-class at the best 4-class config: song F1 0.972 / alarm 0.721 / call 0.711 /
juvenile 0.540 — every category improved, juvenile still weakest.

**OPEN DESIGN DECISION — merging alarm+call.** For Robin, merging them gives **0.920**
(3-class, per-recording, floor 0.578). Biologically defensible: a robin's alarm IS a ticking
call, and the symmetric confusion suggests recordists don't apply the distinction
consistently. BUT it throws away the alarm-vs-calm distinction, which is probably the most
dramatically useful signal for the installation. NOT decided — user's call.

## Modelling ideas TESTED AND FAILED — do not retry (2026-09-01)

A full round of representation/architecture ideas aimed at the alarm/call/juvenile muddle.
**All negative.** Robin, clean XC-only, per-recording, concat AVES2+BirdNET baseline 0.802:
```
mean + BirdNET (BEST, unchanged)          0.802
hierarchical 2-stage (song-vs-rest, then 3-class)   +0.000  (exactly zero)
mean+std pooling + BirdNET                0.764   -0.038
12s longer context + BirdNET              0.780   -0.022
mean+std+ctx + BirdNET (everything)       0.771   -0.031
PCA -> 512-d                              0.809   +0.007 (noise)
PCA -> 64/128/256-d                       0.792/0.806/0.804
smaller adapter hidden=128 / 256          0.780 / 0.785  (both WORSE than 512)
```
- **Hierarchical is pointless** because the joint model ALREADY isolates song at F1 0.972 —
  a dedicated song-vs-rest stage has nothing left to contribute, and the 3-class specialist
  inherits exactly the same difficulty. The muddle is intrinsic to those three categories,
  not a side-effect of song competing for attention.
- **Richer pooling fails because std is simply a weaker representation** (std alone = 0.631
  vs mean 0.740). Concatenating a weak feature set onto a strong one dilutes it. This is NOT
  a dimensionality/overfitting problem — that hypothesis was tested and REJECTED: PCA gave at
  most +0.007 and shrinking adapter capacity made things worse.
- **Longer context is also the most expensive option at inference** (transformer attention is
  quadratic in sequence length; the 12s pass took ~4x the 6s one), so it loses on both axes.

**Conclusion: the representation is about as good as this model class gets on this data.**
The remaining alarm/call/juvenile confusion is a DATA/LABEL problem, not a modelling one.
Productive directions left are more juvenile data, by-ear review of the alarm-vs-call
distinction, or accepting the alarm+call merge (0.920). Stop trying new architectures.

**Still untested ideas for the alarm/call/juvenile muddle:** other AVES2 variants
(`esp-aves2-eat-bio`, `esp-aves2-naturelm-audio-v1-beats`); richer temporal pooling
(mean+std, or attention pooling, instead of plain mean — plain mean discards the temporal
structure that distinguishes call types); longer context windows for AVES2 (it is a
transformer; current phrase cap is 6s and only 3-6s was tested); a hierarchical model
(song-vs-rest is solved, so train a specialist for the remaining three); more juvenile data
(smallest class at 183 units / 43 recordings in Robin); by-ear review of alarm-vs-call labels.

## Robin juvenile pool EXHAUSTED — do not re-attempt this fetch (2026-09-03)

Tried to grow Robin juvenile beyond 36 recordings. Scanned BOTH sources to full depth,
WORLDWIDE (not just Europe). Net gain: **+4 recordings, +5 units. Not worth repeating.**

Availability (verified, worldwide):
```
Xeno-canto: 8156 total robin recordings | begging call 74 | stage:juvenile 92 | nestling 7
Macaulay:   8129 total robin assets     | juvenile 21 (vs 12 in the earlier EU-only scan)
```
Funnel: 61 raw fetched -> 24 passed the BirdNET species filter (~39%, consistent with the
known low species-diagnosticity of begging calls) -> 19 were ALREADY in the dataset -> only
**5 genuinely new**. Final juvenile: 40 recordings / 188 units, in
`data/runs/robin_v2_pattern_layer/` (deduped by clip_id; base was
`robin_xconly_pattern_layer`).

**TWO TRAPS HIT — both worth remembering:**
1. **`stage:"fledgling"` on the XC API is BOGUS.** It returned 8156 = the entire species.
   A control query with `stage:"gibberish12345"` also returned 8156, proving the API
   silently IGNORES unrecognised stage values rather than erroring. ALWAYS run a nonsense
   control before trusting a filtered count. Valid stages seen: juvenile, nestling, adult.
2. **`--skip_existing_dir` must list EVERY previously-fetched directory, not just the
   topically-related ones.** I passed only the two juvenile dirs, so recordings already
   downloaded under the alarm/call/song fetches were re-fetched — 59 of 70 "new" units were
   exact duplicate clip_ids. Caught by asserting on clip_id overlap before merging; ALWAYS
   dedupe by clip_id when merging pattern-layer runs.

**Conclusion, matching Blackbird exactly:** juvenile is genuinely data-ceilinged from
Macaulay+XC. Remaining levers are a different archive entirely (iNaturalist etc.), data
augmentation on the existing 188 units, or accepting juvenile as a demonstration-only /
lower-confidence category. Do NOT spend more time re-scanning these two sources.

## Juvenile data augmentation — TESTED, no reliable benefit (2026-09-03)

Mild augmentation of Robin juvenile: pitch +/-1 semitone (pedalboard) and additive noise at
25 dB SNR, 3 variants per unit, 183 -> 732 juvenile units in the training pool. Deliberately
mild because pitch is itself a call-type cue in birds.

**Methodology (important — reuse this design for any future augmentation test):**
augmented copies inherit the PARENT recording_id so grouped CV keeps them with their
original; they are tagged `is_aug` and appear in TRAIN FOLDS ONLY — the test fold is always
real units, otherwise the score is meaningless. Also tested building prototypes from real
training units only, so augmented near-duplicates don't smear the class centroid.
```
                                          overall   juvenile
baseline, no augmentation                  0.740     0.312
augment train + prototypes                 0.714     0.351
augment train, prototypes from REAL only   0.714     0.379
```
**Read this correctly: the juvenile "gain" is NOT significant.** Juvenile has only 36
recordings (6 per fold); the standard error on a proportion near 0.35 at n=36 is ~+-0.079,
so +0.067 sits inside one SE. The overall DROP of -0.026 is measured on 423 recordings
(SE ~+-0.021) and is the more reliably-estimated number — and it is negative. Do not report
the juvenile figure as a win.

Real-only prototypes DID beat polluted prototypes (0.379 vs 0.351), so if augmentation is
ever revisited, keep that refinement. But augmentation itself is not currently worth it.

## HUMAN LISTENING TEST RESULT + final headline numbers (2026-09-03)

**User took the blind listening test (`data/reviews/robin_listening_test_v1/`) and scored
40% on alarm-vs-call — BELOW the 50% chance line** (n=30, SE ~+-9%, so statistically
indistinguishable from coin-flipping), while acing the song control. An expert who has
worked with this material for months cannot hear the alarm/call distinction.

**USER DECISION: keep alarm and call SEPARATE anyway, for artistic reasons.** This is a
deliberate, informed choice, not an oversight — do not re-propose merging them. It costs
roughly 0.12 accuracy (merged would be 0.920 vs 0.802 for Robin).

**Recordist-leakage check — PASSED.** Only 16% of recordists contribute both alarm and call
(75 alarm-only, 45 call-only), so recordist identity correlates with label and CV grouped by
`recording_id` could in principle leak. Re-ran grouped by RECORDIST (fully disjoint people
across folds): overall 0.802 -> 0.780, alarm 0.749 -> 0.711, call 0.696 -> 0.668, song
unchanged 0.986. A modest ~2-4pt recordist effect exists but the signal is genuinely
acoustic, NOT recordist signature.

### FINAL RESULTS — AVES2 + BirdNET concatenated, per-recording decision
```
ROBIN (clean XC-only, 481 recordings)      BLACKBIRD (alarm-capped, 400 recordings)
  song      0.987                            song      0.966
  alarm     0.749                            call      0.822
  call      0.696                            juvenile  0.690
  juvenile  0.493                            alarm     0.684
  overall   0.802                            overall   0.819
  (by recordist, strictest: 0.780)
majority floor 0.289 | permutation null 0.307 (+-0.013), i.e. ~2.7x chance
```
Blackbird's historically stuck categories improved enormously against where this project sat
before the encoder switch: **alarm 0.423 -> 0.684, juvenile 0.417 -> 0.690.** The juvenile
figure in particular overturns the long-standing "information-limited, do not retry"
conclusion — the information was missing from BirdNET's embeddings, not from the world.

**NOTABLE: the model beats the human expert on alarm-vs-call (~0.69 vs 0.40).** Two readings,
not yet separable: (a) it exploits sub-perceptual acoustic cues, or (b) recordist tags encode
CONTEXT the recordist could see (bird mobbing a predator) that correlates with subtle
acoustics. Either way the ground truth is "a recordist wrote alarm", never verified behaviour.

## Evaluation protocol, test-set sizes, and the NESTED-CV honest numbers (2026-09-03)

User asked the right question: what exactly is the test set, and is there a validation set?

**Protocol.** There is no single held-out test set — it is 6-fold StratifiedGroupKFold,
grouped by `recording_id`, so every recording is held out exactly once and the reported
figure is the mean over 6 rotations.
```
              per-fold train    per-fold test        total test decisions
ROBIN         ~352 recordings   ~70 rec / 230 units  423 recordings
BLACKBIRD     ~414 recordings   ~82 rec / 230 units  497 recordings
```
A typical Robin test fold contains only **5 juvenile recordings** — which is exactly why
juvenile's error bars are enormous and why single-digit moves in that class mean nothing.

**We did NOT have a validation set, and configuration WAS being selected on the reported
folds** (~15 configs compared this session: encoder choice, aggregation, pooling,
hierarchical, PCA, capacity). That is the soft leakage this file already warned about.

**FIXED with proper nested CV** — candidate config chosen on INNER folds
(StratifiedGroupKFold n=4 inside each outer training set) only; outer test touched once:
```
              reported    NESTED (honest)   per-fold range
ROBIN          0.802          0.799          0.743 - 0.871
BLACKBIRD      0.819          0.809          0.759 - 0.880
```
**The bias was negligible (-0.003 / -0.010) because `concat + per-recording` was selected in
6/6 outer folds for BOTH species** on inner data alone. The configuration is robust, not a
noise artifact. Treat **~0.80 +- 0.02** as the defensible headline for both species.

## TAWNY OWL — DONE, and it SOLVES the juvenile problem (2026-09-03)

Third species. `tawowl1` (verified via API). Chosen for maximum sonic contrast with the two
warblers (low nocturnal hooting) AND because it was the only candidate with real juvenile
depth. Dataset `data/raw/owl_combined_v1/`, embeddings `data/runs/owl_pattern_layer/` +
`data/runs/aves2_embeddings/owl_full_aves2.npy`.

**Owl-specific labelling adaptations (now in the shared code):** `hoot` -> song,
`keewick`/`kewick` -> call, `duet` -> EXCLUDED on BOTH the XC and Macaulay sides (a duet is
two birds answering, so the clip carries two vocal types). 17 mixed-content clips dropped
per the Robin precedent -> `data/reviews/tawny_owl_label_exclusions_v1.csv`.

**ACCEPTED SEX CONFOUND (user decision).** For Tawny Owl, male->song 90% and female->call
94% — the "twit-twoo" is a female kewick answered by a male hoot. User decided these are
legitimate distinct vocal types, so we proceed. BUT the song/call figure for this species
partly measures voice-sex discrimination, NOT purely call function. Report it that way; it
is not directly equivalent to Robin's or Blackbird's song/call number.

**Yields were exceptional: 646/692 recordings accepted (93%)**, vs ~30% juvenile acceptance
for Robin. Owlet begging screeches are far more species-diagnostic to BirdNET than passerine
begging. 4029 units. Source balance on the critical song/call pair is CLEAN (song 462ML/487XC,
call 421/609) so the Robin confound did not recur; alarm is 96% XC (Macaulay has only 3).

### RESULTS — nested CV, honest
```
                per-recording    (nested-CV per-unit gave 0.801 — the two agree)
  juvenile        0.908   <- BEST juvenile in the project by a mile
  song            0.865
  call            0.788
  alarm           0.418   <- new weak category
  overall         0.800
majority floor 0.297 | permutation null 0.281 | AVES2 silhouette +0.183 (best yet)
```
AVES2 beat BirdNET again (+0.143 overall) — **third independent replication**.

**THE HEADLINE: juvenile 0.908.** It was 0.493 (Robin) and 0.690 (Blackbird), and was
declared data-ceilinged on both. Given 186 accepted recordings instead of 40, it becomes the
STRONGEST category. This confirms the long-running diagnosis: juvenile was never a modelling
problem, it was a data-availability problem, and choosing a species where the data exists
solves it outright.

**Alarm (0.418) is the trade.** Only 79 recordings — owls lack a distinct passerine-style
alarm call, and Macaulay has almost none. Expect alarm to stay weak for this species.

### CROSS-SPECIES SUMMARY (all AVES2+BirdNET concat, nested CV)
```
              Robin    Blackbird   Tawny Owl
song          0.987      0.966       0.865
call          0.696      0.822       0.788
alarm         0.749      0.684       0.418
juvenile      0.493      0.690       0.908
overall       0.799      0.809       0.800
```
Remarkably consistent overall (~0.80 on all three) but with very different per-category
profiles — each species is strong where its data is deep. No species is good at everything.

## DROPPED FROM THE PROJECT: species pitch-shifting of response audio (2026-09-03)

The "shift the response track's pitch to match how high/low the species sings" feature is
**cut by user decision — do not build on it or re-propose it.** The code still exists and
works (`src/compute_species_pitch_profile.py`, the `--play_response`/`--pitch_profiles_dir`
path in `scripts/infer_clip.py` and `scripts/live_mic.py`, and profiles for blackbird
(reference, 2184 Hz), robin (+14.23 semitones) and wood_pigeon (-24.98)), but it is no
longer part of the installation. Leave it in place; just don't extend it, and don't compute
profiles for new species.

## DEPLOYABLE CHECKPOINTS — all three now on the SAME validated recipe (2026-09-03)

`train_supcon_adapter.py` gained `--aves2_npy` / `--aves2_meta`, which L2-normalise each
encoder separately and concatenate as `L2(aves2_768) || L2(birdnet_1024)` — matching the
evaluation exactly. Without this the trainer could only produce BirdNET-only (1024-d)
checkpoints, i.e. NOT the model any of the reported numbers describe.

**USE THESE:**
```
data/runs/robin_supcon_adapter_v2/        1792-d concat   (nested-CV 0.799)
data/runs/blackbird_supcon_adapter_v2/    1792-d concat   (nested-CV 0.809)
data/runs/tawny_owl_supcon_adapter_v1/    1792-d concat   (nested-CV 0.800)
```
All verified: load cleanly, 4x128 prototypes, identical label order
`["alarm","call","juvenile","song"]`, forward pass OK. Train-set sanity accuracies are
0.99+ — that is MEMORISATION, not generalisation; the honest figures are the nested-CV ones.

The `*_supcon_adapter_v1` dirs are the SUPERSEDED BirdNET-only checkpoints (robin/blackbird);
kept only for rollback, ~16MB total. `tawny_owl_supcon_adapter_v1` is NOT superseded — the
owl was trained on the concat recipe from the start.

**DEPLOYMENT IMPLICATION:** these checkpoints need BOTH encoders at inference. Fine for the
precompute path (labels computed offline, device just plays + looks up). For any live path,
AVES2 is ~95ms/unit but BirdNET's current `birdnet_embedding_array` call measured **6.6s**
per clip because it spins the whole embedding stack per call — a warm-loaded BirdNET would
be needed, and has not been written.

## NEXT STEP AGREED: build the INSTALLATION, not more research (2026-09-03)

Gap analysis: research is in good shape (3 species, all ~0.80, honestly validated) but the
installation is barely started — no curated source-clip library, no precomputed clip->label
lookup, no `live_*.py` runtime, two speakers never wired, NUC never tested. The only runtime
code is `scripts/infer_clip.py` (single-clip demo) and `scripts/live_mic.py` (a test tool for
a microphone the installation deliberately does not have).

Priority order: (1) checkpoints consistent — DONE; (2) precompute path: curated clips ->
one offline classification pass -> lookup table = the device's brain, no models on the NUC;
(3) NUC + two speakers + soak test. **BLOCKED ON AN ARTISTIC DECISION: which source clips
does the installation actually play?** The precompute library is shaped around that answer.

Deliberately NOT doing now: a 4th species, and external validation on the held-out
`data/raw/robin_macaulay_eu_v1` (508 mp3s, a source never trained on). Both worthwhile,
neither moves the exhibition forward.

## Still open (superseded parts of this list resolved above where noted)

- Promote SupCon code into proper committed `src/` (SupCon loss class DONE
in `adapter_model.py`; `train_supcon_adapter.py` DONE, trains a deployable checkpoint +
prototypes, format-compatible with `transform_embeddings.py`/`assign_anchor_labels.py` --
but NOT YET RUN to completion, was interrupted to fetch XC data and re-evaluate first, see
above); nested-CV tuning as the standard path in committed code, not just scratchpad;
alternative embedding backends (perch-hoplite, avex_effnet_bio); decide the alarm-crowding
question before training the final checkpoint, since it affects which dataset to train on.

**Open next steps, not yet decided:**
1. Should the adapter checkpoint from this be saved/reused (e.g., trained on ALL data,
   not just per-fold) for actual downstream use (assign_anchor_labels.py
   --adapter_checkpoint)? CV gives an honest accuracy estimate but doesn't itself produce
   a single deployable checkpoint — a final model would need training on the full dataset.
2. Juvenile: accept it as a demonstration-only/lower-confidence category for now (matches
   earlier "unknown fallback" design), or keep pursuing more juvenile data via other
   levers (e.g. widening beyond Europe, discussed and deferred earlier).
3. Same balanced-pairs fix + full pipeline could/should now be tried on other species
   (Wren, Wood Pigeon, Blue Tit) once/if similar independent Macaulay ground truth is
   pulled for them — this Blackbird result doesn't automatically generalize to other
   species' embedding spaces.

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

## The mic PoC's blocker is now RESOLVED (2026-09-08)

`MIC_POC_PLAN.md` scoped itself to **species only**, explicitly "no call-type, per the
unsolved label problem". **That problem is now solved.** Three species heads exist in
`models/` (blackbird, robin, tawny_owl) at ~0.80 nested-CV accuracy, so the PoC's
`species -> response` map can become `species + vocalisation type -> response`, i.e. 12
combinations rather than 3.

`src/classify_clip.py` is the shared inference core for this: it holds both frozen encoders
warm and returns `(species, vocalisation, confidences)`. Use it for the PoC's still-to-build
`src/live_species.py` / `src/live_respond.py` rather than writing a second inference path.
`scripts/live_soundscape.py` is the MIC-LESS installation runtime (plays a soundscape and
classifies its own buffer, pipelined so segment N+1 is classified while N plays).

The two paths deliberately coexist: the mic PoC is a bench test of whether BirdNET survives
real room acoustics; the soundscape runtime is the exhibition design. Both call the same
classifier.

## Mic PoC (2026-08-09) — plan + recorder

`MIC_POC_PLAN.md` is the plan for a bench PoC on the x86 Linux box: **mic in -> species ->
play a mapped response clip**. Note this deliberately reintroduces the microphone that the
live-installation decision above dropped — it is a bench test of whether BirdNET survives
real acoustic audio in a real room, NOT a reversal of the installation design. Feedback
(response audio retriggering the mic) is handled by muting input during playback + cooldown.
Scope is **species only** — no call-type, per the unsolved label problem.

**BUILT and tested:** `src/live_audio.py` (capture primitives) + `scripts/record_clip.py`
(CLI). Modes: `--list_devices`, `--meter` (noise floor + suggested VOX threshold),
`--seconds` (fixed clip), `--auto` (sound-activated, one file per event), `--analyse`
(chains into `infer_clip.py`). Writes 48 kHz mono PCM_16 to `data/raw/mic_recordings/`
(gitignored) + `recordings_log.csv` of per-clip levels. Verified on real hardware (Mac
built-in mic) and by unit tests on synthetic audio; NOT yet run on the NUC with a USB mic.

Design points worth not relitigating: capture at the device's native rate and resample
once at save time via scipy `resample_poly` (per-block resampling would leave a
discontinuity at every block boundary; scipy avoids dragging librosa/numba into the live
runtime). Recordings are saved WITHOUT peak normalization — absolute level is what the
energy gate is calibrated against. `requirements-deploy.txt` gained `sounddevice`;
`setup_nuc.sh` gained `libportaudio2 portaudio19-dev`.

Still to build (see plan stages 2-4): `src/live_species.py`, `src/live_respond.py`,
`scripts/live_demo.py`, `config/response_map.yaml`.

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
