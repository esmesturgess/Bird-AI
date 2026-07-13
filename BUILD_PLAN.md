# Bird Translation AI Build Plan

## Locked Decisions

- Project type: non-commercial art/research prototype.
- V1 target species: Common Wood Pigeon and Eurasian Wren.
- V1 output: `species`, `species_confidence`, `cluster_id`, `cluster_confidence`, `music_trigger`.
- V1 does not require named biological call types.
- V1 should work on live microphone audio after offline training.
- BirdNET stays frozen for species inference and remains the default embedding baseline.
- Earth Species AVEX / ESP-AVES2 embeddings can be tested as an alternate within-species representation when clustering quality needs improvement.
- Within-species grouping is unsupervised.
- Music mapping is rule-based / artist-designed first.

## V1 Success Criteria

- A live microphone stream can be processed on a small Linux device.
- BirdNET can usually separate wood pigeon from wren in the target environment.
- For each of the two species, the system assigns incoming sounds to repeatable within-species clusters.
- The system returns `unknown` when species or cluster confidence is too low.
- Stable species-plus-cluster detections trigger distinct music responses.

## What We Keep

These pieces are already aligned with the V1 architecture and should stay:

- [README.md](/Users/esmesturgess/Bird Translation AI/README.md)
  - Good description of the current offline vertical slice and artifact layout.
- [src/pipeline.py](/Users/esmesturgess/Bird Translation AI/src/pipeline.py)
  - Keep as the batch orchestration entry point.
- [src/segment.py](/Users/esmesturgess/Bird Translation AI/src/segment.py)
  - Keep as the first pass event segmentation layer.
- [src/birdnet_infer.py](/Users/esmesturgess/Bird Translation AI/src/birdnet_infer.py)
  - Keep as the BirdNET wrapper.
- [src/embed.py](/Users/esmesturgess/Bird Translation AI/src/embed.py)
  - Keep BirdNET embeddings as the default representation.
- [src/features.py](/Users/esmesturgess/Bird Translation AI/src/features.py)
  - Keep interpretable acoustic descriptors and heuristic indices.
- [src/cluster.py](/Users/esmesturgess/Bird Translation AI/src/cluster.py)
  - Keep HDBSCAN as the default training-time clustering method.
- [src/report.py](/Users/esmesturgess/Bird Translation AI/src/report.py)
  - Keep the report/listening workflow for manual review.
- Current `data/` artifact structure
  - Keep the separation between raw audio, events, predictions, embeddings, clusters, and reports.

## What We Add

### 1. Species Scope and Dataset Curation

Add a small config-driven species scope so the repo knows V1 is only:

- `wood_pigeon`
- `wren`

New files:

- `config/v1_species.yaml`
- `data/manifests/train_manifest.csv`
- `data/manifests/val_manifest.csv`
- `data/manifests/test_field_manifest.csv`

Purpose:

- Separate web recordings from field recordings.
- Track source, species, recording conditions, and split membership.
- Keep a held-out field set for realistic evaluation.

### 2. Training Artifacts for Live Inference

The current repo clusters offline, but V1 needs reusable live-time artifacts.

Add:

- per-species cluster prototypes or exemplars
- cluster distance thresholds
- species confidence thresholds
- optional smoothing settings for live triggering

New files/modules:

- `src/artifacts.py`
- `src/calibration.py`
- `models/species_thresholds.json`
- `models/cluster_prototypes.json`
- `models/cluster_thresholds.json`

Purpose:

- Train offline once.
- Deploy small artifacts for fast on-device assignment.
- Avoid online re-clustering on the device.

### 3. Live Inference Path

Add a streaming path that reuses the trained artifacts.

New files/modules:

- `src/live.py`
- `src/audio_stream.py`
- `src/assign.py`

Purpose:

- Capture microphone audio in rolling windows.
- Run BirdNET on short windows.
- Extract BirdNET embeddings for accepted detections.
- Assign to the nearest species-conditioned cluster prototype.
- Emit `unknown` if thresholds fail.

### 4. Noise and Rejection Layer

Add a simple explicit rejection layer before music triggering.

New files/modules:

- `src/reject.py`

Purpose:

- Reject silence, speech-like events, low-energy windows, and outliers.
- Reduce false triggers from wind, people, and other non-target audio.

Recommended V1 behavior:

- energy gate
- duration gate
- BirdNET confidence gate
- cluster-distance gate
- debounce / temporal smoothing across 2-3 consecutive windows

Biodenoising note:

- Keep Earth Species `biodenoising` in the plan for the live environmental-microphone installation stage.
- Do not enable it by default for clean web-audio clustering experiments yet; when we reach live field audio, compare raw vs denoised audio before BirdNET detection and before music triggering.

### 5. Music Trigger Interface

Add a thin interface that turns stable detections into musical events.

New files/modules:

- `src/music_map.py`
- `src/music_trigger.py`
- `config/music_map.yaml`

Purpose:

- Keep the ML system separate from the music system.
- Allow fixed mapping from `species + cluster_id` to a sound, MIDI message, OSC event, or playlist cue.

### 6. Evaluation and Review

Add a small evaluation layer focused on real-world usefulness, not academic perfection.

New files/modules:

- `src/eval_live.py`
- `src/review_clusters.py`

Purpose:

- measure species stability on field audio
- measure cluster repeatability within each species
- review representative clips and optionally attach human-readable notes to clusters

## What We Do Not Build In V1

- Custom species model training
- Joint species-plus-call-type model
- Automatic biological-function labeling
- Online clustering on the Raspberry Pi
- End-to-end music generation from raw embeddings
- More than two species

## Exact Implementation Order

### Phase 1. Lock Species and Data Contracts

Goal:

- Make the repo explicitly aware that V1 is `wood_pigeon + wren`.

Tasks:

1. Add `config/v1_species.yaml`.
2. Add manifest columns for:
   - `species_label`
   - `species_source`
   - `recording_source` (`web` or `field`)
   - `split`
   - `notes`
3. Create train/validation/test-field manifests.
4. Update the README with the V1 scope.

Done when:

- We can point the pipeline at a curated two-species dataset with fixed splits.

### Phase 2. Refactor Batch Pipeline Into Reusable Train Steps

Goal:

- Keep the current batch pipeline, but expose train-time outputs needed for deployment.

Tasks:

1. Split `src/pipeline.py` into clear stages:
   - segmentation
   - BirdNET inference
   - embedding generation
   - clustering
   - artifact export
2. Keep the current offline report generation.
3. Export cluster training metadata after clustering.

Likely touched files:

- `src/pipeline.py`
- `src/cluster.py`
- `src/embed.py`
- new `src/artifacts.py`

Done when:

- A single training run produces both the current report artifacts and deployable cluster artifacts.

### Phase 3. Add Cluster Prototype Export

Goal:

- Convert offline HDBSCAN results into fast inference-time lookup objects.

Tasks:

1. For each species cluster, export one or more prototypes:
   - centroid in embedding space, or
   - representative exemplars
2. Export max-distance or percentile-distance thresholds per cluster.
3. Store species-specific artifact files under `models/`.

Important note:

- HDBSCAN remains the training-time clustering algorithm.
- Live inference should use nearest-prototype assignment, not live HDBSCAN.

Done when:

- We can load a saved artifact and assign a new embedding to `cluster_id` or `unknown`.

Current implementation note:

- `src.pattern_cluster` now exports prototype artifacts for every pattern cluster run:
  - `models/cluster_prototypes.json`
  - `models/cluster_prototypes.npz`
- It also exports centroid-neighbour merge review files:
  - `reports/cluster_merge_suggestions.csv`
  - `reports/same_different_review_pairs.csv`
- These review files are the intended source of same/different judgments for the later species-specific adapter.

### Phase 4. Add Confidence Calibration

Goal:

- Turn raw model scores into safer trigger decisions.

Tasks:

1. Calibrate BirdNET species thresholds on validation clips for wood pigeon and wren.
2. Calibrate cluster-distance thresholds per species.
3. Define one combined decision rule:
   - accept species only above threshold
   - accept cluster only within threshold
   - otherwise emit `unknown`

Likely touched files:

- new `src/calibration.py`
- new `models/species_thresholds.json`
- new `models/cluster_thresholds.json`

Done when:

- We have a reproducible policy for when to trigger and when to abstain.

### Phase 5. Build the Live Audio Path

Goal:

- Add real-time inference using the trained artifacts.

Tasks:

1. Add microphone capture in fixed windows.
2. Reuse the preprocessing/segmentation logic where practical.
3. Run BirdNET on rolling windows.
4. Extract embeddings for accepted detections.
5. Assign the embedding to the nearest cluster prototype.
6. Output a structured event:
   - timestamp
   - species
   - species_confidence
   - cluster_id
   - cluster_confidence
   - decision (`accept` or `unknown`)

Likely touched files:

- new `src/audio_stream.py`
- new `src/live.py`
- new `src/assign.py`

Done when:

- A microphone feed can produce structured detection events in real time.

### Phase 6. Add Rejection and Temporal Smoothing

Goal:

- Prevent noisy live audio from creating too many false triggers.

Tasks:

1. Add energy and duration gates.
2. Add optional speech/noise rejection heuristics.
3. Add temporal smoothing so music triggers only after stable detections across multiple windows.
4. Add cooldown behavior to prevent repeated rapid-fire triggers.

Likely touched files:

- new `src/reject.py`
- `src/live.py`

Done when:

- Outdoor noise no longer causes constant triggering.

### Phase 7. Add Music Mapping and Trigger Output

Goal:

- Connect the detector to the art system without entangling the ML code.

Tasks:

1. Add `config/music_map.yaml`.
2. Map `species + cluster_id` to a symbolic music event.
3. Implement a first output transport:
   - OSC is a good first default
   - MIDI is also fine
4. Keep this layer deterministic and hand-designed in V1.

Likely touched files:

- new `src/music_map.py`
- new `src/music_trigger.py`

Done when:

- Stable detections can trigger distinct musical cues on another device.

### Phase 8. Tighten the Review Loop

Goal:

- Improve clusters and music mapping through listening, not guesswork.

Tasks:

1. Extend the report workflow to show species-specific cluster representatives.
2. Add a small review file for manual annotations:
   - cluster nickname
   - probable call type
   - music note
3. Use reviewed clusters to refine thresholds and the mapping table.

Likely touched files:

- `src/report.py`
- new `src/review_clusters.py`
- `config/music_map.yaml`

Done when:

- We can iteratively improve the system by reviewing a manageable number of representative clips.

## First Coding Sprint

If building starts immediately, the first sprint should be:

1. Add `config/v1_species.yaml`.
2. Add split-aware manifests for wood pigeon and wren.
3. Refactor the training pipeline to export deployable cluster artifacts.
4. Implement nearest-prototype assignment for new embeddings.
5. Add a simple command-line live runner that prints structured events before any music integration.

This gets us to the first meaningful checkpoint:

- offline training works
- deployable artifacts exist
- live microphone events can be printed and inspected

## Second Coding Sprint

1. Add confidence calibration.
2. Add rejection and smoothing.
3. Add music mapping and OSC/MIDI trigger output.
4. Test on held-out field audio.

This gets us to the first art-ready prototype.

## Recommended Repo Shape After V1

```text
config/
  v1_species.yaml
  music_map.yaml
models/
  species_thresholds.json
  cluster_prototypes.json
  cluster_thresholds.json
src/
  artifacts.py
  assign.py
  audio_stream.py
  calibration.py
  live.py
  music_map.py
  music_trigger.py
  reject.py
```

## Guiding Principle

Do not try to make the model "understand bird emotion" in V1.

Make it do four things well:

1. hear a target species
2. assign a stable within-species cluster
3. abstain when uncertain
4. trigger music reliably

That is the right foundation for later call-type naming, biological interpretation, and richer sonification.
