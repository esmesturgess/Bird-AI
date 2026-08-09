# Bird Translation AI: Vertical Slice v0

This repo implements a first end-to-end baseline for exploratory bird-vocalization sonification workflows.

## ▶ Run the demo on the Intel NUC (or any x86 Linux box)

Pull this from GitHub and run the "bird sound in → spoken species out" demo:

```bash
git clone https://github.com/esmesturgess/Bird-AI.git
cd Bird-AI
bash scripts/setup_nuc.sh                      # installs venv, deps, espeak-ng, BirdNET
# copy a blackbird clip onto the box, then:
./.venv/bin/python scripts/infer_clip.py --clip <clip.wav> --speak
```

It detects the species and **says it aloud** (e.g. "Blackbird"). Full step-by-step,
what to copy over, and troubleshooting: **[scripts/README_NUC.md](scripts/README_NUC.md)**.
The trimmed runtime dependencies are in `requirements-deploy.txt`.

> Status: species detection works end-to-end. The call-type step ("Blackbird → alarm")
> is scaffolded but not yet accurate — see `CLAUDE.md`.

If you are re-entering the project and want the quickest accurate overview first, start with:

- [PROJECT_REBRIEF.md](/Users/esmesturgess/Bird%20Translation%20AI/PROJECT_REBRIEF.md)
- [REPO_ZONES.md](/Users/esmesturgess/Bird%20Translation%20AI/REPO_ZONES.md)

## V1 Scope

Phase 1 now locks the first live prototype to:

- Common Wood Pigeon (`wood_pigeon`)
- Eurasian Wren (`wren`)

The current project scope is a **non-commercial** art/research prototype:

- train offline on a larger computer
- run live inference on a Raspberry Pi-class Linux device
- use BirdNET for species inference
- use unsupervised within-species clustering for `cluster_id`
- trigger music from stable `species + cluster_id` detections
- allow `unknown` when confidence is low

The pipeline does **weak supervision**:
- Segments candidate vocal events from long recordings.
- Runs BirdNET-Analyzer inference for top-k species predictions per event.
- Computes interpretable acoustic features and derived indices (`urgency_index`, `song_likeness`).
- Builds baseline offline embeddings.
- Clusters events with HDBSCAN.
- Produces a UMAP visualization and an HTML listening/report view.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install BirdNET-Analyzer separately so `birdnet-analyze` is available in your shell.
For BirdNET embeddings support, install analyzer extras in the same venv:

```bash
pip install "birdnet-analyzer[embeddings]"
```

If embeddings still fall back, ensure these are present in the venv:

```bash
pip install perch-hoplite ml-collections
```

## Run In Colab (Recommended For Storage/Compute)

If you want to keep developing locally but run the pipeline in Colab, use:

- [colab/README_COLAB.md](colab/README_COLAB.md)
- [colab/Bird_Translation_AI_Colab.ipynb](colab/Bird_Translation_AI_Colab.ipynb)

This workflow stores inputs/outputs in Google Drive and avoids filling local disk.

## Input

Put long recordings in:

```text
data/raw/
```

Supported audio formats: `wav, mp3, flac, ogg, m4a, aac`.

## Dataset Curation

Phase 1 adds split-aware recording manifests under `data/manifests/`:

- `train_manifest.csv`
- `val_manifest.csv`
- `test_field_manifest.csv`

These are the source-of-truth curation files for V1 and are separate from the generated
`events_manifest.csv` cache produced during segmentation.

The V1 species scope lives in `config/v1_species.yaml`. For Phase 1 tooling this file is
stored as JSON-compatible YAML so it can be parsed without adding a new config dependency.

Required manifest columns:

- `recording_id`
- `audio_path`
- `species_label`
- `species_common_name`
- `scientific_name`
- `species_source`
- `recording_source`
- `split`
- `notes`

Current allowed values:

- `species_label`: `wood_pigeon`, `wren`
- `recording_source`: `web`, `field`
- `split`: `train`, `val`, `test_field`

You can validate the Phase 1 manifests with:

```bash
./.venv/bin/python -m src.dataset
```

To also verify that all referenced source recordings exist on disk:

```bash
./.venv/bin/python -m src.dataset --require-files
```

## Manifest-Driven Training

Phase 2 makes the training entrypoint manifest-aware.

Preferred workflow:

1. Fill `data/manifests/train_manifest.csv` with the recordings you want to train on.
2. Optionally fill `data/manifests/val_manifest.csv` for later calibration work.
3. Validate the manifests.
4. Run either a species-only BirdNET check or the full training pipeline from the project venv.

Example:

```bash
./.venv/bin/python -m src.pipeline \
  --train_manifest data/manifests/train_manifest.csv \
  --val_manifest data/manifests/val_manifest.csv \
  --input_dir data/raw \
  --output_dir data \
  --birdnet_binary ./.venv/bin/birdnet-analyze \
  --birdnet_embeddings_binary ./.venv/bin/birdnet-embeddings
```

If the train manifest exists but has no rows yet, the pipeline falls back to scanning `data/raw/`.

To stop after BirdNET species predictions and inspect species recognition before clustering:

```bash
./.venv/bin/python -m src.pipeline \
  --train_manifest data/manifests/train_manifest.csv \
  --val_manifest data/manifests/val_manifest.csv \
  --output_dir data \
  --birdnet_binary ./.venv/bin/birdnet-analyze \
  --birdnet_embeddings_binary ./.venv/bin/birdnet-embeddings \
  --stop_after_species
```

For stricter species-only curation, you can bias the per-recording cap toward longer, stronger clips and drop the weak tail:

```bash
./.venv/bin/python -m src.pipeline \
  --train_manifest data/manifests/train_manifest.csv \
  --val_manifest data/manifests/val_manifest.csv \
  --output_dir data/runs/wren_species_full_cap10_quality_v1 \
  --birdnet_binary ./.venv/bin/birdnet-analyze \
  --birdnet_embeddings_binary ./.venv/bin/birdnet-embeddings \
  --stop_after_species \
  --max_events_per_recording 10 \
  --selection_pref_duration 0.55 \
  --selection_score_min 0.04
```

Re-running the same command against the same `output_dir` will reuse the cached segmented events and BirdNET predictions unless the source recordings or selector settings change.

For BirdNET species checks, the pipeline can also build fixed-length inference clips from the original recording around each detected event. This is useful because BirdNET is designed around 3-second snippets, and short event fragments are often harder to identify than a short window with real surrounding context:

```bash
./.venv/bin/python -m src.pipeline \
  --train_manifest data/manifests/train_manifest.csv \
  --val_manifest data/manifests/val_manifest.csv \
  --output_dir data/runs/wren_species_full_cap10_context_soft_v1 \
  --birdnet_binary ./.venv/bin/birdnet-analyze \
  --birdnet_embeddings_binary ./.venv/bin/birdnet-embeddings \
  --stop_after_species \
  --max_events_per_recording 10 \
  --selection_pref_duration 0.45 \
  --selection_score_min 0.03 \
  --birdnet_context_seconds 3.0
```

You can also populate the manifests from a single species folder, for example a Google Drive folder:

```bash
./.venv/bin/python -m src.curate_manifest \
  --source_dir "/path/to/species/folder" \
  --species_label wren \
  --species_common_name "Eurasian Wren" \
  --scientific_name "Troglodytes troglodytes"
```

## xeno-canto Ingestion

To scale beyond manual downloads, the repo now includes a xeno-canto v3 fetcher.

Important:

- xeno-canto API v2 is discontinued
- API v3 requires a real API key on every request
- exact species fetches use tagged queries like `sp:"Troglodytes troglodytes"` rather than the old v2 free-text query style

Set your key in the shell:

```bash
export XC_API_KEY="your_real_xeno_canto_key"
```

Then fetch recordings plus metadata and a manifest-shaped CSV:

```bash
./.venv/bin/python -m src.fetch_xeno_canto \
  --species "Troglodytes troglodytes" \
  --species "Columba palumbus" \
  --output_dir "/path/to/Google Drive/BirdTranslationAI/raw_xeno_canto" \
  --min_quality C \
  --max_recordings_per_species 80 \
  --per_page 100 \
  --sleep_seconds 0.5
```

Useful options:

- `--species_config config/v1_species.yaml`
- `--species_csv path/to/species.csv`
- `--country "United Kingdom"`
- `--area europe`
- `--sound_type song`
- `--quality_tag '>C'`
- `--min_length_seconds 1.0 --max_length_seconds 20.0`
- `--metadata_only`

Outputs:

- `xc_recordings_metadata.csv`: fetched metadata with local file paths and download status
- `xc_recordings_manifest.csv`: manifest-shaped rows for downloaded files
- `xc_fetch_summary.json`: fetch summary by species

## What To Check After A Run

Phase 2 is designed so you can inspect whether the run is behaving sensibly.

Check these first:

- `data/artifacts/train_recordings_used.csv`
- `data/artifacts/training_run_summary.json`
- `data/artifacts/cluster_training_summary.csv`
- `data/preds/master_predictions.csv`
- `data/features/master_features_clustered.csv`
- `data/reports/index.html`

What each file is for:

- `train_recordings_used.csv`: confirms exactly which recordings were included in training
- `training_run_summary.json`: high-level counts and the main output checkpoints
- `cluster_training_summary.csv`: per-cluster event counts and averages, including species-conditioned clusters
- `master_predictions.csv`: BirdNET results before clustering
- `master_features_clustered.csv`: joined event-level feature table with cluster assignments
- `index.html`: listening report for manual review

For species-only runs, check these first:

- `data/artifacts/train_recordings_used.csv`
- `data/artifacts/species_run_summary.json`
- `data/artifacts/species_prediction_summary.csv`
- `data/preds/master_predictions.csv`

Note:

- Phase 2 exports training metadata and report checkpoints.
- Cluster prototypes and deployment artifacts come in the next phase.

## Run

You can still run the original folder-scan flow directly:

```bash
./.venv/bin/python -m src.pipeline --input_dir data/raw --output_dir data
```

```bash
python -m src.pipeline \
  --input_dir data/raw \
  --output_dir data \
  --sr 32000 \
  --min_dur 0.25 \
  --max_dur 4.0 \
  --seg_threshold_db 8 \
  --merge_gap 0.25 \
  --birdiness_min_hz 700 \
  --birdiness_ratio_min 0.0 \
  --seg_rms_min 0.003 \
  --conf_thresh 0.25 \
  --top_k 5 \
  --birdnet_binary birdnet-analyze \
  --birdnet_embeddings_binary birdnet-embeddings \
  --min_species_samples 5 \
  --hdbscan_min_cluster_size 0 \
  --hdbscan_min_samples 0 \
  --embedding_backend birdnet \
  --embedding_fallback_backend mel_time \
  --embedding_mel_frames 96
```

Use `--force` to recompute outputs and ignore cached artifacts.

Notes:
- `--hdbscan_min_cluster_size 0` and `--hdbscan_min_samples 0` use adaptive defaults.
- For very small datasets, try `--hdbscan_min_cluster_size 2 --hdbscan_min_samples 1`.

## Output Structure

```text
data/
  raw/                # input recordings
  events/             # segmented event clips (.wav)
  preds/              # BirdNET per-clip json + master predictions csv
  features/           # per-clip json + master csv
  specs/              # mel spectrogram .png + .npy per clip
  embeddings/         # embeddings.npy + pca/umap + labels + scatter plot
  clusters/           # global + species-conditioned cluster assignments
  reports/            # html report + distribution plots + playlists + cluster mean mel plots
  manifests/          # event manifest cache
```

Key files:
- `data/preds/master_predictions.csv`
- `data/features/master_features.csv`
- `data/features/master_features_clustered.csv`
- `data/clusters/master_clusters.csv`
- `data/reports/index.html`
- `data/reports/cluster_means/cluster_<id>_mean_mel.png`
- `data/embeddings/umap_clusters_global.png`
- `data/embeddings/embedding_meta.json`
- `data/embeddings/embedding_index.csv`

## What Features Mean

- `urgency_index`: z-scored sum of bandwidth, flatness, repetition rate, and RMS.
  - Higher often means harsher/broader/louder/more repetitive.
- `song_likeness`: z-scored sum of tonality proxy, pitch stability, and duration.
  - Higher often means more tonal/stable/phrase-like events.

These are heuristic exploratory scores, not biological truth labels.

## How To Interpret The Report

1. Open `data/reports/index.html`.
2. Check the global UMAP and distributions to spot dataset bias and outliers.
3. For each cluster, listen to representative clips and compare:
   - `urgency_index` (alarm-like vs calmer),
   - `song_likeness` (tonal/song-like vs broadband/noisy).
4. Treat `cluster_id == -1` as noise/outliers.
5. Use cluster representatives as a manual tagging starting point in future iterations.

Representative clips are selected as the most central clips in embedding space, but the selector now also tries to spread picks across source recordings so one recording does not dominate the entire review set.

You can tune that in `src.pattern_cluster` with:

```bash
./.venv/bin/python -m src.pattern_cluster \
  --pattern_run_dir data/runs/wren_pattern_layer_v1_birdnet \
  --output_dir data/runs/wren_pattern_clusters_v1 \
  --representatives_per_cluster 8 \
  --representative_max_per_recording 2 \
  --force
```

## Notes on Embeddings

Default embedding backend is **BirdNET embeddings** (`birdnet`) when available. This clusters on BirdNET model feature vectors (not BirdNET prediction scores).

If BirdNET embedding extraction fails, pipeline can fallback to mel embeddings (`--embedding_fallback_backend mel_time` by default). To disable fallback and fail hard, use:

```bash
--embedding_fallback_backend none
```

You can manually switch to mel embeddings:

```bash
--embedding_backend mel_stats
```

BirdNET is currently used for species predictions. A later slice will add BirdNET-native embeddings (or another documented embedding backend) while keeping the same downstream clustering/report pipeline.
