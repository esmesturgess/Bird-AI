#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${XC_API_KEY:-}" ]]; then
  echo "XC_API_KEY is not set. Set it before running this script." >&2
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVE_ROOT="/Users/esmesturgess/Library/CloudStorage/GoogleDrive-esmesturgessdurden@gmail.com/My Drive/BirdTranslationAI"

RAW_DIR="${DRIVE_ROOT}/raw_xeno_canto_blue_tit_v1"
SPECIES_RUN_DIR="${DRIVE_ROOT}/runs/species_decision/blue_tit_species_aug_uk_birdnet_windows_v1"
PATTERN_RUN_DIR="${DRIVE_ROOT}/runs/pattern_layers/blue_tit_pattern_layer_phrasepool_birdnet_windows_v1"
CLUSTER_RUN_DIR="${DRIVE_ROOT}/runs/within_species_clusters/01_phrase_pool_cluster_runs/blue_tit_pattern_clusters_phrasepool_birdnet_windows_v1"

cd "${PROJECT_DIR}"

echo "Fetching Blue tit recordings from xeno-canto into Google Drive..."
./.venv/bin/python -m src.fetch_xeno_canto \
  --species_csv config/blue_tit_species.csv \
  --output_dir "${RAW_DIR}" \
  --country "United Kingdom" \
  --min_quality C \
  --min_length_seconds 1.0 \
  --max_length_seconds 90.0 \
  --max_recordings_per_species 80 \
  --max_per_recordist 20 \
  --prefer_shorter \
  --per_page 100 \
  --sleep_seconds 0.5 \
  --skip_existing_dir "${RAW_DIR}"

echo "Running BirdNET species-decision stage..."
./.venv/bin/python -m src.pipeline \
  --species_config config/v1_species.yaml \
  --train_manifest "${RAW_DIR}/xc_recordings_manifest.csv" \
  --output_dir "${SPECIES_RUN_DIR}" \
  --birdnet_binary ./.venv/bin/birdnet-analyze \
  --birdnet_embeddings_binary ./.venv/bin/birdnet-embeddings \
  --species_frontend birdnet_windows \
  --birdnet_window_overlap 1.5 \
  --birdnet_window_merge_consecutive 0 \
  --stop_after_species

echo "Building phrase-pooled BirdNET embeddings for Blue tit..."
./.venv/bin/python -m src.pattern_layer \
  --species_run_dir "${SPECIES_RUN_DIR}" \
  --output_dir "${PATTERN_RUN_DIR}" \
  --target_species_bucket "Cyanistes caeruleus" \
  --analysis_unit phrase_pool \
  --embedding_clip_source event \
  --embedding_backend birdnet \
  --embedding_fallback_backend none \
  --phrase_gap_seconds 0.8 \
  --phrase_max_duration_seconds 6.0 \
  --phrase_padding_seconds 0.4 \
  --phrase_pooling duration_confidence

echo "Clustering phrase-pooled Blue tit embeddings with the baseline HDBSCAN settings..."
./.venv/bin/python -m src.pattern_cluster \
  --pattern_run_dir "${PATTERN_RUN_DIR}" \
  --output_dir "${CLUSTER_RUN_DIR}" \
  --min_cluster_size 6 \
  --min_samples 3 \
  --representatives_per_cluster 8 \
  --representative_max_per_recording 2

echo "Blue tit baseline complete."
echo "Review bundle: ${CLUSTER_RUN_DIR}/review_bundle"
echo "HTML report: ${CLUSTER_RUN_DIR}/reports/index.html"
