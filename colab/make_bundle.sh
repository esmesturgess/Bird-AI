#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/colab/dist"
BUNDLE_NAME="bird_translation_ai_bundle.zip"
INCLUDE_RAW=0

for arg in "$@"; do
  case "$arg" in
    --with-raw)
      INCLUDE_RAW=1
      ;;
    *)
      BUNDLE_NAME="$arg"
      ;;
  esac
done

TMP_DIR="$(mktemp -d)"
STAGE_DIR="$TMP_DIR/project"
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$STAGE_DIR"

rsync -a \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude ".DS_Store" \
  "$ROOT_DIR/src" "$STAGE_DIR/"

cp "$ROOT_DIR/requirements.txt" "$STAGE_DIR/requirements.txt"
cp "$ROOT_DIR/README.md" "$STAGE_DIR/README.md"

mkdir -p "$STAGE_DIR/data/raw"
if [[ "$INCLUDE_RAW" -eq 1 && -d "$ROOT_DIR/data/raw" ]]; then
  rsync -a --exclude ".DS_Store" "$ROOT_DIR/data/raw/" "$STAGE_DIR/data/raw/"
fi

cat > "$STAGE_DIR/COLAB_BUNDLE_NOTE.txt" <<'EOF'
This archive was created for Colab runtime execution.
Put audio files in data/raw (or set --input_dir in notebook config).
EOF

mkdir -p "$DIST_DIR"
(
  cd "$STAGE_DIR"
  zip -rq "$DIST_DIR/$BUNDLE_NAME" .
)

echo "Created bundle: $DIST_DIR/$BUNDLE_NAME"
if [[ "$INCLUDE_RAW" -eq 1 ]]; then
  echo "Included data/raw files."
else
  echo "Included empty data/raw directory."
fi
