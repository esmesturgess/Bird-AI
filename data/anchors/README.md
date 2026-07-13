Blackbird anchor workflow

Purpose:
Use a small set of trusted reference clips to check whether a blackbird clustering run lines up with human-meaningful buckets like `song`, `call`, `alarm`, and `juvenile_begging`.

How to use:
1. Put the actual audio files somewhere you control, ideally in Google Drive.
2. Copy `blackbird_anchor_manifest_template.csv` and replace each `audio_path` with the real absolute path to the clip.
3. Run `src.anchor_match` against a blackbird clustering run.

Example:
`./.venv/bin/python -m src.anchor_match --pattern_run_dir "<pattern_run_dir>" --cluster_csv "<cluster_csv>" --anchors_manifest_csv data/anchors/blackbird_anchor_manifest_template.csv --output_dir "<drive_output_dir>" --top_k 5`

Important:
- URLs alone are not enough; the script needs local audio files.
- If the pattern run is already in adapter space, the script will apply the same adapter to the anchor clips before scoring.
- This does not change the clusters by itself. It tells us which clusters the reference clips are closest to, so we can judge whether the current clustering is biologically and musically sensible.
