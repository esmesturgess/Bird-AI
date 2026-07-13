# Run This Project In Colab (Keep Developing Locally)

This setup keeps your local machine as the development environment and uses Colab only for runtime/compute.

## 1. Build a source bundle locally

From the project root:

```bash
chmod +x colab/make_bundle.sh
./colab/make_bundle.sh
```

Optional: include your current `data/raw` audio files in the upload bundle:

```bash
./colab/make_bundle.sh --with-raw
```

Bundle output:

```text
colab/dist/bird_translation_ai_bundle.zip
```

## 2. Open the notebook in Colab

Use [Bird_Translation_AI_Colab.ipynb](Bird_Translation_AI_Colab.ipynb).

Inside the notebook:

1. Set `CONFIG["source_mode"] = "upload_zip"` (or `"github"` once your repo is on GitHub).
2. Mount Google Drive.
3. Upload `colab/dist/bird_translation_ai_bundle.zip` when prompted.
4. Install dependencies and run the pipeline cell.

## 3. Where inputs and outputs live

In the notebook config:

- Inputs: `drive_root/input_subdir` (default `/content/drive/MyDrive/BirdTranslationAI/raw`)
- Outputs: `drive_root/runs/run_name` (default `/content/drive/MyDrive/BirdTranslationAI/runs/run_01`)

This avoids filling Colab runtime storage and keeps outputs persisted in Drive.

## 4. Typical workflow

1. Edit code locally in this repo.
2. Rebuild bundle (`./colab/make_bundle.sh`).
3. Re-run notebook in Colab and upload new bundle.
4. Review outputs from Google Drive.
