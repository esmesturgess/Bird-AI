# Storage Policy

Persistent project outputs should be written to Google Drive, not the local repo.

Default Drive project root:

`/Users/esmesturgess/Library/CloudStorage/GoogleDrive-esmesturgessdurden@gmail.com/My Drive/BirdTranslationAI`

Current run locations:

- Species decision runs: `BirdTranslationAI/runs/species_decision`
- Within-species clustering runs: `BirdTranslationAI/runs/within_species_clusters`
- Downloaded source recordings: `My Drive/Bird_recordings`

The command-line tools now reject `--output_dir` values outside Google Drive by default. If Google Drive moves, set `BIRD_TRANSLATION_GOOGLE_DRIVE_ROOT` to the new Drive root. Only for an intentional tiny local scratch run, set `BIRD_TRANSLATION_ALLOW_LOCAL_OUTPUT=1`.
