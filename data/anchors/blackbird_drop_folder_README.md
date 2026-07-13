Put your blackbird reference clips in these subfolders:

- `song`
- `call`
- `alarm`
- `juvenile_begging`

You can drag in `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, or `.aac` files.

Then I can run:

`python -m src.build_anchor_manifest --audio_dir "<this folder>" --output_csv "<drive csv path>"`

and immediately convert them into an anchor manifest for:

`python -m src.anchor_match ...`

Notes:
- We are treating `subsong` as `song`.
- If a file is in the wrong folder, just move it; the manifest is rebuilt from the folder structure.
