# Trained models

One directory per species. Each contains everything the installation needs to classify
that bird's vocalisations.

| file | what it is |
|---|---|
| `adapter.pt` | the projection head trained for this project (MLP 1792 → 512 → 128) |
| `prototypes.npy` | 4 × 128 class reference points, one per vocalisation type |
| `prototype_labels.json` | label order: `alarm, call, juvenile, song` |
| `adapter_meta.json` | provenance — training data, hyperparameters, class counts |

## Input format — important

These heads take **1792 dimensions**, not 1024. That is
`L2(AVES2 768-d) ‖ L2(BirdNET 1024-d)` — each encoder normalised separately, then
concatenated in that order. Feeding BirdNET alone will load without error and produce
confident nonsense. `src/classify_clip.py` builds the input correctly; use it rather than
assembling features by hand.

## Honest accuracy

Nested cross-validation, grouped by recording so no recording appears in both training and
test. These are per-recording figures.

| | overall | song | call | alarm | juvenile |
|---|---|---|---|---|---|
| blackbird | 0.809 | 0.966 | 0.822 | 0.684 | 0.690 |
| robin | 0.799 | 0.987 | 0.696 | 0.749 | 0.493 |
| tawny_owl | 0.800 | 0.865 | 0.788 | 0.418 | 0.908 |

Chance is ~0.25 and a permutation null measured 0.307, so these are real. But note the
weak cells: Robin juvenile (0.493) and Tawny Owl alarm (0.418) are unreliable, both
because those categories had the fewest recordings.

The `train_set_accuracy_sanity_check` inside `adapter_meta.json` reads 0.99+. That is
memorisation of the training data, **not** accuracy — do not quote it.
