# Repo Zones

This repo currently contains both:

1. **offline training / review code**
2. **future live runtime code or runtime-adjacent code**

The most useful simplification is to think about it in those two zones instead of
as one big blob.

## Zone 1: Offline Training And Review

This is where we:

- gather and curate recordings
- run BirdNET species checks
- build pattern-layer units
- cluster within species
- review clusters by ear
- compare clips to semantic references
- train lightweight adapters
- export cleaned prototypes / semantic libraries

These files are mainly offline:

- [src/pipeline.py](/Users/esmesturgess/Bird%20Translation%20AI/src/pipeline.py)
- [src/pattern_layer.py](/Users/esmesturgess/Bird%20Translation%20AI/src/pattern_layer.py)
- [src/pattern_cluster.py](/Users/esmesturgess/Bird%20Translation%20AI/src/pattern_cluster.py)
- [src/cluster.py](/Users/esmesturgess/Bird%20Translation%20AI/src/cluster.py)
- [src/light_cluster_review.py](/Users/esmesturgess/Bird%20Translation%20AI/src/light_cluster_review.py)
- [src/report.py](/Users/esmesturgess/Bird%20Translation%20AI/src/report.py)
- [src/assign_anchor_labels.py](/Users/esmesturgess/Bird%20Translation%20AI/src/assign_anchor_labels.py)
- [src/anchor_match.py](/Users/esmesturgess/Bird%20Translation%20AI/src/anchor_match.py)
- [src/build_anchor_manifest.py](/Users/esmesturgess/Bird%20Translation%20AI/src/build_anchor_manifest.py)
- [src/curate_manifest.py](/Users/esmesturgess/Bird%20Translation%20AI/src/curate_manifest.py)
- [src/curate_reference_review_set.py](/Users/esmesturgess/Bird%20Translation%20AI/src/curate_reference_review_set.py)
- [src/fetch_xeno_canto.py](/Users/esmesturgess/Bird%20Translation%20AI/src/fetch_xeno_canto.py)
- [src/review_pairs.py](/Users/esmesturgess/Bird%20Translation%20AI/src/review_pairs.py)
- [src/train_adapter.py](/Users/esmesturgess/Bird%20Translation%20AI/src/train_adapter.py)
- [src/transform_embeddings.py](/Users/esmesturgess/Bird%20Translation%20AI/src/transform_embeddings.py)
- [src/apply_manual_anchor_overrides.py](/Users/esmesturgess/Bird%20Translation%20AI/src/apply_manual_anchor_overrides.py)
- [src/cluster_crosswalk.py](/Users/esmesturgess/Bird%20Translation%20AI/src/cluster_crosswalk.py)
- [src/cluster_validity.py](/Users/esmesturgess/Bird%20Translation%20AI/src/cluster_validity.py)
- [src/export_semantic_library.py](/Users/esmesturgess/Bird%20Translation%20AI/src/export_semantic_library.py)

## Zone 2: Shared Core That A Live Runtime Will Reuse

These modules are the closest thing to the future Raspberry Pi core.

- [src/birdnet_infer.py](/Users/esmesturgess/Bird%20Translation%20AI/src/birdnet_infer.py)
- [src/embed.py](/Users/esmesturgess/Bird%20Translation%20AI/src/embed.py)
- [src/preprocess.py](/Users/esmesturgess/Bird%20Translation%20AI/src/preprocess.py)
- [src/segment.py](/Users/esmesturgess/Bird%20Translation%20AI/src/segment.py)
- [src/features.py](/Users/esmesturgess/Bird%20Translation%20AI/src/features.py)
- [src/utils.py](/Users/esmesturgess/Bird%20Translation%20AI/src/utils.py)
- [src/prototypes.py](/Users/esmesturgess/Bird%20Translation%20AI/src/prototypes.py)
- [src/prototype_assign.py](/Users/esmesturgess/Bird%20Translation%20AI/src/prototype_assign.py)
- [src/adapter_model.py](/Users/esmesturgess/Bird%20Translation%20AI/src/adapter_model.py)

Important note:

Even here, the current code is still more research-oriented than true deployment code.
The eventual Pi runtime should be a **smaller subset** built from these shared pieces.

## What Should Eventually Go On The Raspberry Pi

Not the whole repo.

The Raspberry Pi runtime should eventually only need:

1. audio capture / windowing
2. BirdNET species inference
3. embedding extraction
4. optional adapter transform
5. nearest prototype or nearest semantic-family assignment
6. confidence thresholding
7. music triggering

That means the Pi side should become a much smaller runtime package, conceptually like:

- `live_audio.py`
- `live_species.py`
- `live_embed.py`
- `live_assign.py`
- `live_music.py`

Those files do not exist yet as a clean runtime layer.
Right now the repo is still mostly an offline research/training repo.

## Why This Split Matters

Too many files do **not** automatically make the Pi slower.

What makes the Pi slow is:

- which code actually runs
- which models are loaded
- how much audio is processed
- how many embeddings / comparisons we do per second

So the file count is mainly a **clarity** problem, not a direct speed problem.

The right simplification is:

- keep the offline zone for training and review
- keep the live zone tiny and explicit

## Generalization Note

Blackbird should not be the only species feeding semantic learning.

If we want rough semantic-family generalization for unseen species later, we need
examples from **multiple species**. Blackbird is just the clearest starting case.

The right medium-term direction is:

- `Blackbird` as the clearest semantic reference species
- `Wren` as a hard edge case
- `Wood pigeon` as a simpler contrasting vocal style
- then more species as the shared semantic layer matures
