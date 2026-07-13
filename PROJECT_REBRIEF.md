# Bird Translation AI Rebrief

## What This Project Is

This repo is a non-commercial art/research prototype for turning bird vocalizations into musical responses.

The practical goal is:

1. hear a bird sound
2. identify the species
3. group the sound into a repeatable within-species pattern
4. optionally align that pattern with a human-readable call family like `song`, `call`, `alarm`
5. trigger music from that result

The intended long-term deployment path is:

- train and review offline on a larger machine
- export small artifacts
- run live inference on a Raspberry Pi-class device

## Where We Are Now

The repo has moved beyond the original simple offline clustering baseline.

The current working structure is:

1. **Species stage**
   - BirdNET is used as the frozen species front-end.
   - This is the part that decides things like `Troglodytes troglodytes` or `Turdus merula`.

2. **Pattern layer**
   - Accepted species detections are converted into within-species units.
   - These can be raw accepted events or phrase-pooled units built from adjacent events.

3. **Pattern clustering**
   - Embeddings are clustered offline to discover repeatable within-species patterns.
   - This is still mainly HDBSCAN-based.

4. **Human review / weak supervision**
   - We listen to clusters and say which ones are similar.
   - We also gather reference clips such as `song`, `call`, `alarm`, `juvenile_begging`.

5. **Semantic assignment**
   - A clustered or unclustered phrase can be assigned to the nearest semantic anchor family.

6. **Prototype / library export**
   - The eventual deployment target is not “recluster live”.
   - It is “compare a new clip to stored per-species prototypes and return a label or `unknown`”.

## Current State Of The Species

### Blackbird

Blackbird is the clearest semantic example so far.

It has been tested with:

- discovered clusters
- human merge judgments
- reference-based semantic labeling
- unseen Xeno-canto holdout recordings

The strongest current semantic buckets are:

- `song`
- `call`
- `alarm`
- `juvenile_begging`

Blackbird is the best species for understanding the full current pipeline.
It should be treated as the clearest semantic starting point, not as the only
species we eventually train semantic structure from.

### Wren

Wren is harder.

It has revealed two important truths:

1. the **frontend acceptance step** matters a lot
2. `call`, `alarm`, and `juvenile` can blur together more than we hoped

Recent work improved Wren by:

- adding a larger reference set
- allowing target-species top-k fallback acceptance in the pattern layer

That improved coverage, but Wren is still not as semantically stable as Blackbird.

### Wood Pigeon / House Sparrow / Blue Tit

These species have been used to stress-test the general workflow, especially clustering quality.
They are useful for comparison, but they are not as mature as Blackbird in the semantic layer.

## Generalization Direction

If the long-term aim is rough semantic understanding for unseen species, we should
not train that idea from Blackbird alone.

The sensible direction is:

- use `Blackbird` as the clearest semantic reference case
- use `Wren` as a hard semantic ambiguity case
- use `Wood pigeon` as a contrasting vocal style
- add more species once the shared semantic layer becomes clearer

So the plan is not:

- "one perfect blackbird semantic model"

It is:

- "a shared semantic layer learned from several species, starting with the ones we understand best"

## The Actual Architecture

The current system is best understood as:

`audio -> BirdNET species detection -> accepted species units -> embeddings -> pattern grouping -> semantic interpretation -> future music trigger`

Not this:

`audio -> one giant end-to-end custom “animal translator” model`

That distinction matters.

BirdNET is the pretrained acoustic front-end.
This project learns structure **on top of BirdNET**, not instead of BirdNET.

## The Three Most Important Code Paths

If you want to understand the repo quickly, read in this order.

### 1. Batch Species + Clustering Baseline

- [src/pipeline.py](/Users/esmesturgess/Bird%20Translation%20AI/src/pipeline.py)

This is the original all-in-one batch pipeline.
It still matters because it explains the repo’s artifact layout and the basic stages:

- segmentation
- BirdNET inference
- feature extraction
- embeddings
- clustering
- reporting

### 2. Species Run -> Pattern Layer

- [src/pattern_layer.py](/Users/esmesturgess/Bird%20Translation%20AI/src/pattern_layer.py)

This is the most important bridge into the newer workflow.
It takes a completed species run and creates the within-species units we actually cluster and review later.

This file is where to look if you want to understand:

- what counts as an accepted species event
- how phrase pooling works
- how event clips become phrase-level embeddings

### 3. Semantic Anchor Assignment

- [src/assign_anchor_labels.py](/Users/esmesturgess/Bird%20Translation%20AI/src/assign_anchor_labels.py)

This is the current semantic layer.
It answers:

- how do we compare pattern clips to reference clips?
- how do we assign labels like `song` or `alarm`?
- when do we smooth labels at the cluster level?

## The Main Supporting Modules

- [src/birdnet_infer.py](/Users/esmesturgess/Bird%20Translation%20AI/src/birdnet_infer.py)
  BirdNET wrapper logic

- [src/embed.py](/Users/esmesturgess/Bird%20Translation%20AI/src/embed.py)
  embedding backends, especially BirdNET embedding extraction

- [src/cluster.py](/Users/esmesturgess/Bird%20Translation%20AI/src/cluster.py)
  clustering helpers and representative selection

- [src/light_cluster_review.py](/Users/esmesturgess/Bird%20Translation%20AI/src/light_cluster_review.py)
  fast review bundle generation

- [src/transform_embeddings.py](/Users/esmesturgess/Bird%20Translation%20AI/src/transform_embeddings.py)
  pushes BirdNET embeddings through the learned adapter

- [src/train_adapter.py](/Users/esmesturgess/Bird%20Translation%20AI/src/train_adapter.py)
  trains the small metric-learning adapter on reviewed same/different judgments

- [src/export_semantic_library.py](/Users/esmesturgess/Bird%20Translation%20AI/src/export_semantic_library.py)
  exports the smaller semantic library that is closer to what we’d eventually deploy

## What Is Stable vs Experimental

### Stable enough to treat as core

- BirdNET as the species front-end
- pattern-layer artifact creation
- offline clustering and review bundle export
- semantic anchors as a useful interpretation layer

### Still clearly experimental

- cross-species generalization of semantic call families
- Wren `alarm` vs `juvenile_begging`
- how much we should merge `call` and `alarm` for some species
- how much of the semantic layer should be cluster-based vs direct prototype-based

## The Most Useful Mental Model

There are really two loops in this repo.

### Offline training / review loop

1. collect recordings
2. run BirdNET species stage
3. build pattern units
4. cluster and review
5. compare to semantic references
6. export cleaner prototypes

### Future live loop

1. mic hears a sound
2. BirdNET proposes a species
3. pattern embedding is computed
4. nearest reviewed prototype / semantic family is chosen
5. if confidence is low, return `unknown`
6. trigger music

That future live loop is the reason the repo has moved away from “just inspect clusters” and toward semantic libraries and prototypes.

## Recommended Reading Order

If you want to re-understand the code with the least pain:

1. read this file
2. read [REPO_ZONES.md](/Users/esmesturgess/Bird%20Translation%20AI/REPO_ZONES.md)
3. read [README.md](/Users/esmesturgess/Bird%20Translation%20AI/README.md) only for setup and original repo scope
4. read [src/pattern_layer.py](/Users/esmesturgess/Bird%20Translation%20AI/src/pattern_layer.py)
5. read [src/assign_anchor_labels.py](/Users/esmesturgess/Bird%20Translation%20AI/src/assign_anchor_labels.py)
6. then read [src/pipeline.py](/Users/esmesturgess/Bird%20Translation%20AI/src/pipeline.py) as the legacy “full batch” backbone

## Short Honest Summary

This repo is now best described as:

- **BirdNET-powered species detection**
- **within-species pattern discovery**
- **human-in-the-loop semantic interpretation**
- **eventual prototype-based live inference for an art installation**

It is no longer just a generic unsupervised clustering experiment, and it is not yet a finished live installation stack.
