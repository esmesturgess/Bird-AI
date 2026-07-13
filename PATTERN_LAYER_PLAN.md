# Pattern Layer Plan

## Where We Are Now

Current best Wren species-check recipe:

- segment bird-like events from each source recording
- keep at most `10` events per recording
- rank events with a mild duration-aware quality score
- build a real `3.0s` BirdNET inference clip from the original recording around each event
- use BirdNET only for species inference at this stage

Current preferred run:

- output dir: `data/runs/wren_species_full_cap10_context_soft_v1`
- key selector settings:
  - `--max_events_per_recording 10`
  - `--selection_pref_duration 0.45`
  - `--selection_score_min 0.03`
  - `--birdnet_context_seconds 3.0`

Why this is the preferred recipe:

- it keeps more clips than the overly strict selector
- it gives BirdNET real surrounding context instead of silence padding
- it produced a much cleaner Wren species stage than the earlier full-set runs

## What The Next Layer Actually Is

The next layer is not "emotion detection" first.

It should be:

1. `species -> within-species embedding -> cluster_id`
2. `cluster_id -> optional human interpretation`
3. `cluster_id -> music mapping`

That means our next ML output should be:

- `species`
- `cluster_id`
- `cluster_confidence`

Not:

- guaranteed biological labels like `alarm`, `mating`, or `aggressive`

Those labels can come later if the clusters line up clearly enough.

## Why We Are Not Rebuilding BirdNET's Two-Spectrogram Frontend

BirdNET already computes its own audio frontend internally when we ask it for:

- species predictions
- embeddings

So if we use BirdNET embeddings, we are already benefiting from BirdNET's:

- `48 kHz` audio handling
- dual spectrogram frontend
- trained neural representation

We would only need to manually recreate BirdNET's frontend if we wanted to train our own BirdNET-like encoder from scratch.

For this project, that is not the right next step.

## Are We Training An AI?

Yes, but not in the "train a big neural net from scratch" sense.

At the next layer we will be learning:

- within-species clusters in BirdNET embedding space
- cluster thresholds / prototype assignment rules
- later, optional human-readable cluster tags

So the practical setup is:

- pretrained BirdNET = frozen encoder / species front-end
- our project = learns structure on top of BirdNET embeddings

That still counts as machine learning.

## Exact Next-Layer Plan

### Phase A. Freeze The Species Front-End

Goal:

- treat the current Wren species recipe as the stable front-end for clustering experiments

Tasks:

1. Keep `data/runs/wren_species_full_cap10_context_soft_v1` as the reference species-check run.
2. Use only events accepted as Wren for the first clustering pass.
3. Keep `unknown` and non-Wren events out of the Wren clustering training set.

Done when:

- we have a clean Wren-only event table for clustering.

### Phase B. Generate Wren Embeddings

Goal:

- extract BirdNET embeddings for the accepted Wren clips

Tasks:

1. Take the accepted Wren events from the reference species run.
2. Generate BirdNET embeddings for those clips.
3. Save a single table with:
   - `clip_id`
   - `source_file`
   - `start_s`
   - `end_s`
   - `duration_s`
   - `species_top1_conf`
   - embedding vector / embedding file reference

Done when:

- we can inspect a Wren-only embedding dataset.

### Phase C. Cluster Wren Patterns

Goal:

- discover repeatable Wren sound groups

Tasks:

1. Start with HDBSCAN on BirdNET embeddings.
2. Treat noise / outlier points as valid "unclustered" outputs.
3. Save:
   - `cluster_id`
   - membership strength if available
   - exemplar clips per cluster
4. Make a listening report for each cluster.

Done when:

- we can listen to example clips from each Wren cluster and tell whether the grouping feels coherent.

### Phase D. Evaluate Cluster Quality

Goal:

- decide whether the discovered clusters are stable enough to use

Tasks:

1. Check cluster size distribution.
2. Check how many clips become HDBSCAN noise.
3. Listen to top exemplars from each cluster.
4. Compare clusters against interpretable features:
   - duration
   - repetition rate
   - pitch stability
   - tonality proxy
5. Only keep the clustering recipe if the clusters are both:
   - acoustically coherent
   - useful enough for music mapping

Done when:

- we can say "these 3-6 Wren clusters seem real enough to use."

### Phase E. Add Inference-Time Cluster Assignment

Goal:

- turn offline clusters into live-time assignment artifacts

Tasks:

1. Export one or more prototypes / exemplars per cluster.
2. Export distance thresholds per cluster.
3. Add a nearest-prototype assignment step for live inference.
4. Return `unknown_cluster` when no cluster is close enough.

Done when:

- a new accepted Wren event can be assigned to a stable Wren cluster without rerunning HDBSCAN live.

### Phase F. Human Meaning And Music

Goal:

- translate clusters into something useful for your artwork

Tasks:

1. Listen to each cluster's exemplars.
2. Decide whether the cluster sounds like:
   - song phrase
   - chatter/contact
   - sharp alarm-like burst
   - trill/repetition pattern
   - or just "pattern A/B/C"
3. Map each cluster to music behavior.

Important rule:

- do not force biological labels unless the listening evidence is genuinely convincing.

Done when:

- `species + cluster_id` can trigger a distinct musical response.

## What We Should Not Do Yet

- train our own deep species model
- rebuild BirdNET's internal dual-spectrogram encoder ourselves
- claim we are detecting true bird emotion directly
- use k-means as the default just because it is familiar
- jump to two-species clustering before Wren-only clustering looks coherent

## Immediate Next Implementation Order

1. Create a Wren-only accepted-events manifest from the reference species run.
2. Generate BirdNET embeddings for those accepted Wren clips.
3. Run HDBSCAN on those embeddings.
4. Export cluster exemplars and a listening report.
5. Review the clusters before touching music mapping.
