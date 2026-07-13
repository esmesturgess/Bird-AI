## Review Artifacts

These files capture manual listening judgments about which older machine-generated
clusters sounded similar enough to merge.

Important:

- These labels refer to the archived cluster runs listed in each CSV row.
- They should not be assumed to apply directly to newer phrase-pooled runs that
  use different cluster IDs.
- They are intended as seed supervision for a later adapter / metric-learning
  stage, or as a human merge reference when comparing old and new runs.

Files:

- `manual_cluster_group_map_v1.csv`
  - cluster-to-group assignments from manual listening
- `manual_cluster_pair_labels_v1.csv`
  - derived pairwise same/different/unsure labels between old cluster IDs

House sparrow note:

- Cluster `1` was marked as uncertain by the reviewer, so pair labels involving
  that cluster are marked `unsure` rather than forced into `same` or
  `different`.
