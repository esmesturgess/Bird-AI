from __future__ import annotations

from pathlib import Path
from collections import Counter

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .utils import ensure_dir


def _cluster_single_set(
    embeddings: np.ndarray,
    out_prefix: Path,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    use_umap: bool = False,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = embeddings.shape[0]
    if n == 0:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )

    pca_path = out_prefix.with_name(f"{out_prefix.name}_pca.npy")
    umap_path = out_prefix.with_name(f"{out_prefix.name}_umap.npy")
    labels_path = out_prefix.with_name(f"{out_prefix.name}_labels.npy")
    probs_path = out_prefix.with_name(f"{out_prefix.name}_probs.npy")

    if pca_path.exists() and umap_path.exists() and labels_path.exists() and probs_path.exists() and not force:
        pca_emb = np.load(pca_path)
        umap_emb = np.load(umap_path)
        labels = np.load(labels_path)
        probs = np.load(probs_path)
        return labels.astype(np.int32), probs.astype(np.float32), pca_emb, umap_emb

    if n < 2:
        labels = np.full(n, -1, dtype=np.int32)
        probs = np.zeros(n, dtype=np.float32)
        pca_emb = np.zeros((n, 2), dtype=np.float32)
        umap_emb = np.zeros((n, 2), dtype=np.float32)
        if n == 1:
            pca_emb[0, 0] = 0.0
            umap_emb[0, 0] = 0.0
        np.save(pca_path, pca_emb.astype(np.float32))
        np.save(umap_path, umap_emb.astype(np.float32))
        np.save(labels_path, labels.astype(np.int32))
        np.save(probs_path, probs.astype(np.float32))
        return labels, probs, pca_emb, umap_emb

    pca_dims = min(50, embeddings.shape[1], n)
    pca = PCA(n_components=pca_dims, random_state=42)
    pca_emb = pca.fit_transform(embeddings)

    eff_min_cluster_size = min_cluster_size if min_cluster_size is not None else max(2, int(round(0.15 * n)))
    eff_min_cluster_size = int(max(2, min(eff_min_cluster_size, n)))
    eff_min_samples = min_samples if min_samples is not None else max(1, eff_min_cluster_size // 2)
    eff_min_samples = int(max(1, min(eff_min_samples, n)))

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=eff_min_cluster_size,
        min_samples=eff_min_samples,
        prediction_data=False,
    )
    labels = clusterer.fit_predict(pca_emb).astype(np.int32)
    probs = clusterer.probabilities_.astype(np.float32)

    # Small exploratory datasets can end up all-noise in HDBSCAN.
    # Fallback to a simple partition to keep iterative analysis moving.
    if n >= 6 and np.all(labels == -1):
        k = max(2, min(6, int(round(np.sqrt(n)))))
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(pca_emb).astype(np.int32)
        probs = np.ones(n, dtype=np.float32)

    if use_umap and n >= 3:
        try:
            reducer = umap.UMAP(n_components=2, random_state=42)
            umap_emb = reducer.fit_transform(pca_emb)
        except Exception:
            umap_emb = np.pad(pca_emb[:, :2], ((0, 0), (0, max(0, 2 - pca_emb.shape[1]))))
    else:
        umap_emb = np.pad(pca_emb[:, :2], ((0, 0), (0, max(0, 2 - pca_emb.shape[1]))))

    np.save(pca_path, pca_emb.astype(np.float32))
    np.save(umap_path, umap_emb.astype(np.float32))
    np.save(labels_path, labels.astype(np.int32))
    np.save(probs_path, probs.astype(np.float32))
    return labels, probs, pca_emb.astype(np.float32), umap_emb.astype(np.float32)


def _save_cluster_scatter(df: pd.DataFrame, umap_emb: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(10, 7))
    labels = df["cluster_id"].to_numpy()
    urgency = df["urgency_index"].to_numpy() if "urgency_index" in df else np.ones(len(df))
    urg_min = float(np.nanmin(urgency)) if len(urgency) else 0.0
    urg_max = float(np.nanmax(urgency)) if len(urgency) else 1.0
    urg_range = (urg_max - urg_min) + 1e-9
    sizes = 20 + 30 * (urgency - urg_min) / urg_range
    plt.scatter(umap_emb[:, 0], umap_emb[:, 1], c=labels, s=sizes, cmap="tab20", alpha=0.8)
    plt.title("UMAP of Event Embeddings (Color=Cluster, Size=Urgency)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def run_dual_mode_clustering(
    features_df: pd.DataFrame,
    embeddings: np.ndarray,
    embeddings_dir: Path,
    clusters_dir: Path,
    min_species_samples: int = 20,
    hdbscan_min_cluster_size: int | None = None,
    hdbscan_min_samples: int | None = None,
    use_umap: bool = False,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    ensure_dir(embeddings_dir)
    ensure_dir(clusters_dir)
    n = len(features_df)
    if n != embeddings.shape[0]:
        raise ValueError("Embeddings row count must match features row count.")

    global_labels, global_probs, global_pca, global_umap = _cluster_single_set(
        embeddings=embeddings,
        out_prefix=embeddings_dir / "global",
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        use_umap=use_umap,
        force=force,
    )
    global_df = features_df.copy()
    global_df["cluster_id"] = global_labels.astype(int)
    global_df["cluster_prob"] = global_probs.astype(float)
    _save_cluster_scatter(global_df, global_umap, embeddings_dir / "umap_clusters_global.png")

    global_records = pd.DataFrame(
        {
            "clip_id": features_df["clip_id"].to_numpy(),
            "species_bucket": features_df["species_bucket"].to_numpy(),
            "mode": "global",
            "cluster_id": global_labels.astype(int),
            "cluster_prob": global_probs.astype(float),
        }
    )

    species_rows: list[pd.DataFrame] = []
    species_values = sorted(str(s) for s in features_df["species_bucket"].fillna("unknown").unique())
    for species in species_values:
        mask = features_df["species_bucket"].fillna("unknown").astype(str) == species
        idx = np.where(mask.to_numpy())[0]
        if len(idx) < min_species_samples:
            skipped = pd.DataFrame(
                {
                    "clip_id": features_df.iloc[idx]["clip_id"].to_numpy(),
                    "species_bucket": species,
                    "mode": "species_conditioned",
                    "cluster_id": np.full(len(idx), -1, dtype=int),
                    "cluster_prob": np.zeros(len(idx), dtype=float),
                }
            )
            if len(skipped) > 0:
                species_rows.append(skipped)
            continue

        labels, probs, pca_emb, umap_emb = _cluster_single_set(
            embeddings=embeddings[idx],
            out_prefix=clusters_dir / f"species_{species}",
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            use_umap=use_umap,
            force=force,
        )
        subset = features_df.iloc[idx].copy()
        subset["cluster_id"] = labels.astype(int)
        subset["cluster_prob"] = probs.astype(float)
        _save_cluster_scatter(subset, umap_emb, clusters_dir / f"umap_species_{species}.png")
        species_rows.append(
            pd.DataFrame(
                {
                    "clip_id": subset["clip_id"].to_numpy(),
                    "species_bucket": species,
                    "mode": "species_conditioned",
                    "cluster_id": labels.astype(int),
                    "cluster_prob": probs.astype(float),
                }
            )
        )
        np.save((clusters_dir / f"species_{species}_pca.npy"), pca_emb.astype(np.float32))
        np.save((clusters_dir / f"species_{species}_umap.npy"), umap_emb.astype(np.float32))

    species_df = pd.concat(species_rows, ignore_index=True) if species_rows else pd.DataFrame(
        columns=["clip_id", "species_bucket", "mode", "cluster_id", "cluster_prob"]
    )
    master_clusters = pd.concat([global_records, species_df], ignore_index=True)
    master_clusters.to_csv(clusters_dir / "master_clusters.csv", index=False)
    return global_df, master_clusters, global_pca, global_umap


def representative_indices_by_cluster(
    df: pd.DataFrame,
    pca_emb: np.ndarray,
    cluster_col: str = "cluster_id",
    per_cluster: int = 10,
    diversity_col: str | None = None,
    max_per_diversity_value: int = 2,
) -> dict[int, list[int]]:
    reps: dict[int, list[int]] = {}
    labels = sorted(int(v) for v in np.unique(df[cluster_col].to_numpy()))
    for cluster_id in labels:
        idx = np.where(df[cluster_col].to_numpy() == cluster_id)[0]
        if len(idx) == 0:
            reps[cluster_id] = []
            continue
        cluster_vectors = pca_emb[idx]
        center = np.mean(cluster_vectors, axis=0, keepdims=True)
        dist = np.linalg.norm(cluster_vectors - center, axis=1)
        order = np.argsort(dist)
        ordered_idx = idx[order].tolist()

        if diversity_col and diversity_col in df.columns and max_per_diversity_value > 0:
            selected: list[int] = []
            selected_set: set[int] = set()
            diversity_counts: Counter[str] = Counter()
            diversity_values = df.iloc[ordered_idx][diversity_col].fillna("").astype(str).tolist()

            for round_limit in range(1, max_per_diversity_value + 1):
                for row_idx, diversity_value in zip(ordered_idx, diversity_values):
                    if row_idx in selected_set:
                        continue
                    key = diversity_value or "__missing__"
                    if diversity_counts[key] >= round_limit:
                        continue
                    selected.append(row_idx)
                    selected_set.add(row_idx)
                    diversity_counts[key] += 1
                    if len(selected) >= per_cluster:
                        break
                if len(selected) >= per_cluster:
                    break

            if len(selected) < per_cluster:
                for row_idx in ordered_idx:
                    if row_idx in selected_set:
                        continue
                    selected.append(row_idx)
                    selected_set.add(row_idx)
                    if len(selected) >= per_cluster:
                        break

            reps[cluster_id] = selected
            continue

        reps[cluster_id] = ordered_idx[:per_cluster]
    return reps
