from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template

from .utils import ensure_dir, zscore


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Bird Vocalization Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; background: #f7f9fc; color: #222; }
    .card { background: white; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 1px 6px rgba(0,0,0,0.08); }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 1rem; }
    .plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 1rem; }
    img { width: 100%; border-radius: 8px; }
    table { border-collapse: collapse; width: 100%; margin-top: 0.5rem; }
    th, td { border-bottom: 1px solid #eee; text-align: left; padding: 0.25rem; font-size: 0.9rem; }
    .muted { color: #5b6775; }
    h2 { margin-top: 2rem; }
  </style>
</head>
<body>
  <h1>Bird Vocalization Clustering Report</h1>
  <p>Total events: {{ total_events }} | Number of clusters (excluding noise): {{ n_clusters }}</p>
  <div class="card">
    <h3>Global Plots</h3>
    <img src="{{ umap_image }}" alt="UMAP clusters"/>
    {% for plot in dist_plots %}
    <img src="{{ plot }}" alt="distribution plot"/>
    {% endfor %}
  </div>
  {% if analysis_plots %}
  <div class="card">
    <h3>Analysis Views</h3>
    <p class="muted">These help us see whether the clusters are genuinely different vocal patterns or just artifacts of a few recordings.</p>
    <div class="plot-grid">
      {% for plot in analysis_plots %}
      <div>
        <img src="{{ plot["path"] }}" alt="{{ plot["title"] }}"/>
        <p><strong>{{ plot["title"] }}</strong><br>{{ plot["description"] }}</p>
      </div>
      {% endfor %}
    </div>
  </div>
  {% endif %}
  {% if cluster_overview_rows %}
  <div class="card">
    <h3>Cluster Overview</h3>
    <table>
      <thead>
        <tr>
          <th>Cluster</th>
          <th>Events</th>
          <th>Recordings</th>
          <th>Dominant Recording Share</th>
          <th>Mean Duration (s)</th>
          <th>Mean BirdNET Conf</th>
        </tr>
      </thead>
      <tbody>
        {% for row in cluster_overview_rows %}
        <tr>
          <td>{{ row["cluster_id"] }}</td>
          <td>{{ row["n_events"] }}</td>
          <td>{{ row["n_recordings"] }}</td>
          <td>{{ "%.2f"|format(row["dominant_recording_share"]) }}</td>
          <td>{{ "%.3f"|format(row["mean_duration_s"]) }}</td>
          <td>{{ "%.3f"|format(row["mean_species_top1_conf"]) }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% endif %}
  {% if review_sheet_paths %}
  <div class="card">
    <h3>Manual Review Files</h3>
    <p class="muted">Use these to decide whether each cluster sounds coherent and how you want to map it later.</p>
    <ul>
      {% for item in review_sheet_paths %}
      <li><strong>{{ item["label"] }}</strong>: {{ item["path"] }}</li>
      {% endfor %}
    </ul>
  </div>
  {% endif %}
  {% for cluster in clusters %}
  <h2>Cluster {{ cluster["cluster_id"] }} (n={{ cluster["size"] }})</h2>
  <p class="muted">
    Recordings represented: {{ cluster["n_recordings"] }} |
    Dominant recording share: {{ "%.2f"|format(cluster["dominant_recording_share"]) }} |
    Mean BirdNET conf: {{ "%.3f"|format(cluster["mean_species_top1_conf"]) }}
  </p>
  {% if cluster["mean_spec_rel"] %}
  <div class="card">
    <h4>Mean Mel Spectrogram</h4>
    <img src="{{ cluster["mean_spec_rel"] }}" alt="cluster mean mel spectrogram"/>
  </div>
  {% endif %}
  <div class="grid">
    {% for item in cluster["items"] %}
    <div class="card">
      <h4>{{ item["clip_id"] }}</h4>
      <audio controls src="{{ item["audio_rel"] }}"></audio>
      <img src="{{ item["spec_rel"] }}" alt="spectrogram"/>
      <table>
        <tr><th>Urgency</th><td>{{ "%.3f"|format(item["urgency_index"]) }}</td></tr>
        <tr><th>Song-likeness</th><td>{{ "%.3f"|format(item["song_likeness"]) }}</td></tr>
        <tr><th>RMS</th><td>{{ "%.4f"|format(item["rms"]) }}</td></tr>
        <tr><th>Bandwidth</th><td>{{ "%.1f"|format(item["spectral_bandwidth"]) }}</td></tr>
        <tr><th>Flatness</th><td>{{ "%.4f"|format(item["spectral_flatness"]) }}</td></tr>
        <tr><th>Repetition (Hz)</th><td>{{ "%.3f"|format(item["repetition_rate_hz"]) }}</td></tr>
      </table>
    </div>
    {% endfor %}
  </div>
  {% endfor %}
</body>
</html>
""".strip()


def _resample_time_axis(x: np.ndarray, target_frames: int) -> np.ndarray:
    if x.shape[1] == target_frames:
        return x
    if x.shape[1] <= 1:
        return np.repeat(x, repeats=target_frames, axis=1)
    src = np.linspace(0.0, 1.0, x.shape[1], endpoint=True)
    dst = np.linspace(0.0, 1.0, target_frames, endpoint=True)
    out = np.empty((x.shape[0], target_frames), dtype=np.float32)
    for band in range(x.shape[0]):
        out[band] = np.interp(dst, src, x[band]).astype(np.float32)
    return out


def create_cluster_mean_spectrograms(
    df: pd.DataFrame,
    specs_dir: Path,
    reports_dir: Path,
    target_frames: int = 96,
) -> dict[int, Path]:
    out_dir = reports_dir / "cluster_means"
    ensure_dir(out_dir)
    if "cluster_id" not in df or "spec_npy" not in df:
        return {}

    out_paths: dict[int, Path] = {}
    for cluster_id, group in df.groupby("cluster_id"):
        mats: list[np.ndarray] = []
        for spec_npy in group["spec_npy"].dropna().astype(str):
            spec_path = specs_dir / spec_npy
            if not spec_path.exists():
                continue
            try:
                mel_db = np.load(spec_path)
            except Exception:
                continue
            if mel_db.ndim != 2 or mel_db.shape[0] < 2 or mel_db.shape[1] < 1:
                continue
            mats.append(_resample_time_axis(mel_db.astype(np.float32), target_frames=target_frames))
        if not mats:
            continue
        mean_mel = np.mean(np.stack(mats, axis=0), axis=0).astype(np.float32)
        cid = int(cluster_id)
        np.save(out_dir / f"cluster_{cid}_mean_mel.npy", mean_mel)
        png_path = out_dir / f"cluster_{cid}_mean_mel.png"
        plt.figure(figsize=(6, 3))
        plt.imshow(mean_mel, origin="lower", aspect="auto", cmap="magma")
        plt.title(f"Cluster {cid}: Mean Mel Spectrogram")
        plt.xlabel("Normalized time")
        plt.ylabel("Mel bin")
        plt.colorbar(label="dB")
        plt.tight_layout()
        plt.savefig(png_path, dpi=120)
        plt.close()
        out_paths[cid] = png_path
    return out_paths


def create_distribution_plots(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    ensure_dir(out_dir)
    columns = [
        ("rms", "RMS Distribution"),
        ("spectral_bandwidth", "Bandwidth Distribution"),
        ("repetition_rate_hz", "Repetition Rate Distribution"),
        ("urgency_index", "Urgency Index Distribution"),
    ]
    paths: list[Path] = []
    for col, title in columns:
        if col not in df:
            continue
        plt.figure(figsize=(8, 4))
        plt.hist(df[col].dropna(), bins=40, color="#2a6f97", alpha=0.85)
        plt.title(title)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        out_path = out_dir / f"dist_{col}.png"
        plt.savefig(out_path, dpi=120)
        plt.close()
        paths.append(out_path)
    return paths


def _cluster_order(df: pd.DataFrame) -> list[int]:
    return sorted(int(v) for v in df["cluster_id"].dropna().astype(int).unique().tolist())


def build_cluster_overview_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    if "cluster_id" not in df:
        return []

    overview_rows: list[dict[str, object]] = []
    for cluster_id in _cluster_order(df):
        subset = df[df["cluster_id"] == cluster_id]
        if subset.empty:
            continue
        recording_counts = subset["recording_id"].astype(str).value_counts() if "recording_id" in subset else pd.Series(dtype=int)
        dominant_count = int(recording_counts.iloc[0]) if not recording_counts.empty else 0
        overview_rows.append(
            {
                "cluster_id": int(cluster_id),
                "n_events": int(len(subset)),
                "n_recordings": int(subset["recording_id"].nunique()) if "recording_id" in subset else 0,
                "dominant_recording_id": str(recording_counts.index[0]) if not recording_counts.empty else "",
                "dominant_recording_share": float(dominant_count / max(1, len(subset))),
                "mean_duration_s": float(subset["duration_s"].mean()) if "duration_s" in subset else 0.0,
                "mean_species_top1_conf": float(subset["species_top1_conf"].mean()) if "species_top1_conf" in subset else 0.0,
            }
        )
    return overview_rows


def create_cluster_recording_heatmap(
    df: pd.DataFrame,
    reports_dir: Path,
    max_recordings: int = 24,
) -> Path | None:
    if "cluster_id" not in df or "recording_id" not in df:
        return None

    counts = (
        df.groupby(["recording_id", "cluster_id"]).size().unstack(fill_value=0).sort_index(axis=1)
    )
    if counts.empty:
        return None

    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
    counts = counts.head(max_recordings)
    matrix = counts.to_numpy(dtype=float)

    plt.figure(figsize=(max(7, 1.2 * matrix.shape[1]), max(5, 0.45 * matrix.shape[0])))
    plt.imshow(matrix, aspect="auto", cmap="YlGnBu")
    plt.colorbar(label="Events from recording")
    plt.xticks(range(matrix.shape[1]), [str(c) for c in counts.columns.tolist()])
    plt.yticks(range(matrix.shape[0]), counts.index.tolist(), fontsize=8)
    plt.xlabel("Cluster ID")
    plt.ylabel("Recording ID")
    plt.title("Cluster vs Recording Count Heatmap")

    if matrix.shape[0] <= 24 and matrix.shape[1] <= 12:
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = int(matrix[row_idx, col_idx])
                if value > 0:
                    plt.text(col_idx, row_idx, str(value), ha="center", va="center", fontsize=8, color="#17324d")

    plt.tight_layout()
    out_path = reports_dir / "cluster_recording_heatmap.png"
    plt.savefig(out_path, dpi=140)
    plt.close()
    return out_path


def create_cluster_feature_heatmap(
    df: pd.DataFrame,
    reports_dir: Path,
) -> Path | None:
    feature_cols = [
        "duration_s",
        "species_top1_conf",
        "spectral_bandwidth",
        "spectral_flatness",
        "repetition_rate_hz",
        "pitch_stability",
        "song_likeness",
        "urgency_index",
    ]
    keep_cols = [col for col in feature_cols if col in df.columns]
    if "cluster_id" not in df or not keep_cols:
        return None

    mean_df = df.groupby("cluster_id")[keep_cols].mean().sort_index()
    if mean_df.empty:
        return None

    norm = mean_df.copy()
    for col in keep_cols:
        norm[col] = zscore(norm[col].to_numpy(dtype=float))

    matrix = norm.to_numpy(dtype=float)
    plt.figure(figsize=(max(8, 1.1 * matrix.shape[1]), max(4, 0.7 * matrix.shape[0])))
    plt.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    plt.colorbar(label="Cluster mean z-score")
    plt.xticks(range(matrix.shape[1]), keep_cols, rotation=30, ha="right")
    plt.yticks(range(matrix.shape[0]), [str(c) for c in mean_df.index.tolist()])
    plt.xlabel("Feature")
    plt.ylabel("Cluster ID")
    plt.title("Cluster Feature Profile Heatmap")

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            plt.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.1f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    out_path = reports_dir / "cluster_feature_heatmap.png"
    plt.savefig(out_path, dpi=140)
    plt.close()
    return out_path


def create_cluster_quality_plot(df: pd.DataFrame, reports_dir: Path) -> Path | None:
    if "cluster_id" not in df or "species_top1_conf" not in df or "duration_s" not in df:
        return None

    cluster_ids = _cluster_order(df)
    if not cluster_ids:
        return None

    conf_values = [df.loc[df["cluster_id"] == cluster_id, "species_top1_conf"].dropna().to_numpy() for cluster_id in cluster_ids]
    dur_values = [df.loc[df["cluster_id"] == cluster_id, "duration_s"].dropna().to_numpy() for cluster_id in cluster_ids]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].boxplot(conf_values, tick_labels=[str(cid) for cid in cluster_ids], patch_artist=True)
    axes[0].set_title("BirdNET Confidence by Cluster")
    axes[0].set_xlabel("Cluster ID")
    axes[0].set_ylabel("Top-1 confidence")
    axes[0].set_ylim(0.0, 1.05)

    axes[1].boxplot(dur_values, tick_labels=[str(cid) for cid in cluster_ids], patch_artist=True)
    axes[1].set_title("Original Event Duration by Cluster")
    axes[1].set_xlabel("Cluster ID")
    axes[1].set_ylabel("Duration (s)")

    plt.tight_layout()
    out_path = reports_dir / "cluster_quality_by_cluster.png"
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def create_cluster_source_balance_plot(df: pd.DataFrame, reports_dir: Path) -> Path | None:
    if "cluster_id" not in df or "recording_id" not in df:
        return None

    rows: list[dict[str, float | int]] = []
    for cluster_id in _cluster_order(df):
        subset = df[df["cluster_id"] == cluster_id]
        if subset.empty:
            continue
        counts = subset["recording_id"].astype(str).value_counts()
        dominant_share = float(counts.iloc[0] / len(subset)) if not counts.empty else 0.0
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "n_recordings": int(subset["recording_id"].nunique()),
                "dominant_share": dominant_share,
            }
        )

    if not rows:
        return None

    plot_df = pd.DataFrame(rows)
    x = np.arange(len(plot_df))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.bar(x - width / 2, plot_df["n_recordings"], width=width, color="#457b9d")
    ax1.set_ylabel("Distinct recordings")
    ax1.set_xlabel("Cluster ID")
    ax1.set_xticks(x, [str(v) for v in plot_df["cluster_id"].tolist()])
    ax1.set_title("Cluster Source Balance")

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, plot_df["dominant_share"], width=width, color="#e76f51")
    ax2.set_ylabel("Dominant recording share")
    ax2.set_ylim(0.0, 1.05)

    plt.tight_layout()
    out_path = reports_dir / "cluster_source_balance.png"
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def export_cluster_review_sheets(
    df: pd.DataFrame,
    representatives: dict[int, list[int]],
    reports_dir: Path,
    specs_dir: Path,
) -> dict[str, Path]:
    ensure_dir(reports_dir)
    overview_rows = build_cluster_overview_rows(df)

    cluster_rows: list[dict[str, object]] = []
    exemplar_rows: list[dict[str, object]] = []

    overview_by_cluster = {int(row["cluster_id"]): row for row in overview_rows}
    for cluster_id in _cluster_order(df):
        cluster_summary = overview_by_cluster.get(int(cluster_id), {})
        playlist_path = (reports_dir / "playlists" / f"cluster_{int(cluster_id)}.m3u").resolve()
        rep_indices = representatives.get(int(cluster_id), [])
        first_rep = df.iloc[rep_indices[0]] if rep_indices else None
        cluster_rows.append(
            {
                "cluster_id": int(cluster_id),
                "n_events": int(cluster_summary.get("n_events", 0)),
                "n_recordings": int(cluster_summary.get("n_recordings", 0)),
                "dominant_recording_id": str(cluster_summary.get("dominant_recording_id", "")),
                "dominant_recording_share": float(cluster_summary.get("dominant_recording_share", 0.0)),
                "mean_duration_s": float(cluster_summary.get("mean_duration_s", 0.0)),
                "mean_species_top1_conf": float(cluster_summary.get("mean_species_top1_conf", 0.0)),
                "playlist_path": playlist_path.as_posix(),
                "first_representative_clip": str(first_rep["clip_id"]) if first_rep is not None else "",
                "review_keep_for_v1": "",
                "review_cluster_name": "",
                "review_possible_bio_label": "",
                "review_musical_mood": "",
                "review_coherence_1to5": "",
                "review_recording_bias_1to5": "",
                "review_notes": "",
            }
        )

        for exemplar_rank, row_idx in enumerate(rep_indices, start=1):
            row = df.iloc[row_idx]
            exemplar_rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "exemplar_rank": int(exemplar_rank),
                    "clip_id": str(row["clip_id"]),
                    "recording_id": str(row.get("recording_id", "")),
                    "duration_s": float(row.get("duration_s", 0.0)),
                    "species_top1_conf": float(row.get("species_top1_conf", 0.0)),
                    "audio_path": str(row.get("embedding_audio_path", "")),
                    "spec_png_path": (specs_dir / str(row["spec_png"])).resolve().as_posix() if pd.notna(row.get("spec_png")) else "",
                    "review_same_pattern_yes_no": "",
                    "review_strength_1to5": "",
                    "review_possible_subtype": "",
                    "review_notes": "",
                }
            )

    cluster_sheet_path = reports_dir / "cluster_review_sheet.csv"
    exemplar_sheet_path = reports_dir / "representative_review_sheet.csv"
    pd.DataFrame(cluster_rows).to_csv(cluster_sheet_path, index=False)
    pd.DataFrame(exemplar_rows).to_csv(exemplar_sheet_path, index=False)

    guide_path = reports_dir / "review_guide.md"
    guide_text = (
        "# Cluster Review Guide\n\n"
        "Use `cluster_review_sheet.csv` to decide which clusters feel coherent enough to keep for V1.\n\n"
        "Suggested checks:\n"
        "1. Listen to the playlist for each cluster.\n"
        "2. Check whether the representatives sound like the same vocal pattern.\n"
        "3. Watch for recording bias: if almost every exemplar comes from one source recording, treat the cluster cautiously.\n"
        "4. Only assign a biological label if it feels genuinely defensible after listening.\n"
        "5. It is completely fine to keep a cluster as `pattern_A`, `pattern_B`, etc. for now.\n"
    )
    guide_path.write_text(guide_text, encoding="utf-8")

    return {
        "cluster_review_sheet": cluster_sheet_path,
        "representative_review_sheet": exemplar_sheet_path,
        "review_guide": guide_path,
    }


def export_review_bundle(
    df: pd.DataFrame,
    representatives: dict[int, list[int]],
    reports_dir: Path,
    specs_dir: Path,
) -> Path:
    bundle_dir = reports_dir.parent / "review_bundle"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    ensure_dir(bundle_dir)

    readme_path = bundle_dir / "README.md"
    readme_path.write_text(
        "# Review Bundle\n\n"
        "Each `cluster_*` folder contains representative audio clips (`.wav`) and matching spectrogram images (`.png`).\n"
        "The numeric prefix (`01`, `02`, etc.) is the exemplar rank within that cluster.\n",
        encoding="utf-8",
    )

    for cluster_id, idx_list in sorted(representatives.items(), key=lambda item: item[0]):
        cluster_dir = bundle_dir / f"cluster_{int(cluster_id)}"
        ensure_dir(cluster_dir)
        for exemplar_rank, row_idx in enumerate(idx_list, start=1):
            row = df.iloc[row_idx]
            clip_id = str(row["clip_id"])

            audio_src = Path(str(row.get("embedding_audio_path", "")))
            if audio_src.exists():
                audio_dst = cluster_dir / f"{exemplar_rank:02d}_{clip_id}.wav"
                shutil.copy2(audio_src, audio_dst)

            spec_name = str(row.get("spec_png", ""))
            if spec_name:
                spec_src = specs_dir / spec_name
                if spec_src.exists():
                    spec_dst = cluster_dir / f"{exemplar_rank:02d}_{clip_id}.png"
                    shutil.copy2(spec_src, spec_dst)

    return bundle_dir


def write_cluster_playlists(df: pd.DataFrame, events_dir: Path, reports_dir: Path) -> None:
    playlists_dir = reports_dir / "playlists"
    ensure_dir(playlists_dir)
    for cluster_id, group in df.groupby("cluster_id"):
        m3u = playlists_dir / f"cluster_{int(cluster_id)}.m3u"
        with m3u.open("w", encoding="utf-8") as f:
            f.write("#EXTM3U\n")
            for _, row in group.iterrows():
                f.write((events_dir / row["event_wav"]).resolve().as_posix() + "\n")


def generate_html_report(
    df: pd.DataFrame,
    representatives: dict[int, list[int]],
    reports_dir: Path,
    events_dir: Path,
    specs_dir: Path,
    umap_image: Path,
    dist_plots: list[Path],
    cluster_mean_specs: dict[int, Path] | None = None,
    analysis_plots: list[dict[str, str]] | None = None,
    review_sheet_paths: dict[str, Path] | None = None,
    cluster_overview_rows: list[dict[str, object]] | None = None,
) -> Path:
    ensure_dir(reports_dir)
    clusters_payload = []
    for cluster_id in sorted(representatives.keys()):
        idx_list = representatives[cluster_id]
        rows = df.iloc[idx_list] if idx_list else df.iloc[[]]
        mean_spec_rel = ""
        full_cluster = df[df["cluster_id"] == cluster_id]
        recording_counts = full_cluster["recording_id"].astype(str).value_counts() if "recording_id" in full_cluster else pd.Series(dtype=int)
        dominant_share = float(recording_counts.iloc[0] / max(1, len(full_cluster))) if not recording_counts.empty else 0.0
        if cluster_mean_specs:
            mean_path = cluster_mean_specs.get(int(cluster_id))
            if mean_path is not None:
                mean_spec_rel = mean_path.resolve().as_posix()
        items = []
        for _, row in rows.iterrows():
            audio_abs = (events_dir / row["event_wav"]).resolve()
            spec_abs = (specs_dir / row["spec_png"]).resolve()
            items.append(
                {
                    "clip_id": row["clip_id"],
                    "audio_rel": audio_abs.as_posix(),
                    "spec_rel": spec_abs.as_posix(),
                    "urgency_index": row["urgency_index"],
                    "song_likeness": row["song_likeness"],
                    "rms": row["rms"],
                    "spectral_bandwidth": row["spectral_bandwidth"],
                    "spectral_flatness": row["spectral_flatness"],
                    "repetition_rate_hz": row["repetition_rate_hz"],
                }
            )
        clusters_payload.append(
            {
                "cluster_id": cluster_id,
                "size": int((df["cluster_id"] == cluster_id).sum()),
                "n_recordings": int(full_cluster["recording_id"].nunique()) if "recording_id" in full_cluster else 0,
                "dominant_recording_share": dominant_share,
                "mean_species_top1_conf": float(full_cluster["species_top1_conf"].mean()) if "species_top1_conf" in full_cluster else 0.0,
                "mean_spec_rel": mean_spec_rel,
                "items": items,
            }
        )

    template = Template(HTML_TEMPLATE)
    html = template.render(
        total_events=len(df),
        n_clusters=int(sum(1 for c in df["cluster_id"].unique() if c != -1)),
        umap_image=umap_image.resolve().as_posix(),
        dist_plots=[p.resolve().as_posix() for p in dist_plots],
        analysis_plots=analysis_plots or [],
        cluster_overview_rows=cluster_overview_rows or [],
        review_sheet_paths=[
            {"label": key.replace("_", " ").title(), "path": path.resolve().as_posix()}
            for key, path in (review_sheet_paths or {}).items()
        ],
        clusters=clusters_payload,
    )
    out_path = reports_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path
