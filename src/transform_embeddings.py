from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .adapter_model import EmbeddingAdapter, require_torch
from .utils import ensure_dir, google_drive_roots, require_google_drive_output, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform a saved pattern-layer embedding set through a trained adapter MLP."
    )
    parser.add_argument("--pattern_run_dir", type=Path, required=True)
    parser.add_argument("--adapter_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def _load_checkpoint(checkpoint_path: Path):
    require_torch()
    import torch

    checkpoint = torch.load(checkpoint_path.as_posix(), map_location="cpu")
    model = EmbeddingAdapter(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        dropout=float(checkpoint.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return checkpoint, model


def _copy_required_manifests(src_dir: Path, dst_dir: Path) -> None:
    ensure_dir(dst_dir / "manifests")
    for name in ("accepted_events_manifest.csv", "accepted_events_summary.json"):
        src = src_dir / "manifests" / name
        if not src.exists():
            raise FileNotFoundError(f"Required manifest not found: {src.as_posix()}")
        shutil.copy2(src, dst_dir / "manifests" / name)


def _copy_embedding_index(src_dir: Path, dst_dir: Path) -> None:
    ensure_dir(dst_dir / "embeddings")
    src = src_dir / "embeddings" / "embedding_index.csv"
    if not src.exists():
        raise FileNotFoundError(f"Embedding index not found: {src.as_posix()}")
    shutil.copy2(src, dst_dir / "embeddings" / "embedding_index.csv")


def _maybe_remap_summary_paths(summary: dict[str, object]) -> dict[str, object]:
    summary = dict(summary)
    embedding_dir = Path(str(summary.get("embedding_dir", "")).strip()).expanduser()
    species_run_dir = Path(str(summary.get("species_run_dir", "")).strip()).expanduser()
    if embedding_dir.exists():
        return summary

    species_run_name = species_run_dir.name
    if not species_run_name:
        return summary

    for drive_root in google_drive_roots():
        candidate_species_dir = drive_root / "My Drive" / "BirdTranslationAI" / "runs" / "species_decision" / species_run_name
        candidate_embedding_dir = candidate_species_dir / "events"
        if candidate_embedding_dir.exists():
            summary["embedding_dir"] = candidate_embedding_dir.as_posix()
            summary["species_run_dir"] = candidate_species_dir.as_posix()
            return summary
    return summary


def main() -> None:
    args = parse_args()
    pattern_run_dir = args.pattern_run_dir.resolve()
    adapter_checkpoint = args.adapter_checkpoint.expanduser().resolve()
    output_dir = require_google_drive_output(args.output_dir, purpose="--output_dir")

    source_embeddings_path = pattern_run_dir / "embeddings" / "accepted_embeddings.npy"
    source_meta_path = pattern_run_dir / "embeddings" / "embedding_meta.json"
    if not source_embeddings_path.exists():
        raise FileNotFoundError(f"Source embeddings not found: {source_embeddings_path.as_posix()}")
    if not source_meta_path.exists():
        raise FileNotFoundError(f"Source embedding meta not found: {source_meta_path.as_posix()}")

    checkpoint, model = _load_checkpoint(adapter_checkpoint)
    source_embeddings = np.load(source_embeddings_path).astype(np.float32)
    if source_embeddings.ndim != 2:
        raise ValueError("Expected a 2D embedding matrix.")
    if int(source_embeddings.shape[1]) != int(checkpoint["input_dim"]):
        raise ValueError(
            f"Adapter expects input_dim={checkpoint['input_dim']}, but source embeddings have dim={source_embeddings.shape[1]}."
        )

    require_torch()
    import torch

    with torch.no_grad():
        transformed = model(torch.from_numpy(source_embeddings)).cpu().numpy().astype(np.float32)

    ensure_dir(output_dir)
    _copy_required_manifests(pattern_run_dir, output_dir)
    _copy_embedding_index(pattern_run_dir, output_dir)

    embeddings_dir = output_dir / "embeddings"
    np.save(embeddings_dir / "accepted_embeddings.npy", transformed)

    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    summary_path = output_dir / "manifests" / "accepted_events_summary.json"
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_payload = _maybe_remap_summary_paths(summary_payload)
    save_json(summary_path, summary_payload)

    source_meta.update(
        {
            "backend": "birdnet_adapter",
            "requested_backend": "birdnet_adapter",
            "embedding_dim": int(transformed.shape[1]),
            "adapter_checkpoint": adapter_checkpoint.as_posix(),
            "adapter_input_dim": int(checkpoint["input_dim"]),
            "adapter_hidden_dim": int(checkpoint["hidden_dim"]),
            "adapter_output_dim": int(checkpoint["output_dim"]),
            "adapter_dropout": float(checkpoint.get("dropout", 0.0)),
            "adapter_margin": float(checkpoint.get("margin", 0.0)),
            "source_pattern_run_dir": pattern_run_dir.as_posix(),
            "source_embedding_dim": int(source_embeddings.shape[1]),
            "adapted_summary_embedding_dir": str(summary_payload.get("embedding_dir", "")),
        }
    )
    save_json(embeddings_dir / "embedding_meta.json", source_meta)

    save_json(
        output_dir / "embeddings" / "transform_summary.json",
        {
            "artifact_type": "adapted_embeddings",
            "source_pattern_run_dir": pattern_run_dir.as_posix(),
            "adapter_checkpoint": adapter_checkpoint.as_posix(),
            "n_embeddings": int(source_embeddings.shape[0]),
            "source_dim": int(source_embeddings.shape[1]),
            "output_dim": int(transformed.shape[1]),
        },
    )
    print(f"Saved adapted embeddings to {(embeddings_dir / 'accepted_embeddings.npy').as_posix()}")


if __name__ == "__main__":
    main()
