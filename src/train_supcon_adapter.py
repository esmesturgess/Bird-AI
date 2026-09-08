"""Train a deployable EmbeddingAdapter with Supervised Contrastive Loss on class-labelled
BirdNET embeddings, and save both the adapter checkpoint AND the per-class prototypes that
nearest-prototype inference needs.

Unlike train_adapter.py (pairwise loss, same/different pair CSV), this trains on per-clip
CLASS labels using stratified batches + SupConLoss (validated 2026-08-13 as the best
Blackbird approach). The checkpoint is written in the SAME format as train_adapter.py so
it is drop-in compatible with transform_embeddings.py / assign_anchor_labels.py
(--adapter_checkpoint).

This trains ONE final model on ALL provided data (no held-out fold). The honest accuracy
estimate comes from the separate nested-CV evaluation, not from this run's own train-set
numbers (which are printed only as a sanity check).

    ./.venv/bin/python -m src.train_supcon_adapter \
        --pattern_run_dir data/runs/blackbird_macaulay_final_pattern_layer \
        --labels_csv data/raw/blackbird_macaulay_combined_v1/call_type_ground_truth.csv \
        --label_column call_type_ground_truth \
        --output_dir data/runs/blackbird_supcon_adapter_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .adapter_model import EmbeddingAdapter, SupConLoss, require_torch
from .utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a deployable SupCon adapter + prototypes.")
    p.add_argument("--pattern_run_dir", type=Path, required=True,
                   help="Dir with embeddings/accepted_embeddings.npy, embeddings/embedding_index.csv, "
                        "manifests/accepted_events_manifest.csv")
    p.add_argument("--labels_csv", type=Path, required=True,
                   help="CSV mapping recording_id -> class label.")
    p.add_argument("--label_column", type=str, default="call_type_ground_truth")
    p.add_argument("--aves2_npy", type=Path, default=None,
                   help="Optional AVES2 embeddings (.npy) to CONCATENATE with the BirdNET ones. "
                        "Requires --aves2_meta. Each source is L2-normalised before concatenation. "
                        "This is the validated best configuration (see CLAUDE.md): the two encoders "
                        "are complementary, and concat beats either alone on all three species.")
    p.add_argument("--aves2_meta", type=Path, default=None,
                   help="CSV with a clip_id column aligned row-for-row with --aves2_npy.")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=400)
    p.add_argument("--batch_per_class", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--hidden_dim", type=int, default=512,  # validated 2026-08-14 via nested-CV
                    help="Default 512: nested-CV architecture sweep on Blackbird found a consistent "
                         "preference for hidden_dim=512 over 256 (see CLAUDE.md).")
    p.add_argument("--output_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def main() -> None:
    require_torch()
    import torch

    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    pr = args.pattern_run_dir
    emb = np.load(pr / "embeddings/accepted_embeddings.npy").astype(np.float32)
    idx = pd.read_csv(pr / "embeddings/embedding_index.csv")
    man = pd.read_csv(pr / "manifests/accepted_events_manifest.csv")
    labels_df = pd.read_csv(args.labels_csv)

    rec_of_clip = dict(zip(man["clip_id"], man["recording_id"]))
    label_of_rec = dict(zip(labels_df["recording_id"], labels_df[args.label_column]))

    rows, missing = [], 0
    for _, r in idx.iterrows():
        rid = rec_of_clip.get(r["clip_id"])
        lab = label_of_rec.get(rid)
        if rid is None or lab is None or (isinstance(lab, float) and np.isnan(lab)):
            missing += 1
            continue
        rows.append((int(r["embedding_row"]), str(lab), str(r["clip_id"])))
    if not rows:
        raise SystemExit("No embeddings could be matched to labels — check label_column / ids.")

    embedding_sources = ["birdnet"]
    if args.aves2_npy is not None:
        if args.aves2_meta is None:
            raise SystemExit("--aves2_npy requires --aves2_meta")
        av = np.load(args.aves2_npy).astype(np.float32)
        av_meta = pd.read_csv(args.aves2_meta)
        av_row = {c: i for i, c in enumerate(av_meta["clip_id"])}
        kept = [(bi, lab, cid) for bi, lab, cid in rows if cid in av_row]
        dropped = len(rows) - len(kept)
        if not kept:
            raise SystemExit("No clip_ids shared between the BirdNET index and --aves2_meta.")
        rows = kept
        Xb = _l2(np.stack([emb[i] for i, _, _ in rows]))
        Xa = _l2(np.stack([av[av_row[cid]] for _, _, cid in rows]))
        X = np.concatenate([Xa, Xb], axis=1).astype(np.float32)  # AVES2 first, matching the eval
        embedding_sources = ["aves2", "birdnet"]
        print(f"concatenated AVES2 {Xa.shape} + BirdNET {Xb.shape} -> {X.shape} "
              f"(dropped {dropped} clips absent from the AVES2 set)")
    else:
        X = np.stack([emb[i] for i, _, _ in rows]).astype(np.float32)
    y = np.array([lab for _, lab, _ in rows])
    classes = sorted(set(y))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"training clips: {len(y)} (unmatched skipped: {missing}) | classes: {classes}")
    print("class counts:", dict(pd.Series(y).value_counts()))

    # --- train adapter with SupCon on stratified batches (all data, final model) ---
    idx_by_class = {c: np.where(y == c)[0] for c in classes}
    model = EmbeddingAdapter(input_dim=X.shape[1], hidden_dim=args.hidden_dim,
                             output_dim=args.output_dim, dropout=args.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = SupConLoss(temperature=args.temperature)
    model.train()
    for it in range(args.iterations):
        opt.zero_grad()
        batch_idx, batch_lab = [], []
        for c in classes:
            sel = rng.choice(idx_by_class[c], args.batch_per_class)
            batch_idx.append(sel)
            batch_lab.extend([class_to_idx[c]] * args.batch_per_class)
        xb = torch.tensor(X[np.concatenate(batch_idx)])
        yb = torch.tensor(np.array(batch_lab))
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
    model.eval()

    # --- build per-class prototypes in adapter space (what nearest-prototype needs) ---
    with torch.no_grad():
        Z = model(torch.tensor(X)).numpy()
    protos = _l2(np.stack([_l2(Z)[y == c].mean(0) for c in classes]))

    # train-set sanity check (NOT the honest accuracy — that's the nested-CV estimate)
    sims = _l2(Z) @ protos.T
    train_pred = np.array([classes[i] for i in sims.argmax(1)])
    train_acc = float((train_pred == y).mean())

    # --- save, in the format transform_embeddings.py / assign_anchor_labels.py expect ---
    out = args.output_dir
    ensure_dir(out)
    checkpoint_path = out / "adapter.pt"
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "input_dim": int(X.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "output_dim": int(args.output_dim),
            "dropout": float(args.dropout),
            "margin": 0.0,  # unused for SupCon; kept for loader compatibility
        },
        checkpoint_path,
    )
    np.save(out / "prototypes.npy", protos.astype(np.float32))
    save_json(out / "prototype_labels.json", {"labels": classes})
    save_json(out / "adapter_meta.json", {
        "artifact_type": "embedding_adapter",
        "loss": "supcon",
        "embedding_sources": embedding_sources,
        "input_layout": ("L2(aves2_768) || L2(birdnet_1024)" if len(embedding_sources) == 2
                         else "birdnet_1024"),
        "aves2_npy": args.aves2_npy.as_posix() if args.aves2_npy else None,
        "checkpoint_path": checkpoint_path.as_posix(),
        "pattern_run_dir": pr.as_posix(),
        "labels_csv": args.labels_csv.as_posix(),
        "label_column": args.label_column,
        "n_clips": int(len(y)),
        "classes": classes,
        "class_counts": {c: int((y == c).sum()) for c in classes},
        "hyperparameters": {
            "iterations": args.iterations, "batch_per_class": args.batch_per_class,
            "temperature": args.temperature, "hidden_dim": args.hidden_dim,
            "output_dim": args.output_dim, "dropout": args.dropout,
            "lr": args.lr, "weight_decay": args.weight_decay, "seed": args.seed,
        },
        "train_set_accuracy_sanity_check": train_acc,
        "note": "train_set_accuracy is NOT a generalization estimate; see nested-CV eval in CLAUDE.md.",
    })
    print(f"\nSaved adapter checkpoint: {checkpoint_path}")
    print(f"Saved prototypes: {out/'prototypes.npy'}  shape={protos.shape}  labels={classes}")
    print(f"train-set sanity-check accuracy (NOT generalization): {train_acc:.3f}")


if __name__ == "__main__":
    main()
