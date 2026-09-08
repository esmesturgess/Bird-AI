"""Honest cross-validated evaluation of the SupCon adapter vs raw-BirdNET-space
nearest-prototype baseline, on independently-labelled (Macaulay/Xeno-canto) ground truth.

Uses StratifiedGroupKFold (grouped by recording_id, so no clip from the same recording
leaks across train/test) with FIXED hyperparameters -- this does not re-tune anything.
The hyperparameters/architecture (hidden_dim=512, output_dim=128, temperature=0.1,
batch_per_class=16) were already selected via a separate nested-CV sweep on Blackbird
(see CLAUDE.md); this script applies that settled recipe to get an honest per-category
generalization estimate for a (possibly different) species/dataset.

    ./.venv/bin/python -m src.evaluate_supcon_adapter \
        --pattern_run_dir data/runs/wren_combined_v1_pattern_layer \
        --labels_csv data/raw/wren_xc_eu_v1/call_type_ground_truth.csv \
        --labels_csv data/raw/wren_macaulay_eu_v1/call_type_ground_truth.csv \
        --label_column call_type_ground_truth \
        --output_json data/runs/wren_supcon_eval_v1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .adapter_model import EmbeddingAdapter, SupConLoss, require_torch
from .utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pattern_run_dir", type=Path, required=True)
    p.add_argument("--labels_csv", type=Path, action="append", required=True,
                    help="Repeatable. Each CSV must have recording_id + --label_column.")
    p.add_argument("--label_column", type=str, default="call_type_ground_truth")
    p.add_argument("--output_json", type=Path, required=True)
    p.add_argument("--n_splits", type=int, default=6)
    p.add_argument("--iterations", type=int, default=400)
    p.add_argument("--batch_per_class", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--output_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def _prototypes(z: np.ndarray, y: np.ndarray, classes: list[str]) -> np.ndarray:
    return _l2(np.stack([_l2(z)[y == c].mean(0) for c in classes]))


def _predict(z: np.ndarray, protos: np.ndarray, classes: list[str]) -> np.ndarray:
    sims = _l2(z) @ protos.T
    return np.array([classes[i] for i in sims.argmax(1)])


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
    gt = pd.concat([pd.read_csv(c) for c in args.labels_csv], ignore_index=True)
    gt = gt.drop_duplicates(subset="recording_id")

    rec_of_clip = dict(zip(man["clip_id"], man["recording_id"]))
    label_of_rec = dict(zip(gt["recording_id"], gt[args.label_column]))

    rows, missing = [], 0
    for _, r in idx.iterrows():
        rid = rec_of_clip.get(r["clip_id"])
        lab = label_of_rec.get(rid)
        if rid is None or lab is None or (isinstance(lab, float) and np.isnan(lab)):
            missing += 1
            continue
        rows.append((int(r["embedding_row"]), str(rid), str(lab)))
    if not rows:
        raise SystemExit("No embeddings could be matched to labels — check label_column / ids.")
    X = np.stack([emb[i] for i, _, _ in rows]).astype(np.float32)
    groups = np.array([rid for _, rid, _ in rows])
    y = np.array([lab for _, _, lab in rows])
    classes = sorted(set(y))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"clips: {len(y)} (unmatched skipped: {missing}) | recordings: {len(set(groups))} "
          f"| classes: {classes}")
    print("class counts (clips):", dict(pd.Series(y).value_counts()))

    skf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_results = []
    confusion = {"raw": {}, "adapter": {}}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y, groups)):
        Xtr, ytr = X[train_idx], y[train_idx]
        Xte, yte = X[test_idx], y[test_idx]

        # --- raw-space baseline ---
        raw_protos = _prototypes(Xtr, ytr, classes)
        raw_pred = _predict(Xte, raw_protos, classes)
        raw_acc = float((raw_pred == yte).mean())

        # --- train SupCon adapter on this fold's train portion only ---
        idx_by_class = {c: np.where(ytr == c)[0] for c in classes}
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
            xb = torch.tensor(Xtr[np.concatenate(batch_idx)])
            yb = torch.tensor(np.array(batch_lab))
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            Ztr = model(torch.tensor(Xtr)).numpy()
            Zte = model(torch.tensor(Xte)).numpy()
        ad_protos = _prototypes(Ztr, ytr, classes)
        ad_pred = _predict(Zte, ad_protos, classes)
        ad_acc = float((ad_pred == yte).mean())

        per_cat_raw = {c: float((raw_pred[yte == c] == c).mean()) for c in classes}
        per_cat_ad = {c: float((ad_pred[yte == c] == c).mean()) for c in classes}
        fold_results.append({
            "fold": fold, "n_train": len(train_idx), "n_test": len(test_idx),
            "raw_overall": raw_acc, "adapter_overall": ad_acc,
            "raw_per_category": per_cat_raw, "adapter_per_category": per_cat_ad,
        })
        for true_c in classes:
            for pred_c in classes:
                mask = yte == true_c
                if mask.sum() == 0:
                    continue
                confusion["raw"].setdefault(true_c, {}).setdefault(pred_c, 0)
                confusion["raw"][true_c][pred_c] += int(((raw_pred == pred_c) & mask).sum())
                confusion["adapter"].setdefault(true_c, {}).setdefault(pred_c, 0)
                confusion["adapter"][true_c][pred_c] += int(((ad_pred == pred_c) & mask).sum())
        print(f"fold {fold}: raw={raw_acc:.3f} adapter={ad_acc:.3f} "
              f"(train={len(train_idx)} test={len(test_idx)})")

    majority_floor = float(pd.Series(y).value_counts(normalize=True).max())
    overall_raw = float(np.mean([f["raw_overall"] for f in fold_results]))
    overall_ad = float(np.mean([f["adapter_overall"] for f in fold_results]))
    per_cat_raw_mean = {c: float(np.mean([f["raw_per_category"][c] for f in fold_results])) for c in classes}
    per_cat_ad_mean = {c: float(np.mean([f["adapter_per_category"][c] for f in fold_results])) for c in classes}

    print(f"\n=== AGGREGATE (mean over {args.n_splits} folds) ===")
    print(f"majority floor: {majority_floor:.3f}")
    print(f"{'category':12s} {'raw':>8s} {'adapter':>8s}")
    for c in classes:
        print(f"{c:12s} {per_cat_raw_mean[c]:8.3f} {per_cat_ad_mean[c]:8.3f}")
    print(f"{'overall':12s} {overall_raw:8.3f} {overall_ad:8.3f}")

    print("\n=== confusion matrix (adapter, summed over folds, rows=true) ===")
    for true_c in classes:
        row = confusion["adapter"].get(true_c, {})
        total = sum(row.values())
        print(f"  {true_c:10s} " + " ".join(f"{c}={row.get(c,0)}({row.get(c,0)/total*100:.0f}%)" for c in classes if total))

    ensure_dir(args.output_json.parent)
    save_json(args.output_json, {
        "pattern_run_dir": pr.as_posix(),
        "labels_csv": [c.as_posix() for c in args.labels_csv],
        "n_clips": int(len(y)),
        "n_recordings": int(len(set(groups))),
        "classes": classes,
        "class_counts_clips": {c: int((y == c).sum()) for c in classes},
        "majority_floor": majority_floor,
        "hyperparameters": {
            "n_splits": args.n_splits, "iterations": args.iterations,
            "batch_per_class": args.batch_per_class, "temperature": args.temperature,
            "hidden_dim": args.hidden_dim, "output_dim": args.output_dim,
            "dropout": args.dropout, "lr": args.lr, "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "fold_results": fold_results,
        "aggregate": {
            "raw_overall": overall_raw, "adapter_overall": overall_ad,
            "raw_per_category": per_cat_raw_mean, "adapter_per_category": per_cat_ad_mean,
        },
        "confusion_matrix": confusion,
    })
    print(f"\nSaved: {args.output_json}")


if __name__ == "__main__":
    main()
