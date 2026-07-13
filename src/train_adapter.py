from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np

from .adapter_model import EmbeddingAdapter, PairContrastiveLoss, require_torch
from .utils import ensure_dir, require_google_drive_output, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small MLP adapter on top of frozen BirdNET embeddings using reviewed same/different pairs."
    )
    parser.add_argument(
        "--pair_csv",
        type=Path,
        default=Path("data/reviews/clip_pair_labels_v1.csv"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/models/adapter_v1"),
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--margin", type=float, default=0.35)
    parser.add_argument("--val_fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path.as_posix()}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_embedding_arrays(pattern_run_dirs: list[Path]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for pattern_run_dir in pattern_run_dirs:
        path = pattern_run_dir / "embeddings" / "accepted_embeddings.npy"
        if not path.exists():
            raise FileNotFoundError(f"Embedding array not found: {path.as_posix()}")
        arrays[pattern_run_dir.as_posix()] = np.load(path).astype(np.float32)
    return arrays


def _build_training_arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]]]:
    filtered = [row for row in rows if str(row.get("review_label", "")).strip().lower() in {"same", "different"}]
    if not filtered:
        raise ValueError("No trainable same/different rows found in pair CSV.")

    pattern_run_dirs = sorted({Path(str(row["pattern_run_dir"])).resolve() for row in filtered})
    embedding_arrays = _load_embedding_arrays(pattern_run_dirs)

    x1_rows: list[np.ndarray] = []
    x2_rows: list[np.ndarray] = []
    y_rows: list[float] = []

    for row in filtered:
        pattern_run_dir = Path(str(row["pattern_run_dir"])).resolve().as_posix()
        emb = embedding_arrays[pattern_run_dir]
        idx_a = int(float(row["embedding_row_a"]))
        idx_b = int(float(row["embedding_row_b"]))
        if idx_a < 0 or idx_a >= emb.shape[0] or idx_b < 0 or idx_b >= emb.shape[0]:
            continue
        x1_rows.append(emb[idx_a])
        x2_rows.append(emb[idx_b])
        y_rows.append(1.0 if str(row["review_label"]).strip().lower() == "same" else 0.0)

    x1 = np.vstack(x1_rows).astype(np.float32)
    x2 = np.vstack(x2_rows).astype(np.float32)
    y = np.asarray(y_rows, dtype=np.float32)
    return x1, x2, y, filtered


def _split_indices(n_items: int, *, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(int(seed))
    indices = list(range(n_items))
    rng.shuffle(indices)
    val_count = int(round(float(val_fraction) * n_items))
    val_count = max(1, min(val_count, max(1, n_items - 1))) if n_items > 1 else 0
    val_idx = np.asarray(indices[:val_count], dtype=np.int64)
    train_idx = np.asarray(indices[val_count:], dtype=np.int64)
    if train_idx.size == 0:
        train_idx = val_idx.copy()
    return train_idx, val_idx


def main() -> None:
    args = parse_args()
    require_torch()
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    args.output_dir = require_google_drive_output(args.output_dir, purpose="--output_dir")

    rows = _read_csv_rows(args.pair_csv)
    x1, x2, y, filtered_rows = _build_training_arrays(rows)
    input_dim = int(x1.shape[1])

    train_idx, val_idx = _split_indices(len(y), val_fraction=float(args.val_fraction), seed=int(args.seed))
    train_dataset = TensorDataset(
        torch.from_numpy(x1[train_idx]),
        torch.from_numpy(x2[train_idx]),
        torch.from_numpy(y[train_idx]),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(x1[val_idx]),
        torch.from_numpy(x2[val_idx]),
        torch.from_numpy(y[val_idx]),
    ) if len(val_idx) else None

    train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(args.batch_size), shuffle=False) if val_dataset is not None else None

    torch.manual_seed(int(args.seed))
    model = EmbeddingAdapter(
        input_dim=input_dim,
        hidden_dim=int(args.hidden_dim),
        output_dim=int(args.output_dim),
        dropout=float(args.dropout),
    )
    criterion = PairContrastiveLoss(margin=float(args.margin))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    best_state = None
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_x1, batch_x2, batch_y in train_loader:
            optimizer.zero_grad()
            z1 = model(batch_x1)
            z2 = model(batch_x2)
            loss = criterion(z1, z2, batch_y)
            loss.backward()
            optimizer.step()
            batch_size = int(batch_y.shape[0])
            train_loss_sum += float(loss.item()) * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / max(1, train_count)

        val_loss = train_loss
        pos_cos = 0.0
        neg_cos = 0.0
        pos_count = 0
        neg_count = 0
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for batch_x1, batch_x2, batch_y in val_loader:
                    z1 = model(batch_x1)
                    z2 = model(batch_x2)
                    loss = criterion(z1, z2, batch_y)
                    cos = torch.nn.functional.cosine_similarity(z1, z2, dim=-1)
                    mask_pos = batch_y > 0.5
                    mask_neg = ~mask_pos
                    if int(mask_pos.sum().item()) > 0:
                        pos_cos += float(cos[mask_pos].sum().item())
                        pos_count += int(mask_pos.sum().item())
                    if int(mask_neg.sum().item()) > 0:
                        neg_cos += float(cos[mask_neg].sum().item())
                        neg_count += int(mask_neg.sum().item())
                    batch_size = int(batch_y.shape[0])
                    val_loss_sum += float(loss.item()) * batch_size
                    val_count += batch_size
            val_loss = val_loss_sum / max(1, val_count)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_mean_positive_cosine": float(pos_cos / max(1, pos_count)) if pos_count else 0.0,
                "val_mean_negative_cosine": float(neg_cos / max(1, neg_count)) if neg_count else 0.0,
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    ensure_dir(args.output_dir)
    checkpoint_path = args.output_dir / "adapter.pt"
    history_path = args.output_dir / "history.json"
    metadata_path = args.output_dir / "adapter_meta.json"

    final_state = best_state if best_state is not None else model.state_dict()
    torch.save(
        {
            "state_dict": final_state,
            "input_dim": input_dim,
            "hidden_dim": int(args.hidden_dim),
            "output_dim": int(args.output_dim),
            "dropout": float(args.dropout),
            "margin": float(args.margin),
        },
        checkpoint_path,
    )
    save_json(history_path, {"history": history})
    save_json(
        metadata_path,
        {
            "artifact_type": "embedding_adapter",
            "checkpoint_path": checkpoint_path.as_posix(),
            "pair_csv": args.pair_csv.resolve().as_posix(),
            "n_pairs_total": int(len(filtered_rows)),
            "n_pairs_train": int(len(train_idx)),
            "n_pairs_val": int(len(val_idx)),
            "input_dim": int(input_dim),
            "hidden_dim": int(args.hidden_dim),
            "output_dim": int(args.output_dim),
            "dropout": float(args.dropout),
            "margin": float(args.margin),
            "best_val_loss": float(best_val_loss),
        },
    )
    print(f"Saved adapter checkpoint to {checkpoint_path.as_posix()}")
    print(f"Saved training history to {history_path.as_posix()}")
    print(f"Saved metadata to {metadata_path.as_posix()}")


if __name__ == "__main__":
    main()
