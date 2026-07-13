from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - import error is handled at runtime.
    torch = None
    nn = None
    F = None


def require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise ImportError(
            "PyTorch is required for adapter training. Install it in the project environment, "
            "for example with `pip install torch`."
        )


class EmbeddingAdapter(nn.Module):
    """
    Small MLP that gently reshapes frozen BirdNET embeddings into a clustering-friendlier space.

    The important idea is not to replace BirdNET, but to learn a light projection on top of it
    from same/different judgments.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.10,
    ) -> None:
        require_torch()
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.dropout = float(dropout)

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, x):  # type: ignore[override]
        z = self.net(x)
        return F.normalize(z, dim=-1)


class PairContrastiveLoss(nn.Module):
    """
    Cosine-based contrastive objective.

    - positive pairs are pulled together
    - negative pairs are only pushed apart once they get closer than the margin
    """

    def __init__(self, margin: float = 0.35) -> None:
        require_torch()
        super().__init__()
        self.margin = float(margin)

    def forward(self, z1, z2, labels):  # type: ignore[override]
        cos = F.cosine_similarity(z1, z2, dim=-1)
        labels = labels.float()
        positive = labels * (1.0 - cos)
        negative = (1.0 - labels) * F.relu(cos - self.margin)
        return (positive + negative).mean()
