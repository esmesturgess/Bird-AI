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


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al. 2020, the "SupCon-out" formulation).

    Operates on a whole batch of L2-normalized embeddings with integer class labels.
    For each anchor, every other same-class item in the batch is a positive and every
    different-class item is a negative, compared via a temperature-scaled softmax over
    the full batch at once (not one pair at a time). This is more sample-efficient than
    the pairwise objective, which matters for scarce classes.

    Validated 2026-08-13 to beat PairContrastiveLoss on all Blackbird call-type categories
    (see CLAUDE.md); this is the same implementation, promoted from the eval scratchpad.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        require_torch()
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, z, labels):  # type: ignore[override]
        # z: (N, D) already L2-normalized (EmbeddingAdapter normalizes its output).
        n = z.shape[0]
        labels = labels.view(-1, 1)
        same = torch.eq(labels, labels.T).float()
        not_self = 1.0 - torch.eye(n, device=z.device)
        pos_mask = same * not_self

        sim = (z @ z.T) / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()  # numerical stability
        exp_sim = torch.exp(sim) * not_self  # exclude self from the denominator
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)

        pos_count = pos_mask.sum(1).clamp(min=1e-12)
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_count
        return (-mean_log_prob_pos).mean()
