"""
Class-balanced Focal Loss with label smoothing.

Chosen because the dataset is imbalanced (52-100 images/class). Focal loss
down-weights easy majority-class samples; the per-class alpha weights are
derived from inverse class frequency; label smoothing regularizes the small
dataset against over-confidence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_alpha_from_counts(counts) -> torch.Tensor:
    """Inverse-frequency alpha weights, normalized to mean 1."""
    counts = torch.as_tensor(counts, dtype=torch.float32)
    w = counts.sum() / (len(counts) * counts)
    return w / w.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.ls = label_smoothing

    def forward(self, logits, target):
        # per-sample cross entropy (with optional label smoothing & alpha)
        ce = F.cross_entropy(
            logits, target,
            weight=self.alpha if self.alpha is not None else None,
            label_smoothing=self.ls,
            reduction="none",
        )
        # pt = probability assigned to the true class
        pt = torch.exp(-F.cross_entropy(logits, target, reduction="none"))
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()
