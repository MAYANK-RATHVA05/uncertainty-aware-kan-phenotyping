"""Numerically stable focal loss for binary labels and unnormalized logits."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BinaryFocalLoss(nn.Module):
    """Binary focal loss operating directly on logits.

    ``alpha`` weights the positive class; ``1 - alpha`` weights the negative
    class. ``gamma=0`` reduces to alpha-weighted binary cross entropy. Targets
    must contain exactly 0 and 1 and match the logits' shape; probabilities or
    continuous depression scores are not converted into labels. Inputs must be
    nonempty, finite tensors on the same device. The default reduction is mean.

    Args:
        alpha: Positive-class weight in [0, 1], default 0.25.
        gamma: Nonnegative focusing exponent, default 2.
        reduction: One of ``"none"``, ``"mean"`` or ``"sum"``.
    """

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ) -> None:
        super().__init__()
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be finite and in [0, 1]")
        if not math.isfinite(gamma) or gamma < 0.0:
            raise ValueError("gamma must be finite and nonnegative")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Return focal loss without applying sigmoid to the input logits."""
        if not isinstance(logits, Tensor) or not isinstance(targets, Tensor):
            raise TypeError("logits and targets must be torch tensors")
        if not logits.is_floating_point():
            raise ValueError("logits must have a floating-point dtype")
        if logits.shape != targets.shape:
            raise ValueError("logits and targets must have identical shapes")
        if logits.numel() == 0:
            raise ValueError("logits and targets must be nonempty")
        if logits.device != targets.device:
            raise ValueError("logits and targets must be on the same device")
        if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
            raise ValueError("logits and targets must contain only finite values")
        if not torch.all((targets == 0) | (targets == 1)):
            raise ValueError("targets must contain binary labels 0 and 1")

        targets = targets.to(dtype=logits.dtype)
        cross_entropy = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        # -expm1(-BCE) computes 1 - p_t accurately even for confident examples.
        focal_weight = (-torch.expm1(-cross_entropy)).pow(self.gamma)
        class_weight = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        loss = class_weight * focal_weight * cross_entropy
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
