"""
MixUp / CutMix batch augmentation.

Both are strong regularizers for small, high-capacity settings: they synthesize
convex combinations of images (MixUp) or paste a rectangular patch of one image
onto another (CutMix), with the loss weighted by the mixing ratio ``lam``.

Usage (training only):
    x, y_a, y_b, lam = mixup_cutmix(x, y, config)
    loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
When augmentation is skipped, ``y_a == y_b`` and ``lam == 1.0`` so the caller's
weighted-loss expression reduces to the ordinary loss.
"""
import numpy as np
import torch


def _rand_bbox(h: int, w: int, lam: float):
    """Random box whose area is (1 - lam) of the image (standard CutMix)."""
    cut = np.sqrt(1.0 - lam)
    cw, ch = int(w * cut), int(h * cut)
    cx, cy = np.random.randint(w), np.random.randint(h)
    x1, x2 = np.clip(cx - cw // 2, 0, w), np.clip(cx + cw // 2, 0, w)
    y1, y2 = np.clip(cy - ch // 2, 0, h), np.clip(cy + ch // 2, 0, h)
    return x1, y1, x2, y2


def mixup_cutmix(x: torch.Tensor, y: torch.Tensor, cfg):
    """Return (mixed_x, y_a, y_b, lam). A no-op returns (x, y, y, 1.0)."""
    if not getattr(cfg, "USE_MIXUP", False) or x.size(0) < 2 \
            or np.random.rand() > cfg.MIXUP_PROB:
        return x, y, y, 1.0

    perm = torch.randperm(x.size(0), device=x.device)
    if np.random.rand() < cfg.MIXUP_SWITCH_PROB:                 # CutMix
        lam = float(np.random.beta(cfg.CUTMIX_ALPHA, cfg.CUTMIX_ALPHA))
        h, w = x.shape[-2:]
        x1, y1, x2, y2 = _rand_bbox(h, w, lam)
        x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (h * w))            # exact area ratio
    else:                                                        # MixUp
        lam = float(np.random.beta(cfg.MIXUP_ALPHA, cfg.MIXUP_ALPHA))
        x = lam * x + (1.0 - lam) * x[perm]
    return x, y, y[perm], lam
