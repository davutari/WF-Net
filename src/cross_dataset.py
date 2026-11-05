"""
Cross-dataset generalization.

Evaluates a model trained on WasteSnap (source) on an unseen external waste
dataset (target, Garbage12) without any fine-tuning. Because the two taxonomies
differ, target class folders are mapped to source class indices via
config.CROSS_LABEL_MAPS and only the shared classes are scored. Optionally the
logits are masked to the shared classes (closed-set protocol).
"""
import os

import torch
from torch.utils.data import DataLoader

import config
from src.data import WasteDataset
from src.metrics import compute_metrics


def _norm(name: str) -> str:
    """Normalize a folder name for robust matching (lowercase, alnum only)."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def build_target(target_dir, source_class_names, label_map):
    """
    Return (paths, source_labels, shared_source_indices) for the target set,
    keeping only folders that map to a known source class.
    """
    label_map = {_norm(k): v for k, v in label_map.items()}
    src_index = {name: i for i, name in enumerate(source_class_names)}
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    paths, labels, shared = [], [], set()
    for d in sorted(os.listdir(target_dir)):
        full = os.path.join(target_dir, d)
        if not os.path.isdir(full):
            continue
        src_name = label_map.get(_norm(d))
        if src_name is None or src_name not in src_index:
            continue                                   # unmapped -> skipped
        si = src_index[src_name]
        shared.add(si)
        for f in sorted(os.listdir(full), key=str.upper):   # same order on every file system
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(full, f))
                labels.append(si)
    return paths, labels, sorted(shared)


@torch.no_grad()
def evaluate_cross(model, paths, labels, shared_idx, n_classes, device,
                   tta=True, restrict=True):
    """Evaluate `model` on the external target samples.

    Returns (metrics, y_true, y_pred, y_prob) so callers can also build a
    confusion matrix / ROC curve for the external set, not just scalar metrics.
    """
    ds = WasteDataset(paths, labels, train=False)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS,
                        pin_memory=torch.cuda.is_available())
    mask = None
    if restrict:
        mask = torch.full((n_classes,), float("-inf"), device=device)
        mask[torch.tensor(shared_idx, device=device)] = 0.0

    model.eval()
    all_prob, all_lab = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        if tta:
            logits = (logits + model(torch.flip(x, dims=[3]))) / 2.0
        if mask is not None:
            logits = logits + mask                     # keep only shared classes
        all_prob.append(torch.softmax(logits, dim=1).cpu())
        all_lab.append(y)

    prob = torch.cat(all_prob).numpy()
    lab = torch.cat(all_lab).numpy()
    pred = prob.argmax(1)
    return compute_metrics(lab, pred, prob, n_classes), lab, pred, prob
