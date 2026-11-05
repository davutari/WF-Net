"""
Training / evaluation engine for a single (model, fold).

Implements the two-phase transfer-learning schedule:
  Phase 1 (FREEZE_EPOCHS): backbone frozen, only head/attention trained.
  Phase 2:                 whole network fine-tuned with a small backbone LR.
Model selection is done on the validation macro-F1 (early stopping); the
selected checkpoint is then evaluated ONCE on the held-out test fold.
"""
import copy
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.data import WasteDataset
from src.losses import FocalLoss, class_alpha_from_counts
from src.metrics import compute_metrics
from src.mixup import mixup_cutmix
from src.models import build_model, split_param_groups
from src.utils import (seed_worker, count_parameters, ModelEMA, durable_torch_save,
                       rng_state, set_rng_state)


def _loader(paths, labels, idx, train, generator=None):
    ds = WasteDataset([paths[i] for i in idx], [labels[i] for i in idx], train)
    return DataLoader(
        ds, batch_size=config.BATCH_SIZE, shuffle=train,
        num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker, generator=generator, drop_last=False,
        # keep worker processes alive across epochs (Windows re-spawns them otherwise)
        persistent_workers=config.NUM_WORKERS > 0,
    )


@torch.no_grad()
def _evaluate(model, loader, device, n_classes, tta=False):
    """Evaluate; if tta=True, average softmax over the image and its h-flip."""
    model.eval()
    all_prob, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        prob = torch.softmax(model(x), dim=1)
        if tta:
            prob = prob + torch.softmax(model(torch.flip(x, dims=[3])), dim=1)
            prob = prob / 2.0
        all_prob.append(prob.cpu())
        all_labels.append(y)
    prob = torch.cat(all_prob).numpy()
    labels = torch.cat(all_labels).numpy()
    pred = prob.argmax(1)
    metrics = compute_metrics(labels, pred, prob, n_classes)
    return metrics, labels, pred, prob


def train_one_fold(display_name, paths, labels, tr_idx, val_idx, test_idx,
                   class_counts, n_classes, fold_id, epochs=None,
                   cross_targets=None, state_path=None, state_every_min=None):
    """Train + evaluate one (model, fold).

    If `state_path` is given, the complete training state (model, EMA, optimizer,
    scheduler, AMP scaler, best weights, early-stopping counters, history and
    every RNG) is saved crash-safely at the end of an epoch whenever
    `state_every_min` minutes have passed, and always when training ends. A call
    with an existing state file continues from the last saved epoch, following
    the same trajectory as an uninterrupted run (with num_workers=0).
    """
    device = config.DEVICE
    epochs = epochs or config.EPOCHS
    every_s = 60.0 * (config.TRAIN_STATE_EVERY_MIN if state_every_min is None
                      else state_every_min)

    gen = torch.Generator()
    gen.manual_seed(config.SEED + fold_id)

    tr_loader = _loader(paths, labels, tr_idx, True, gen)
    val_loader = _loader(paths, labels, val_idx, False)
    test_loader = _loader(paths, labels, test_idx, False)

    model = build_model(display_name, n_classes).to(device)
    n_params = count_parameters(model)
    ema = ModelEMA(model, config.EMA_DECAY) if config.USE_EMA else None

    alpha = class_alpha_from_counts(class_counts).to(device)
    criterion = FocalLoss(gamma=config.FOCAL_GAMMA, alpha=alpha,
                          label_smoothing=config.LABEL_SMOOTHING)

    backbone_params, head_params = split_param_groups(model)
    for p in backbone_params:      # phase-1: freeze backbone
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": config.LR_HEAD},
         {"params": backbone_params, "lr": config.LR_BACKBONE}],
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=config.USE_AMP and torch.cuda.is_available())

    best_score, best_state, patience = -1.0, None, 0
    start_epoch, finished, history, elapsed_before = 0, False, [], 0.0

    # ---------------- resume from a saved training state ----------------
    if state_path and os.path.exists(state_path):
        st = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        if ema is not None and st.get("ema") is not None:
            ema.ema.load_state_dict(st["ema"])
            ema.updates = st["ema_updates"]
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        best_score, best_state, patience = st["best_score"], st["best_state"], st["patience"]
        start_epoch, finished = st["epoch"] + 1, st["finished"]
        history, elapsed_before = st["history"], st["elapsed"]
        set_rng_state(st["rng"], gen)
        if start_epoch > config.FREEZE_EPOCHS:       # unfreeze already happened
            for p in backbone_params:
                p.requires_grad = True
        tqdm.write(f"[{display_name}] fold{fold_id}: resumed after epoch {start_epoch} "
                   f"({'training finished' if finished else 'continuing'})")
        del st

    def save_state(epoch, done):
        if not state_path:
            return
        durable_torch_save({
            "model": model.state_dict(),
            "ema": ema.ema.state_dict() if ema is not None else None,
            "ema_updates": ema.updates if ema is not None else 0,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "best_score": best_score,
            "best_state": best_state, "patience": patience, "epoch": epoch,
            "finished": done, "history": history,
            "elapsed": elapsed_before + (time.time() - t0), "rng": rng_state(gen),
        }, state_path)

    t0 = time.time()
    last_save = time.time()
    epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs):
        if finished:
            break
        t_ep = time.time()
        if epoch == config.FREEZE_EPOCHS:          # phase-2: unfreeze backbone
            for p in backbone_params:
                p.requires_grad = True

        model.train()
        loss_sum, n_batches = 0.0, 0
        pbar = tqdm(tr_loader, desc=f"[{display_name}] fold{fold_id} ep{epoch+1}/{epochs}",
                    leave=False)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x, y_a, y_b, lam = mixup_cutmix(x, y, config)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=config.USE_AMP and torch.cuda.is_available()):
                # MixUp/CutMix-aware supervised loss (no-op mixing -> plain loss)
                def ce_fn(logits):
                    return lam * criterion(logits, y_a) + \
                        (1.0 - lam) * criterion(logits, y_b)
                loss = ce_fn(model(x))
                if getattr(model, "deep_supervision", False) and model.training:
                    loss = loss + model.distillation_loss(ce_fn)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            loss_sum += float(loss.detach())
            n_batches += 1
        scheduler.step()

        # early stopping / checkpointing tracks the evaluated weights (EMA if on)
        eval_model = ema.ema if ema is not None else model
        val_metrics, *_ = _evaluate(eval_model, val_loader, device, n_classes)
        score = val_metrics[config.MONITOR]
        improved = score > best_score
        if improved:
            best_score = score
            best_state = copy.deepcopy(eval_model.state_dict())
            patience = 0
        else:
            patience += 1
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / max(n_batches, 1),
                        "val_macro_f1": float(val_metrics["macro_f1"]),
                        "val_accuracy": float(val_metrics["accuracy"]),
                        "lr_head": optimizer.param_groups[0]["lr"],
                        "seconds": time.time() - t_ep})

        # persistent one-line-per-epoch summary (tqdm.write coexists with the
        # transient batch bar) so progress through the 40 epochs is visible.
        tqdm.write(
            f"[{display_name}] fold{fold_id} ep{epoch+1}/{epochs}  "
            f"val_{config.MONITOR}={score:.4f}  best={best_score:.4f}  "
            f"patience={patience}/{config.EARLY_STOP_PATIENCE}"
            + ("  <- best" if improved else ""))

        stop = patience >= config.EARLY_STOP_PATIENCE
        if stop:
            tqdm.write(f"[{display_name}] fold{fold_id}: early stop at ep{epoch+1}")
        done = stop or epoch == epochs - 1
        if done or time.time() - last_save >= every_s:
            save_state(epoch, done)
            last_save = time.time()
        if stop:
            break

    # final evaluation on the untouched test fold with the best checkpoint
    eval_model = ema.ema if ema is not None else model
    if best_state is not None:
        eval_model.load_state_dict(best_state)
    test_metrics, y_true, y_pred, y_prob = _evaluate(
        eval_model, test_loader, device, n_classes, tta=config.USE_TTA)
    test_metrics["train_time_s"] = elapsed_before + (time.time() - t0)
    test_metrics["n_params"] = n_params
    test_metrics["history"] = history

    # cross-dataset generalization on external targets (model still in memory)
    cross_results = {}
    t_cross = time.time()
    if cross_targets:
        from src.cross_dataset import evaluate_cross
        for name, data in cross_targets.items():
            cm, yt, yp, yprob = evaluate_cross(
                eval_model, data["paths"], data["labels"], data["shared"],
                n_classes, device, tta=config.USE_TTA,
                restrict=config.CROSS_RESTRICT_TO_SHARED)
            cross_results[name] = {"metrics": cm, "y_true": yt,
                                   "y_pred": yp, "y_prob": yprob}
    test_metrics["cross_time_s"] = time.time() - t_cross

    del model, eval_model, ema
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return test_metrics, y_true, y_pred, y_prob, best_state, cross_results
