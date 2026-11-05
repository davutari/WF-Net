"""
Per-image branch weighting of the CATF fusion: does the weighting follow branch
accuracy? Runs AFTER a full training run; needs the saved proposed-model
checkpoints (<run>/checkpoints/ckpt_WF_Net__Proposed__fold*.pt).

For every out-of-fold test image it records the fusion read-out's attention
mass on the ConvNeXt-branch tokens vs the Swin-branch tokens (w_convnext +
w_swin = 1), together with the two auxiliary branch heads' predictions.

Outputs
  1. Variation:  distribution of w_convnext across images (does it vary per
                 image?) + per-class differences (Kruskal-Wallis).
  2. Weight vs accuracy: does the weighting follow the more accurate branch on
                 that image?  Spearman rho between w_convnext and
                 (p_convnext(true) - p_swin(true)) from the auxiliary heads, and
                 Mann-Whitney U between "only ConvNeXt-head correct" and
                 "only Swin-head correct" images.
  3. Visual:     histogram, per-class box plot, weight-vs-accuracy scatter, and
                 a gallery of the images with the highest ConvNeXt and Swin
                 shares with Grad-CAM of both branches.
  4. (optional) the same weights on the external Garbage12 images.

Called from run_experiments.py: run_branch_weighting(run_dir, out_dir)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats
from torch.utils.data import DataLoader

import config
from src.viz import save_fig
from src.data import WasteDataset, build_transforms
from src.gradcam import GradCAM, denormalize
from src.models import WFNet

CKPT_FMT = "ckpt_WF_Net__Proposed__fold{}.pt"


@torch.no_grad()
def _collect(model, paths, labels, device, batch=16):
    """Fused/aux predictions + branch weights for a list of images."""
    ds = WasteDataset(paths, labels, train=False)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    out = {k: [] for k in ("w_a", "pred", "conf", "pa_true", "pb_true",
                           "aux_a_pred", "aux_b_pred")}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        ta, tb = model.branch_a(x), model.branch_b(x)
        logits = model.fusion(ta, tb)                     # capture=True -> _branch_share
        prob = logits.softmax(1)
        pa = model.aux_a(model._pool(ta)).softmax(1)
        pb = model.aux_b(model._pool(tb)).softmax(1)
        idx = torch.arange(len(y), device=device)
        out["w_a"].append(model.fusion._branch_share[:, 0].cpu())
        out["pred"].append(prob.argmax(1).cpu())
        out["conf"].append(prob.max(1).values.cpu())
        out["pa_true"].append(pa[idx, y].cpu())
        out["pb_true"].append(pb[idx, y].cpu())
        out["aux_a_pred"].append(pa.argmax(1).cpu())
        out["aux_b_pred"].append(pb.argmax(1).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def _load_model(ckpt, n_classes, catf_mode, device):
    model = WFNet(n_classes, pretrained=False, catf_mode=catf_mode)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.to(device).eval()
    model.fusion.capture = True
    if not getattr(model, "deep_supervision", False):
        raise RuntimeError("Checkpoint has no auxiliary heads (deep supervision off).")
    return model


def _gallery(model, df, class_names, device, out_path, k):
    """Images with the highest ConvNeXt and Swin shares, with Grad-CAM of both branches."""
    tf = build_transforms(train=False)
    cam_a = GradCAM(model, model.branch_a.attn)
    cam_b = GradCAM(model, model.branch_b.attn)
    picks = [("Highest ConvNeXt share", df.nlargest(k, "w_convnext")),
             ("Highest Swin share", df.nsmallest(k, "w_convnext"))]
    rows = sum(len(p) for _, p in picks)
    fig, axes = plt.subplots(rows, 3, figsize=(7.5, 2.5 * rows))
    axes = np.atleast_2d(axes)
    r = 0
    for title, sub in picks:
        for _, row in sub.iterrows():
            x = tf(Image.open(row.path).convert("RGB")).unsqueeze(0).to(device)
            x.requires_grad_(True)
            with torch.enable_grad():
                ca, _ = cam_a(x)
                cb, _ = cam_b(x)
            img = denormalize(x[0].detach(), config.MEAN, config.STD)
            axes[r, 0].imshow(img)
            axes[r, 0].set_title(f"{title}\nT:{class_names[row.true]} P:{class_names[row.pred]}",
                                 fontsize=7)
            axes[r, 1].imshow(img)
            axes[r, 1].imshow(ca[0], cmap="jet", alpha=0.5)
            axes[r, 1].set_title(f"ConvNeXt CAM  w={row.w_convnext:.2f}", fontsize=7)
            axes[r, 2].imshow(img)
            axes[r, 2].imshow(cb[0], cmap="jet", alpha=0.5)
            axes[r, 2].set_title(f"Swin CAM  w={row.w_swin:.2f}", fontsize=7)
            for a in axes[r]:
                a.axis("off")
            r += 1
    plt.tight_layout()
    save_fig(out_path)


def _analyse(df, class_names, tag, out_dir, lines):
    """Variation + weight-vs-accuracy statistics and figures for one image set."""
    w = df.w_convnext.values
    lines.append(f"== {tag}: {len(df)} images ==")
    q1, q3 = np.percentile(w, [25, 75])
    lines.append(f"  w_convnext mean {w.mean():.3f}  std {w.std():.3f}  "
                 f"IQR [{q1:.3f}, {q3:.3f}]  range [{w.min():.3f}, {w.max():.3f}]")

    groups = [g.w_convnext.values for _, g in df.groupby("true") if len(g) >= 3]
    if len(groups) >= 2:
        h, p = stats.kruskal(*groups)
        lines.append(f"  Per-class differences (Kruskal-Wallis): H={h:.2f}, p={p:.3g}")
    pc = (df.groupby("true").w_convnext.agg(["mean", "std", "count"])
          .rename(index=lambda i: class_names[i]).sort_values("mean"))
    pc.to_csv(os.path.join(out_dir, f"per_class_branch_weighting_{tag}.csv"))
    for name, r in pc.iterrows():
        lines.append(f"    {name:28s} w_convnext {r['mean']:.3f} ± {r['std']:.3f} (n={int(r['count'])})")

    delta = df.pa_true.values - df.pb_true.values
    rho, p_rho = stats.spearmanr(w, delta)
    lines.append(f"  Weight vs accuracy: Spearman rho(w_convnext, p_convnext(true)-p_swin(true)) "
                 f"= {rho:.3f}, p={p_rho:.3g}  (rho>0 => attention follows the branch with "
                 f"the higher true-class probability)")
    a_only = df[df.aux_a_correct & ~df.aux_b_correct].w_convnext
    b_only = df[~df.aux_a_correct & df.aux_b_correct].w_convnext
    if len(a_only) >= 3 and len(b_only) >= 3:
        u, p_u = stats.mannwhitneyu(a_only, b_only, alternative="greater")
        lines.append(f"  Only-ConvNeXt-head-correct (n={len(a_only)}) mean w {a_only.mean():.3f} "
                     f"vs only-Swin-head-correct (n={len(b_only)}) mean w {b_only.mean():.3f}; "
                     f"Mann-Whitney U (one-sided) p={p_u:.3g}")
    else:
        lines.append(f"  Too few disagreement images for Mann-Whitney "
                     f"(A-only={len(a_only)}, B-only={len(b_only)}).")
    ok, bad = df[df.correct].w_convnext, df[~df.correct].w_convnext
    if len(bad) >= 3:
        lines.append(f"  Fused correct mean w {ok.mean():.3f} vs incorrect {bad.mean():.3f}")
    lines.append("")

    # figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].hist(w, bins=30, color="#2b5d9b", alpha=0.85)
    ax[0].axvline(0.5, color="grey", ls=":")
    ax[0].set_xlabel("ConvNeXt-branch attention share  (1 - Swin share)")
    ax[0].set_ylabel("images")
    ax[0].set_title(f"Per-image branch weighting ({tag})")
    order = pc.index.tolist()
    data = [df[df.true == class_names.index(n)].w_convnext.values for n in order]
    try:                                   # matplotlib >= 3.10 API
        ax[1].boxplot(data, orientation="horizontal", tick_labels=order)
    except TypeError:                      # older matplotlib
        ax[1].boxplot(data, vert=False, labels=order)
    ax[1].axvline(0.5, color="grey", ls=":")
    ax[1].tick_params(axis="y", labelsize=7)
    ax[1].set_xlabel("ConvNeXt-branch attention share")
    ax[1].set_title("By class")
    ax[2].scatter(delta, w, s=6, alpha=0.35, color="#2b5d9b")
    ax[2].axhline(0.5, color="grey", ls=":")
    ax[2].axvline(0, color="grey", ls=":")
    ax[2].set_xlabel("p_ConvNeXt(true) - p_Swin(true)   (auxiliary heads)")
    ax[2].set_ylabel("ConvNeXt-branch attention share")
    ax[2].set_title(f"Weight vs accuracy  (Spearman rho={rho:.2f})")
    plt.tight_layout()
    save_fig(os.path.join(out_dir, f"branch_weighting_{tag}.png"))


def run_branch_weighting(run_dir, out_dir, device=None, cross=True, cross_limit=2000,
                         gallery=4, limit=None, log=print):
    """Analyse the proposed model's checkpoints of a finished run (runner layout)."""
    import json
    from src.runner import safe
    device = torch.device(device or config.DEVICE)
    with open(os.path.join(run_dir, "run_meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    class_names, n_folds = meta["class_names"], meta["n_folds"]
    os.makedirs(out_dir, exist_ok=True)
    lines = [f"Branch-weighting analysis | run={os.path.basename(os.path.normpath(run_dir))} "
             f"| seed={meta['seed']} | CATF_MODE={config.CATF_MODE}", ""]

    rows, last_model = [], None
    for f in range(n_folds):
        ckpt = os.path.join(run_dir, "checkpoints", CKPT_FMT.format(f))
        jdir = os.path.join(run_dir, "folds", safe(config.PROPOSED_NAME), f"fold{f}")
        if not (os.path.exists(ckpt) and os.path.exists(os.path.join(jdir, "preds.npz"))):
            log(f"[branch weighting] fold {f}: no checkpoint/predictions -> skipped")
            continue
        model = _load_model(ckpt, len(class_names), config.CATF_MODE, device)
        d = np.load(os.path.join(jdir, "preds.npz"), allow_pickle=True)
        tpaths = [str(p) for p in d["test_paths"]]
        tlabels = [int(y) for y in d["y_true"]]
        if limit:
            tpaths, tlabels = tpaths[:limit], tlabels[:limit]
        r = _collect(model, tpaths, tlabels, device)
        for j, (p, y) in enumerate(zip(tpaths, tlabels)):
            rows.append({"path": p, "fold": f, "true": y,
                         "pred": int(r["pred"][j]), "conf": float(r["conf"][j]),
                         "w_convnext": float(r["w_a"][j]), "w_swin": 1 - float(r["w_a"][j]),
                         "pa_true": float(r["pa_true"][j]), "pb_true": float(r["pb_true"][j]),
                         "aux_a_pred": int(r["aux_a_pred"][j]),
                         "aux_b_pred": int(r["aux_b_pred"][j])})
        last_model = model
        log(f"[branch weighting] fold {f}: {len(tpaths)} images")
    if not rows:
        raise RuntimeError("No proposed-model checkpoints found in " + run_dir)

    df = pd.DataFrame(rows)
    df["correct"] = df.pred == df.true
    df["aux_a_correct"] = df.aux_a_pred == df.true
    df["aux_b_correct"] = df.aux_b_pred == df.true
    df.to_csv(os.path.join(out_dir, "branch_weighting_per_image.csv"), index=False)
    _analyse(df, class_names, "in-domain", out_dir, lines)
    _gallery(last_model, df[df.fold == df.fold.max()], class_names, device,
             os.path.join(out_dir, "branch_weighting_gallery_gradcam.png"), gallery)

    if cross and os.path.isdir(config.GARBAGE12_DIR):
        from src.cross_dataset import build_target
        cpaths, clabels, _ = build_target(config.GARBAGE12_DIR, class_names,
                                          config.CROSS_LABEL_MAPS["Garbage12"])
        rng = np.random.default_rng(0)
        sel = rng.choice(len(cpaths), size=min(cross_limit, len(cpaths)), replace=False)
        r = _collect(last_model, [cpaths[i] for i in sel], [clabels[i] for i in sel], device)
        cdf = pd.DataFrame({"path": [cpaths[i] for i in sel],
                            "true": [clabels[i] for i in sel],
                            "pred": r["pred"], "w_convnext": r["w_a"], "w_swin": 1 - r["w_a"],
                            "pa_true": r["pa_true"], "pb_true": r["pb_true"],
                            "aux_a_pred": r["aux_a_pred"], "aux_b_pred": r["aux_b_pred"]})
        cdf["correct"] = cdf.pred == cdf.true
        cdf["aux_a_correct"] = cdf.aux_a_pred == cdf.true
        cdf["aux_b_correct"] = cdf.aux_b_pred == cdf.true
        cdf.to_csv(os.path.join(out_dir, "branch_weighting_per_image_garbage12.csv"), index=False)
        lines.append("(External set: model of the last fold, unrestricted logits; the "
                     "branch weights do not depend on the closed-set mask.)")
        _analyse(cdf, class_names, "garbage12", out_dir, lines)

    txt = "\n".join(lines)
    with open(os.path.join(out_dir, "branch_weighting_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    log(txt)
    return out_dir
