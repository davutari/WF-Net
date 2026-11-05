"""
How does the single-label WF-Net behave on multi-object scenes?

A controlled *probe* (not a solution): pairs of out-of-fold test images of two
different classes are placed side by side (each resized to 112x224) to form a
synthetic two-object image. For every pair we record whether
  * the top-1 prediction is one of the two present classes,
  * BOTH classes appear in the top-2 predictions,
  * how much probability mass falls on the two present classes,
  * which side wins (left/right position bias),
and draw class-specific Grad-CAM maps for both classes on a few examples to
show whether the model localises both objects. This quantifies the limitation
and motivates a multi-label head / detection front-end (future work).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

import config
from src.data import build_transforms
from src.gradcam import GradCAM, denormalize
from src.viz import save_fig

CKPT_FMT = "ckpt_WF_Net__Proposed__fold{}.pt"


def _composite(p_left, p_right, size=224):
    a = Image.open(p_left).convert("RGB").resize((size // 2, size))
    b = Image.open(p_right).convert("RGB").resize((size // 2, size))
    canvas = Image.new("RGB", (size, size))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (size // 2, 0))
    return canvas


def run_multiobject(run_dir, out_dir, pairs_per_fold=200, gallery=4, device=None,
                    log=print):
    from src.models import WFNet
    from src.runner import safe

    device = torch.device(device or config.DEVICE)
    with open(os.path.join(run_dir, "run_meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    class_names, n_folds = meta["class_names"], meta["n_folds"]
    tf = build_transforms(train=False)
    os.makedirs(out_dir, exist_ok=True)

    rows, examples, last_model = [], [], None
    for f in range(n_folds):
        ck = os.path.join(run_dir, "checkpoints", CKPT_FMT.format(f))
        pp = os.path.join(run_dir, "folds", safe(config.PROPOSED_NAME), f"fold{f}", "preds.npz")
        if not (os.path.exists(ck) and os.path.exists(pp)):
            log(f"[multi-object] fold {f}: no checkpoint -> skipped")
            continue
        d = np.load(pp, allow_pickle=True)
        tpaths, ty = [str(p) for p in d["test_paths"]], d["y_true"].astype(int)
        model = WFNet(len(class_names), pretrained=False).to(device)
        model.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
        model.eval()
        rng = np.random.default_rng(f)
        n_done = 0
        while n_done < pairs_per_fold:
            i, j = rng.choice(len(tpaths), size=2, replace=False)
            if ty[i] == ty[j]:
                continue
            img = _composite(tpaths[i], tpaths[j])
            with torch.no_grad():
                prob = torch.softmax(model(tf(img).unsqueeze(0).to(device)), 1)[0].cpu().numpy()
            top2 = list(np.argsort(-prob)[:2])
            a, b = int(ty[i]), int(ty[j])
            rows.append({"fold": f, "left": class_names[a], "right": class_names[b],
                         "top1": class_names[top2[0]],
                         "top1_in_pair": top2[0] in (a, b),
                         "both_in_top2": set(top2) == {a, b},
                         "pair_prob_mass": float(prob[a] + prob[b]),
                         "winner": "left" if prob[a] > prob[b] else "right"})
            if len(examples) < gallery and set(top2) == {a, b}:
                examples.append((img, a, b))
            n_done += 1
        last_model = model
        log(f"[multi-object] fold {f}: {n_done} composites")
    if not rows:
        raise RuntimeError("No WF-Net checkpoints found in " + run_dir)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "multiobject_pairs.csv"), index=False)
    s = {"pairs": len(df),
         "top1_is_one_of_the_two": df.top1_in_pair.mean(),
         "both_classes_in_top2": df.both_in_top2.mean(),
         "mean_prob_mass_on_pair": df.pair_prob_mass.mean(),
         "left_wins_share": (df.winner == "left").mean(),
         "chance_both_in_top2": 2 / (len(class_names) * (len(class_names) - 1))}
    by_cls = (pd.concat([df.assign(cls=df.left), df.assign(cls=df.right)])
              .groupby("cls").both_in_top2.mean().sort_values())
    by_cls.to_csv(os.path.join(out_dir, "multiobject_by_class.csv"))
    lines = ["Synthetic two-object composites (out-of-fold test images, side by side)", ""]
    lines += [f"  {k:28s}: {v:.4f}" if isinstance(v, float) else f"  {k:28s}: {v}"
              for k, v in s.items()]
    lines += ["", "  both-in-top-2 rate per class:"]
    lines += [f"    {c:28s} {v:.3f}" for c, v in by_cls.items()]
    txt = "\n".join(lines)
    with open(os.path.join(out_dir, "multiobject_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    log(txt)

    # class-specific Grad-CAM for both objects on a few successful composites
    if examples and last_model is not None:
        cam = GradCAM(last_model, last_model.gradcam_target_layer())
        fig, axes = plt.subplots(len(examples), 3, figsize=(8, 2.7 * len(examples)),
                                 squeeze=False)
        for r, (img, a, b) in enumerate(examples):
            x = tf(img).unsqueeze(0).to(device)
            x.requires_grad_(True)
            ca, _ = cam(x, torch.tensor([a], device=device))
            cb, _ = cam(x, torch.tensor([b], device=device))
            disp = denormalize(x[0].detach(), config.MEAN, config.STD)
            axes[r, 0].imshow(disp)
            axes[r, 0].set_title(f"{class_names[a]} | {class_names[b]}", fontsize=7)
            for col, (c, cmap_) in enumerate([(a, ca), (b, cb)], start=1):
                axes[r, col].imshow(disp)
                axes[r, col].imshow(cmap_[0], cmap="jet", alpha=0.5)
                axes[r, col].set_title(f"Grad-CAM for '{class_names[c]}'", fontsize=7)
            for ax in axes[r]:
                ax.axis("off")
        plt.tight_layout()
        save_fig(os.path.join(out_dir, "multiobject_gradcam.png"))
