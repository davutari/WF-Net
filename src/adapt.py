"""
Mitigating the cross-dataset drop with a handful of target labels.

For every fold model of WF-Net (trained on WasteSnap only) the fused
representation (input of the classification head) is extracted for all mapped
Garbage12 images. Then, with k labelled target images per shared class
(k = 1, 5, 10, 20; several random draws), two light-weight adaptation methods
that leave the network frozen are compared with the zero-shot result on the
remaining (query) images:

  * zero-shot   : the source classifier, closed-set on the shared classes
  * prototype   : nearest class mean (cosine) of the k support features
  * linear probe: logistic regression trained on the k support features

This quantifies how much of the drop is recoverable without retraining.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import config
from src.data import WasteDataset
from src.viz import save_fig

CKPT_FMT = "ckpt_WF_Net__Proposed__fold{}.pt"


@torch.no_grad()
def _features(model, paths, labels, device, batch=64):
    feats, logits = [], []
    hook = model.fusion.head.register_forward_hook(
        lambda mod, inp, out: feats.append(inp[0].float().cpu()))
    loader = DataLoader(WasteDataset(paths, labels, train=False), batch_size=batch,
                        shuffle=False, num_workers=config.NUM_WORKERS)
    for x, _ in loader:
        logits.append(model(x.to(device)).float().cpu())
    hook.remove()
    return torch.cat(feats).numpy(), torch.cat(logits).numpy()


def _f1(y, p, shared):
    return f1_score(y, p, labels=shared, average="macro", zero_division=0)


def run_adaptation(run_dir, out_dir, shots=(1, 5, 10, 20), draws=5, device=None,
                   limit=None, log=print):
    from src.cross_dataset import build_target
    from src.models import WFNet

    device = torch.device(device or config.DEVICE)
    with open(os.path.join(run_dir, "run_meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    class_names, n_folds = meta["class_names"], meta["n_folds"]
    paths, labels, shared = build_target(config.GARBAGE12_DIR, class_names,
                                         config.CROSS_LABEL_MAPS["Garbage12"])
    shared = [int(s) for s in shared]
    if limit:
        rng = np.random.default_rng(0)
        sel = sorted(rng.choice(len(paths), size=min(limit, len(paths)), replace=False))
        paths, labels = [paths[i] for i in sel], [labels[i] for i in sel]
    y = np.asarray(labels)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for f in range(n_folds):
        ck = os.path.join(run_dir, "checkpoints", CKPT_FMT.format(f))
        if not os.path.exists(ck):
            log(f"[adapt] fold {f}: no checkpoint -> skipped")
            continue
        model = WFNet(len(class_names), pretrained=False).to(device)
        model.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
        model.eval()
        X, logits = _features(model, paths, labels, device)
        mask = np.full(len(class_names), -np.inf)
        mask[shared] = 0.0
        zero_pred = (logits + mask).argmax(1)
        log(f"[adapt] fold {f}: features {X.shape}, zero-shot macro-F1 "
            f"{_f1(y, zero_pred, shared):.4f}")
        for k in shots:
            for d in range(draws):
                rng = np.random.default_rng(10_000 * f + 100 * k + d)
                sup = np.concatenate([rng.choice(np.where(y == c)[0], size=k, replace=False)
                                      for c in shared if (y == c).sum() > k])
                qry = np.setdiff1d(np.arange(len(y)), sup)
                res = {"zero-shot": zero_pred[qry]}
                # prototypes (cosine)
                Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
                sup_cls = [c for c in shared if (y[sup] == c).any()]
                protos = np.stack([Xn[sup][y[sup] == c].mean(0) for c in sup_cls])
                res["prototype"] = np.asarray(sup_cls)[(Xn[qry] @ protos.T).argmax(1)]
                # linear probe
                sc = StandardScaler().fit(X[sup])
                clf = LogisticRegression(max_iter=2000, C=1.0)
                clf.fit(sc.transform(X[sup]), y[sup])
                res["linear probe"] = clf.predict(sc.transform(X[qry]))
                for method, pred in res.items():
                    rows.append({"fold": f, "shots": k, "draw": d, "method": method,
                                 "macro_f1": _f1(y[qry], pred, shared),
                                 "accuracy": float((pred == y[qry]).mean())})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not rows:
        raise RuntimeError("No WF-Net checkpoints found in " + run_dir)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "adaptation_runs.csv"), index=False)
    s = (df.groupby(["method", "shots"]).agg(macro_f1_mean=("macro_f1", "mean"),
                                             macro_f1_std=("macro_f1", "std"),
                                             accuracy_mean=("accuracy", "mean"))
         .reset_index())
    s.to_csv(os.path.join(out_dir, "adaptation_summary.csv"), index=False)

    plt.figure(figsize=(6.5, 4.2))
    for method, g in s.groupby("method"):
        plt.errorbar(g.shots, g.macro_f1_mean, yerr=g.macro_f1_std, fmt="o-", capsize=3,
                     label=method)
    plt.xscale("log")
    plt.xticks(list(shots), [str(k) for k in shots])
    plt.xlabel("labelled Garbage12 images per class (k)")
    plt.ylabel("macro-F1 on the remaining images")
    plt.title("Few-shot adaptation of frozen WF-Net to Garbage12")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    save_fig(os.path.join(out_dir, "adaptation_curve.png"))

    lines = ["Few-shot adaptation to Garbage12 (frozen WF-Net; mean ± std over "
             f"folds x {draws} draws; query = images not used as support)", ""]
    for r in s.itertuples():
        lines.append(f"  {r.method:13s} k={r.shots:<3d} macro-F1 {r.macro_f1_mean:.4f} "
                     f"± {r.macro_f1_std:.4f}  acc {r.accuracy_mean:.4f}")
    txt = "\n".join(lines)
    with open(os.path.join(out_dir, "adaptation_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    log(txt)
