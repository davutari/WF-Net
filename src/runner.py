"""
Resumable cross-validation runner.

Every (model, fold) job is independent: it gets its own deterministic seed
(derived from the base seed, the model name and the fold), and its outputs are
written to  <out>/folds/<model>/fold<k>/  as soon as it finishes. The file
metrics.json is written LAST and atomically, so it marks a finished job: when the
runner is restarted after an interruption, finished jobs are skipped and the
run continues where it stopped. After all jobs, `aggregate()` rebuilds every
table/figure from disk (also usable on its own).

Because jobs share no state, they could equally be dispatched in parallel on
several GPUs / nodes -- the property used in the HPC discussion of the paper.
"""
import collections
import json
import os
import time
import zlib
from datetime import datetime

import numpy as np
import pandas as pd
import torch

import config
from src import viz
from src.data import discover_dataset, make_folds
from src.metrics import full_report, get_confusion
from src.stats import critical_difference, friedman_nemenyi
from src.utils import (durable_json, durable_savez, durable_torch_save,
                       seed_everything, write_run_snapshot)

METRIC_KEYS = ["accuracy", "precision_macro", "recall_macro", "macro_f1",
               "weighted_f1", "cohen_kappa", "mcc", "roc_auc_macro"]


def safe(name):
    return "".join(c if c.isalnum() else "_" for c in name)


def job_seed(base_seed, model, fold):
    """Deterministic, order-independent seed for one (model, fold) job."""
    return int(base_seed) * 1000 + fold * 97 + zlib.crc32(model.encode()) % 89


def fmt_hms(seconds):
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")


def job_is_valid(jdir, cross_names, ckpt=None):
    """(ok, reason) -- a finished job must have a readable metrics.json, loadable
    prediction files (in-domain + every external target) and, if required, its
    checkpoint. Anything damaged by a power cut is detected here and redone."""
    mpath = os.path.join(jdir, "metrics.json")
    if not os.path.exists(mpath):
        return False, "not started"
    try:
        with open(mpath, encoding="utf-8") as fh:
            rec = json.load(fh)
        assert "macro_f1" in rec
        d = np.load(os.path.join(jdir, "preds.npz"), allow_pickle=True)
        assert len(d["y_true"]) == len(d["y_prob"]) == len(d["test_paths"]) > 0
        for ds in cross_names:
            c = np.load(os.path.join(jdir, f"cross_{safe(ds)}.npz"))
            assert len(c["y_true"]) == len(c["y_prob"]) > 0
        if ckpt is not None:
            assert os.path.exists(ckpt) and os.path.getsize(ckpt) > 1_000_000
    except Exception as e:
        return False, f"incomplete/damaged files ({type(e).__name__})"
    return True, "ok"


def _set_dirs(out_dir):
    config.RESULTS_DIR = out_dir
    config.FIG_DIR = os.path.join(out_dir, "figures")
    config.CKPT_DIR = os.path.join(out_dir, "checkpoints")
    for d in (out_dir, config.FIG_DIR, config.CKPT_DIR):
        os.makedirs(d, exist_ok=True)


def _subsample(paths, labels, per_class, seed):
    """Debug only: keep `per_class` images of every class."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == c]
        keep += sorted(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
    return [paths[i] for i in keep], [labels[i] for i in keep]


def build_cross_targets(class_names, data_dir, enabled, limit=None, log=print):
    from src.cross_dataset import build_target
    targets = {}
    if not enabled:
        return targets
    for name, tdir in config.CROSS_DATASETS.items():
        if not tdir or not os.path.isdir(tdir):
            log(f"[cross] '{name}' not found, skipped: {tdir}")
            continue
        if os.path.abspath(tdir) == os.path.abspath(data_dir):
            log(f"[cross] '{name}' is the training set itself, skipped.")
            continue
        paths, labels, shared = build_target(tdir, class_names, config.CROSS_LABEL_MAPS[name])
        if not paths:
            log(f"[cross] '{name}': no mappable classes, skipped.")
            continue
        if limit:
            rng = np.random.default_rng(0)
            sel = sorted(rng.choice(len(paths), size=min(limit, len(paths)), replace=False))
            paths, labels = [paths[i] for i in sel], [labels[i] for i in sel]
        log(f"[cross] '{name}': {len(paths)} images, {len(shared)} shared classes")
        targets[name] = {"paths": paths, "labels": labels, "shared": shared}
    return targets


# --------------------------------------------------------------------------- #
def run_cv(out_dir, models, seed, n_folds=5, epochs=None, data_dir=None, cross=True,
           num_workers=None, ckpt_models=None, debug_per_class=None, cross_limit=None,
           log=print):
    """Train/evaluate every (model, fold) job that is not finished yet, then aggregate."""
    from src.engine import train_one_fold

    data_dir = data_dir or config.DATA_DIR
    config.SEED = seed
    config.N_FOLDS = n_folds
    if epochs:
        config.EPOCHS = epochs
    if num_workers is not None:
        config.NUM_WORKERS = num_workers
    ckpt_models = set(ckpt_models if ckpt_models is not None else [config.PROPOSED_NAME])
    _set_dirs(out_dir)
    write_run_snapshot(config, os.path.join(out_dir, "run_config.txt"))

    paths, labels, class_names = discover_dataset(data_dir)
    if debug_per_class:
        paths, labels = _subsample(paths, labels, debug_per_class, seed)
    n_classes = len(class_names)
    counts = collections.Counter(labels)
    class_counts = [counts[i] for i in range(n_classes)]
    folds = list(make_folds(labels))[:n_folds]
    log(f"Data: {data_dir} | {len(paths)} images, {n_classes} classes | "
        f"seed {seed} | {n_folds} folds | max epochs {config.EPOCHS}")
    cross_targets = build_cross_targets(class_names, data_dir, cross, cross_limit, log)

    meta = {"data_dir": data_dir, "seed": seed, "n_folds": n_folds, "models": models,
            "class_names": class_names, "n_images": len(paths),
            "cross": {k: {"n": len(v["paths"]), "shared": list(map(int, v["shared"]))}
                      for k, v in cross_targets.items()}}
    durable_json(os.path.join(out_dir, "run_meta.json"), meta)

    total = len(models) * len(folds)
    done_n = 0
    for m in models:
        for fid, (tr, val, te) in enumerate(folds):
            jdir = os.path.join(out_dir, "folds", safe(m), f"fold{fid}")
            ckpt = os.path.join(config.CKPT_DIR, f"ckpt_{safe(m)}_fold{fid}.pt")
            done_n += 1
            ok, why = job_is_valid(jdir, list(cross_targets), ckpt if m in ckpt_models else None)
            if ok:
                log(f"[{done_n}/{total}] {m} fold{fid}: already finished -> skipped")
                continue
            if why != "not started":
                log(f"[{done_n}/{total}] {m} fold{fid}: {why} -> job is (re)run")
                if os.path.exists(os.path.join(jdir, "metrics.json")):
                    os.remove(os.path.join(jdir, "metrics.json"))
            os.makedirs(jdir, exist_ok=True)
            state_path = os.path.join(jdir, "train_state.pt")
            seed_everything(job_seed(seed, m, fid), config.DETERMINISTIC)
            t0 = time.time()
            metrics, y_true, y_pred, y_prob, state, cross_res = train_one_fold(
                m, paths, labels, tr, val, te, class_counts, n_classes, fid,
                epochs=config.EPOCHS, cross_targets=cross_targets, state_path=state_path)
            durable_savez(os.path.join(jdir, "preds.npz"),
                          y_true=y_true, y_pred=y_pred, y_prob=y_prob,
                          test_idx=np.asarray(te),
                          test_paths=np.array([paths[i] for i in te]))
            cross_metrics = {}
            for ds, cres in cross_res.items():
                durable_savez(os.path.join(jdir, f"cross_{safe(ds)}.npz"),
                              y_true=cres["y_true"], y_pred=cres["y_pred"],
                              y_prob=cres["y_prob"],
                              shared=np.array(cross_targets[ds]["shared"]))
                cross_metrics[ds] = {k: float(cres["metrics"].get(k, float("nan")))
                                     for k in METRIC_KEYS}
            if m in ckpt_models and state is not None:
                durable_torch_save(state, ckpt)
            del state
            rec = {"model": m, "fold": fid,
                   **{k: float(metrics.get(k, float("nan"))) for k in METRIC_KEYS},
                   "n_params": int(metrics["n_params"]),
                   "train_time_s": float(metrics["train_time_s"]),
                   # compute time accumulated across resumes (training + test + external eval)
                   "wall_s": float(metrics["train_time_s"]) + float(metrics.get("cross_time_s", 0.0)),
                   "session_s": time.time() - t0, "cross": cross_metrics,
                   "history": metrics.get("history", []),
                   "finished": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            durable_json(os.path.join(jdir, "metrics.json"), rec)      # done marker (last)
            if os.path.exists(state_path):          # training state no longer needed
                os.remove(state_path)
            msg = (f"[{done_n}/{total}] {m} fold{fid}: macro-F1={rec['macro_f1']:.4f} "
                   f"acc={rec['accuracy']:.4f} ({fmt_hms(rec['wall_s'])})")
            for ds, cm_ in cross_metrics.items():
                msg += f" | {ds} macro-F1={cm_['macro_f1']:.4f}"
            log(msg)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    aggregate(out_dir, log=log)


# --------------------------------------------------------------------------- #
def _load_jobs(out_dir, models, n_folds):
    recs, missing = [], []
    for m in models:
        for f in range(n_folds):
            p = os.path.join(out_dir, "folds", safe(m), f"fold{f}", "metrics.json")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as fh:
                    recs.append(json.load(fh))
            else:
                missing.append((m, f))
    return recs, missing


def aggregate(out_dir, log=print):
    """Rebuild all tables, figures and pooled prediction files from the job folders."""
    with open(os.path.join(out_dir, "run_meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    _set_dirs(out_dir)
    models, n_folds, class_names = meta["models"], meta["n_folds"], meta["class_names"]
    n_classes = len(class_names)
    recs, missing = _load_jobs(out_dir, models, n_folds)
    if missing:
        log(f"[aggregate] {len(missing)} unfinished job(s) -> partial tables: {missing[:5]}")
    complete = [m for m in models if all((m, f) not in missing for f in range(n_folds))]
    if not complete:
        log("[aggregate] nothing complete yet.")
        return

    pf = pd.DataFrame([{k: r[k] for k in ["model", "fold", *METRIC_KEYS, "n_params",
                                           "train_time_s", "wall_s"]} for r in recs])
    pf = pf[pf.model.isin(complete)]
    pf.to_csv(os.path.join(out_dir, "per_fold_metrics.csv"), index=False)

    summary = []
    for m in complete:
        sub = pf[pf.model == m]
        rec = {"model": m}
        for k in METRIC_KEYS:
            rec[f"{k}_mean"] = sub[k].mean()
            rec[f"{k}_std"] = sub[k].std()
        rec["n_params_M"] = sub.n_params.iloc[0] / 1e6
        rec["train_time_s_mean"] = sub.train_time_s.mean()
        summary.append(rec)
    summary_df = pd.DataFrame(summary).sort_values("macro_f1_mean", ascending=False)
    summary_df.to_csv(os.path.join(out_dir, "summary_meanstd.csv"), index=False)
    viz.plot_model_comparison(summary_df, "macro_f1",
                              os.path.join(config.FIG_DIR, "comparison_macro_f1.png"))
    lines = ["===== SUMMARY (macro-F1 mean ± std over folds) ====="]
    lines += [f"  {r.model:36s} {r.macro_f1_mean:.4f} ± {r.macro_f1_std:.4f}"
              for r in summary_df.itertuples()]

    # pooled per-model predictions, confusion matrices, reports, ROC curves
    roc_curves = []
    cross_pool = collections.defaultdict(dict)
    for m in complete:
        yt, yp, ypr, fo = [], [], [], []
        cpool = collections.defaultdict(lambda: {"y_true": [], "y_pred": [], "y_prob": [],
                                                 "fold": [], "shared": None})
        for f in range(n_folds):
            jdir = os.path.join(out_dir, "folds", safe(m), f"fold{f}")
            d = np.load(os.path.join(jdir, "preds.npz"), allow_pickle=True)
            yt += d["y_true"].tolist()
            yp += d["y_pred"].tolist()
            ypr += d["y_prob"].tolist()
            fo += [f] * len(d["y_true"])
            for ds in meta["cross"]:
                cp = os.path.join(jdir, f"cross_{safe(ds)}.npz")
                if os.path.exists(cp):
                    c = np.load(cp)
                    pool = cpool[ds]
                    pool["y_true"] += c["y_true"].tolist()
                    pool["y_pred"] += c["y_pred"].tolist()
                    pool["y_prob"] += c["y_prob"].tolist()
                    pool["fold"] += [f] * len(c["y_true"])
                    pool["shared"] = c["shared"]
        np.savez_compressed(os.path.join(out_dir, f"preds_{safe(m)}.npz"),
                            y_true=np.array(yt), y_pred=np.array(yp), y_prob=np.array(ypr),
                            fold=np.array(fo), class_names=np.array(class_names))
        viz.plot_confusion(get_confusion(yt, yp, n_classes), class_names,
                           f"Confusion Matrix - {m}",
                           os.path.join(config.FIG_DIR, f"cm_{safe(m)}.png"))
        with open(os.path.join(out_dir, f"report_{safe(m)}.txt"), "w", encoding="utf-8") as fh:
            fh.write(full_report(yt, yp, class_names))
        fpr, tpr, auc_ = viz.plot_roc_curves(yt, ypr, class_names, f"ROC - {m}",
                                             os.path.join(config.FIG_DIR, f"roc_{safe(m)}.png"))
        roc_curves.append((m, fpr, tpr, auc_))
        for ds, pool in cpool.items():
            if pool["y_true"]:
                np.savez_compressed(
                    os.path.join(out_dir, f"crosspreds_{safe(ds)}_{safe(m)}.npz"),
                    y_true=np.array(pool["y_true"]), y_pred=np.array(pool["y_pred"]),
                    y_prob=np.array(pool["y_prob"]), fold=np.array(pool["fold"]),
                    shared=pool["shared"], class_names=np.array(class_names))
                cross_pool[ds][m] = pool
    viz.plot_roc_comparison(roc_curves, os.path.join(config.FIG_DIR, "roc_comparison.png"))

    # external (cross-dataset) tables
    for ds in meta["cross"]:
        rows = [{"model": r["model"], "fold": r["fold"],
                 **{k: r["cross"][ds][k] for k in METRIC_KEYS}}
                for r in recs if r["model"] in complete and ds in r.get("cross", {})]
        if not rows:
            continue
        cdf = pd.DataFrame(rows)
        cdf.to_csv(os.path.join(out_dir, f"cross_{safe(ds)}_per_fold.csv"), index=False)
        cs = (cdf.groupby("model")[METRIC_KEYS].agg(["mean", "std"]))
        cs.columns = [f"{a}_{b}" for a, b in cs.columns]
        cs = cs.reset_index().sort_values("macro_f1_mean", ascending=False)
        cs.to_csv(os.path.join(out_dir, f"cross_{safe(ds)}_summary.csv"), index=False)
        viz.plot_model_comparison(cs, "macro_f1",
                                  os.path.join(config.FIG_DIR, f"cross_{safe(ds)}_macro_f1.png"))
        lines += ["", f"===== EXTERNAL [{ds}] macro-F1 mean ± std ====="]
        lines += [f"  {r.model:36s} {r.macro_f1_mean:.4f} ± {r.macro_f1_std:.4f}"
                  for r in cs.itertuples()]
        prop = cross_pool[ds].get(config.PROPOSED_NAME)
        if prop:
            shared = [int(s) for s in prop["shared"]]
            yt, yp = np.array(prop["y_true"]), np.array(prop["y_pred"])
            cm = get_confusion(yt, yp, n_classes)[np.ix_(shared, shared)]
            viz.plot_confusion(cm, [class_names[i] for i in shared],
                               f"Cross-dataset Confusion - {ds} (WF-Net)",
                               os.path.join(config.FIG_DIR, f"cross_{safe(ds)}_cm.png"))
            viz.plot_roc_curves(yt, np.array(prop["y_prob"]), class_names,
                                f"Cross-dataset ROC - {ds} (WF-Net)",
                                os.path.join(config.FIG_DIR, f"cross_{safe(ds)}_roc.png"))
            with open(os.path.join(out_dir, f"cross_{safe(ds)}_report.txt"), "w",
                      encoding="utf-8") as fh:
                fh.write(full_report(yt, yp, class_names))
        _cross_per_class(out_dir, ds, cross_pool[ds], class_names)

    # learning curves (over-fitting diagnostics)
    _learning_curves(out_dir, recs, complete)

    # significance (Friedman / Nemenyi / CD diagram)
    if len(complete) >= 3:
        f1 = pd.DataFrame({m: pf[pf.model == m].sort_values("fold").macro_f1.values
                           for m in complete})
        stat, p, nemenyi, ranks = friedman_nemenyi(f1)
        cd = critical_difference(len(complete), n_folds)
        with open(os.path.join(out_dir, "significance.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"Friedman chi2 = {stat:.4f}, p = {p:.6f}\n")
            fh.write(f"Critical difference (Nemenyi, a=0.05) = {cd:.4f}\n\n")
            fh.write("Mean ranks (lower = better):\n" + ranks.to_string() + "\n")
            if nemenyi is not None:
                fh.write("\nNemenyi post-hoc p-values:\n" + nemenyi.round(4).to_string() + "\n")
        if nemenyi is not None:
            nemenyi.to_csv(os.path.join(out_dir, "nemenyi_pvalues.csv"))
        try:
            viz.plot_cd_diagram(ranks, cd, os.path.join(config.FIG_DIR, "cd_diagram.png"))
        except Exception as e:                     # figure only; never fatal
            log(f"[cd-diagram] skipped: {e}")
        lines += ["", f"Friedman chi2={stat:.3f}, p={p:.5f} | Nemenyi CD={cd:.3f}"]

    # Grad-CAM panel for the proposed model (last fold checkpoint)
    if config.PROPOSED_NAME in complete:
        try:
            _gradcam(out_dir, n_folds - 1, class_names)
        except Exception as e:
            log(f"[gradcam] skipped: {e}")

    # timing (sum of job wall times per model)
    t = pf.groupby("model").wall_s.sum().reindex(complete)
    trows = [{"model": m, "duration_hms": fmt_hms(s), "seconds": round(s, 1)}
             for m, s in t.items()]
    trows.append({"model": "TOTAL (compute)", "duration_hms": fmt_hms(t.sum()),
                  "seconds": round(float(t.sum()), 1)})
    pd.DataFrame(trows).to_csv(os.path.join(out_dir, "timing.csv"), index=False)
    lines += ["", "===== TIMING (sum of job wall-clock times) ====="]
    lines += [f"  {r['model']:36s} {r['duration_hms']:>14s}" for r in trows]

    report = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    log(report)


def _cross_per_class(out_dir, ds, pools, class_names):
    """Per-class F1 on the external set for every model."""
    from sklearn.metrics import f1_score
    rows = []
    for m, pool in pools.items():
        shared = [int(s) for s in pool["shared"]]
        f1 = f1_score(pool["y_true"], pool["y_pred"], labels=shared, average=None,
                      zero_division=0)
        rows.append({"model": m, **{class_names[c]: round(float(v), 4)
                                    for c, v in zip(shared, f1)}})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"cross_{safe(ds)}_per_class_f1.csv"),
                                  index=False)


def _learning_curves(out_dir, recs, complete):
    """Per-epoch training loss / validation macro-F1 of every job."""
    import matplotlib.pyplot as plt
    rows = [{"model": r["model"], "fold": r["fold"], **h}
            for r in recs if r["model"] in complete for h in r.get("history", [])]
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "learning_curves.csv"), index=False)
    show = [m for m in [config.PROPOSED_NAME, "Ablation: Dual (concat)",
                        "Ablation: CATF cross-attention", "Swin-T"] if m in set(df.model)]
    if not show:
        show = list(df.model.unique())[:4]
    fig, ax = plt.subplots(2, len(show), figsize=(4.2 * len(show), 6.2), squeeze=False)
    for j, m in enumerate(show):
        sub = df[df.model == m]
        for f, g in sub.groupby("fold"):
            ax[0, j].plot(g.epoch, g.train_loss, color="#9db4d6", lw=0.9)
            ax[1, j].plot(g.epoch, g.val_macro_f1, color="#9db4d6", lw=0.9)
        mean = sub.groupby("epoch")[["train_loss", "val_macro_f1"]].mean()
        ax[0, j].plot(mean.index, mean.train_loss, color="#1f3b73", lw=2, label="mean")
        ax[1, j].plot(mean.index, mean.val_macro_f1, color="#1f3b73", lw=2, label="mean")
        ax[0, j].axvline(config.FREEZE_EPOCHS + 0.5, color="grey", ls=":", lw=1)
        ax[1, j].axvline(config.FREEZE_EPOCHS + 0.5, color="grey", ls=":", lw=1)
        ax[0, j].set_title(m, fontsize=9)
        ax[1, j].set_xlabel("epoch")
    ax[0, 0].set_ylabel("training loss (MixUp/CutMix)")
    ax[1, 0].set_ylabel("validation macro-F1")
    ax[0, 0].legend(fontsize=7)
    plt.tight_layout()
    viz.save_fig(os.path.join(config.FIG_DIR, "learning_curves.png"))


def _gradcam(out_dir, fold, class_names, k=6):
    from PIL import Image
    from src.data import build_transforms
    from src.gradcam import GradCAM, denormalize
    from src.models import WFNet
    ck = os.path.join(config.CKPT_DIR, f"ckpt_{safe(config.PROPOSED_NAME)}_fold{fold}.pt")
    if not os.path.exists(ck):
        return
    d = np.load(os.path.join(out_dir, "folds", safe(config.PROPOSED_NAME), f"fold{fold}",
                             "preds.npz"), allow_pickle=True)
    model = WFNet(len(class_names), pretrained=False).to(config.DEVICE)
    model.load_state_dict(torch.load(ck, map_location=config.DEVICE, weights_only=True))
    cam_engine = GradCAM(model, model.gradcam_target_layer())
    tf = build_transforms(train=False)
    imgs, cams, preds, trues = [], [], [], []
    for p, y in list(zip(d["test_paths"], d["y_true"]))[:k]:
        x = tf(Image.open(str(p)).convert("RGB")).unsqueeze(0).to(config.DEVICE)
        x.requires_grad_(True)
        cam, pred = cam_engine(x)
        cams.append(cam[0])
        preds.append(int(pred[0]))
        trues.append(int(y))
        imgs.append(denormalize(x[0].detach(), config.MEAN, config.STD))
    viz.plot_gradcam_panel(imgs, cams, preds, trues, class_names,
                           os.path.join(config.FIG_DIR, "gradcam_proposed.png"))
