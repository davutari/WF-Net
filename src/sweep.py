"""
Sensitivity of WF-Net to the self-distillation hyper-parameters.

One-at-a-time (OAT) design around the default (AUX_WEIGHT, KD_WEIGHT,
KD_TEMPERATURE): the default plus two alternative levels of each parameter
(7 settings). Each setting is trained on the same folds of the main split with
the same per-fold seed, so differences come from the hyper-parameters only.
Every finished (setting, fold) is written to its own crash-safe file and an
interrupted training continues from its last saved epoch -> resumable.
"""
import collections
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from src.viz import save_fig


def oat_configs(levels, default):
    out = [dict(default)]
    for key, vals in levels.items():
        for v in vals:
            if v != default[key]:
                c = dict(default)
                c[key] = v
                out.append(c)
    return out


def cfg_id(c):
    return f"aux{c['AUX_WEIGHT']}_kd{c['KD_WEIGHT']}_T{c['KD_TEMPERATURE']}"


def summarize(out_dir, levels, default, log=print):
    csv = os.path.join(out_dir, "sweep_distill.csv")
    df = pd.read_csv(csv)
    g = (df.groupby(["config", "AUX_WEIGHT", "KD_WEIGHT", "KD_TEMPERATURE"])
         .agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
              accuracy_mean=("accuracy", "mean"), n_folds=("fold", "nunique"))
         .reset_index().sort_values("macro_f1_mean", ascending=False))
    g.to_csv(os.path.join(out_dir, "sweep_distill_summary.csv"), index=False)
    spread = g.macro_f1_mean.max() - g.macro_f1_mean.min()
    txt = g.to_string(index=False) + f"\n\nRange of mean macro-F1 across settings: {spread:.4f}"
    with open(os.path.join(out_dir, "sweep_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    log(txt)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, key in zip(axes, levels):
        others = [k for k in levels if k != key]
        sub = g[(g[others[0]] == default[others[0]]) & (g[others[1]] == default[others[1]])]
        sub = sub.sort_values(key)
        ax.errorbar(sub[key], sub.macro_f1_mean, yerr=sub.macro_f1_std.fillna(0),
                    fmt="o-", capsize=4, color="#2b5d9b")
        ax.axvline(default[key], color="grey", ls=":", lw=1)
        ax.set_xlabel(key)
        ax.set_title(f"vary {key} (others at default)", fontsize=9)
    axes[0].set_ylabel("macro-F1 (mean ± std over folds)")
    plt.tight_layout()
    save_fig(os.path.join(out_dir, "sweep_distill.png"))


def _collect(out_dir):
    """Rebuild sweep_distill.csv from the per-job result files (crash-safe)."""
    import glob
    import json
    rows = []
    for fp in sorted(glob.glob(os.path.join(out_dir, "jobs", "*.json"))):
        try:
            with open(fp, encoding="utf-8") as fh:
                rows.append(json.load(fh))
        except Exception:
            pass                                  # unreadable -> job is redone
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(os.path.join(out_dir, "sweep_distill.csv"), index=False)
    return df


def run_sweep(out_dir, seed, folds_to_run, levels, epochs=None, debug_per_class=None,
              max_configs=None, log=print):
    from src.data import discover_dataset, make_folds
    from src.engine import train_one_fold
    from src.runner import _subsample, _set_dirs, job_seed
    from src.utils import durable_json, seed_everything, write_run_snapshot

    default = {"AUX_WEIGHT": config.AUX_WEIGHT, "KD_WEIGHT": config.KD_WEIGHT,
               "KD_TEMPERATURE": config.KD_TEMPERATURE}
    cfgs = oat_configs(levels, default)[:max_configs]
    _set_dirs(out_dir)
    jobs_dir = os.path.join(out_dir, "jobs")
    os.makedirs(jobs_dir, exist_ok=True)
    prev = _collect(out_dir)
    done = set(zip(prev.config, prev.fold)) if len(prev) else set()
    todo = [(c, f) for c in cfgs for f in folds_to_run if (cfg_id(c), f) not in done]
    log(f"[sweep] {len(cfgs)} settings x {len(folds_to_run)} folds; {len(todo)} remaining")

    if todo:
        config.SEED, config.N_FOLDS = seed, 5                 # main split
        if epochs:
            config.EPOCHS = epochs
        write_run_snapshot(config, os.path.join(out_dir, "run_config.txt"))
        paths, labels, class_names = discover_dataset(config.DATA_DIR)
        if debug_per_class:
            paths, labels = _subsample(paths, labels, debug_per_class, seed)
        counts = collections.Counter(labels)
        class_counts = [counts[i] for i in range(len(class_names))]
        folds = list(make_folds(labels))
        try:
            for i, (c, f) in enumerate(todo, 1):
                for k, v in c.items():
                    setattr(config, k, v)       # read at call time by distillation_loss
                seed_everything(job_seed(seed, config.PROPOSED_NAME, f), config.DETERMINISTIC)
                tr, val, te = folds[f]
                stem = os.path.join(jobs_dir, f"{cfg_id(c)}_fold{f}")
                metrics, *_ = train_one_fold(config.PROPOSED_NAME, paths, labels, tr, val, te,
                                             class_counts, len(class_names), f,
                                             epochs=config.EPOCHS, cross_targets=None,
                                             state_path=stem + "_state.pt")
                row = {"config": cfg_id(c), "fold": f, **c,
                       "macro_f1": float(metrics["macro_f1"]),
                       "accuracy": float(metrics["accuracy"]),
                       "minutes": round(float(metrics["train_time_s"]) / 60, 2)}
                durable_json(stem + ".json", row)            # done marker
                if os.path.exists(stem + "_state.pt"):
                    os.remove(stem + "_state.pt")
                log(f"[sweep {i}/{len(todo)}] {cfg_id(c)} fold{f}: "
                    f"macro-F1={row['macro_f1']:.4f} ({row['minutes']} min)")
        finally:
            for k, v in default.items():          # always restore the defaults
                setattr(config, k, v)
    _collect(out_dir)
    summarize(out_dir, levels, default, log)
