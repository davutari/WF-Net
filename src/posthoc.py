"""
Post-hoc statistical analysis (no training, CPU only).

Computes the following from the files a run already saved
(per_fold_metrics.csv, preds_*.npz, crosspreds_*.npz):

  * Bootstrap 95% CIs of macro-F1 / accuracy for every model
  * Paired comparisons of the reference model against every other one:
      - paired bootstrap CI of the macro-F1 / accuracy difference
      - McNemar test on per-image correctness (uses all N images)
      - Wilcoxon signed-rank on per-fold macro-F1
      - corrected resampled t-test for (repeated) k-fold CV
        (Nadeau & Bengio 2003; Bouckaert & Frank 2004)
    with Holm-Bonferroni correction across the comparisons.
  * The same paired tests on the external (cross-dataset) set, e.g.
    concatenation vs CATF on Garbage12

Several runs (e.g. seeds 42 / 1337 / 2024 = repeated 3x5 CV) can be passed at
once; per-fold tests then pool all (repeat, fold) pairs.

Called from run_experiments.py:  run_posthoc([run_dir, ...], out_dir)
"""
import glob
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.viz import save_fig

REF_DEFAULT = "WF-Net (Proposed)"


def _safe(name):
    return "".join(c if c.isalnum() else "_" for c in name)


# --------------------------------------------------------------------------- #
# fast metrics (numpy) -- used inside the bootstrap loops
# --------------------------------------------------------------------------- #
def macro_f1(y, p, labels, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1[labels].mean())


def accuracy(y, p):
    return float((y == p).mean())


def bootstrap_ci(y, p, labels, k, idx_mat):
    """Percentile CI of macro-F1 and accuracy over bootstrap index matrix."""
    f1s = np.array([macro_f1(y[i], p[i], labels, k) for i in idx_mat])
    accs = np.array([accuracy(y[i], p[i]) for i in idx_mat])
    return (macro_f1(y, p, labels, k), np.percentile(f1s, 2.5), np.percentile(f1s, 97.5),
            accuracy(y, p), np.percentile(accs, 2.5), np.percentile(accs, 97.5))


def paired_bootstrap(y, pa, pb, labels, k, idx_mat):
    """CI + two-sided p of (a - b) macro-F1 and accuracy on the same resamples."""
    d_f1 = np.array([macro_f1(y[i], pa[i], labels, k) - macro_f1(y[i], pb[i], labels, k)
                     for i in idx_mat])
    d_acc = np.array([accuracy(y[i], pa[i]) - accuracy(y[i], pb[i]) for i in idx_mat])

    def _p(d):                                  # two-sided bootstrap p-value
        return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))

    return {"dF1": macro_f1(y, pa, labels, k) - macro_f1(y, pb, labels, k),
            "dF1_lo": np.percentile(d_f1, 2.5), "dF1_hi": np.percentile(d_f1, 97.5),
            "p_boot_F1": _p(d_f1),
            "dAcc": accuracy(y, pa) - accuracy(y, pb),
            "dAcc_lo": np.percentile(d_acc, 2.5), "dAcc_hi": np.percentile(d_acc, 97.5),
            "p_boot_acc": _p(d_acc)}


def mcnemar(y, pa, pb):
    """McNemar on discordant pairs; exact binomial when few discordants."""
    ca, cb = (pa == y), (pb == y)
    b = int(np.sum(ca & ~cb))          # a right, b wrong
    c = int(np.sum(~ca & cb))          # a wrong, b right
    if b + c == 0:
        return b, c, 1.0
    if b + c < 25:
        p = stats.binomtest(min(b, c), b + c, 0.5).pvalue
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p = stats.chi2.sf(chi2, 1)
    return b, c, float(p)


def wilcoxon_p(da, db):
    d = np.asarray(da) - np.asarray(db)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(da, db).pvalue)
    except ValueError:
        return float("nan")


def corrected_t(da, db, k):
    """Corrected resampled t-test for r x k-fold CV (Nadeau-Bengio correction).

    var is inflated by n_test/n_train = 1/(k-1) to account for the overlap of
    training sets across folds; df = r*k - 1.
    """
    d = np.asarray(da) - np.asarray(db)
    n = len(d)
    if n < 2 or np.allclose(d.var(ddof=1), 0):
        return float(d.mean()), float("nan")
    var = d.var(ddof=1)
    t = d.mean() / np.sqrt((1.0 / n + 1.0 / (k - 1)) * var)
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return float(d.mean()), float(p)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (NaNs kept as NaN)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_run(run_dir):
    pf = pd.read_csv(os.path.join(run_dir, "per_fold_metrics.csv"))
    preds = {}
    for f in glob.glob(os.path.join(run_dir, "preds_*.npz")):
        d = np.load(f, allow_pickle=True)
        preds[f] = d
    # map file -> model name via per_fold model list
    by_model = {}
    for m in pf.model.unique():
        f = os.path.join(run_dir, f"preds_{_safe(m)}.npz")
        if f in preds:
            by_model[m] = preds[f]
    cross = {}
    # longest matching model name wins, so similarly named models never collide
    names = sorted(pf.model.unique(), key=lambda n: len(_safe(n)), reverse=True)
    for f in glob.glob(os.path.join(run_dir, "crosspreds_*.npz")):
        base = os.path.basename(f)[len("crosspreds_"):-len(".npz")]
        m = next((n for n in names if base.endswith("_" + _safe(n))), None)
        if m is not None:
            ds = base[: -len("_" + _safe(m))]
            cross.setdefault(ds, {})[m] = np.load(f, allow_pickle=True)
    return pf, by_model, cross


def fold_ensemble(d):
    """Average fold-models' probabilities per external image -> one pred each."""
    fold = d["fold"]
    folds = np.unique(fold)
    n = int((fold == folds[0]).sum())
    prob = np.mean([d["y_prob"][fold == f] for f in folds], axis=0)
    y = d["y_true"][fold == folds[0]]
    assert len(y) == n
    return y.astype(int), prob.argmax(1).astype(int)


# --------------------------------------------------------------------------- #
def run_posthoc(runs, out, ref=REF_DEFAULT, boot=2000, seed=0, log=print):
    """Statistics over one run (all its models) or several runs (repeated CV)."""
    from types import SimpleNamespace
    args = SimpleNamespace(runs=list(runs), ref=ref, boot=boot, out=out, seed=seed)

    rng = np.random.default_rng(args.seed)
    out = args.out or os.path.join(args.runs[0], "posthoc")
    os.makedirs(out, exist_ok=True)
    lines = [f"Post-hoc statistics  ({datetime.now():%Y-%m-%d %H:%M})",
             f"Runs ({len(args.runs)}): " + ", ".join(os.path.basename(os.path.normpath(r))
                                                     for r in args.runs),
             f"Reference model: {args.ref} | bootstrap resamples: {args.boot}", ""]

    runs = [load_run(r) for r in args.runs]
    models = sorted(set.intersection(*[set(pf.model.unique()) for pf, _, _ in runs]))
    if args.ref not in models:
        raise ValueError(f"Reference '{args.ref}' not in all runs. Models: {models}")
    others = [m for m in models if m != args.ref]

    # ---- per-fold table pooled over repeats: key = (repeat, fold) ----
    pf_all = pd.concat([pf.assign(repeat=i) for i, (pf, _, _) in enumerate(runs)])
    k_folds = int(pf_all.fold.nunique())
    n_pairs = len(pf_all[pf_all.model == args.ref])
    min_p = 2 * 0.5 ** n_pairs
    lines.append(f"Per-fold pairs per comparison: {n_pairs} "
                 f"({len(runs)} repeat(s) x {k_folds} folds). Smallest two-sided "
                 f"Wilcoxon p attainable with n={n_pairs}: {min_p:.4g}"
                 + ("  <-- cannot reach 0.05; use repeated CV / McNemar / bootstrap."
                    if min_p > 0.05 else ""))
    lines.append("")

    def fold_vec(m, col="macro_f1"):
        s = pf_all[pf_all.model == m].sort_values(["repeat", "fold"])
        return s[col].values

    # ---- bootstrap CIs (primary run = first) ----
    _, preds0, cross0 = runs[0]
    any_d = preds0[args.ref]
    y0 = any_d["y_true"].astype(int)
    k = len(any_d["class_names"])
    labels = np.arange(k)
    idx_mat = rng.integers(0, len(y0), size=(args.boot, len(y0)))

    ci_rows = []
    for m in models:
        d = preds0[m]
        assert np.array_equal(d["y_true"].astype(int), y0), f"sample order differs: {m}"
        f1, f1lo, f1hi, acc, alo, ahi = bootstrap_ci(y0, d["y_pred"].astype(int),
                                                    labels, k, idx_mat)
        row = {"model": m, "macro_f1_pooled": f1, "f1_ci_lo": f1lo, "f1_ci_hi": f1hi,
               "accuracy_pooled": acc, "acc_ci_lo": alo, "acc_ci_hi": ahi,
               "macro_f1_foldmean": fold_vec(m).mean(),
               "macro_f1_foldstd": fold_vec(m).std(ddof=1)}
        if len(runs) > 1:          # pooled F1 of every repeat -> spread across seeds
            reps = [macro_f1(r[1][m]["y_true"].astype(int),
                             r[1][m]["y_pred"].astype(int), labels, k) for r in runs]
            row["macro_f1_repeat_mean"] = float(np.mean(reps))
            row["macro_f1_repeat_std"] = float(np.std(reps, ddof=1))
        ci_rows.append(row)
    ci = pd.DataFrame(ci_rows).sort_values("macro_f1_pooled", ascending=False)
    ci.to_csv(os.path.join(out, "bootstrap_ci.csv"), index=False)
    lines.append("== Bootstrap 95% CI (pooled out-of-fold predictions, run 1) ==")
    for _, r in ci.iterrows():
        lines.append(f"  {r.model:36s} macro-F1 {r.macro_f1_pooled:.4f} "
                     f"[{r.f1_ci_lo:.4f}, {r.f1_ci_hi:.4f}]   acc {r.accuracy_pooled:.4f} "
                     f"[{r.acc_ci_lo:.4f}, {r.acc_ci_hi:.4f}]")
    lines.append("")

    # ---- paired comparisons: ref vs each other model (in-domain) ----
    pref = preds0[args.ref]["y_pred"].astype(int)
    rows = []
    for m in others:
        pb = preds0[m]["y_pred"].astype(int)
        r = {"comparison": f"{args.ref} vs {m}", "other": m}
        r.update(paired_bootstrap(y0, pref, pb, labels, k, idx_mat))
        b, c, p_mc = mcnemar(y0, pref, pb)
        r.update({"mcnemar_ref_only_correct": b, "mcnemar_other_only_correct": c,
                  "p_mcnemar": p_mc})
        r["dF1_foldmean"], r["p_corrected_t"] = corrected_t(fold_vec(args.ref),
                                                            fold_vec(m), k_folds)
        r["p_wilcoxon"] = wilcoxon_p(fold_vec(args.ref), fold_vec(m))
        rows.append(r)
    pw = pd.DataFrame(rows)
    for col in ("p_boot_F1", "p_mcnemar", "p_corrected_t", "p_wilcoxon"):
        pw[col + "_holm"] = holm(pw[col].values)
    pw = pw.sort_values("dF1")
    pw.to_csv(os.path.join(out, "pairwise_tests.csv"), index=False)

    lines.append(f"== Paired tests: {args.ref} minus other (in-domain) ==")
    lines.append("   dF1 [95% CI] | p_boot | p_McNemar | p_corr-t | p_Wilcoxon  "
                 "(Holm-adjusted in CSV)")
    for _, r in pw.iterrows():
        lines.append(f"  vs {r.other:34s} {r.dF1:+.4f} [{r.dF1_lo:+.4f},{r.dF1_hi:+.4f}] | "
                     f"{r.p_boot_F1:.4f} | {r.p_mcnemar:.4f} | {r.p_corrected_t:.4f} | "
                     f"{r.p_wilcoxon:.4f}")
    sd = pw[pw.other == "Ablation: WF-Net w/o self-distill"]
    if not sd.empty:
        r = sd.iloc[0]
        lines.append("")
        lines.append(f"  [self-distillation] dF1={r.dF1:+.4f} "
                     f"[{r.dF1_lo:+.4f},{r.dF1_hi:+.4f}], p_boot={r.p_boot_F1:.4f}, "
                     f"p_McNemar={r.p_mcnemar:.4f}, p_corr-t={r.p_corrected_t:.4f}, "
                     f"p_Wilcoxon={r.p_wilcoxon:.4f}")
    lines.append("")

    _forest(pw, args.ref, os.path.join(out, "paired_differences_forest.png"))

    # ---- external (cross-dataset) paired tests on the fold-ensemble ----
    for ds, mp in cross0.items():
        if args.ref not in mp:
            continue
        shared = mp[args.ref]["shared"].astype(int)
        y_ext, p_ref = fold_ensemble(mp[args.ref])
        idx_ext = rng.integers(0, len(y_ext), size=(args.boot, len(y_ext)))
        crow = []
        for m, d in mp.items():
            y_m, p_m = fold_ensemble(d)
            assert np.array_equal(y_m, y_ext), f"external order differs: {m}"
            rr = {"model": m, "ensemble_macro_f1_shared": macro_f1(y_ext, p_m, shared, k),
                  "ensemble_accuracy": accuracy(y_ext, p_m)}
            if m != args.ref:
                rr.update(paired_bootstrap(y_ext, p_ref, p_m, shared, k, idx_ext))
                b, c, p_mc = mcnemar(y_ext, p_ref, p_m)
                rr.update({"p_mcnemar": p_mc})
            crow.append(rr)
        cdf = pd.DataFrame(crow).sort_values("ensemble_macro_f1_shared", ascending=False)
        cfile = os.path.join(args.runs[0], f"cross_{ds}_per_fold.csv")
        if os.path.exists(cfile):          # per-fold Wilcoxon on the external set
            cpf = pd.read_csv(cfile)
            refv = cpf[cpf.model == args.ref].sort_values("fold").macro_f1.values
            cdf["p_wilcoxon_perfold"] = [
                wilcoxon_p(refv, cpf[cpf.model == m].sort_values("fold").macro_f1.values)
                if m != args.ref and len(cpf[cpf.model == m]) == len(refv) else np.nan
                for m in cdf.model]
        cdf.to_csv(os.path.join(out, f"cross_{ds}_pairwise_tests.csv"), index=False)
        lines.append(f"== External set '{ds}' (5-fold-ensemble predictions, "
                     f"{len(y_ext)} images, macro-F1 over {len(shared)} shared classes) ==")
        for _, r in cdf.iterrows():
            extra = ""
            if r.model != args.ref:
                extra = (f" | ref-minus {r.dF1:+.4f} [{r.dF1_lo:+.4f},{r.dF1_hi:+.4f}] "
                         f"p_boot={r.p_boot_F1:.4f} p_McNemar={r.p_mcnemar:.4f}")
            lines.append(f"  {r.model:36s} F1 {r.ensemble_macro_f1_shared:.4f} "
                         f"acc {r.ensemble_accuracy:.4f}{extra}")
        lines.append("")

    txt = "\n".join(lines)
    with open(os.path.join(out, "posthoc_summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    log(txt)
    log(f"Saved to: {out}")
    return out


def _forest(pw, ref, path):
    """Forest plot of paired macro-F1 differences (ref minus other) with 95% CI."""
    pw = pw.sort_values("dF1")
    y = np.arange(len(pw))
    plt.figure(figsize=(8, 0.42 * len(pw) + 1.5))
    err = [pw.dF1 - pw.dF1_lo, pw.dF1_hi - pw.dF1]
    sig = pw.p_boot_F1_holm < 0.05
    plt.errorbar(pw.dF1, y, xerr=err, fmt="o", color="#2b5d9b", ecolor="#7aa0cc",
                 capsize=3)
    plt.scatter(pw.dF1[sig], y[sig], color="#c0392b", zorder=3,
                label="significant (Holm-adjusted bootstrap p<0.05)")
    plt.axvline(0, color="grey", lw=1, ls=":")
    plt.yticks(y, pw.other, fontsize=8)
    plt.xlabel(f"macro-F1 difference ({ref} minus model), 95% paired bootstrap CI")
    plt.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    save_fig(path)

