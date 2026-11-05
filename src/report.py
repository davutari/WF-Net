"""
SUMMARY.md -- one page that collects the main numbers of the study, grouped by
topic. Rebuilt after every step from whatever results exist (missing ones are
marked "not available yet").
"""
import heapq
import os
from datetime import datetime

import pandas as pd

import config

NA = "_not available yet_"


def _csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def _txt(path):
    return open(path, encoding="utf-8").read() if os.path.exists(path) else None


def _row(df, model):
    if df is None or "model" not in df:
        return None
    sub = df[df.model == model]
    return sub.iloc[0] if len(sub) else None


def _pm(r, key):
    return f"{r[key + '_mean']:.4f} ± {r[key + '_std']:.4f}" if r is not None else NA


def _block(text, keep=None):
    if not text:
        return [NA]
    lines = text.strip().splitlines()
    if keep:
        lines = [ln for ln in lines if keep(ln)]
    return ["```"] + lines + ["```"]


def lpt_makespan(durations, k):
    """Longest-processing-time-first schedule of independent jobs on k workers."""
    loads = [0.0] * k
    heapq.heapify(loads)
    for d in sorted(durations, reverse=True):
        heapq.heappush(loads, heapq.heappop(loads) + d)
    return max(loads)


def _job_durations(dirs):
    """Measured compute time (s) of every independent training job of the study."""
    jobs = []
    for d in [dirs["main"], dirs["garbage12"]] + list(dirs["repeated_runs"]):
        pf = _csv(os.path.join(d, "per_fold_metrics.csv"))
        if pf is not None and "wall_s" in pf:
            jobs += pf.wall_s.tolist()
    sw = _csv(os.path.join(dirs["distillation_sensitivity"], "sweep_distill.csv"))
    if sw is not None and "minutes" in sw:
        jobs += (sw.minutes * 60).tolist()
    return jobs


def build_report(root, dirs, log=print):
    P = config.PROPOSED_NAME
    L = ["# Results summary", "",
         f"_Created: {datetime.now():%Y-%m-%d %H:%M}; source: `{root}`_", "",
         "The numbers are taken automatically from the CSV/TXT files in the result folders.", ""]

    main_dir = dirs["main"]
    s = _csv(os.path.join(main_dir, "summary_meanstd.csv"))
    c = _csv(os.path.join(main_dir, "cross_Garbage12_summary.csv"))

    # 1. main table ------------------------------------------------------------
    L += ["## 1. Main results (5-fold CV, seed 42)", ""]
    if s is not None:
        L += ["| Model | macro-F1 | Accuracy | Garbage12 macro-F1 | Params (M) |",
              "|---|---|---|---|---|"]
        for r in s.itertuples():
            L.append(f"| {r.model} | {r.macro_f1_mean:.4f} ± {r.macro_f1_std:.4f} | "
                     f"{r.accuracy_mean:.4f} ± {r.accuracy_std:.4f} | "
                     f"{_pm(_row(c, r.model), 'macro_f1')} | {r.n_params_M:.2f} |")
        L += [""] + _block(_txt(os.path.join(main_dir, "significance.txt")),
                           lambda ln: ln.startswith(("Friedman", "Critical")))
    else:
        L.append(NA)

    # 2. fusion vs concatenation ------------------------------------------------
    L += ["", "## 2. Fusion vs concatenation, cross-attention", ""]
    for m in [P, "Ablation: Dual (concat)", "Ablation: CATF cross-attention"]:
        L.append(f"- **{m}**: WasteSnap: {_pm(_row(s, m), 'macro_f1')} | "
                 f"Garbage12: {_pm(_row(c, m), 'macro_f1')}")
    pw = _csv(os.path.join(dirs["stats_main"], "pairwise_tests.csv"))
    cpw = _csv(os.path.join(dirs["stats_main"], "cross_Garbage12_pairwise_tests.csv"))
    for name, df, col in [("WasteSnap, paired tests", pw, "other"),
                          ("Garbage12, 5-fold ensemble predictions", cpw, "model")]:
        L.append(f"- {name}:")
        if df is None:
            L.append(f"  - {NA}")
            continue
        for other in ["Ablation: Dual (concat)", "Ablation: CATF cross-attention"]:
            r = df[df[col] == other]
            if len(r) and "dF1" in r:
                r = r.iloc[0]
                L.append(f"  - WF-Net − {other}: ΔF1 = {r.dF1:+.4f} [{r.dF1_lo:+.4f}, "
                         f"{r.dF1_hi:+.4f}], p_boot = {r.p_boot_F1:.4f}, "
                         f"p_McNemar = {r.p_mcnemar:.4f}")

    # 3. statistics ---------------------------------------------------------------
    L += ["", "## 3. Statistics", ""]
    for title, key in [("Single run (all models)", "stats_main"),
                       ("3×5 repeated CV (key models)", "stats_repeated")]:
        L.append(f"**{title}**")
        L += _block(_txt(os.path.join(dirs[key], "posthoc_summary.txt")),
                    lambda ln: ln.startswith(("Per-fold pairs", "  vs ", "  [self-distillation")))
        L.append("")

    # 4. backbone choice ------------------------------------------------------------
    L += ["## 4. Backbone choice: lightweight variant", ""]
    fl = _csv(os.path.join(dirs["cost"], "flops.csv"))
    for m in [P, "Ablation: WF-Net-Lite"]:
        f = _row(fl, m)
        cost = f"{f.params_M:.2f} M, {f.GFLOPs:.2f} GFLOPs" if f is not None else NA
        L.append(f"- {m}: WasteSnap {_pm(_row(s, m), 'macro_f1')} | Garbage12 "
                 f"{_pm(_row(c, m), 'macro_f1')} | {cost}")

    # 5. branch weighting ------------------------------------------------------------
    L += ["", "## 5. Branch weighting", ""]
    L += _block(_txt(os.path.join(dirs["branch_weighting"], "branch_weighting_summary.txt")),
                lambda ln: ln.startswith("==") or any(k in ln for k in
                                                       ("Weight vs accuracy", "Mann-Whitney",
                                                        "Kruskal", "w_convnext mean")))

    # 6. domain shift ----------------------------------------------------------------
    L += ["", "## 6. Domain shift: few-shot adaptation", ""]
    L += _block(_txt(os.path.join(dirs["few_shot"], "adaptation_summary.txt")))
    pc = _csv(os.path.join(main_dir, "cross_Garbage12_per_class_f1.csv"))
    L += ["", "Garbage12 per-class F1:"]
    if pc is not None:
        cols = list(pc.columns)
        L += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in pc.iterrows():
            L.append("| " + " | ".join(str(v) for v in r.values) + " |")
    else:
        L.append(NA)

    # 7. multi-object ----------------------------------------------------------------
    L += ["", "## 7. Multi-object scene test", ""]
    L += _block(_txt(os.path.join(dirs["multi_object"], "multiobject_summary.txt")),
                lambda ln: not ln.startswith("    "))

    # 8. compute ---------------------------------------------------------------------
    L += ["", "## 8. Computational cost", ""]
    cm = dirs["cost"]
    lt = _csv(os.path.join(cm, "latency_throughput.csv"))
    if fl is not None:
        df = fl.merge(lt, on="model", how="left") if lt is not None else fl
        cols = ["model", "params_M", "GFLOPs"] + [k for k in
                ["lat_median_ms_fp32", "lat_median_ms_amp", "throughput_ips_fp32",
                 "throughput_ips_amp", "throughput_ips_amp_with_tta"] if k in df]
        L += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in df[cols].iterrows():
            L.append("| " + " | ".join(str(v) for v in r.values) + " |")
        if lt is None:
            L.append(f"\nLatency/throughput: {NA}")
    else:
        L.append(NA)
    cs = _csv(os.path.join(cm, "cpu_scaling.csv"))
    if cs is not None:
        best = cs.loc[cs.speedup.idxmax()]
        L.append(f"\n- CPU inference strong scaling: highest speed-up {best.speedup:.2f}× "
                 f"({int(best.threads)} threads, parallel efficiency {100 * best.efficiency:.0f}%).")
    ls = _csv(os.path.join(cm, "loader_scaling.csv"))
    if ls is not None:
        best = ls.loc[ls.images_per_s.idxmax()]
        L.append(f"- Data-loading pipeline: {ls.images_per_s.iloc[0]:.0f} → "
                 f"{best.images_per_s:.0f} images/s ({int(best.workers)} workers, "
                 f"{best.speedup_vs_0:.2f}×).")
    bs = _csv(os.path.join(cm, "batch_scaling.csv"))
    if bs is not None:
        b = bs[(bs.model == P) & (bs.precision == "amp")]
        if len(b):
            top = b.loc[b.throughput_ips.idxmax()]
            L.append(f"- WF-Net AMP: highest throughput {top.throughput_ips:.0f} images/s "
                     f"(batch {int(top.batch)}, peak memory {top.peak_mem_mib:.0f} MiB).")

    # 9. HPC projection ---------------------------------------------------------------
    L += ["", "## 9. Training workload and parallel scheduling projection", ""]
    jobs = _job_durations(dirs)
    if jobs:
        tot = sum(jobs)
        L.append(f"Measured independent training jobs: **{len(jobs)}**, total compute "
                 f"**{tot / 3600:.2f} GPU-hours**. The jobs are independent (each has its own "
                 "seed), so they can be distributed over GPUs. **Projection** of a longest-"
                 "processing-time-first (LPT) schedule with the measured job durations:")
        L += ["", "| GPUs | Projected time (hours) | Speed-up | Efficiency |",
              "|---|---|---|---|"]
        for k in (1, 2, 4, 8, 16):
            mk = lpt_makespan(jobs, k)
            L.append(f"| {k} | {mk / 3600:.2f} | {tot / mk:.2f}× | {100 * tot / mk / k:.0f}% |")
    else:
        L.append(NA)

    # 10-12. sweep / garbage12 / learning curves ----------------------------------------
    L += ["", "## 10. Distillation sensitivity", ""]
    L += _block(_txt(os.path.join(dirs["distillation_sensitivity"], "sweep_summary.txt")))
    L += ["", "## 11. Training on Garbage12", ""]
    g = _csv(os.path.join(dirs["garbage12"], "summary_meanstd.csv"))
    if g is not None:
        for r in g.itertuples():
            L.append(f"- {r.model}: macro-F1 {r.macro_f1_mean:.4f} ± {r.macro_f1_std:.4f}, "
                     f"accuracy {r.accuracy_mean:.4f} ± {r.accuracy_std:.4f}")
    else:
        L.append(NA)
    L += ["", "## 12. Learning curves", ""]
    lc = _csv(os.path.join(main_dir, "learning_curves.csv"))
    if lc is not None:
        for m in [P, "Ablation: Dual (concat)"]:
            sub = lc[lc.model == m]
            if len(sub):
                best_ep = sub.loc[sub.groupby("fold").val_macro_f1.idxmax()].epoch
                L.append(f"- {m}: best validation epoch on average {best_ep.mean():.1f} "
                         f"(over the folds {best_ep.min()}-{best_ep.max()}); curve: "
                         "`02_main_seed42/figures/learning_curves.png`")
    else:
        L.append(NA)

    # 13. timing ------------------------------------------------------------------------
    L += ["", "## 13. Durations", ""]
    for name, d in [("Main run", main_dir), ("Garbage12", dirs["garbage12"])] + \
            [(f"Repeated {os.path.basename(p)}", p) for p in dirs["repeated_runs"]]:
        t = _csv(os.path.join(d, "timing.csv"))
        if t is not None:
            tot = t[t.model.str.startswith("TOTAL")]
            if len(tot):
                L.append(f"- {name}: {tot.iloc[0].duration_hms} (total compute time of the jobs)")

    path = os.path.join(root, "SUMMARY.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    log(f"[report] {path}")
    return path
