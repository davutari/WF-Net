"""Recomputes the numbers of the paper from the per-image predictions in predictions/.

No training and no images are needed. Every check prints MATCH or MISMATCH against the value
printed in the paper, which is listed in expected_values.json or in this file.

    python verify_statistics.py          (about five minutes on a laptop)

Protocol of the paired tests: exact McNemar test on correct and incorrect outcomes, and a paired
bootstrap of the difference in pooled macro-F1 (10,000 resamples, stratified by fold). Holm correction
within each family. Corrected resampled t-tests over folds.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest, t as student_t

HERE = os.path.dirname(os.path.abspath(__file__))
PRED = os.path.join(HERE, "predictions")
EXPECTED = json.load(open(os.path.join(HERE, "expected_values.json"), encoding="utf-8"))
RESULTS = {"match": 0, "mismatch": 0}

REF = "WF_Net__Proposed_"
BASE = ["VGG16", "ResNet50", "InceptionV3", "Xception", "DenseNet121", "MobileNetV2", "EfficientNetB0",
        "InceptionResNetV2", "ViT_B_16", "Swin_T"]
ABL = ["Ablation__CNN_only", "Ablation__Transformer_only", "Ablation__Dual__concat_",
       "Ablation__Dual_CATF_w_o_attn", "Ablation__WF_Net_w__CBAM", "Ablation__WF_Net_w_o_self_distill",
       "Ablation__WF_Net_Lite"]
# Seed of each bootstrap of Table 5 and Section 4.5, as used in the original analysis.
BOOT_SEED = {m: i for i, m in enumerate(BASE)}
BOOT_SEED.update({"Ablation__CNN_only": 10, "Ablation__Transformer_only": 11, "Ablation__Dual__concat_": 12,
                  "Ablation__Dual_CATF_w_o_attn": 13, "Ablation__WF_Net_w__CBAM": 14,
                  "Ablation__WF_Net_w_o_self_distill": 15, "Ablation__WF_Net_Lite": 17})


def check(name, value, expected, tol=0.00501):
    ok = abs(value - expected) <= tol
    RESULTS["match" if ok else "mismatch"] += 1
    print(f"  {'MATCH' if ok else 'MISMATCH':8s} {name}: computed {value:.4f}, paper {expected}")


def check_p(name, value, printed):
    """printed: a float, or a string such as '<0.001'."""
    if isinstance(printed, str) and printed.startswith("<"):
        ok = value < float(printed[1:])
    else:
        ok = abs(value - float(printed)) <= 0.5 * 10 ** (-len(str(printed).split(".")[1])) + 1e-9
    RESULTS["match" if ok else "mismatch"] += 1
    print(f"  {'MATCH' if ok else 'MISMATCH':8s} {name}: computed p={value:.2g}, paper {printed}")


def check_eq(name, value, expected):
    ok = value == expected
    RESULTS["match" if ok else "mismatch"] += 1
    print(f"  {'MATCH' if ok else 'MISMATCH':8s} {name}: computed {value}, paper {expected}")


def macro_f1(y, p, n_classes, labels=None):
    """Same definition as sklearn.metrics.f1_score(average='macro', zero_division=0)."""
    cm = np.bincount(y * n_classes + p, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    den = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    f1 = np.divide(2 * tp, den, out=np.zeros(n_classes), where=den > 0)
    return f1[labels].mean() if labels is not None else f1[den > 0].mean()


def boot(y, a, b, strata, seed, n_classes, B=10000, labels=None):
    """Paired stratified bootstrap of macro-F1(a) - macro-F1(b), in points."""
    rng = np.random.default_rng(seed)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    d0 = macro_f1(y, a, n_classes, labels) - macro_f1(y, b, n_classes, labels)
    ds = np.empty(B)
    for r in range(B):
        ix = np.concatenate([rng.choice(g, len(g)) for g in groups])
        ds[r] = macro_f1(y[ix], a[ix], n_classes, labels) - macro_f1(y[ix], b[ix], n_classes, labels)
    return d0 * 100, np.percentile(ds, 2.5) * 100, np.percentile(ds, 97.5) * 100


def mcnemar(y, a, b):
    n10 = int(((a == y) & (b != y)).sum())
    n01 = int(((a != y) & (b == y)).sum())
    return n10, n01, (binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if n10 + n01 else 1.0)


def holm(ps):
    ps = np.asarray(ps, float)
    adj = np.empty(len(ps))
    running = 0.0
    for j, i in enumerate(np.argsort(ps)):
        running = max(running, min(1.0, (len(ps) - j) * ps[i]))
        adj[i] = running
    return adj


def corrected_t(d, ntest=154, ntrain=523):
    """Nadeau-Bengio corrected resampled t-test; repeated-CV form when d holds several repetitions."""
    d = np.asarray(d, float)
    t = d.mean() / np.sqrt((1.0 / len(d) + ntest / ntrain) * d.var(ddof=1))
    return 2 * student_t.sf(abs(t), len(d) - 1)


def load(run, model, n_folds):
    """Predictions of all folds, concatenated in fold order, and the per-fold metrics."""
    y, p, fold, idx, probs, mets = [], [], [], [], [], []
    for f in range(n_folds):
        base = os.path.join(PRED, run, model, f"fold{f}")
        z = np.load(base + ".npz", allow_pickle=False)
        y.append(z["y_true"].astype(np.int64))
        p.append(z["y_pred"].astype(np.int64))
        fold.append(np.full(len(z["y_true"]), f))
        if "test_idx" in z.files:
            idx.append(z["test_idx"])
        probs.append(z["y_prob"])
        mets.append(json.load(open(base + ".json", encoding="utf-8")))
    return dict(y=np.concatenate(y), p=np.concatenate(p), fold=np.concatenate(fold),
                idx=np.concatenate(idx) if idx else None, probs=probs, mets=mets)


def by_index(a):
    o = np.argsort(a["idx"])
    return dict(y=a["y"][o], p=a["p"][o], fold=a["fold"][o], idx=a["idx"][o], mets=a["mets"])


def mean_sd(a, key):
    v = np.array([m[key] for m in a["mets"]]) * 100
    return v.mean(), v.std(ddof=1)


def fold_f1(a):
    return np.array([m["macro_f1"] for m in a["mets"]]) * 100


# ------------------------------------------------------------------------------- Tables 3 and 6
print("Table 3 and Table 6: 5-fold results on WasteSnap (mean and sd over folds)")
MAIN = {m: load("wastesnap_seed42", m, 5) for m in BASE + ABL + [REF]}
for table in ("table3", "table6"):
    for model, rec in EXPECTED[table].items():
        for key, (mu, sd) in rec.items():
            a, b = mean_sd(MAIN[model], key)
            check(f"{table} {model} {key} mean", a, mu)
            check(f"{table} {model} {key} sd", b, sd)

# ------------------------------------------------------------------------------- Table 5
print("\nTable 5 and Section 4.5: pooled paired comparisons with WF-Net on the 770 test images")
y, fold = MAIN[REF]["y"], MAIN[REF]["fold"]
K = 9
rows = {}
for m in BASE + ABL:
    assert np.array_equal(MAIN[m]["y"], y)
    d, lo, hi = boot(y, MAIN[REF]["p"], MAIN[m]["p"], fold, BOOT_SEED[m], K)
    n10, n01, p = mcnemar(y, MAIN[REF]["p"], MAIN[m]["p"])
    rows[m] = dict(d=d, lo=lo, hi=hi, b=n10, c=n01, p_mcn=p, p_nb=corrected_t(fold_f1(MAIN[REF]) - fold_f1(MAIN[m])),
                   d_mean=(fold_f1(MAIN[REF]) - fold_f1(MAIN[m])).mean())
for fam in (BASE, ABL):
    for key in ("p_mcn", "p_nb"):
        for m, h in zip(fam, holm([rows[x][key] for x in fam])):
            rows[m][key + "_holm"] = h
for m, rec in EXPECTED["table5"].items():
    r = rows[m]
    check(f"table5 {m} delta F1", r["d"], rec["d"])
    check(f"table5 {m} CI low", r["lo"], rec["lo"])
    check(f"table5 {m} CI high", r["hi"], rec["hi"])
    check_eq(f"table5 {m} b", r["b"], rec["b"])
    check_eq(f"table5 {m} c", r["c"], rec["c"])
    check_p(f"table5 {m} p_McN (Holm)", r["p_mcn_holm"], rec["p_mcn"])
    check_p(f"table5 {m} p_NB (Holm)", r["p_nb_holm"], rec["p_nb"])
for m, (d, lo, hi) in {"Ablation__Dual__concat_": (-0.22, -1.86, 1.42),
                       "Ablation__WF_Net_w_o_self_distill": (0.53, -1.38, 2.46),
                       "Ablation__WF_Net_w__CBAM": (-0.82, -2.66, 1.02),
                       "Ablation__Dual_CATF_w_o_attn": (-0.23, -2.14, 1.71)}.items():
    check(f"Section 4.5 {m} delta F1", rows[m]["d"], d)
    check(f"Section 4.5 {m} CI low", rows[m]["lo"], lo)
    check(f"Section 4.5 {m} CI high", rows[m]["hi"], hi)
    check_p(f"Section 4.5 {m} p_Holm", rows[m]["p_mcn_holm"], "1.0")
check("Section 4.5 no-attention variant, difference of fold means", rows["Ablation__Dual_CATF_w_o_attn"]["d_mean"], -0.32)
check("Section 4.5 WF-Net-Lite, pooled difference", rows["Ablation__WF_Net_Lite"]["d"], 6.91)
check("Section 4.5 WF-Net-Lite, difference of the rounded fold means", rows["Ablation__WF_Net_Lite"]["d_mean"], 6.96, tol=0.0101)
check_p("Section 4.5 WF-Net-Lite p_Holm", rows["Ablation__WF_Net_Lite"]["p_mcn_holm"], "<0.001")

# ------------------------------------------------------------------------------- repeated CV
print("\nTable 5 and Section 4.4: 3x5 repeated cross-validation")
KEY = ["ViT_B_16", "Swin_T", "VGG16"]
NOSD = "Ablation__WF_Net_w_o_self_distill"
REPS = {s: {m: load(f"wastesnap_seed{s}", m, 5) for m in KEY + [REF, NOSD]} for s in (1337, 2024)}
REPS[42] = {m: MAIN[m] for m in KEY + [REF, NOSD]}
ref15 = np.concatenate([fold_f1(REPS[s][REF]) for s in (42, 1337, 2024)])
ps = []
for m in KEY:
    o15 = np.concatenate([fold_f1(REPS[s][m]) for s in (42, 1337, 2024)])
    check(f"table5 {m} delta F1 over 15 folds", (ref15 - o15).mean(), EXPECTED["table5"][m]["d15"])
    ps.append(corrected_t(ref15 - o15))
for m, h in zip(KEY, holm(ps)):
    check_p(f"table5 {m} p_BF (Holm)", h, EXPECTED["table5"][m]["p_bf"])
check("Section 4.4 WF-Net macro-F1 over three repetitions", ref15.mean(), 88.84)
d15 = ref15 - np.concatenate([fold_f1(REPS[s][NOSD]) for s in (42, 1337, 2024)])
check("Section 4.5 effect of self-distillation over 15 folds", d15.mean(), 1.34)
check_p("Section 4.5 effect of self-distillation, corrected t-test", corrected_t(d15), "0.34")

# ------------------------------------------------------------------------------- Table 7
print("\nTable 7 and Section 4.7: zero-shot transfer to the 8,213 mapped Garbage12 images")
CROSS = {m: load("garbage12_zero_shot_seed42", m, 5) for m in EXPECTED["table7"]}
for model, rec in EXPECTED["table7"].items():
    for key, (mu, sd) in rec.items():
        a, b = mean_sd(CROSS[model], key)
        check(f"table7 {model} {key} mean", a, mu)
        check(f"table7 {model} {key} sd", b, sd)
acc = {m: mean_sd(CROSS[m], "accuracy")[0] for m in BASE}
f1 = {m: mean_sd(CROSS[m], "macro_f1")[0] for m in BASE}
check_eq("Section 4.7 strongest baseline on accuracy", max(acc, key=acc.get), "Swin_T")
check_eq("Section 4.7 strongest baseline on macro-F1", max(f1, key=f1.get), "ViT_B_16")
# The margins of the paper are differences of the rounded means of Table 7.
check("Section 4.7 accuracy margin", round(mean_sd(CROSS[REF], "accuracy")[0], 2) - round(acc["Swin_T"], 2), 2.85)
check("Section 4.7 macro-F1 margin", round(mean_sd(CROSS[REF], "macro_f1")[0], 2) - round(f1["ViT_B_16"], 2), 3.22)
z0 = np.load(os.path.join(PRED, "garbage12_zero_shot_seed42", REF, "fold0.npz"), allow_pickle=False)
shared = np.asarray(z0["shared"], dtype=int)
yc = z0["y_true"].astype(np.int64)
ens = {m: np.mean(CROSS[m]["probs"], axis=0).argmax(1).astype(np.int64)
       for m in (REF, "Ablation__Dual__concat_")}
fa, fb = macro_f1(yc, ens[REF], K, shared) * 100, macro_f1(yc, ens["Ablation__Dual__concat_"], K, shared) * 100
check("Section 4.7 ensemble macro-F1, WF-Net", fa, 48.21)
check("Section 4.7 ensemble macro-F1, concatenation", fb, 46.87)
d, lo, hi = boot(yc, ens[REF], ens["Ablation__Dual__concat_"], yc, 100, K, B=2000, labels=shared)
check("Section 4.7 ensemble difference", d, 1.35)
check("Section 4.7 ensemble CI low", lo, 0.65)
check("Section 4.7 ensemble CI high", hi, 2.07)
check_p("Section 4.7 ensemble McNemar", mcnemar(yc, ens[REF], ens["Ablation__Dual__concat_"])[2], "0.003")

# ------------------------------------------------------------------------------- Table 8
print("\nTable 8 and Section 4.8: complete Garbage12 dataset (15,515 images, 3 folds)")
G12 = {m: by_index(load("garbage12_seed42", m, 3))
       for m in (REF, "Ablation__Dual__concat_", "ViT_B_16", "Swin_T")}
ARMS = {a: by_index(load("followup_garbage12_a4000", a, 3)) for a in ("nomix", "eased", "nosd", "default40")}
for model, rec in EXPECTED["table8"].items():
    a = ARMS.get(model) or G12[model]
    for key in ("accuracy", "macro_f1"):
        mu, sd = mean_sd(a, key)
        check(f"table8 {model} {key} mean", mu, rec[key][0])
        check(f"table8 {model} {key} sd", sd, rec[key][1])
    check(f"table8 {model} minutes per fold", np.mean([m["train_time_s"] for m in a["mets"]]) / 60, rec["minutes"], tol=0.0501)
c0 = np.load(os.path.join(PRED, "followup_garbage12_a4000", "default", "fold0.npz"), allow_pickle=False)
g0 = np.load(os.path.join(PRED, "garbage12_seed42", REF, "fold0.npz"), allow_pickle=False)
check_eq("Section 4.8 control re-run of fold 0 reproduces the predictions exactly",
         bool(np.array_equal(c0["y_pred"], g0["y_pred"])), True)
ref = G12[REF]
K12 = 12
y12, f12 = ref["y"], ref["fold"]
stats = {}
for a, arm in ARMS.items():
    assert np.array_equal(arm["idx"], ref["idx"]) and np.array_equal(arm["y"], y12)
    stats[a] = (boot(y12, arm["p"], ref["p"], f12, 0, K12), mcnemar(y12, arm["p"], ref["p"]))
hp = dict(zip(ARMS, holm([stats[a][1][2] for a in ARMS])))
for a, (d, lo, hi), p in (("nomix", (-0.16, -0.44, 0.12), "0.22"), ("eased", (0.28, -0.02, 0.59), "0.17"),
                          ("default40", (0.58, 0.31, 0.85), "<0.001"), ("nosd", (-0.86, -1.19, -0.53), "<0.001")):
    check(f"Section 4.8 {a} difference", stats[a][0][0], d)
    check(f"Section 4.8 {a} CI low", stats[a][0][1], lo)
    check(f"Section 4.8 {a} CI high", stats[a][0][2], hi)
    check_p(f"Section 4.8 {a} p_Holm", hp[a], p)
for a, other, (d, lo, hi) in (("eased", "Ablation__Dual__concat_", (-1.05, -1.34, -0.77)),
                              ("eased", "ViT_B_16", (-1.31, -1.61, -1.01)),
                              ("default40", "Ablation__Dual__concat_", (-0.76, None, None)),
                              ("default40", "ViT_B_16", (-1.02, None, None))):
    r = boot(y12, ARMS[a]["p"], G12[other]["p"], f12, 0, K12)
    check(f"Section 4.8 {a} vs {other} difference", r[0], d)
    if lo is not None:
        check(f"Section 4.8 {a} vs {other} CI low", r[1], lo)
        check(f"Section 4.8 {a} vs {other} CI high", r[2], hi)
gap = macro_f1(y12, G12["Ablation__Dual__concat_"]["p"], K12) * 100 - macro_f1(y12, ref["p"], K12) * 100
check("Section 4.8 gap of the default run to the concatenation variant", gap, 1.34)
ts = [corrected_t(fold_f1(ARMS[a]) - fold_f1(ref), ntest=5172, ntrain=8791) for a in ARMS]
check_eq("Section 4.8 corrected t-tests over the folds, all p > 0.39", bool(min(ts) > 0.39), True)

# ------------------------------------------------------------------------------- Discussion
print("\nSection 5: distillation pilot")
wf, lite = by_index(MAIN[REF]), by_index(MAIN["Ablation__WF_Net_Lite"])
kd = by_index(load("followup_lite_a4000", "kd", 5))
assert np.array_equal(kd["idx"], lite["idx"]) and np.array_equal(kd["y"], lite["y"])
check("pilot student macro-F1", mean_sd(kd, "macro_f1")[0], 82.36)
d, lo, hi = boot(kd["y"], kd["p"], lite["p"], kd["fold"], 0, K)
check("pilot gain", d, 0.95)
check("pilot CI low", lo, -0.99)
check("pilot CI high", hi, 2.87)
check_p("pilot McNemar", mcnemar(kd["y"], kd["p"], lite["p"])[2], "0.47")
check_p("pilot corrected t-test over the folds", corrected_t(fold_f1(kd) - fold_f1(lite)), "0.71")
r = boot(kd["y"], kd["p"], wf["p"], kd["fold"], 0, K)
check("pilot gap to WF-Net", -r[0], 5.96)
check("pilot gap CI low", r[1], -8.58)
check("pilot gap CI high", r[2], -3.44)
check_p("pilot gap McNemar", mcnemar(kd["y"], kd["p"], wf["p"])[2], "<0.001")
for t, v in zip(by_index(load("followup_lite_a4000", "kd", 5))["mets"], fold_f1(MAIN[REF])):
    check("pilot teacher reproduces the published fold", t["teacher_macro_f1"] * 100, v, tol=1e-6)
print(f"\n{RESULTS['match']} checks match, {RESULTS['mismatch']} do not.")
sys.exit(1 if RESULTS["mismatch"] else 0)
