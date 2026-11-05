"""Plotting: confusion matrix, per-model comparison bars, Grad-CAM panels."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config


def _ensure_dir(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def save_fig(path, dpi=300):
    """Save the current figure as 300-dpi PNG *and* vector PDF, then close it."""
    _ensure_dir(path)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    root, ext = os.path.splitext(path)
    if ext.lower() == ".png":
        plt.savefig(root + ".pdf", bbox_inches="tight")
    plt.close()


def plot_confusion(cm, class_names, title, out_path):
    _ensure_dir(out_path)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_fig(out_path)


def plot_model_comparison(summary_df, metric, out_path):
    _ensure_dir(out_path)
    df = summary_df.sort_values(f"{metric}_mean", ascending=True)
    plt.figure(figsize=(9, 6))
    plt.barh(df["model"], df[f"{metric}_mean"],
             xerr=df[f"{metric}_std"], capsize=3, color="#4C72B0")
    plt.xlabel(metric.replace("_", " ").title())
    plt.title(f"Model comparison ({metric}, mean ± std over folds)")
    plt.tight_layout()
    save_fig(out_path)


def plot_roc_curves(y_true, y_prob, class_names, title, out_path):
    """One-vs-rest ROC per class + macro-average, with AUCs in the legend.

    Returns (fpr_grid, mean_tpr, macro_auc) so callers can overlay the
    macro-average curves of several models on a single comparison plot.
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    _ensure_dir(out_path)

    n_classes = len(class_names)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr_grid = np.linspace(0.0, 1.0, 200)
    mean_tpr = np.zeros_like(fpr_grid)
    valid = 0

    plt.figure(figsize=(8, 7))
    for c in range(n_classes):
        if y_bin[:, c].sum() == 0:           # class absent from this set -> skip
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
        plt.plot(fpr, tpr, lw=1.2, alpha=0.7,
                 label=f"{class_names[c]} (AUC={auc(fpr, tpr):.3f})")
        mean_tpr += np.interp(fpr_grid, fpr, tpr)
        valid += 1

    mean_tpr /= max(valid, 1)
    mean_tpr[0] = 0.0
    macro_auc = auc(fpr_grid, mean_tpr)
    plt.plot(fpr_grid, mean_tpr, color="navy", lw=2.6, linestyle="--",
             label=f"Macro-average (AUC={macro_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle=":")

    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    save_fig(out_path)
    return fpr_grid, mean_tpr, macro_auc


def plot_roc_comparison(roc_data, out_path):
    """Overlay every model's macro-average ROC. roc_data: [(name, fpr, tpr, auc)]."""
    _ensure_dir(out_path)
    plt.figure(figsize=(8, 7))
    for name, fpr, tpr, a in sorted(roc_data, key=lambda r: -r[3]):
        plt.plot(fpr, tpr, lw=1.6, label=f"{name} (AUC={a:.3f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle=":")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Macro-average ROC comparison (mean over test folds)")
    plt.legend(loc="lower right", fontsize=6)
    plt.tight_layout()
    save_fig(out_path)


def plot_cd_diagram(mean_ranks, cd, out_path, title=None):
    """Demsar critical-difference diagram.

    Places every model on an average-rank axis (best = rank 1 on the right) and
    joins models whose mean-rank gap is < CD (not significantly different) with
    thick cliques bars. `mean_ranks` is a pandas Series {model: mean_rank}.
    """
    _ensure_dir(out_path)
    ranks = mean_ranks.sort_values()
    names = list(ranks.index)
    vals = list(ranks.values)
    k = len(names)
    lo, hi = 1, k                                   # rank axis bounds

    # cliques: maximal groups of consecutive models within CD of each other
    cliques = []
    i = 0
    while i < k:
        j = i
        while j + 1 < k and (vals[j + 1] - vals[i]) < cd:
            j += 1
        if j > i:
            cliques.append((i, j))
        i = j + 1 if j > i else i + 1

    half = (k + 1) // 2
    fig_h = 2.2 + 0.32 * k
    plt.figure(figsize=(9, fig_h))
    ax = plt.gca()
    ax.set_xlim(hi + 0.5, lo - 0.5)                 # reversed: rank 1 on the right
    ax.set_ylim(0, fig_h)
    top = fig_h - 0.6
    ax.plot([lo, hi], [top, top], "k-", lw=1.5)     # main axis
    for r in range(lo, hi + 1):
        ax.plot([r, r], [top, top + 0.12], "k-", lw=1)
        ax.text(r, top + 0.22, str(r), ha="center", va="bottom", fontsize=8)

    # leaders down to each model label, split left/right for readability
    for idx, (name, v) in enumerate(zip(names, vals)):
        left = idx < half
        y = top - 0.5 - (idx if left else (k - 1 - idx)) * ((top - 0.8) / max(half, 1))
        x_txt = hi + 0.4 if left else lo - 0.4
        ax.plot([v, v], [top, y], "k-", lw=0.8)
        ax.plot([v, x_txt], [y, y], "k-", lw=0.8)
        ax.text(x_txt + (0.1 if left else -0.1), y,
                f"{name} ({v:.2f})", ha="left" if left else "right",
                va="center", fontsize=8)

    # clique bars just below the axis
    for ci, (a, b) in enumerate(cliques):
        y = top - 0.18 - ci * 0.16
        ax.plot([vals[a], vals[b]], [y, y], "-", color="crimson", lw=4,
                solid_capstyle="round")

    # CD reference bar
    ax.plot([hi, hi - cd], [top + 0.6, top + 0.6], "k-", lw=2)
    ax.plot([hi, hi], [top + 0.55, top + 0.65], "k-", lw=1)
    ax.plot([hi - cd, hi - cd], [top + 0.55, top + 0.65], "k-", lw=1)
    ax.text((hi - cd / 2), top + 0.72, f"CD = {cd:.2f}", ha="center",
            va="bottom", fontsize=9)

    ax.set_title(title or "Critical-difference diagram (Nemenyi, α=0.05)")
    ax.axis("off")
    plt.tight_layout()
    save_fig(out_path)


def plot_gradcam_panel(images, cams, preds, trues, class_names, out_path):
    _ensure_dir(out_path)
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i in range(n):
        axes[0, i].imshow(images[i])
        axes[0, i].set_title(f"T:{class_names[trues[i]]}", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(images[i])
        axes[1, i].imshow(cams[i], cmap="jet", alpha=0.5)
        axes[1, i].set_title(f"P:{class_names[preds[i]]}", fontsize=8)
        axes[1, i].axis("off")
    plt.tight_layout()
    save_fig(out_path)
