"""
Evaluation metrics for imbalanced multi-class classification.

Reports accuracy, macro/weighted precision-recall-F1, Cohen's kappa, MCC and
macro one-vs-rest ROC-AUC.
"""
import warnings

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, cohen_kappa_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report,
)


def compute_metrics(y_true, y_pred, y_prob=None, n_classes=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    pr_m, rc_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    _, _, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": pr_m,
        "recall_macro": rc_m,
        "macro_f1": f1_m,
        "weighted_f1": f1_w,
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    if y_prob is not None:
        try:
            labels = list(range(n_classes)) if n_classes else None
            # A class absent from y_true (e.g. the 2 classes missing from the
            # cross-dataset, or a rare class not sampled into a tiny val split)
            # makes its one-vs-rest AUC undefined -> sklearn warns. That is
            # expected and handled (nan), so silence just this warning here.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UndefinedMetricWarning)
                out["roc_auc_macro"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro",
                    labels=labels)
        except ValueError:
            out["roc_auc_macro"] = float("nan")
    return out


def full_report(y_true, y_pred, class_names, labels=None):
    # `labels` pins the full class set so a report over a subset (e.g. a
    # cross-dataset with only the shared classes present) still aligns with
    # class_names instead of raising a size-mismatch error.
    if labels is None:
        labels = list(range(len(class_names)))
    return classification_report(y_true, y_pred, labels=labels,
                                 target_names=class_names,
                                 zero_division=0, digits=4)


def get_confusion(y_true, y_pred, n_classes):
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
