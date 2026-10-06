"""Multi-label metrics, threshold tuning, figures and error dumps shared by every model.

Headline metric: macro-F1 over the six emotions, the official SemEval-2025
Task 11 Track A metric. Macro averaging gives surprise (4% of texts) the same
weight as sadness (26%), so a model cannot score well by ignoring rare emotions.
Micro-F1, exact-match accuracy and per-emotion P/R/F1 are reported alongside.
"""
import json
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, multilabel_confusion_matrix,
                             precision_recall_fscore_support)

from src.config import EMOTIONS, FIGURES_DIR, RESULTS_DIR

EXPERIMENTS_CSV = RESULTS_DIR / "experiments.csv"


def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    metrics = {
        "macro_f1": f, "macro_precision": p, "macro_recall": r,
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "exact_match": accuracy_score(y_true, y_pred),  # all six labels right
    }
    per_label = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics |= {f"f1_{e}": v for e, v in zip(EMOTIONS, per_label)}
    return metrics


def tune_thresholds(y_true, probs, grid=np.arange(0.05, 0.95, 0.01)) -> np.ndarray:
    """Pick, per emotion, the probability cut-off that maximises that emotion's F1 on dev.

    A single 0.5 cut-off is wrong for rare labels: a model trained on 4% positives
    rarely becomes 50% sure, so surprise would almost never be predicted.
    """
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    return np.array([
        grid[np.argmax([f1_score(y_true[:, j], probs[:, j] >= t, zero_division=0) for t in grid])]
        for j in range(y_true.shape[1])
    ])


def log_experiment(name: str, config: dict, dev_metrics: dict, test_metrics: dict) -> None:
    """Add/replace one row of results/experiments.csv (the experiment table in the report)."""
    row = {"experiment": name, "time": datetime.now().isoformat(timespec="seconds"),
           "config": json.dumps(config)}
    row |= {f"dev_{k}": round(v, 4) for k, v in dev_metrics.items()}
    row |= {f"test_{k}": round(v, 4) for k, v in test_metrics.items()}
    df = pd.DataFrame([row])
    if EXPERIMENTS_CSV.exists():
        df = pd.concat([pd.read_csv(EXPERIMENTS_CSV), df]).drop_duplicates("experiment", keep="last")
    df.to_csv(EXPERIMENTS_CSV, index=False)


def save_report(name: str, test_df: pd.DataFrame, y_pred, probs=None) -> None:
    """Per-emotion confusion matrices (figure) and every misclassified test text (CSV)."""
    out = RESULTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    y_true, y_pred = test_df[EMOTIONS].to_numpy(), np.asarray(y_pred)

    cms = multilabel_confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, len(EMOTIONS), figsize=(16, 3))
    for ax, emotion, cm in zip(axes, EMOTIONS, cms):
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(emotion)
        ax.set_xticks([0, 1], ["pred no", "pred yes"])
        ax.set_yticks([0, 1], ["true no", "true yes"])
    fig.suptitle(f"{name}: per-emotion confusion matrices (test)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"cm_{name}.png", dpi=150)
    plt.close(fig)

    def names(rows):
        return ["+".join(e for e, v in zip(EMOTIONS, row) if v) or "none" for row in rows]

    errors = pd.DataFrame({"id": test_df["id"], "text": test_df["text"],
                           "gold": names(y_true), "pred": names(y_pred)})
    if probs is not None:
        for j, e in enumerate(EMOTIONS):
            errors[f"p_{e}"] = np.asarray(probs)[:, j].round(3)
    errors[errors["gold"] != errors["pred"]].to_csv(out / "errors.csv", index=False)
    (out / "test_metrics.json").write_text(json.dumps(compute_metrics(y_true, y_pred), indent=2))
