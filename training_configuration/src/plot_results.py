"""
plot_results.py

Generate visual outputs required for presentation:
- ROC curves per class
- Confusion matrix grid per class
- Training history plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve, auc


LABELS_DEFAULT = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--thresholds", default="best_thresholds_f1.json")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    project_dir = Path(cfg["project_dir"])
    labels = list(cfg.get("target_labels", LABELS_DEFAULT))

    out_dir = project_dir / "results" / args.tag
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    test_true = np.load(out_dir / "test_true.npy")
    test_prob = np.load(out_dir / "test_prob.npy")

    thresholds_path = out_dir / args.thresholds
    if thresholds_path.exists():
        with open(thresholds_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)
    else:
        thresholds = {label: 0.5 for label in labels}

    # ROC curve
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(test_true[:, i].astype(int), test_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Test Set")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "roc_curves_test.png", dpi=200)
    plt.close()

    # Confusion matrices per class
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    for i, label in enumerate(labels):
        thr = thresholds.get(label, 0.5)
        y_true = test_true[:, i].astype(int)
        y_pred = (test_prob[:, i] >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
        disp.plot(ax=axes[i], values_format="d", colorbar=False)
        axes[i].set_title(f"{label}\nthr={thr:.2f}")

    for j in range(len(labels), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Confusion Matrices by Class - Test Set")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrices_test.png", dpi=200)
    plt.close()

    # Training history
    hist_path = out_dir / "training_history.csv"
    if hist_path.exists():
        hist = pd.read_csv(hist_path)

        plt.figure(figsize=(8, 5))
        plt.plot(hist["epoch"], hist["train_loss"], label="Train loss")
        plt.plot(hist["epoch"], hist["val_loss"], label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "training_loss.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(hist["epoch"], hist["val_macro_auc_roc"], label="Validation macro AUC-ROC")
        plt.plot(hist["epoch"], hist["val_macro_f1"], label="Validation macro F1")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title("Validation Metrics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "validation_metrics.png", dpi=200)
        plt.close()

    print(f"Figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
