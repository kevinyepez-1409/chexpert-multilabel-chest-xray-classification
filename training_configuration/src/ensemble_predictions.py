"""
ensemble_predictions.py

Average predictions from trained runs and evaluate the ensemble.

Example:
python src/ensemble_predictions.py --config configs/chexpert_kaggle.yaml --tags convnext_tiny_kaggle80k_320 efficientnet_b0_kaggle80k_320 --out-tag ensemble_kaggle80k
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) else np.nan


def optimize_thresholds(y_true, y_prob, labels):
    grid = np.arange(0.05, 0.96, 0.01)
    out = {}
    for i, label in enumerate(labels):
        best_thr = 0.5
        best_f1 = -1
        for thr in grid:
            pred = (y_prob[:, i] >= thr).astype(int)
            score = f1_score(y_true[:, i], pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thr = float(thr)
        out[label] = best_thr
    return out


def evaluate(y_true, y_prob, labels, threshold):
    if isinstance(threshold, dict):
        pred = np.zeros_like(y_true, dtype=int)
        for i, label in enumerate(labels):
            pred[:, i] = (y_prob[:, i] >= threshold[label]).astype(int)
    else:
        pred = (y_prob >= threshold).astype(int)

    rows = []
    for i, label in enumerate(labels):
        yt = y_true[:, i].astype(int)
        yp = pred[:, i].astype(int)
        ys = y_prob[:, i]
        rows.append({
            "label": label,
            "auc_roc": roc_auc_score(yt, ys),
            "auc_pr": average_precision_score(yt, ys),
            "f1": f1_score(yt, yp, zero_division=0),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall_sensitivity": recall_score(yt, yp, zero_division=0),
            "specificity": specificity(yt, yp),
        })
    df = pd.DataFrame(rows)
    summary = {
        "macro_auc_roc": float(df["auc_roc"].mean()),
        "macro_auc_pr": float(df["auc_pr"].mean()),
        "macro_f1": float(df["f1"].mean()),
        "macro_precision": float(df["precision"].mean()),
        "macro_recall": float(df["recall_sensitivity"].mean()),
        "macro_specificity": float(df["specificity"].mean()),
    }
    return df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--out-tag", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    project_dir = Path(cfg["project_dir"])
    labels = list(cfg["target_labels"])

    result_dirs = [project_dir / "results" / tag for tag in args.tags]

    val_true_ref = np.load(result_dirs[0] / "val_true.npy")
    test_true_ref = np.load(result_dirs[0] / "test_true.npy")

    val_probs = []
    test_probs = []

    for d in result_dirs:
        val_true = np.load(d / "val_true.npy")
        test_true = np.load(d / "test_true.npy")
        if not np.array_equal(val_true_ref, val_true):
            raise ValueError(f"val_true mismatch: {d}")
        if not np.array_equal(test_true_ref, test_true):
            raise ValueError(f"test_true mismatch: {d}")
        val_probs.append(np.load(d / "val_prob.npy"))
        test_probs.append(np.load(d / "test_prob.npy"))

    val_prob = np.mean(val_probs, axis=0)
    test_prob = np.mean(test_probs, axis=0)

    out_dir = project_dir / "results" / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "val_true.npy", val_true_ref)
    np.save(out_dir / "test_true.npy", test_true_ref)
    np.save(out_dir / "val_prob.npy", val_prob)
    np.save(out_dir / "test_prob.npy", test_prob)

    thresholds = optimize_thresholds(val_true_ref, val_prob, labels)

    with open(out_dir / "best_thresholds_f1.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    metrics_05, summary_05 = evaluate(test_true_ref, test_prob, labels, 0.5)
    metrics_opt, summary_opt = evaluate(test_true_ref, test_prob, labels, thresholds)

    metrics_05.to_csv(out_dir / "test_metrics_threshold_0p5.csv", index=False)
    metrics_opt.to_csv(out_dir / "test_metrics_optimized_thresholds.csv", index=False)

    summary = pd.DataFrame([
        {"threshold": "0.5", **summary_05},
        {"threshold": "optimized_on_validation", **summary_opt},
    ])
    summary["models"] = " + ".join(args.tags)
    summary.to_csv(out_dir / "summary_test.csv", index=False)

    print(summary.to_string(index=False))
    print(f"Saved ensemble to: {out_dir}")


if __name__ == "__main__":
    main()
