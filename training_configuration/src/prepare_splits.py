"""
prepare_splits.py

Build patient-wise CheXpert Kaggle splits for multilabel chest X-ray classification.

Main design decisions:
1) Default to frontal-only radiographs to avoid mixing frontal and lateral views.
2) Split by patient_id to avoid patient leakage.
3) Convert uncertainty labels to binary labels using configurable strategies.
4) Save CSVs compatible with the training script:
   local_ready/train_<strategy>_local.csv
   local_ready/val_<strategy>_local.csv
   local_ready/test_<strategy>_local.csv
   local_ready/official_valid_<strategy>_local.csv
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml


PATIENT_RE = re.compile(r"(patient\d+)", re.IGNORECASE)
STUDY_RE = re.compile(r"(study\d+)", re.IGNORECASE)


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_patient_id(path_str: str) -> str:
    m = PATIENT_RE.search(str(path_str))
    if not m:
        raise ValueError(f"Cannot extract patient_id from path: {path_str}")
    return m.group(1)


def extract_study_id(path_str: str) -> str:
    m = STUDY_RE.search(str(path_str))
    return m.group(1) if m else "unknown_study"


def resolve_image_path(path_str: str, archive_root: Path) -> Path | None:
    """
    Resolve Kaggle CheXpert Path values against possible local extraction layouts.

    CSV Path usually looks like:
        CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg

    User's extracted folder often looks like:
        archive/train/patient00001/study1/view1_frontal.jpg

    This function tries both.
    """
    raw = str(path_str).replace("\\", "/")
    stripped = raw.replace("CheXpert-v1.0-small/", "")

    candidates = [
        archive_root / raw,
        archive_root / stripped,
        archive_root.parent / raw,
        archive_root.parent / stripped,
    ]

    for c in candidates:
        if c.exists():
            return c

    return None


def apply_uncertainty_strategy(df: pd.DataFrame, labels: List[str], strategy: str) -> pd.DataFrame:
    """
    Convert CheXpert labels to binary {0,1}.

    Original values:
    - 1.0: positive
    - 0.0: negative
    - -1.0: uncertain
    - NaN: blank / not mentioned

    Strategies:
    - uzero: uncertain -> 0, NaN -> 0
    - uone: uncertain -> 1, NaN -> 0
    - hybrid2: Atelectasis and Edema uncertain -> 1; the others uncertain -> 0; NaN -> 0
    """
    out = df.copy()

    hybrid_positive_uncertain = {"Atelectasis", "Edema"}

    for label in labels:
        if label not in out.columns:
            raise ValueError(f"Missing label column in CSV: {label}")

        vals = out[label].copy()

        if strategy == "uzero":
            vals = vals.fillna(0).replace(-1, 0)
        elif strategy == "uone":
            vals = vals.fillna(0).replace(-1, 1)
        elif strategy == "hybrid2":
            vals = vals.fillna(0)
            if label in hybrid_positive_uncertain:
                vals = vals.replace(-1, 1)
            else:
                vals = vals.replace(-1, 0)
        else:
            raise ValueError(f"Unsupported uncertainty_strategy: {strategy}")

        vals = vals.astype(float).astype(int)

        invalid = set(vals.unique()) - {0, 1}
        if invalid:
            raise ValueError(f"Invalid binary values for {label}: {invalid}")

        out[label] = vals

    return out


def prepare_dataframe(csv_path: Path, archive_root: Path, labels: List[str],
                      uncertainty_strategy: str, view_filter: str,
                      split_source_name: str) -> pd.DataFrame:
    print(f"[LOAD] {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[LOAD] rows={len(df)} columns={len(df.columns)}")

    if "Path" not in df.columns:
        raise ValueError("Input CSV must contain a 'Path' column.")

    if view_filter.lower() != "all":
        if "Frontal/Lateral" not in df.columns:
            raise ValueError("CSV missing 'Frontal/Lateral' column required for view filtering.")
        before = len(df)
        df = df[df["Frontal/Lateral"].astype(str).str.lower() == view_filter.lower()].copy()
        print(f"[FILTER] view={view_filter}: {before} -> {len(df)} rows")

    df["patient_id"] = df["Path"].apply(extract_patient_id)
    df["study_id"] = df["Path"].apply(extract_study_id)
    df["source_split"] = split_source_name

    print("[PATH] resolving local image paths...")
    resolved = []
    missing = 0
    for p in df["Path"].astype(str):
        rp = resolve_image_path(p, archive_root)
        if rp is None:
            missing += 1
            resolved.append(None)
        else:
            resolved.append(str(rp))

    df["local_image_path"] = resolved
    before = len(df)
    df = df[df["local_image_path"].notna()].copy()
    print(f"[PATH] existing images: {len(df)}/{before}; missing={missing}")

    df = apply_uncertainty_strategy(df, labels, uncertainty_strategy)

    keep_cols = [
        "Path",
        "local_image_path",
        "patient_id",
        "study_id",
        "source_split",
        "Sex",
        "Age",
        "Frontal/Lateral",
        "AP/PA",
    ] + labels

    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].reset_index(drop=True)


def patient_wise_target_split(df: pd.DataFrame, target_train: int, target_val: int,
                              target_test: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Randomly selects patients until row targets are reached.

    Because patients are indivisible, final row counts may be slightly above target.
    Remaining patients are returned as unused.
    """
    rng = random.Random(seed)

    patient_sizes = df.groupby("patient_id").size().to_dict()
    patients = list(patient_sizes.keys())
    rng.shuffle(patients)

    buckets = {"train": [], "val": [], "test": [], "unused": []}
    targets = {"train": target_train, "val": target_val, "test": target_test}
    counts = {"train": 0, "val": 0, "test": 0}

    current = "train"

    for pid in patients:
        if current == "train" and counts["train"] >= targets["train"]:
            current = "val"
        if current == "val" and counts["val"] >= targets["val"]:
            current = "test"
        if current == "test" and counts["test"] >= targets["test"]:
            current = "unused"

        buckets[current].append(pid)

        if current in counts:
            counts[current] += patient_sizes[pid]

    split_dfs = {}
    for split, pids in buckets.items():
        split_dfs[split] = df[df["patient_id"].isin(pids)].copy().reset_index(drop=True)

    return split_dfs["train"], split_dfs["val"], split_dfs["test"], split_dfs["unused"]


def label_stats(df: pd.DataFrame, labels: List[str]) -> Dict[str, Dict[str, int | float]]:
    stats = {}
    for label in labels:
        pos = int(df[label].sum())
        total = int(len(df))
        stats[label] = {
            "positive": pos,
            "negative": int(total - pos),
            "prevalence": float(pos / total) if total else 0.0,
        }
    return stats


def verify_patient_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, int]:
    train_pat = set(train_df["patient_id"].astype(str))
    val_pat = set(val_df["patient_id"].astype(str))
    test_pat = set(test_df["patient_id"].astype(str))

    overlaps = {
        "train_val": len(train_pat & val_pat),
        "train_test": len(train_pat & test_pat),
        "val_test": len(val_pat & test_pat),
    }

    if any(overlaps.values()):
        raise ValueError(f"Patient leakage detected: {overlaps}")

    return overlaps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-train-rows", type=int, default=None)
    parser.add_argument("--target-val-rows", type=int, default=None)
    parser.add_argument("--target-test-rows", type=int, default=None)
    parser.add_argument("--view-filter", default=None, choices=["Frontal", "Lateral", "all"])
    parser.add_argument("--strategy", default=None, help="Override output strategy name.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    project_dir = Path(cfg["project_dir"])
    archive_root = Path(cfg["archive_root"])
    train_csv = Path(cfg["train_csv"])
    valid_csv = Path(cfg["valid_csv"])
    labels = list(cfg["target_labels"])
    uncertainty_strategy = cfg.get("uncertainty_strategy", "hybrid2")
    view_filter = args.view_filter or cfg.get("view_filter", "Frontal")
    strategy = args.strategy or cfg.get("strategy", "kaggle80k_hybrid2_frontal")
    seed = int(cfg.get("seed", 42))

    target_train = int(args.target_train_rows or cfg.get("target_train_rows", 80000))
    target_val = int(args.target_val_rows or cfg.get("target_val_rows", 10000))
    target_test = int(args.target_test_rows or cfg.get("target_test_rows", 10000))

    project_dir.mkdir(parents=True, exist_ok=True)
    local_ready = project_dir / "local_ready"
    reports_dir = project_dir / "reports"
    local_ready.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PREPARE CHEXPERT KAGGLE SPLITS")
    print("=" * 80)
    print(f"project_dir: {project_dir}")
    print(f"archive_root: {archive_root}")
    print(f"strategy: {strategy}")
    print(f"view_filter: {view_filter}")
    print(f"uncertainty_strategy: {uncertainty_strategy}")
    print(f"targets: train={target_train}, val={target_val}, test={target_test}")

    train_full = prepare_dataframe(
        train_csv, archive_root, labels, uncertainty_strategy, view_filter, "kaggle_train"
    )

    official_valid = prepare_dataframe(
        valid_csv, archive_root, labels, uncertainty_strategy="uzero", view_filter=view_filter, split_source_name="kaggle_official_valid"
    )

    print("[SPLIT] Creating patient-wise train/val/test splits from Kaggle train.csv...")
    train_df, val_df, test_df, unused_df = patient_wise_target_split(
        train_full, target_train, target_val, target_test, seed
    )

    overlaps = verify_patient_overlap(train_df, val_df, test_df)

    split_map = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "official_valid": official_valid,
        "unused": unused_df,
    }

    summary = {
        "strategy": strategy,
        "view_filter": view_filter,
        "uncertainty_strategy": uncertainty_strategy,
        "labels": labels,
        "overlaps": overlaps,
        "splits": {},
    }

    for split, df in split_map.items():
        print(f"\n[{split.upper()}]")
        print(f"rows={len(df)} patients={df['patient_id'].nunique() if len(df) else 0}")
        print(df[labels].sum().astype(int).to_string() if len(df) else "empty")
        summary["splits"][split] = {
            "rows": int(len(df)),
            "patients": int(df["patient_id"].nunique()) if len(df) else 0,
            "label_stats": label_stats(df, labels) if len(df) else {},
        }

    train_df.to_csv(local_ready / f"train_{strategy}_local.csv", index=False)
    val_df.to_csv(local_ready / f"val_{strategy}_local.csv", index=False)
    test_df.to_csv(local_ready / f"test_{strategy}_local.csv", index=False)
    official_valid.to_csv(local_ready / f"official_valid_{strategy}_local.csv", index=False)
    unused_df.to_csv(local_ready / f"unused_{strategy}_local.csv", index=False)

    with open(reports_dir / f"split_summary_{strategy}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[DONE] Saved split CSVs to:")
    print(local_ready)
    print("[DONE] Saved summary to:")
    print(reports_dir / f"split_summary_{strategy}.json")


if __name__ == "__main__":
    main()
