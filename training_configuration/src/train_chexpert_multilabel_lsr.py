"""
train_chexpert_multilabel_lsr.py

Train multilabel CNN models for CheXpert Kaggle frontal chest X-rays.

Supported models:
- convnext_tiny
- efficientnet_b0
- densenet121

Outputs:
- models/<model>_<tag>_<strategy>_best.pt
- results/<tag>/training_history.csv
- results/<tag>/val_true.npy, val_prob.npy
- results/<tag>/test_true.npy, test_prob.npy
- results/<tag>/test_metrics_threshold_0p5.csv
- results/<tag>/test_metrics_optimized_thresholds.csv
- results/<tag>/summary_test.csv
- results/<tag>/confusion_matrix_summary.csv
"""

from __future__ import annotations

import argparse
import json
import random
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

try:
    import yaml
except ImportError as exc:
    raise SystemExit("Missing dependency: pip install pyyaml") from exc


@dataclass
class Config:
    project_dir: str
    strategy: str
    target_labels: Tuple[str, ...]
    img_size: int = 320
    batch_size: int = 8
    num_workers: int = 0
    seed: int = 42
    epochs: int = 35
    patience: int = 6
    freeze_epochs: int = 2
    head_lr: float = 3e-4
    backbone_lr: float = 2e-5
    head_lr_finetune: float = 1e-4
    weight_decay: float = 5e-5
    dropout: float = 0.30
    pos_weight_mode: str = "sqrt"
    pos_weight_cap: float = 4.0
    use_amp: bool = True
    use_weighted_sampler: bool = False
    horizontal_flip_p: float = 0.20
    rotation_degrees: float = 7.0
    translate: float = 0.03
    scale_min: float = 0.97
    scale_max: float = 1.03


class CheXpertDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, target_labels: List[str], transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.target_labels = target_labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["local_image_path"]).convert("RGB")
        labels = row[self.target_labels].astype("float32").values
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(labels, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--strategy", default=None)
    p.add_argument("--model-name", default="convnext_tiny",
                   choices=["convnext_tiny", "efficientnet_b0", "densenet121"])
    p.add_argument("--tag", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--min-specificity", type=float, default=None)
    return p.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    with open(args.config, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)

    if args.strategy is not None:
        d["strategy"] = args.strategy
    if args.batch_size is not None:
        d["batch_size"] = args.batch_size
    if args.epochs is not None:
        d["epochs"] = args.epochs
    if args.patience is not None:
        d["patience"] = args.patience

    d["target_labels"] = tuple(d["target_labels"])

    allowed = set(Config.__dataclass_fields__.keys())
    d = {k: v for k, v in d.items() if k in allowed}
    return Config(**d)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def load_split(project_dir: Path, split: str, strategy: str) -> pd.DataFrame:
    path = project_dir / "local_ready" / f"{split}_{strategy}_local.csv"
    print(f"[DATA] Loading {split}: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Missing split CSV: {path}")
    return pd.read_csv(path)


def verify_dataframes(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                      labels: List[str]) -> None:
    """Verify split integrity.

    LSR-compatible behavior:
    - train labels may be soft values in [0, 1]
    - val/test labels must remain binary {0, 1}, because metrics, thresholds,
      confusion matrices, accuracy, precision, recall and F1 require hard labels.
    """
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if "local_image_path" not in df.columns:
            raise ValueError(f"{name} split missing local_image_path column.")
        missing = (~df["local_image_path"].apply(lambda p: Path(str(p)).exists())).sum()
        print(f"[DATA] {name}: rows={len(df)}, patients={df['patient_id'].nunique() if 'patient_id' in df.columns else 'NA'}, missing_images={missing}")
        if missing:
            raise FileNotFoundError(f"{missing} missing images in {name}")

        for label in labels:
            if label not in df.columns:
                raise ValueError(f"{name} split missing label column: {label}")

            values = df[label].dropna().astype(float)

            if name == "train":
                bad = values[(values < 0.0) | (values > 1.0)]
                if len(bad):
                    raise ValueError(
                        f"{name}:{label} has values outside [0,1]. "
                        f"Examples: {bad.head(10).tolist()}"
                    )
            else:
                unique = set(values.unique())
                invalid = unique - {0, 1, 0.0, 1.0}
                if invalid:
                    short = list(sorted(invalid))[:20]
                    raise ValueError(
                        f"{name}:{label} must be binary for evaluation. "
                        f"Invalid examples: {short}"
                    )

    if "patient_id" in train_df.columns:
        train_pat = set(train_df["patient_id"].astype(str))
        val_pat = set(val_df["patient_id"].astype(str))
        test_pat = set(test_df["patient_id"].astype(str))
        overlaps = {
            "train_val": len(train_pat & val_pat),
            "train_test": len(train_pat & test_pat),
            "val_test": len(val_pat & test_pat),
        }
        print(f"[DATA] patient overlaps: {overlaps}")
        if any(overlaps.values()):
            raise ValueError(f"Patient-wise leakage detected: {overlaps}")
    else:
        warnings.warn("patient_id missing; cannot verify patient-wise leakage.")

def get_model_and_transforms(model_name: str, num_labels: int, cfg: Config):
    if model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(in_features, num_labels))
        mean, std = weights.transforms().mean, weights.transforms().std

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(in_features, num_labels))
        mean, std = weights.transforms().mean, weights.transforms().std

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(in_features, num_labels))
        mean, std = weights.transforms().mean, weights.transforms().std

    else:
        raise ValueError(model_name)

    train_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(p=cfg.horizontal_flip_p),
        transforms.RandomRotation(degrees=cfg.rotation_degrees),
        transforms.RandomAffine(
            degrees=0,
            translate=(cfg.translate, cfg.translate),
            scale=(cfg.scale_min, cfg.scale_max),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return model, train_tf, eval_tf


def get_feature_module(model: nn.Module) -> nn.Module:
    if hasattr(model, "features"):
        return model.features
    raise ValueError("Model does not expose .features; freeze/unfreeze not implemented.")


def get_classifier_module(model: nn.Module) -> nn.Module:
    if hasattr(model, "classifier"):
        return model.classifier
    raise ValueError("Model does not expose .classifier.")


def freeze_features(model: nn.Module, freeze: bool) -> None:
    for p in get_feature_module(model).parameters():
        p.requires_grad = not freeze


def compute_pos_weight(train_df: pd.DataFrame, labels: List[str], mode: str, cap: float) -> Optional[torch.Tensor]:
    y = train_df[labels].astype(float).values
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    standard = neg / np.maximum(pos, 1)

    if mode == "none":
        return None
    if mode == "standard":
        values = standard
    elif mode == "sqrt":
        values = np.sqrt(standard)
    elif mode == "capped":
        values = np.minimum(standard, cap)
    else:
        raise ValueError(f"Unsupported pos_weight_mode: {mode}")

    for label, value in zip(labels, values):
        print(f"[POSW] {label}: {value:.3f}")

    return torch.tensor(values, dtype=torch.float32)


def build_sampler(train_df: pd.DataFrame, labels: List[str]) -> WeightedRandomSampler:
    y = train_df[labels].astype(float).values
    pos_counts = y.sum(axis=0)
    class_weights = 1.0 / np.maximum(pos_counts, 1)
    sample_weight = 1.0 + (y * class_weights).sum(axis=1)
    sample_weight = sample_weight / sample_weight.mean()
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weight, dtype=torch.double),
        num_samples=len(sample_weight),
        replacement=True,
    )


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp) -> float:
    model.train()
    running = 0.0

    for images, targets in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += float(loss.item()) * images.size(0)

    return running / len(loader.dataset)


@torch.no_grad()
def validate_loss(model, loader, criterion, device) -> float:
    model.eval()
    running = 0.0

    for images, targets in tqdm(loader, desc="Val loss", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        running += float(loss.item()) * images.size(0)

    return running / len(loader.dataset)


@torch.no_grad()
def collect_predictions(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []

    for images, targets in tqdm(loader, desc="Predicting", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        ys.append(targets.cpu().numpy())
        ps.append(probs)

    return np.concatenate(ys), np.concatenate(ps)


def safe_confusion(y_t: np.ndarray, y_p: np.ndarray) -> Tuple[int, int, int, int]:
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, labels: List[str], threshold=0.5) -> pd.DataFrame:
    if np.isscalar(threshold):
        y_pred = (y_prob >= float(threshold)).astype(int)
    else:
        y_pred = np.zeros_like(y_prob, dtype=int)
        for i, label in enumerate(labels):
            y_pred[:, i] = (y_prob[:, i] >= threshold[label]).astype(int)

    rows = []

    for i, label in enumerate(labels):
        y_t = y_true[:, i].astype(int)
        y_p = y_pred[:, i].astype(int)
        y_s = y_prob[:, i]

        try:
            auc_roc = roc_auc_score(y_t, y_s)
        except ValueError:
            auc_roc = np.nan
        try:
            auc_pr = average_precision_score(y_t, y_s)
        except ValueError:
            auc_pr = np.nan

        tn, fp, fn, tp = safe_confusion(y_t, y_p)

        rows.append({
            "label": label,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "accuracy": accuracy_score(y_t, y_p),
            "precision": precision_score(y_t, y_p, zero_division=0),
            "recall_sensitivity": recall_score(y_t, y_p, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
            "f1": f1_score(y_t, y_p, zero_division=0),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "positive_n": int(y_t.sum()),
            "negative_n": int(len(y_t) - y_t.sum()),
        })

    return pd.DataFrame(rows)


def summary_from_metrics(metrics: pd.DataFrame, strategy: str, tag: str, threshold_name: str) -> Dict:
    return {
        "strategy": strategy,
        "tag": tag,
        "threshold": threshold_name,
        "macro_auc_roc": float(np.nanmean(metrics["auc_roc"])),
        "macro_auc_pr": float(np.nanmean(metrics["auc_pr"])),
        "macro_f1": float(np.nanmean(metrics["f1"])),
        "macro_recall": float(np.nanmean(metrics["recall_sensitivity"])),
        "macro_precision": float(np.nanmean(metrics["precision"])),
        "macro_specificity": float(np.nanmean(metrics["specificity"])),
        "macro_accuracy": float(np.nanmean(metrics["accuracy"])),
    }


def optimize_thresholds(y_true: np.ndarray, y_prob: np.ndarray, labels: List[str],
                        min_specificity: Optional[float] = None) -> Dict[str, float]:
    thresholds = {}
    grid = np.arange(0.05, 0.96, 0.01)

    for i, label in enumerate(labels):
        y_t = y_true[:, i].astype(int)
        y_s = y_prob[:, i]
        best_thr = 0.5
        best_f1 = -1.0

        for thr in grid:
            y_p = (y_s >= thr).astype(int)
            tn, fp, fn, tp = safe_confusion(y_t, y_p)
            spec = tn / (tn + fp) if (tn + fp) else 0.0

            if min_specificity is not None and spec < min_specificity:
                continue

            score = f1_score(y_t, y_p, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thr = float(thr)

        thresholds[label] = best_thr
        print(f"[THRESH] {label}: {best_thr:.2f} val_f1={best_f1:.4f}")

    return thresholds


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    tag = args.tag or f"{args.model_name}_{cfg.strategy}"
    labels = list(cfg.target_labels)

    set_seed(cfg.seed)

    project_dir = Path(cfg.project_dir)
    out_dir = project_dir / "results" / tag
    model_dir = project_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TRAIN CHEXPERT MULTILABEL")
    print("=" * 80)
    print(f"[CONFIG] strategy={cfg.strategy} model={args.model_name} tag={tag}")
    print(f"[CONFIG] img_size={cfg.img_size} batch_size={cfg.batch_size} epochs={cfg.epochs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.use_amp and device.type == "cuda")
    print(f"[SYS] device={device}")
    if torch.cuda.is_available():
        print(f"[SYS] GPU={torch.cuda.get_device_name(0)}")
    print(f"[SYS] AMP={use_amp}")

    train_df = load_split(project_dir, "train", cfg.strategy)
    val_df = load_split(project_dir, "val", cfg.strategy)
    test_df = load_split(project_dir, "test", cfg.strategy)
    verify_dataframes(train_df, val_df, test_df, labels)

    model, train_tf, eval_tf = get_model_and_transforms(args.model_name, len(labels), cfg)
    model = model.to(device)

    train_ds = CheXpertDataset(train_df, labels, train_tf)
    val_ds = CheXpertDataset(val_df, labels, eval_tf)
    test_ds = CheXpertDataset(test_df, labels, eval_tf)

    sampler = build_sampler(train_df, labels) if cfg.use_weighted_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda")
    )

    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda")
    )

    pos_weight = compute_pos_weight(train_df, labels, cfg.pos_weight_mode, cfg.pos_weight_cap)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    freeze_features(model, True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.head_lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.3, patience=2, min_lr=1e-7
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auc = -np.inf
    best_epoch = -1
    no_improve = 0
    history = []
    phase = "frozen_head"
    checkpoint_path = model_dir / f"{args.model_name}_{tag}_{cfg.strategy}_best.pt"

    for epoch in range(1, cfg.epochs + 1):
        if epoch == cfg.freeze_epochs + 1:
            print("[TRAIN] Unfreezing backbone.")
            freeze_features(model, False)
            phase = "fine_tuning"
            optimizer = torch.optim.AdamW(
                [
                    {"params": get_feature_module(model).parameters(), "lr": cfg.backbone_lr},
                    {"params": get_classifier_module(model).parameters(), "lr": cfg.head_lr_finetune},
                ],
                weight_decay=cfg.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.3, patience=3, min_lr=1e-7
            )
            no_improve = 0

        start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp)
        val_loss = validate_loss(model, val_loader, criterion, device)
        val_true, val_prob = collect_predictions(model, val_loader, device)

        val_metrics = multilabel_metrics(val_true, val_prob, labels, threshold=0.5)
        val_auc = float(np.nanmean(val_metrics["auc_roc"]))
        val_f1 = float(np.nanmean(val_metrics["f1"]))

        scheduler.step(val_auc)

        row = {
            "epoch": epoch,
            "phase": phase,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_auc_roc": val_auc,
            "val_macro_auc_pr": float(np.nanmean(val_metrics["auc_pr"])),
            "val_macro_f1": val_f1,
            "lr_groups": str([g["lr"] for g in optimizer.param_groups]),
            "time_sec": time.time() - start,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | {phase} | "
            f"train={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_auc={val_auc:.4f} | val_f1={val_f1:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "model_name": args.model_name,
                    "target_labels": labels,
                    "best_epoch": best_epoch,
                    "best_val_auc": best_auc,
                },
                checkpoint_path,
            )
            print(f"  ✓ Saved best checkpoint (epoch={epoch}, val_auc={best_auc:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement: {no_improve}/{cfg.patience}")
            if no_improve >= cfg.patience:
                print("[TRAIN] Early stopping.")
                break

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[EVAL] Loaded best checkpoint: epoch={checkpoint['best_epoch']} val_auc={checkpoint['best_val_auc']:.4f}")

    val_true, val_prob = collect_predictions(model, val_loader, device)
    test_true, test_prob = collect_predictions(model, test_loader, device)

    np.save(out_dir / "val_true.npy", val_true)
    np.save(out_dir / "val_prob.npy", val_prob)
    np.save(out_dir / "test_true.npy", test_true)
    np.save(out_dir / "test_prob.npy", test_prob)

    print("[THRESH] Optimizing thresholds on validation set...")
    thresholds = optimize_thresholds(val_true, val_prob, labels, min_specificity=args.min_specificity)

    with open(out_dir / "best_thresholds_f1.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    metrics_05 = multilabel_metrics(test_true, test_prob, labels, threshold=0.5)
    metrics_opt = multilabel_metrics(test_true, test_prob, labels, threshold=thresholds)

    metrics_05.to_csv(out_dir / "test_metrics_threshold_0p5.csv", index=False)
    metrics_opt.to_csv(out_dir / "test_metrics_optimized_thresholds.csv", index=False)

    confusion_rows = []
    for _, row in metrics_opt.iterrows():
        confusion_rows.append({
            "label": row["label"],
            "tp": int(row["tp"]),
            "fp": int(row["fp"]),
            "tn": int(row["tn"]),
            "fn": int(row["fn"]),
        })
    pd.DataFrame(confusion_rows).to_csv(out_dir / "confusion_matrix_summary.csv", index=False)

    summary = [
        summary_from_metrics(metrics_05, cfg.strategy, tag, "0.5"),
        summary_from_metrics(metrics_opt, cfg.strategy, tag, "optimized_on_validation"),
    ]

    pd.DataFrame(summary).to_csv(out_dir / "summary_test.csv", index=False)
    with open(out_dir / "summary_test.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[RESULTS]")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\n[DONE] Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
