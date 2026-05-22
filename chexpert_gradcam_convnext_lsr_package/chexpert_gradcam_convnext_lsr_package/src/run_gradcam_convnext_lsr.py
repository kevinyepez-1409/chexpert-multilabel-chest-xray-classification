"""
Grad-CAM demo for the CheXpert ConvNeXt-Tiny + Hybrid2 LSR model.

This script loads the trained ConvNeXt-Tiny LSR checkpoint, predicts five
CheXpert labels from one frontal chest X-ray, and saves a Grad-CAM heatmap.

Default model:
    convnext_tiny_kaggle80k_320_lsr

Default strategy:
    kaggle80k_hybrid2_lsr_frontal

Run from PowerShell:
    conda activate chexpert_v4
    cd "C:\\Users\\User\\Documents\\BIOMEDICINA\\Noveno\\Redes Neuronales\\chexpert_kaggle_final_project\\chexpert_kaggle_final_project"
    python src\\run_gradcam_convnext_lsr.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

import matplotlib.pyplot as plt
from matplotlib import cm


DEFAULT_PROJECT_DIR = Path(
    r"C:\Users\User\Documents\BIOMEDICINA\Noveno\Redes Neuronales\chexpert_kaggle_final_project"
)
DEFAULT_MODEL_NAME = "convnext_tiny"
DEFAULT_TAG = "convnext_tiny_kaggle80k_320_lsr"
DEFAULT_STRATEGY = "kaggle80k_hybrid2_lsr_frontal"
DEFAULT_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]


class GradCAM:
    """Minimal Grad-CAM implementation for CNN-like feature maps."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: Tuple[torch.Tensor],
        grad_output: Tuple[torch.Tensor],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        activations = self.activations
        gradients = self.gradients

        # ConvNeXt target layer should produce [B, C, H, W].
        if activations.ndim != 4 or gradients.ndim != 4:
            raise RuntimeError(
                f"Expected 4D feature maps, got activations={activations.shape}, gradients={gradients.shape}"
            )

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam_tensor = (weights * activations).sum(dim=1, keepdim=True)
        cam_tensor = F.relu(cam_tensor)
        cam_tensor = F.interpolate(cam_tensor, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam_tensor[0, 0].detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def build_convnext_tiny(num_labels: int, dropout: float = 0.40):
    """Build the same ConvNeXt-Tiny head used during training."""
    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = models.convnext_tiny(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_labels))

    mean = weights.transforms().mean
    std = weights.transforms().std

    # Last feature block of ConvNeXt. Suitable for Grad-CAM.
    target_layer = model.features[-1]
    return model, mean, std, target_layer


def load_checkpoint(
    checkpoint_path: Path,
    model_name: str,
    fallback_labels: List[str],
    device: torch.device,
):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    labels = checkpoint.get("target_labels", fallback_labels)
    config = checkpoint.get("config", {})
    dropout = float(config.get("dropout", 0.40))
    img_size = int(config.get("img_size", 320))

    if model_name != "convnext_tiny":
        raise ValueError("This demo script is configured for model_name='convnext_tiny'.")

    model, mean, std, target_layer = build_convnext_tiny(num_labels=len(labels), dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, labels, mean, std, target_layer, img_size, checkpoint


def load_thresholds(results_dir: Path, tag: str, labels: List[str]) -> Dict[str, float]:
    path = results_dir / tag / "best_thresholds_f1.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {label: 0.5 for label in labels}


def select_image(
    test_csv: Path,
    labels: List[str],
    image_path: Optional[str],
    sample_index: Optional[int],
    seed: int,
):
    if image_path:
        return Path(image_path), None

    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    df = pd.read_csv(test_csv)
    if "local_image_path" not in df.columns:
        raise ValueError("The test CSV must contain a 'local_image_path' column.")

    if sample_index is not None:
        if sample_index < 0 or sample_index >= len(df):
            raise IndexError(f"sample_index must be between 0 and {len(df)-1}")
        row = df.iloc[sample_index]
    else:
        rng = random.Random(seed)
        row = df.iloc[rng.randrange(len(df))]

    selected_path = Path(row["local_image_path"])
    true_labels = {label: int(row[label]) for label in labels if label in row.index}
    return selected_path, true_labels


def preprocess_image(image_path: Path, img_size: int, mean, std, device: torch.device):
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    return image, input_tensor


def overlay_cam_on_image(image: Image.Image, cam: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    image_resized = image.resize((cam.shape[1], cam.shape[0]))
    base = np.asarray(image_resized).astype(np.float32) / 255.0
    heatmap = cm.get_cmap("jet")(cam)[..., :3]
    overlay = (1.0 - alpha) * base + alpha * heatmap
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


def save_single_gradcam(
    image: Image.Image,
    overlay: np.ndarray,
    class_name: str,
    probability: float,
    output_path: Path,
    img_size: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image.resize((img_size, img_size)), cmap="gray")
    ax1.set_title("Original chest X-ray")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(overlay)
    ax2.set_title(f"Grad-CAM: {class_name}\nProbability: {probability:.3f}")
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_all_classes_grid(
    image: Image.Image,
    cams: Dict[str, np.ndarray],
    probs: Dict[str, float],
    output_path: Path,
    img_size: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(cams)
    fig = plt.figure(figsize=(4 * n, 4))
    for i, (label, cam) in enumerate(cams.items(), start=1):
        ax = fig.add_subplot(1, n, i)
        overlay = overlay_cam_on_image(image, cam)
        ax.imshow(overlay)
        ax.set_title(f"{label}\nP={probs[label]:.3f}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM for ConvNeXt-Tiny LSR CheXpert model")
    parser.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tag", type=str, default=DEFAULT_TAG)
    parser.add_argument("--strategy", type=str, default=DEFAULT_STRATEGY)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--class-name", type=str, default=None, help="Pathology name to explain. Default: highest probability.")
    parser.add_argument("--all-classes", action="store_true", help="Also generate a Grad-CAM grid for all labels.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    models_dir = project_dir / "models"
    results_dir = project_dir / "results"
    local_ready_dir = project_dir / "local_ready"

    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else models_dir / f"{args.model_name}_{args.tag}_{args.strategy}_best.pt"
    test_csv = local_ready_dir / f"test_{args.strategy}_local.csv"
    output_dir = Path(args.output_dir) if args.output_dir else project_dir / "deliverables" / "gradcam_convnext_lsr"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("GRAD-CAM DEMO - CHEXPERT CONVNEXT-TINY LSR")
    print("=" * 80)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test CSV: {test_csv}")
    print(f"Output dir: {output_dir}")

    model, labels, mean, std, target_layer, img_size, checkpoint = load_checkpoint(
        checkpoint_path=checkpoint_path,
        model_name=args.model_name,
        fallback_labels=DEFAULT_LABELS,
        device=device,
    )
    thresholds = load_thresholds(results_dir=results_dir, tag=args.tag, labels=labels)

    image_path, true_labels = select_image(
        test_csv=test_csv,
        labels=labels,
        image_path=args.image_path,
        sample_index=args.sample_index,
        seed=args.seed,
    )
    image, input_tensor = preprocess_image(image_path, img_size, mean, std, device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs_array = torch.sigmoid(logits).detach().cpu().numpy()[0]

    prob_by_label = {label: float(probs_array[i]) for i, label in enumerate(labels)}
    pred_table = pd.DataFrame(
        {
            "label": labels,
            "probability": [prob_by_label[label] for label in labels],
            "threshold": [float(thresholds.get(label, 0.5)) for label in labels],
        }
    )
    pred_table["prediction"] = pred_table["probability"] >= pred_table["threshold"]
    if true_labels is not None:
        pred_table["true_label"] = [true_labels.get(label, np.nan) for label in labels]
    pred_table = pred_table.sort_values("probability", ascending=False)

    prediction_csv = output_dir / "prediction_table.csv"
    pred_table.to_csv(prediction_csv, index=False)
    print("\nSelected image:")
    print(image_path)
    print("\nPrediction table:")
    print(pred_table.to_string(index=False))
    print(f"\nSaved prediction table: {prediction_csv}")

    if args.class_name:
        if args.class_name not in labels:
            raise ValueError(f"Unknown class name: {args.class_name}. Available labels: {labels}")
        class_name = args.class_name
    else:
        class_name = pred_table.iloc[0]["label"]

    class_idx = labels.index(class_name)
    gradcam = GradCAM(model, target_layer)
    cam = gradcam(input_tensor, class_idx)
    overlay = overlay_cam_on_image(image, cam)
    safe_class_name = class_name.replace(" ", "_").replace("/", "_")
    fig_path = output_dir / f"gradcam_{safe_class_name}.png"
    save_single_gradcam(
        image=image,
        overlay=overlay,
        class_name=class_name,
        probability=prob_by_label[class_name],
        output_path=fig_path,
        img_size=img_size,
    )
    print(f"Saved Grad-CAM figure: {fig_path}")

    if args.all_classes:
        cams = {}
        for label in labels:
            cams[label] = gradcam(input_tensor, labels.index(label))
        grid_path = output_dir / "gradcam_all_classes.png"
        save_all_classes_grid(image=image, cams=cams, probs=prob_by_label, output_path=grid_path, img_size=img_size)
        print(f"Saved all-class Grad-CAM grid: {grid_path}")

    gradcam.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
