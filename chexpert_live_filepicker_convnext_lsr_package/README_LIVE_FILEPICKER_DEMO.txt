LIVE FILE EXPLORER DEMO - CHEXPERT CONVNEXT-TINY LSR
======================================================

This package contains a Jupyter notebook for a live presentation demo.

Notebook:
  notebooks/04_live_filepicker_prediction_convnext_lsr.ipynb

Purpose:
  1. Open Windows File Explorer from the notebook.
  2. Let the user select a chest X-ray image.
  3. Load the trained ConvNeXt-Tiny + Hybrid2 LSR checkpoint.
  4. Predict five CheXpert labels.
  5. Apply optimized validation thresholds.
  6. Generate a Grad-CAM explanation for the highest-probability class, or for a manually selected class.

Recommended location:
  Copy this folder to:
  C:\Users\User\Documents\BIOMEDICINA\Noveno\Redes Neuronales\chexpert_kaggle_final_project\deliverables\live_filepicker_demo_convnext_lsr

How to run:
  1. Open VS Code or Jupyter.
  2. Select the chexpert_v4 kernel/environment.
  3. Open notebooks/04_live_filepicker_prediction_convnext_lsr.ipynb.
  4. Run cells from top to bottom.
  5. When the file explorer opens, select any CXR image.

Expected model checkpoint:
  C:\Users\User\Documents\BIOMEDICINA\Noveno\Redes Neuronales\chexpert_kaggle_final_project\models\convnext_tiny_convnext_tiny_kaggle80k_320_lsr_kaggle80k_hybrid2_lsr_frontal_best.pt

Expected output folder:
  C:\Users\User\Documents\BIOMEDICINA\Noveno\Redes Neuronales\chexpert_kaggle_final_project\deliverables\live_filepicker_demo_convnext_lsr\outputs

Notes:
  - If the file explorer does not appear, check the taskbar because it may open behind VS Code/Jupyter.
  - If tkinter does not work in the current environment, use the manual path fallback cell.
  - The notebook is for educational demonstration only, not clinical diagnosis.
