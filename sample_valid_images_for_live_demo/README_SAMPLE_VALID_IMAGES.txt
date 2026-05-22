
Sample validation images for live model testing

Source:
C:\Users\User\Documents\BIOMEDICINA\Noveno\Redes Neuronales\archive\valid

CSV:
C:\Users\User\Documents\BIOMEDICINA\Noveno\Redes Neuronales\archive\valid.csv

Number of copied images:
20

Purpose:
These images come from the official Kaggle CheXpert valid folder and were not used for model training.
They can be used in the live inference notebook to demonstrate how the ConvNeXt-Tiny LSR model classifies unseen chest X-rays.

Recommended notebook:
deliverables/live_filepicker_demo_convnext_lsr/04_live_filepicker_prediction_convnext_lsr.ipynb

How to use:
1. Open the live inference notebook.
2. Run the setup cells.
3. Use the file picker to select one image from this folder.
4. The notebook will display:
   - The chest X-ray
   - Predicted probabilities
   - Binary predictions using optimized thresholds
   - Grad-CAM explanation

Labels:
Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion
