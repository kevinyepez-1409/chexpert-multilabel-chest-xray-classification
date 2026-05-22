GRAD-CAM DEMO FOR CHEXPERT CONVNEXT-TINY LSR MODEL
===================================================

Purpose
-------
This package creates a simple visual explainability demo for the trained model:

  ConvNeXt-Tiny + Hybrid2 LSR
  tag: convnext_tiny_kaggle80k_320_lsr
  strategy: kaggle80k_hybrid2_lsr_frontal

It loads the saved checkpoint, selects one frontal chest X-ray from the test set,
computes the predicted probabilities for the 5 CheXpert labels, and generates a
Grad-CAM heatmap for the selected pathology.

This is intended for the project presentation as a visual example of:

  Input X-ray -> CNN prediction -> predicted probabilities -> Grad-CAM explanation

Important note
--------------
Grad-CAM is generated from one individual ConvNeXt-Tiny model, not from the ensemble.
That is methodologically acceptable because Grad-CAM requires a single neural network
and a target convolutional feature layer. If your final table uses an ensemble, you can
say that Grad-CAM was generated using the best individual ConvNeXt model for visual
interpretability.

Expected project structure
--------------------------
Your project should be located here:

  C:\Users\User\Documents\BIOMEDICINA\Noveno\Redes Neuronales\chexpert_kaggle_final_project

Expected checkpoint:

  models\convnext_tiny_convnext_tiny_kaggle80k_320_lsr_kaggle80k_hybrid2_lsr_frontal_best.pt

Expected test CSV:

  local_ready\test_kaggle80k_hybrid2_lsr_frontal_local.csv

How to use the notebook
-----------------------
1. Copy the notebook:

   notebooks\02_demo_gradcam_convnext_lsr.ipynb

   into:

   C:\Users\User\Documents\BIOMEDICINA\Noveno\Redes Neuronales\chexpert_kaggle_final_project\notebooks

2. Open it with Jupyter or VS Code.
3. Select the chexpert_v4 environment/kernel.
4. Run the cells in order.

How to use the script from PowerShell
-------------------------------------
1. Copy the script:

   src\run_gradcam_convnext_lsr.py

   into:

   C:\Users\User\Documents\BIOMEDICINA\Noveno\Redes Neuronales\chexpert_kaggle_final_project\chexpert_kaggle_final_project\src

2. Activate the environment:

   conda activate chexpert_v4

3. Go to the project code folder:

   cd "C:\Users\User\Documents\BIOMEDICINA\Noveno\Redes Neuronales\chexpert_kaggle_final_project\chexpert_kaggle_final_project"

4. Run the default demo:

   python src\run_gradcam_convnext_lsr.py

Optional examples
-----------------
Explain a specific class:

   python src\run_gradcam_convnext_lsr.py --class-name "Pleural Effusion"

Use a specific row from the test CSV:

   python src\run_gradcam_convnext_lsr.py --sample-index 25

Use a specific image path:

   python src\run_gradcam_convnext_lsr.py --image-path "C:\path\to\view1_frontal.jpg" --class-name "Edema"

Generate a grid with Grad-CAM for all five labels:

   python src\run_gradcam_convnext_lsr.py --all-classes

Outputs
-------
The script saves results to:

  C:\Users\User\Documents\BIOMEDICINA\Noveno\Redes Neuronales\chexpert_kaggle_final_project\deliverables\gradcam_convnext_lsr

Expected files:

  prediction_table.csv
  gradcam_<class_name>.png
  gradcam_all_classes.png              if --all-classes is used

Presentation suggestion
-----------------------
Use the image with two panels:

  Original chest X-ray | Grad-CAM heatmap

Suggested caption:

  "Grad-CAM visualization for the ConvNeXt-Tiny LSR model. The heatmap highlights
  image regions that contributed most strongly to the selected pathology prediction."

Limitations
-----------
Grad-CAM is an interpretability aid, not a clinical proof. It should be used to support
model transparency, but it does not replace radiologist evaluation.
