# 🏠 Indonesian Traditional House Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📋 Overview

This project aims to classify images of Indonesian traditional houses into five distinct categories: **Balinese, Batak, Dayak, Javanese, and Minangkabau**.

Leveraging deep learning techniques, specifically transfer learning with **EfficientNetV2M**, this model is designed to handle real-world challenges such as class imbalance, noisy labels, and varying image qualities. The pipeline includes advanced preprocessing (CLAHE), a multi-stage training strategy with hard-negative mining, and post-processing threshold optimization to maximize the F1-Macro score.

## 🧠 Key Features & Methodology

### 1. Data Preprocessing & Cleaning
* **Label Correction**: Manual relabeling was performed on misclassified training images (e.g., Balinese pagodas mislabeled as Javanese) to ensure ground truth quality.
* **Image Enhancement**:
    * **Denoising & CLAHE**: Applied Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance local contrast and definition in house structures.
    * **Augmentation**: Random flip, brightness, and contrast adjustments during training to improve generalization.

### 2. Model Architecture
We utilized **EfficientNetV2M** pretrained on ImageNet as the backbone, fine-tuned for this specific task.
* **Backbone**: Frozen initially, then partially unfrozen for fine-tuning.
* **Custom Head**:
    * `GlobalAveragePooling2D`
    * `GaussianNoise(0.08)` for robustness.
    * `Dense(256, ReLU)` + `Dropout(0.5)` to prevent overfitting.
    * Output Layer with L2 regularization.

### 3. Advanced Training Strategy
* **Handling Imbalance**: Utilized **Weighted Focal Loss** to focus learning on hard-to-classify examples and minority classes (Batak, Dayak).
* **Three-Stage Training**:
    1.  **Warmup**: Train only the custom head with a frozen backbone.
    2.  **Fine-tuning**: Unfreeze the top 30% of EfficientNetV2M layers with a lower learning rate (`1e-5`).
    3.  **Hard Fine-tuning**: Identified consistently misclassified validation images and performed an additional training pass specifically on these "hard" examples.

### 4. Post-Processing
* **Threshold Tuning**: Instead of a default 0.5 softmax threshold, we optimized classification thresholds per class to maximize the Macro F1-Score on the validation set.

## 📂 Dataset Structure

The project expects the following directory structure:

```
/kaggle/input/lomba-ui/
├── Train/
│   ├── balinese/
│   ├── batak/
│   ├── dayak/
│   ├── javanese/
│   └── minangkabau/
└── Test/
    └── (test images)
```

## 🛠️ Installation & Requirements

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/yourusername/indonesian-house-classification.git](https://github.com/yourusername/indonesian-house-classification.git)
cd indonesian-house-classification
pip install -r requirements.txt
```

**Key Requirements:**
* TensorFlow 2.x
* NumPy
* Pandas
* OpenCV (`opencv-python-headless`)
* scikit-learn
* Matplotlib & Seaborn

## 🚀 Usage

### Training
The training pipeline is encapsulated in the main notebook/script. It will automatically:
1. Load and clean the data.
2. execute the 3-stage training process.
3. Save the best models as `.keras` files (`model_EFF2V.keras`, `model_EFF2V_FINETUNED.keras`, `model_EFF2V_HARD_FINETUNED.keras`).

### Inference
To generate predictions on the Test set using the optimized thresholds:

```python
# Ensure the best model is loaded
model.load_weights("model_EFF2V_HARD_FINETUNED.keras")

# The script will automatically generate 'submission_thresholded.csv'
# containing the image IDs and their predicted styles.
```

## 📊 Results & Evaluation

The model is evaluated primarily on the **Macro F1-Score** due to class imbalance.

| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| Balinese | ... | ... | ... |
| Batak | ... | ... | ... |
| Dayak | ... | ... | ... |
| Javanese | ... | ... | ... |
| Minangkabau| ... | ... | ... |
| **Macro Avg**| **...** | **...** | **...** |

*(Note: Actual results depend on the final training run).*

The confusion matrices during training demonstrate significant improvement after applying Weighted Focal Loss and Hard Fine-tuning, particularly in distinguishing between visually similar Javanese and Balinese architectures.
