# Cancerous Image Classification Using Deep Learning

![Project Banner](https://img.shields.io/badge/Deep%20Learning-Cancer%20Detection-blue)

## Overview
This project focuses on the classification of histopathological images to detect cancerous tissue using deep learning techniques. The goal is to develop an automated system that can accurately classify medical images as cancerous or non-cancerous, aiding early diagnosis and improving healthcare outcomes.

The study leverages the PatchCamelyon (PCam) dataset, a large-scale benchmark dataset of histopathology image patches, to train convolutional neural networks (CNN) for binary classification and further explores multi-class classification and image segmentation techniques.

---

## Table of Contents
- [Background](#background)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training--evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Acknowledgments](#acknowledgments)
- [References](#references)

---

## Background
Early cancer detection is critical for effective treatment and improved patient survival. Histopathological imaging remains the gold standard for diagnosis but is time-consuming and dependent on expert pathologists. Deep learning, especially CNNs, offers promising automated solutions to classify cancerous tissue from image data with high accuracy.

---

## Dataset
- **PatchCamelyon (PCam)** dataset consists of 327,680 RGB image patches (96x96 pixels) extracted from lymph node sections.
- Images are labeled binary: positive if the central 32x32 pixel region contains tumor tissue, negative otherwise.
- Dataset splits:
  - Training: 262,144 images
  - Validation: 32,768 images
  - Testing: 32,768 images
- Balanced classes with equal positive and negative samples.

---

## Methodology

### Data Preprocessing
- Image normalization by scaling pixel values to [0, 1].
- Mean subtraction to center data.
- Data augmentation: random rotations (±30°), horizontal and vertical flips, scaling, cropping, brightness and contrast adjustments.
- Addressing class imbalance through oversampling and weighted loss functions.

### Model Development
- A custom CNN model designed with three convolutional layers followed by fully connected layers.
- Binary classification using sigmoid activation in the output layer.
- Multi-class classification and clustering using Gaussian Mixture Models (GMM).
- Image segmentation with Expectation-Maximization (EM) algorithm to identify distinct tissue regions.

---

## Model Architecture

| Layer               | Configuration                      |
|---------------------|----------------------------------|
| Conv Layer 1        | Input channels: 3, Output: 32, Kernel: 3x3 |
| Conv Layer 2        | Input: 32, Output: 64, Kernel: 3x3          |
| Conv Layer 3        | Input: 64, Output: 128, Kernel: 3x3         |
| Fully Connected 1   | Input features: 128 × 12 × 12, Output: 256  |
| Fully Connected 2   | Input: 256, Output: 1 (Sigmoid activation)  |
| Dropout             | Applied for regularization                    |

---

## Training & Evaluation
- Loss function: Binary Cross-Entropy.
- Optimizer: Adam.
- Training for 50 epochs.
- Metrics: Validation accuracy, loss, precision-recall curve, ROC curve, confusion matrix.

---

## Results

| Metric                  | Result                   |
|-------------------------|--------------------------|
| Test Accuracy           | ~81.2%                   |
| Validation Accuracy     | Improved steadily over epochs |
| Precision-Recall Curve  | High precision and recall balance |
| ROC Curve AUC          | Strong discriminative ability |
| Confusion Matrix        | Balanced true positive and negative rates |

Multi-class classification achieved 77.46% validation accuracy using CNN + GMM clustering. EM algorithm effectively segmented images into meaningful clusters.

---

## Conclusion
The project demonstrates the successful application of CNNs and statistical methods to classify and segment histopathological images for cancer detection. The binary classification model achieved strong accuracy, and clustering provided insights into spatial tumor distributions. This work lays the foundation for automated, scalable cancer diagnostic tools leveraging deep learning.

---

## Acknowledgments
Thanks to Dr. Vibhor Kumar for guidance, colleagues Shashank Shekhar Pathak and Manoj Telrandhe for support, and Indraprastha Institute of Information Technology Delhi for resources.

---

## References
- PatchCamelyon Dataset and PCam: [GitHub Link](https://github.com/basveeling/pcam)
- [B. Ahn et al., Nature Communications, 2024](https://doi.org/10.1038/s41467-024-48667-6)
- [V. Badrinarayanan et al., IEEE TPAMI, 2017](https://doi.org/10.1109/TPAMI.2016.2644615)
- [Babak Ehteshami Bejnordi et al., JAMA, 2017](https://doi.org/10.1001/jama.2017.14585)
- And others as listed in the report.

---

## Contact
**Anand Kumar**  
M.Tech Student, Computer Science & Engineering  
Indraprastha Institute of Information Technology Delhi  
Email: [anand23111@iiitd.ac.in](mailto:anand23111@iiitd.ac.in)

---


*This README was generated based on the capstone report submitted on August 12, 2024.*

