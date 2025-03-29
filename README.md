# Network Traffic Anomaly Detection using Vision Transformer (ViT)

## Overview
This project focuses on network traffic anomaly detection using a pretrained **ViT-Base-Patch16** model from the Transformers library. The model is trained on the **NSL-KDD dataset**, a widely used benchmark for network intrusion detection. The goal is to classify network traffic as either normal or anomalous, with further classification into specific attack types.

## Dataset
- **Source**: [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)
- **Dataset Details**:
  - Contains **125,973 records** with **41 features**.
  - Includes **normal** and **attack traffic samples**.
  - Attacks are categorized into four types:
    - **DoS (Denial of Service Attacks)**
    - **Probe (Surveillance and Probing Attacks)**
    - **R2L (Remote-to-Local Attacks)**
    - **U2R (User-to-Root Attacks)**

## Model & Methodology
- **Pretrained Model**: `vit_base_patch16_224` from **Hugging Face Transformers**
- **Libraries Used**:
  - Transformers
  - PyTorch
  - Scikit-learn
  - Pandas, NumPy
  - Matplotlib, Seaborn (for visualization)
- **Feature Engineering**:
  - One-hot encoding for categorical features.
  - Normalization of numerical features.
  - Dimensionality reduction using PCA (optional).
- **Training Pipeline**:
  - Split dataset into **train (70%)**, **validation (15%)**, and **test (15%)**.
  - Train the **Vision Transformer model** on network traffic data.
  - Optimize using **Adam optimizer** and **Cross-Entropy loss**.
  - Use **ReduceLROnPlateau** for dynamic learning rate adjustment.

## Evaluation Metrics
- **Accuracy**: Measures the overall correctness of predictions.
- **Precision**: Proportion of true positive detections among all positive predictions.
- **Recall (Sensitivity)**: Ability to correctly identify anomalous traffic.
- **F1-score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: Visualizes true vs. false classifications.
- **ROC-AUC Score**: Evaluates the model’s discrimination power.

## Expected Model Output
The model outputs the probability of network traffic being normal or anomalous. If classified as anomalous, it further identifies the specific attack type (DoS, Probe, R2L, or U2R). Below is a sample confusion matrix visualization:


## Installation & Usage
### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install torch transformers pandas numpy scikit-learn matplotlib seaborn
```

### Running the Model
1. Clone the repository:
   ```bash
   git clone https://github.com/nandnir204/NetworkAnamolyDetection.git
   cd network-anomaly-detection
   ```
2. Train-Test Model:
   ```bash
   python networkanamolydetection.py
   ```

## Future Work
To improve model performance, future enhancements may include:
- Experimenting with **Transformer-based hybrid models**.
- Using advanced feature selection techniques.
- Expanding the dataset with **real-time traffic data**.
- Deploying the model as an **API for live anomaly detection**.

## Contact
For any queries, reach out to **Nandini Rajoria** at **nandinirajoria204@gmail.com**.

