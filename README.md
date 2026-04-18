# ridge-gourd-test-app




# 🌿 Hybrid AI Ridge Gourd Disease Detection

## Overview
This repository contains the training codes, deployment scripts, and the exported model for an advanced AI Plant Disease Detection System specifically designed for **Ridge Gourd** leaves. 

The system utilizes a state-of-the-art hybrid deep learning architecture, combining the spatial feature extraction capabilities of a CNN with the robust classification power of gradient boosting, all packaged into a single web-deployable file.

**Live Web App:** [Insert your Streamlit App URL here]

## 🧠 Model Architecture & "Distillation"
This project implements a unique hybrid pipeline to maximize accuracy while maintaining web compatibility:
1. **Feature Extractor:** `DenseNet121` (Pre-trained on ImageNet). Used to capture complex, multi-scale venation and lesion patterns from the leaves.
2. **Classifier:** `XGBoost`.
3. **Knowledge Distillation:** To deploy the hybrid model natively via Streamlit without needing independent XGBoost and Keras endpoints, the trained XGBoost logic was "distilled" into sequential Keras Dense layers. 
   - *Result:* A single, seamless `.h5` file that retains the high predictive power of the XGBoost classifier but runs natively in TensorFlow.

## 📊 Performance Metrics
The model was rigorously tested on a holdout validation set of 1,800 images:
* **Pre-Distillation (XGBoost):** 91.17% Accuracy
* **Post-Distillation (Keras End-to-End):** 91.89% Accuracy

## 🎯 Supported Classifications
The model is trained to detect the following classes:
* `Healthy`
* `Leaf_Minor_Infestation`
* `Mosaic_Virus`

## 💻 Tech Stack
* **Deep Learning:** TensorFlow / Keras, DenseNet121
* **Machine Learning:** XGBoost, Scikit-Learn
* **Web Framework:** Streamlit
* **Image Processing:** OpenCV, Pillow, NumPy
* **Deployment:** Streamlit Community Cloud

## 🚀 How to Run Locally

**1. Clone the repository:**
```bash
git clone [https://github.com/YourUsername/ridge-gourd-test-app.git](https://github.com/YourUsername/ridge-gourd-test-app.git)
cd ridge-gourd-test-app
```

**2. Install required dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit application:**
```bash
streamlit run app.py
```

## 📁 Repository Structure
* `app.py`: The main Streamlit web application script.
* `requirements.txt`: Python dependencies required to run the app.
* `hybridcnn_distilled_webapp_v2.h5`: The exported, distilled Keras model.


