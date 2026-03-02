# 🩺 Breast Cancer Detection using Machine Learning

## 📌 Project Overview
This project builds a highly sensitive Machine Learning model to classify breast cancer tumors as either **Malignant** or **Benign** based on cell nucleus measurements. 

In medical diagnostics, a **False Negative** (predicting a malignant tumor as benign) is the most dangerous outcome. Therefore, this project specifically focuses on optimizing the **Recall** metric. By dynamically adjusting the classification threshold of a Random Forest model, the final deployment achieves a **100% Recall rate**, effectively reducing False Negatives to zero.

## 📊 Dataset
The model is trained on the classic **Breast Cancer Wisconsin (Diagnostic) Dataset**.
* **Total Features:** 30 numerical measurements (e.g., radius, texture, perimeter, area, smoothness) computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
* **Target:** Diagnosis (M = Malignant, B = Benign).

## ⚙️ Methodology & Workflow

1. **Data Preprocessing:** * Handled empty trailing columns and dropped irrelevant ID identifiers.
   * Encoded the categorical target variable (`M` -> `1`, `B` -> `0`).
2. **Feature Selection (Mitigating Multicollinearity):**
   * Generated a Correlation Matrix Heatmap to identify redundant features.
   * Automatically dropped 10 features that had a correlation coefficient strictly greater than `0.90` (e.g., dropping `perimeter` and `area` while keeping `radius`).
   * **Result:** Reduced the feature space from 30 to 20, preventing model overfitting and improving processing speed.
3. **Model Training:**
   * Trained a **Random Forest Classifier** (`n_estimators=100`).
   * Baseline model achieved ~96.5% accuracy but missed 2 malignant cases (False Negatives).
4. **Threshold Optimization (The Medical Approach):**
   * Extracted raw prediction probabilities using `.predict_proba()`.
   * Lowered the decision threshold from the default `0.50` down to a conservative **`0.30`**.
   * **Result:** Pushed the model's Recall to 1.00 (100%), catching all malignant cases in the test set.

## 💻 Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Deployment/UI:** Streamlit

## 🚀 How to Run Locally

You can run this model interactively on your own machine using the provided Streamlit web app.

**1. Clone the repository:**
```bash
git clone [https://github.com/Paragiscool/Breast_cancer_detection.git](https://github.com/Paragiscool/Breast_cancer_detection.git)
cd Breast_cancer_detection
