# Conformal Prediction for Classifiers (ML Lab)

A reliability layer for machine learning classifiers using **Split-Conformal Prediction** applied to the Pima Indians Diabetes dataset.

## Live Demo
🔗 **Live Interactive Demo**: `[https://conformalprediction.streamlit.app](https://conformalprediction.streamlit.app/)` 

---

## Problem Overview

Standard classification models output point predictions (e.g. "Diabetic" or "Not Diabetic") and raw softmax probabilities. However, raw probabilities are often uncalibrated and overconfident. In clinical applications like diabetes screening, a false point prediction can lead to misdiagnosis or delayed care.

This project introduces a **split-conformal prediction layer** on top of 5 machine learning classifiers. Instead of outputting a single point label, conformal prediction outputs a **prediction set** guaranteed to contain the true ground-truth label with a user-specified confidence level (e.g., 90% or 80%).

---

## Method: Split-Conformal Prediction in Plain Words

1. **Data Splitting**:
   We split the dataset into three parts:
   - **Train Set (60%)**: Used to fit model weights.
   - **Calibration Set (20%)**: Held out from model training; used solely to compute non-conformity scores and calculate uncertainty thresholds.
   - **Test Set (20%)**: Held out for unbiased evaluation.

2. **Non-Conformity Scores**:
   For each sample $i$ in the calibration set with true label $y_i$, we calculate a non-conformity score:
   $$\text{Score}_i = 1 - P(\text{true class} \mid X_i)$$
   A higher score indicates that the model was less confident in the true class.

3. **Conformal Threshold**:
   Given a user-defined error rate $\alpha$ (e.g. $\alpha = 0.1$ for 90% target coverage), we find the $q$-quantile threshold of the calibration scores:
   $$q = \text{Quantile}\left(\text{Scores}, \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$$

4. **Prediction Set Construction**:
   For any new patient $X_{\text{new}}$, the prediction set includes all classes $c$ whose predicted probability meets or exceeds $1 - q$:
   $$\text{Prediction Set} = \{c \in \{0, 1\} \mid P(Y=c \mid X_{\text{new}}) \ge 1 - q\}$$

   - **Set Size 1** (e.g., `["Diabetic"]` or `["Not Diabetic"]`): The model is confident.
   - **Set Size 2** (e.g., `["Not Diabetic", "Diabetic"]`) or **0**: The model is uncertain, signaling that a medical doctor's review is needed.

---

## Benchmark Evaluation Results

Evaluation metrics on the test split across 5 classifiers at significance levels $\alpha = 0.1$ (target 90% coverage) and $\alpha = 0.2$ (target 80% coverage):

| Model | Alpha ($\alpha$) | Accuracy | Coverage | Avg Set Size | Uncertain % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.1 | 74.03% | 86.36% | 1.32 | 32.47% |
| **Logistic Regression** | 0.2 | 74.03% | 78.57% | 1.09 | 9.09% |
| **Decision Tree** | 0.1 | 70.13% | 87.66% | 1.49 | 49.35% |
| **Decision Tree** | 0.2 | 70.13% | 81.82% | 1.26 | 25.97% |
| **Gaussian Naive Bayes** | 0.1 | 68.83% | 83.77% | 1.36 | 36.36% |
| **Gaussian Naive Bayes** | 0.2 | 68.83% | 77.92% | 1.16 | 15.58% |
| **K-Nearest Neighbors** | 0.1 | 71.43% | 92.21% | 1.42 | 42.21% |
| **K-Nearest Neighbors** | 0.2 | 71.43% | 81.82% | 1.20 | 20.13% |
| **Random Forest** | 0.1 | 72.73% | 90.26% | 1.40 | 40.26% |
| **Random Forest** | 0.2 | 72.73% | 77.92% | 1.12 | 11.69% |

---

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Preprocess Dataset**:
   ```bash
   python src/preprocess.py
   ```

3. **Train Models**:
   ```bash
   python src/models.py
   ```

4. **Evaluate Conformal Metrics**:
   ```bash
   python src/evaluate.py
   ```

5. **Generate Plots**:
   ```bash
   python src/plots.py
   ```

6. **Launch Streamlit Interactive Application**:
   ```bash
   streamlit run app/streamlit_app.py
   ```

7. **Run Test Suite**:
   ```bash
   pytest -q
   ```

---

## Dataset Credit

Data source: **Pima Indians Diabetes Database**, National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), via UCI Machine Learning Repository and Kaggle.
