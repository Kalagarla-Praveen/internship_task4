
# Logistic Regression Binary Classification Report

This project demonstrates the application of **Logistic Regression** for binary classification using the Breast Cancer Wisconsin dataset.

---

## 📁 Dataset Information

- Source: Breast Cancer Wisconsin Diagnostic Dataset
- Features: 30 numerical features computed from digitized images of a fine needle aspirate (FNA) of a breast mass.
- Target:
  - `M` = Malignant (encoded as 1)
  - `B` = Benign (encoded as 0)

---

## ⚙️ Tools Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

## 📊 Steps Performed

### 1. Data Preprocessing
- Removed irrelevant columns (`id`, `Unnamed: 32`)
- Converted categorical diagnosis labels to binary (`M` → 1, `B` → 0)
- Standardized features using `StandardScaler`

### 2. Model Training
- Trained a **Logistic Regression** model using `sklearn.linear_model.LogisticRegression`

### 3. Model Evaluation
- **Confusion Matrix**:
  ```
  [[70,  1],
   [ 2, 41]]
  ```
- **Classification Report**:
  - Precision (Malignant): 0.98
  - Recall (Malignant): 0.95
  - F1-score (Malignant): 0.96
  - **Overall Accuracy**: 97%
- **ROC AUC Score**: **0.997**

### 4. Threshold Tuning
- Default threshold: `0.5`
- Custom threshold tested: `0.3`
- Classification performance was evaluated to observe trade-offs between **precision** and **recall**

---

## 🧠 Sigmoid Function in Logistic Regression

The sigmoid function is used to convert the linear combination of features into a probability:

```
σ(z) = 1 / (1 + e^(-z))
```

- If `σ(z) >= threshold` → Class 1 (Malignant)
- If `σ(z) < threshold` → Class 0 (Benign)

Changing the threshold alters model sensitivity and specificity.

---

## ✅ Conclusion

Logistic Regression performed exceptionally well on this binary classification task with:
- High accuracy
- Excellent AUC
- Interpretability for medical diagnosis

This model serves as a solid baseline for more advanced classifiers.
```
