# 🛡Credit Card Fraud Detection using Machine Learning – Securelytics Inc.

## Project Overview

Securelytics Inc. is developing a scalable AI solution to detect fraudulent financial transactions across its digital credit platform. The goal is to identify suspicious activity in real-time using advanced machine learning techniques and anomaly detection.

This project demonstrates the ability to:

- Design and train fraud-specific ML models  
- Handle extreme class imbalance  
- Perform real-time risk scoring  
- Deploy models in scalable environments  

---

## Goal

To design, train, and evaluate ML models that:

- Accurately detect fraudulent credit card transactions  
- Reduce false positives while maximizing fraud recall  
- Provide a risk score usable in real-time financial systems  

---

## Intended Audience

- Senior Data Scientists in FinTech  
- Risk Management and Fraud Teams  
- Product Managers in financial products  
- Cloud Engineering Teams deploying AI pipelines  

---

## Strategy & Pipeline

### I. Preprocessing
- Handle imbalance with SMOTE  
- Standardize `Amount` and `Time`  
- Remove outliers  

### II. Feature Engineering
- Analyze transaction frequency per customer  
- Derive features like rapid succession patterns  

### III. Modeling
- Logistic Regression (baseline)  
- Random Forest and XGBoost  
- Isolation Forest, One-Class SVM  
- LSTM (optional, sequential pattern analysis)  

### IV. Evaluation
- Confusion Matrix  
- Precision, Recall, F1-Score  
- ROC-AUC and Cost-sensitive analysis  

### V. Deployment
- Streamlit UI for real-time classification  
- RESTful Flask API for integration  
- Containerization (optional)

---

## Challenges

- Fraud class < 0.2% of total  
- Preventing high false positive rate  
- Maintaining real-time speed  
- Ensuring scalability for production  

---

## Dataset Used

**Dataset Name:** Credit Card Fraud Detection  
**Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
**Size:** 284,807 transactions (492 frauds)  
**Features:**
- `Time`, `Amount`, `Class` (target)
- `V1` to `V28` anonymized PCA features  

---

## Sample Implementation Steps

### 1️⃣ Load and Preprocess Data
```python
df = pd.read_csv('/content/creditcard.csv')
df['Amount'] = StandardScaler().fit_transform(df[['Amount']])
df['Time'] = StandardScaler().fit_transform(df[['Time']])
```

### 2️⃣ Train/Test Split
```python
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
```

### 3️⃣ Train Random Forest & Evaluate
```python
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
```

### 4️⃣ Apply SMOTE and Retrain
```python
sm = SMOTE()
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
model.fit(X_resampled, y_resampled)
```

### 5️⃣ Save the Model
```python
joblib.dump(model, 'linkedin_project/fraud_model.pkl')
```

---

## Visual Outputs

- Feature Importance (Gini/SHAP)  
- ROC Curve  
- Precision-Recall Curve  
- Streamlit UI for simulation

---

## References

- [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- [imbalanced-learn.org](https://imbalanced-learn.org/)  
- [XGBoost Documentation](https://xgboost.readthedocs.io/)  
- [SHAP Explainability](https://github.com/slundberg/shap)  
- [Streamlit Documentation](https://docs.streamlit.io/)  
