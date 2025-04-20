# 💳 Credit Card Fraud Detection  
Efficient credit card fraud detection using custom features, class imbalance handling (no SMOTE), and cost-sensitive threshold tuning. Includes ready-to-use prediction tools and thorough evaluation. Built with Python, scikit-learn, and pandas for a Kaggle competition.

---

## 🧠 Project Overview  
This project develops a machine learning model to detect fraudulent credit card transactions efficiently. The model balances minimizing false positives while ensuring high accuracy in fraud detection.

---

## 📊 Dataset  
The dataset includes credit card transactions with features such as:
- Transaction amount  
- Time elapsed between transactions  
- Anonymized features (V1–V28) from PCA transformation  
- Binary class label (0: Normal, 1: Fraud)  

📁 **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
📌 Contains 284,807 transactions with only 492 frauds (~0.17%)  

---

## 🧪 Methodology  

### 🔍 Data Preprocessing  
- Exploratory data analysis to understand feature distributions  
- Feature scaling for Amount and Time  
- Feature engineering to create meaningful derived features  
- Handling highly imbalanced class distribution  

### ⚖️ Class Imbalance Handling  
Since SMOTE was not used, two alternative approaches were implemented:  
1. Random undersampling of the majority class  
2. Data augmentation with synthetic minority class samples  

### 🤖 Model Development  
- Trained **Random Forest Classifier**  
- Implemented **threshold tuning** for optimal performance  
- Evaluated using **precision**, **recall**, **ROC-AUC**, and **PR-AUC**  
- Cost-based optimization to minimize business impact of false classifications  

### 🧩 Feature Importance Analysis  
Identified the most predictive features for fraud detection:  
- [List top 5 features from analysis]  

---

## 📈 Results  
- **Final model:** `{best_model_name}`  
- **Optimal threshold:** `{final_threshold:.2f}`  
- **Key performance metrics:**  
  - Precision: `{tp / (tp + fp) if (tp + fp) > 0 else 0:.4f}`  
  - Recall: `{tp / (tp + fn) if (tp + fn) > 0 else 0:.4f}`  
  - False Positive Rate: `{fp / (fp + tn) if (fp + tn) > 0 else 0:.4f}`  

---

## 💼 Business Impact  
- Estimated cost savings from fraud prevention  
- Balance between false positives (customer friction) and false negatives (fraud losses)  

---

## 🛠️ Usage  
```python
# Example of using the model to evaluate a new transaction
def evaluate_transaction(transaction, model, threshold=0.5):
    # Preprocess transaction data
    # Make prediction
    fraud_prob = model.predict_proba(transaction)[:, 1][0]
    is_fraud = fraud_prob >= threshold

    return {
        'fraud_probability': fraud_prob,
        'is_fraud': is_fraud,
        'threshold_used': threshold
    }
