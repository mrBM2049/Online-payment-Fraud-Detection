# 💳 Online Payment Fraud Detection Ensemble Model

This repository houses the code and resources for a **machine learning project** focused on building a robust system to detect **fraudulent online payment transactions**.  
The solution uses an **Ensemble Model** built from multiple powerful classifiers for **high-accuracy risk assessment**.

---

## 🧠 Project Explanation & Purpose

This project develops a **high-performance system** for online payment fraud detection.  
At its core lies an **Ensemble Model** (a “super-model” combining three individual models: **Logistic Regression**, **XGBoost**, and **Random Forest**) trained to analyze the characteristics of financial transactions.

The model’s main purpose is to **accurately assess the probability of fraud in real-time**, helping financial institutions flag and prevent malicious transfers.  

Additionally, a **Streamlit web application** provides an **interactive interface** for testing and validating predictions using custom transaction data.

---

## 🚀 Setup & Installation

Follow these steps to set up the project environment and run the application.

### 🧩 Step 1: Clone the Repository

```bash
git clone mrBM2049/Online-payment-Fraud-Detection
cd "Online-payment-Fraud-Detection"
```

---

### 🧱 Step 2: Create and Activate a Virtual Environment

It is best practice to install dependencies inside an isolated environment (`.venv`).

#### For Windows:
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

#### For macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 💾 Step 3: Download Data and Model Files

This project depends on a **dataset** and a **trained model file** that must be downloaded manually.

- **Download Dataset:**  
  Place the file in the project root directory and ensure it is named `new_data.csv`.

  📂 **Dataset Link:**  
  [Download new_data.csv](https://drive.google.com/file/d/127JqP3WGjBVihR-ZcUR86T3wwy3_g63v/view?usp=sharing)


- **Download Trained Ensemble Model (If not in repository):**  
  Place the file in the project root directory and ensure it is named `Ensemble_Fraud_Detection_Model.joblib`.

  🤖 **Model Link:**  
  [Download Ensemble_Fraud_Detection_Model.joblib](https://drive.google.com/drive/folders/1m2o_gxVcRLGJLcpifdDhzzJfuK0djaof?usp=sharing)

---

### 📦 Step 4: Install Dependencies

With your virtual environment activated, install all required Python libraries:

```bash
pip install -r requirements.txt
```

---

### ▶️ Step 5: Run the Streamlit Application

Launch the Streamlit app to open the interactive fraud detection dashboard in your web browser:

```bash
streamlit run app.py
```

---

## 📖 Model Training Resources (From Original Notebook)

The original model was built and trained using several critical preprocessing and model selection steps.

### ⚙️ Data Preprocessing

- **One-Hot Encoding:**  
  Converted categorical column `type` (e.g., CASH_OUT, TRANSFER) into numeric features.

- **Feature Removal:**  
  Removed non-predictive identifier columns (`nameOrig`, `nameDest`).

- **Handling Imbalance:**  
  Configured `XGBoost` with `scale_pos_weight=773` to account for the extreme rarity of fraudulent transactions.

---

### 🧩 Key Models

The final ensemble combines three trained classifiers using a **Soft Voting strategy**:

| Model | Description |
|--------|--------------|
| **Logistic Regression** | Baseline linear model providing probabilistic interpretation. |
| **Random Forest Classifier** | Tree-based ensemble offering feature richness and interpretability. |
| **XGBoost Classifier (Tuned)** | Gradient Boosting model delivering the best individual performance. |

---

### 📊 Evaluation Metric: ROC AUC Score

The **ROC AUC Score** is used as the primary evaluation metric.  
It’s ideal for **imbalanced datasets** because it measures the model’s ability to distinguish fraud vs. non-fraud transactions **independently of classification thresholds**.

---

## 🏁 Summary

This project combines:
- Rigorous **data preprocessing**
- A powerful **ensemble learning approach**
- A **Streamlit-based user interface** for real-time fraud detection testing

It demonstrates how multiple ML algorithms can complement each other to produce **robust and explainable fraud detection results**.

---

## COLAB:
https://colab.research.google.com/drive/19x_4VE5nQ4aFSYjEh-bQWG14Sog9usWB?usp=sharing
