# Heart-Risk-Prediction
A machine learning web application that predicts heart disease risk based on user health parameters using supervised ML models. Deployed on Hugging Face Spaces.
# ❤️ Heart Disease Risk Detector (Meta-Ensemble ML)

A machine learning–based web application that predicts the **risk of heart disease**
using patient health parameters.  
The application is deployed on **Hugging Face Spaces** and built using a **meta-ensemble model**.

🔗 **Live Demo**: https://whefjhgsdcjwgugf-risk-detector-vx9.hf.space/

---

## 🚀 Features
- User-friendly web interface
- Predicts heart disease risk instantly
- Uses a powerful **meta-ensemble ML model**
- Deployed on Hugging Face Spaces
- Suitable for real-world clinical risk screening (educational use)

---

## 🧠 Machine Learning Details
- **Type**: Supervised Learning (Binary Classification)
- **Base Models**:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - AdaBoost
  - XGBoost
- **Meta Model**: MLP (Neural Network)
- **Final Model File**:  
  `Meta-MLP_Base-GB-AdaB-XGB-RF_full.pkl` (~2 GB)

---

## 📊 Input Parameters
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise Induced Angina
- Oldpeak (ST Depression)
- Slope of ST Segment
- Number of Major Vessels
- Thalassemia

---

## 🛠 Tech Stack
- Python
- Flask / Gradio
- Scikit-learn
- NumPy, Pandas
- Hugging Face Spaces
- HTML, CSS

---

## 📦 Model & Data Notice (Important)

Due to **GitHub file size limitations**, the trained meta-ensemble model  
(~2 GB `.pkl` file) is **not included** in this repository.

The complete application, including the model, is hosted and executed on
**Hugging Face Spaces**:

🔗 https://whefjhgsdcjwgugf-risk-detector-vx9.hf.space/

This repository contains:
- Source code
- UI files
- Documentation
- Deployment configuration

---

## ⚙️ Project Structure
