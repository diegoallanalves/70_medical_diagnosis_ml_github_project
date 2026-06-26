# 🏥 Medical Diagnosis Anomaly Detection with Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-green)

## 📌 Overview

This portfolio project demonstrates an end-to-end machine learning
workflow using **synthetic medical data**. It showcases data generation,
preprocessing, clustering, anomaly detection, classification, and
interactive visualisation.

> **Disclaimer:** This project uses **100% synthetic (fake) data** for
> educational and portfolio purposes only. It must **not** be used for
> real medical diagnosis or clinical decision-making.

------------------------------------------------------------------------

## ✨ Features

-   Synthetic patient dataset generation
-   Data cleaning and preprocessing
-   Feature scaling
-   K-Means clustering
-   Isolation Forest anomaly detection
-   Random Forest classification
-   Interactive Plotly visualisations
-   3D patient cluster visualisation
-   Model metrics export (JSON)

------------------------------------------------------------------------

## 🛠️ Technologies

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Plotly
-   PyCharm
-   Git

------------------------------------------------------------------------

## 📂 Project Structure

``` text
70_medical_diagnosis_ml_github_project/
│
├── data/
├── outputs/
├── src/
│   ├── generate_fake_data.py
│   ├── medical_diagnosis_algorithm.py
│   └── utils.py
├── README.md
├── requirements.txt
└── .gitignore
```

------------------------------------------------------------------------

## 🚀 Installation

``` bash
git clone https://github.com/diegoallanalves/70_medical_diagnosis_ml_github_project.git

cd 70_medical_diagnosis_ml_github_project

python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ Running the Project

Generate the synthetic dataset:

``` bash
python src/generate_fake_data.py
```

Run the machine learning pipeline:

``` bash
python src/medical_diagnosis_algorithm.py
```

The project generates:

-   Interactive HTML charts
-   Model evaluation metrics
-   Classified patient dataset
-   Detected anomalies

------------------------------------------------------------------------

## 🤖 Machine Learning Pipeline

``` text
Synthetic Data
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
K-Means Clustering
      │
      ▼
Isolation Forest
      │
      ▼
Random Forest Classifier
      │
      ▼
Interactive Visualisations
```

------------------------------------------------------------------------

## 📊 Outputs

-   3D Patient Clusters
-   Anomaly Detection Scatter Plot
-   Feature Importance Chart
-   Model Metrics (JSON)
-   Classified Dataset (CSV)

------------------------------------------------------------------------

## 📈 Future Improvements

-   Streamlit dashboard
-   XGBoost & LightGBM models
-   SHAP explainability
-   Docker deployment
-   CI/CD with GitHub Actions
-   Azure deployment
-   SQL database integration

------------------------------------------------------------------------

## 👨‍💻 Author

**Diego Alves**

MSc in Data Science • Python • Machine Learning • Data Engineering
