# Medical Diagnosis Anomaly Detection Demo

This is a portfolio and educational Python project for GitHub. It uses fake synthetic medical-style data to demonstrate data preparation, clustering, anomaly detection, classification, and interactive charts.

Warning: This project uses fake data only. It is not medical advice and must not be used for real diagnosis.

## What the project does

The project creates fake patient records with variables such as age, BMI, blood pressure, cholesterol, glucose, heart rate, smoking status, exercise level, family history, and symptom score.

It then uses:

- K-Means for clustering patient profiles
- Isolation Forest for anomaly detection
- Random Forest for synthetic diagnostic risk classification
- Plotly for interactive charts
- Scikit-learn for machine learning
- Pandas and NumPy for data handling

## How to run in PyCharm

1. Open PyCharm.
2. Open this folder as a project.
3. Create a virtual environment.
4. Install the requirements:

```bash
pip install -r requirements.txt
```

5. Generate the fake dataset:

```bash
python src/generate_fake_data.py
```

6. Run the machine learning algorithm:

```bash
python src/medical_diagnosis_algorithm.py
```

7. Open the charts in the outputs folder.

## Charts included

### 3D patient clusters

File: `outputs/3d_patient_clusters.html`

This chart shows each synthetic patient in a 3D space using glucose, cholesterol, and systolic blood pressure. The colours represent the K-Means cluster assigned to each patient. It helps show whether patients with similar health indicators naturally group together.

How to read it: points that are close together have similar glucose, cholesterol, and blood pressure values. Separate groups suggest different patient profiles in the synthetic dataset.

### Risk distribution

File: `outputs/risk_distribution.html`

This bar chart shows how many synthetic patients are classified as Low Risk, Medium Risk, or High Risk. It gives a quick overview of the balance of the generated dataset.

How to read it: a very large difference between categories means the dataset is imbalanced. In real machine learning projects, this matters because models can become better at predicting the largest class and weaker at predicting smaller classes.

### Anomaly scatter chart

File: `outputs/anomaly_scatter.html`

This chart compares glucose and cholesterol values and highlights records identified by Isolation Forest as normal or anomalous. The marker size is based on the symptom score.

How to read it: anomaly points usually appear far from the main group, especially where glucose, cholesterol, or symptom scores are unusually high.

### Age and BMI risk chart

File: `outputs/age_bmi_risk_scatter.html`

This scatter chart shows the relationship between age and BMI, coloured by diagnostic risk. It also marks whether each patient was detected as normal or anomalous.

How to read it: it helps identify whether higher-risk synthetic patients are concentrated in specific age or BMI ranges.

### Feature correlation heatmap

File: `outputs/feature_correlation_heatmap.html`

This heatmap shows the correlation between numerical features such as age, BMI, blood pressure, cholesterol, glucose, heart rate, and symptom score.

How to read it: values closer to 1 mean two features tend to increase together. Values closer to -1 mean one feature tends to decrease when the other increases. Values near 0 mean there is little linear relationship.

### Feature importance chart

File: `outputs/feature_importance.html`

This chart shows which features the Random Forest model used most when predicting the synthetic diagnostic risk label.

How to read it: longer bars represent features with stronger influence in the model. This helps explain which variables were most useful for classification in the synthetic dataset.

### Confusion matrix

File: `outputs/confusion_matrix.html`

This chart compares the model's predicted risk classes against the actual synthetic risk labels.

How to read it: values on the diagonal are correct predictions. Values outside the diagonal are mistakes, showing where the model confused one risk level with another.

## Explanation

The dataset is synthetic. A risk label is created using simple rules based on fake health indicators. The model then learns to classify records as Low Risk, Medium Risk, or High Risk.

The anomaly detection model identifies unusual records, such as very high glucose, high cholesterol, high blood pressure, or unusual symptom combinations.

The charts help explain the synthetic dataset, the patient clusters, the anomalies, the feature relationships, and the Random Forest model performance.

## GitHub description

Synthetic medical diagnosis ML project using Python, fake data, K-Means clustering, Isolation Forest anomaly detection, Random Forest classification, and interactive Plotly visualisations.
