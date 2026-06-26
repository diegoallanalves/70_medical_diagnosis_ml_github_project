import numpy as np
import pandas as pd
from utils import DATA_FILE, ensure_directories

def generate_synthetic_medical_data(rows=1200, random_state=42):
    np.random.seed(random_state)

    age = np.random.randint(18, 85, rows)
    bmi = np.random.normal(27, 5, rows).clip(16, 48)
    systolic_bp = np.random.normal(125, 18, rows).clip(85, 220)
    diastolic_bp = np.random.normal(80, 12, rows).clip(50, 140)
    cholesterol = np.random.normal(195, 38, rows).clip(100, 360)
    glucose = np.random.normal(105, 32, rows).clip(60, 320)
    heart_rate = np.random.normal(78, 13, rows).clip(45, 150)

    smoking = np.random.choice([0, 1], rows, p=[0.72, 0.28])
    family_history = np.random.choice([0, 1], rows, p=[0.65, 0.35])
    exercise_level = np.random.choice([0, 1, 2], rows, p=[0.30, 0.45, 0.25])
    symptoms_score = np.random.poisson(2, rows).clip(0, 10)

    risk_score = (
        (age > 55).astype(int)
        + (bmi > 30).astype(int)
        + (systolic_bp > 140).astype(int)
        + (cholesterol > 240).astype(int)
        + (glucose > 140).astype(int)
        + smoking
        + family_history
        + (exercise_level == 0).astype(int)
        + (symptoms_score > 4).astype(int)
    )

    diagnostic_risk = np.where(
        risk_score >= 6, "High Risk",
        np.where(risk_score >= 3, "Medium Risk", "Low Risk")
    )

    df = pd.DataFrame({
        "patient_id": [f"P{str(i).zfill(5)}" for i in range(1, rows + 1)],
        "age": age,
        "bmi": bmi.round(1),
        "systolic_bp": systolic_bp.round(0),
        "diastolic_bp": diastolic_bp.round(0),
        "cholesterol": cholesterol.round(0),
        "glucose": glucose.round(0),
        "heart_rate": heart_rate.round(0),
        "smoking": smoking,
        "family_history": family_history,
        "exercise_level": exercise_level,
        "symptoms_score": symptoms_score,
        "diagnostic_risk": diagnostic_risk
    })

    anomaly_count = min(35, max(1, round(rows * 0.03)))
    anomaly_indices = np.random.choice(df.index, size=anomaly_count, replace=False)
    df.loc[anomaly_indices, "glucose"] = np.random.randint(230, 340, len(anomaly_indices))
    df.loc[anomaly_indices, "cholesterol"] = np.random.randint(290, 390, len(anomaly_indices))
    df.loc[anomaly_indices, "systolic_bp"] = np.random.randint(175, 230, len(anomaly_indices))
    df.loc[anomaly_indices, "symptoms_score"] = np.random.randint(7, 11, len(anomaly_indices))
    df.loc[anomaly_indices, "diagnostic_risk"] = "High Risk"

    return df

if __name__ == "__main__":
    ensure_directories()
    data = generate_synthetic_medical_data()
    data.to_csv(DATA_FILE, index=False)
    print(f"Fake medical dataset created at {DATA_FILE}")
