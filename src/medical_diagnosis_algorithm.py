import os

import pandas as pd
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from utils import DATA_FILE, OUTPUTS_DIR, ensure_directories, save_json

SHOW_CHARTS = os.getenv("SHOW_CHARTS", "1") != "0"

FEATURES = [
    "age", "bmi", "systolic_bp", "diastolic_bp", "cholesterol",
    "glucose", "heart_rate", "smoking", "family_history",
    "exercise_level", "symptoms_score"
]

def load_data(path=DATA_FILE):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run src/generate_fake_data.py first."
        )
    return pd.read_csv(path)

def prepare_features(df):
    x = df[FEATURES]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x, x_scaled, scaler

def save_chart(fig, filename):
    chart_path = OUTPUTS_DIR / filename
    fig.write_html(chart_path)

    if SHOW_CHARTS:
        fig.show()

    return chart_path

def run_clustering(df, x_scaled):
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(x_scaled)
    return df, kmeans

def run_anomaly_detection(df, x_scaled):
    model = IsolationForest(contamination=0.04, random_state=42)
    df["anomaly_flag"] = model.fit_predict(x_scaled)
    df["anomaly_label"] = df["anomaly_flag"].map({1: "Normal", -1: "Anomaly"})
    return df, model

def run_classification(df):
    x = df[FEATURES]
    y = df["diagnostic_risk"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )

    classifier = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "classification_report": classification_report(
            y_test, predictions, target_names=label_encoder.classes_, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "labels": label_encoder.classes_.tolist()
    }

    return classifier, label_encoder, metrics

def create_3d_cluster_chart(df):
    fig = px.scatter_3d(
        df,
        x="glucose",
        y="cholesterol",
        z="systolic_bp",
        color="cluster",
        hover_data=["patient_id", "age", "bmi", "diagnostic_risk", "anomaly_label"],
        title="3D Patient Clusters: Glucose, Cholesterol and Blood Pressure"
    )
    fig.update_traces(marker={"size": 4, "opacity": 0.75})
    fig.update_layout(
        autosize=True,
        height=760,
        margin={"l": 0, "r": 0, "b": 0, "t": 55},
        title={"x": 0.5, "xanchor": "center"},
        legend={"orientation": "h", "y": -0.04, "x": 0.5, "xanchor": "center"},
        scene={
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.65, "y": 1.65, "z": 1.15}},
            "xaxis": {"title": "Glucose"},
            "yaxis": {"title": "Cholesterol"},
            "zaxis": {"title": "Systolic BP"},
        },
    )
    return save_chart(fig, "3d_patient_clusters.html")

def create_risk_distribution_chart(df):
    risk_order = ["Low Risk", "Medium Risk", "High Risk"]
    risk_counts = (
        df["diagnostic_risk"]
        .value_counts()
        .reindex(risk_order)
        .reset_index()
    )
    risk_counts.columns = ["diagnostic_risk", "patients"]

    fig = px.bar(
        risk_counts,
        x="diagnostic_risk",
        y="patients",
        color="diagnostic_risk",
        text="patients",
        title="Patient Risk Distribution",
        labels={"diagnostic_risk": "Diagnostic Risk", "patients": "Number of Patients"}
    )
    fig.update_traces(textposition="outside")
    return save_chart(fig, "risk_distribution.html")

def create_anomaly_chart(df):
    fig = px.scatter(
        df,
        x="glucose",
        y="cholesterol",
        color="anomaly_label",
        size="symptoms_score",
        hover_data=["patient_id", "age", "systolic_bp", "diagnostic_risk"],
        title="Anomaly Detection: Glucose vs Cholesterol"
    )
    return save_chart(fig, "anomaly_scatter.html")

def create_age_bmi_risk_chart(df):
    fig = px.scatter(
        df,
        x="age",
        y="bmi",
        color="diagnostic_risk",
        symbol="anomaly_label",
        hover_data=["patient_id", "glucose", "cholesterol", "systolic_bp"],
        title="Age and BMI by Diagnostic Risk",
        labels={"age": "Age", "bmi": "BMI", "diagnostic_risk": "Diagnostic Risk"}
    )
    return save_chart(fig, "age_bmi_risk_scatter.html")

def create_correlation_heatmap(df):
    correlation_df = df[FEATURES].corr().round(2)

    fig = px.imshow(
        correlation_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Feature Correlation Heatmap"
    )
    return save_chart(fig, "feature_correlation_heatmap.html")

def create_feature_importance_chart(classifier):
    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": classifier.feature_importances_
    }).sort_values("importance", ascending=False)

    fig = px.bar(
        importance_df,
        x="importance",
        y="feature",
        orientation="h",
        title="Random Forest Feature Importance"
    )
    return save_chart(fig, "feature_importance.html")

def create_confusion_matrix_chart(metrics):
    matrix_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=metrics["labels"],
        columns=metrics["labels"]
    )

    fig = px.imshow(
        matrix_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Random Forest Confusion Matrix",
        labels={"x": "Predicted Risk", "y": "Actual Risk", "color": "Patients"}
    )
    return save_chart(fig, "confusion_matrix.html")

def main():
    ensure_directories()

    df = load_data()
    x, x_scaled, scaler = prepare_features(df)

    df, kmeans = run_clustering(df, x_scaled)
    df, anomaly_model = run_anomaly_detection(df, x_scaled)
    classifier, label_encoder, metrics = run_classification(df)

    df.to_csv(OUTPUTS_DIR / "patient_results_with_clusters_and_anomalies.csv", index=False)
    save_json(metrics, OUTPUTS_DIR / "model_metrics.json")

    chart_paths = [
        create_3d_cluster_chart(df),
        create_risk_distribution_chart(df),
        create_anomaly_chart(df),
        create_age_bmi_risk_chart(df),
        create_correlation_heatmap(df),
        create_feature_importance_chart(classifier),
        create_confusion_matrix_chart(metrics),
    ]

    print("Algorithm completed successfully.")
    print("Results saved in the outputs folder.")
    print(f"Model accuracy: {metrics['accuracy']}")
    print("Charts created:")
    for chart_path in chart_paths:
        print(f"- {chart_path}")

if __name__ == "__main__":
    main()
