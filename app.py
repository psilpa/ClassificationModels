import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix
)

# ------------------------------
# Try importing XGBoost safely
# ------------------------------
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="ML Model Evaluation", layout="wide")
st.title("📊 Classification Model Evaluation Dashboard")
st.write("Upload a test dataset to evaluate multiple ML classification models.")

# ------------------------------
# File Upload
# ------------------------------
uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

if "HeartDisease" not in df.columns:
    st.error("Dataset must contain 'HeartDisease' column as target.")
    st.stop()

# ------------------------------
# Data Preprocessing
# ------------------------------
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# Define Models
# ------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

if xgb_available:
    models["XGBoost"] = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )

# ------------------------------
# Train & Evaluate
# ------------------------------
results = []

confusion_matrices = {}

for name, model in models.items():
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1 Score": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred)
    }

    results.append(metrics)
    confusion_matrices[name] = confusion_matrix(y, y_pred)

# ------------------------------
# Metrics Table
# ------------------------------
st.subheader("📈 Model Performance Comparison")

results_df = pd.DataFrame(results)
st.dataframe(
    results_df.style.format("{:.4f}", subset=results_df.columns[1:])
)

# ------------------------------
# Confusion Matrices
# ------------------------------
st.subheader("🔍 Confusion Matrices")

cols = st.columns(3)
i = 0

for model_name, cm in confusion_matrices.items():
    with cols[i % 3]:
        st.write(f"**{model_name}**")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    i += 1
