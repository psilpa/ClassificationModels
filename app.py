import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix
)

# ------------------------------
# Safe XGBoost Import
# ------------------------------
try:
    from xgboost import XGBClassifier
    xgb_imported = True
except Exception:
    xgb_imported = False

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="ML Classification Models", layout="wide")
st.title("📊 Classification Model Evaluation Dashboard")

if not xgb_imported:
    st.info(
        "ℹ️ XGBoost is implemented but disabled in this deployment "
        "due to OpenMP runtime limitations."
    )

# ------------------------------
# Load Models, Scaler & Features
# ------------------------------
@st.cache_resource
def load_artifacts():
    models = {}

    model_files = {
        "Logistic Regression": "logistic.pkl",
        "Decision Tree": "decision_tree.pkl",
        "KNN": "knn.pkl",
        "Naive Bayes": "naive_bayes.pkl",
        "Random Forest": "random_forest.pkl"
    }

    if xgb_imported:
        model_files["XGBoost"] = "xgboost.pkl"

    for name, file in model_files.items():
        try:
            with open(f"model/saved_models/{file}", "rb") as f:
                models[name] = pickle.load(f)
        except Exception:
            st.warning(f"⚠️ {name} model could not be loaded.")

    with open("model/saved_models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("model/saved_models/feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    return models, scaler, feature_columns


models, scaler, feature_columns = load_artifacts()

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
# Preprocessing
# ------------------------------
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X = pd.get_dummies(X, drop_first=True)

# 🔑 ALIGN FEATURES
X = X.reindex(columns=feature_columns, fill_value=0)

X_scaled = scaler.transform(X)

# ------------------------------
# Evaluate Models
# ------------------------------
results = []
conf_matrices = {}

for name, model in models.items():
    try:
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_p
