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
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="ML Classification Model Comparison",
    layout="centered"
)

st.title("📊 Machine Learning Classification Models")
st.write("Upload test data and evaluate different ML classification models.")

# ------------------------------
# Dataset Upload
# ------------------------------
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "HeartDisease" not in df.columns:
        st.error("Dataset must contain 'HeartDisease' column as target.")
        st.stop()

    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    X = pd.get_dummies(X, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------------------
    # Model Selection
    # ------------------------------
    model_name = st.selectbox(
        "Select Machine Learning Model",
        (
            "Logistic Regression",
            "Decision Tree",
            "K-Nearest Neighbors",
            "Naive Bayes",
            "Random Forest (Ensemble)",
            "XGBoost (Ensemble)"
        )
    )

    # ------------------------------
    # Model Initialization
    # ------------------------------
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)

    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)

    elif model_name == "K-Nearest Neighbors":
        model = KNeighborsClassifier(n_neighbors=5)

    elif model_name == "Naive Bayes":
        model = GaussianNB()

    elif model_name == "Random Forest (Ensemble)":
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    elif model_name == "XGBoost (Ensemble)":
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )

    # ------------------------------
    # Train Model
    # ------------------------------
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # ------------------------------
    # Metrics Calculation
    # ------------------------------
    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    # ------------------------------
    # Display Metrics
    # ------------------------------
    st.subheader("📈 Model Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("AUC Score", f"{auc:.4f}")
    col3.metric("Precision", f"{precision:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    # ------------------------------
    # Confusion Matrix
    # ------------------------------
    st.subheader("🔍 Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

    # ------------------------------
    # Classification Report
    # ------------------------------
    st.subheader("📄 Classification Report")
    st.text(classification_report(y, y_pred))

else:
    st.info("Please upload a CSV file to begin.")
