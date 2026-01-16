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
# Page Config
# ------------------------------
st.set_page_config(page_title="ML Model Evaluation", layout="wide")
st.title("📊 Classification Model Evaluation Dashboard")

# ------------------------------
# Load Models (ONCE)
# ------------------------------
@st.cache_resource
def load_models():
    models = {}
    model_names = [
        "logistic",
        "decision_tree",
        "knn",
        "naive_bayes",
        "random_forest",
        "xgboost"
    ]

    for name in model_names:
        try:
            with open(f"model/saved_models/{name}.pkl", "rb") as f:
                models[name] = pickle.load(f)
        except:
            pass

    with open("model/saved_models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return models, scaler


models, scaler = load_models()

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

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X = pd.get_dummies(X, drop_first=True)
X_scaled = scaler.transform(X)

# ------------------------------
# Evaluation
# ------------------------------
results = []
conf_matrices = {}

for name, model in models.items():
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    composite_score = (accuracy + auc + f1 + mcc) / 4

    results.append({
        "Model": name.replace("_", " ").title(),
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc,
        "Composite Score": composite_score
    })

    conf_matrices[name] = confusion_matrix(y, y_pred)


# ------------------------------
# Display Metrics
# ------------------------------
st.subheader("📈 Model Performance Comparison")
results_df = pd.DataFrame(results)
st.dataframe(results_df.style.format("{:.4f}", subset=results_df.columns[1:]))

# ------------------------------
# Confusion Matrices
# ------------------------------
st.subheader("🔍 Confusion Matrices")

cols = st.columns(3)
i = 0

for name, cm in conf_matrices.items():
    with cols[i % 3]:
        st.write(f"**{name.replace('_', ' ').title()}**")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    i += 1
    
results_df = pd.DataFrame(results)

best_model_row = results_df.loc[
    results_df["Composite Score"].idxmax()
]

st.success(
    f"✅ **Recommended Model:** {best_model_row['Model']}\n\n"
    f"📌 Composite Score: {best_model_row['Composite Score']:.4f}\n\n"
    "This recommendation is based on Accuracy, AUC, F1 Score, and MCC."
)

st.subheader("📈 Model Performance Comparison")

st.dataframe(
    results_df.style.format("{:.4f}", subset=results_df.columns[1:])
)

with st.expander("ℹ️ How is the best model selected?"):
    st.markdown("""
    The best model is selected using a **Composite Score**, calculated as:

    **(Accuracy + AUC + F1 Score + MCC) / 4**

    This ensures:
    - Balanced performance
    - Robustness to class imbalance
    - Avoidance of accuracy-only bias
    """)

