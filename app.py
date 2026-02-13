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
    xgb_imported = True

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
        with open(f"model/saved_models/{file}", "rb") as f:
                models[name] = pickle.load(f)
        try:
            with open(f"model/saved_models/{file}", "rb") as f:
                models[name] = pickle.load(f)
        except Exception:
            if name == XGBoost:
                import xgboost as xgb
                bst = xgb.Booster()
                bst.load_model('xgboost.json')
            st.warning(f"⚠️ {name} model could not be loaded.")

    with open("model/saved_models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("model/saved_models/feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    return models, scaler, feature_columns


models, scaler, feature_columns = load_artifacts()

# ------------------------------
# Sample File Download
# ------------------------------
df = pd.read_csv("data/validation_data.csv")
st.download_button("Download CSV", df.to_csv(index=False), "validation_data.csv", "text/csv")

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
        y_prob = model.predict_proba(X_scaled)[:, 1]
    except Exception:
        continue

    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    composite = (accuracy + auc + f1 + mcc) / 4

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc,
        "Composite Score": composite
    })

    conf_matrices[name] = confusion_matrix(y, y_pred)

# ------------------------------
# Results Table
# ------------------------------
st.subheader("📈 Model Performance Comparison")
results_df = pd.DataFrame(results)
results_df.index = range(1, len(results_df) + 1)
results_df.index.name = "S.No"

st.dataframe(
    results_df.style.format("{:.4f}", subset=results_df.columns[1:])
)

# ------------------------------
# Best Model Recommendation
# ------------------------------
best_row = results_df.loc[results_df["Composite Score"].idxmax()]

st.success(
    f"✅ **Recommended Model:** {best_row['Model']}  \n"
    f"📌 Composite Score: {best_row['Composite Score']:.4f}"
)

with st.expander("ℹ️ How is the best model selected?"):
    st.markdown("""
    **Composite Score = (Accuracy + AUC + F1 Score + MCC) / 4**

    This ensures balanced and robust model selection.
    """)

# ------------------------------
# Confusion Matrices
# ------------------------------
st.subheader("🔍 Confusion Matrices")

cols = st.columns(3)
i = 0

for name, cm in conf_matrices.items():
    with cols[i % 3]:
        st.write(f"**{name}**")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    i += 1
