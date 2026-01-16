import pandas as pd
import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Safe XGBoost Import
# ------------------------------
try:
    from xgboost import XGBClassifier
    xgb_imported = True
except Exception:
    xgb_imported = False


# ------------------------------
# Load Dataset
# ------------------------------
df = pd.read_csv("../data/heart.csv")

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

os.makedirs("saved_models", exist_ok=True)

# Save scaler
with open("saved_models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ------------------------------
# Models
# ------------------------------
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "xgboost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
}

# ------------------------------
# Train & Save
# ------------------------------
for name, model in models.items():
    model.fit(X_scaled, y)
    with open(f"saved_models/{name}.pkl", "wb") as f:
        pickle.dump(model, f)

print("✅ All models trained and saved successfully")
