import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
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
    xgb_available = True
except Exception:
    xgb_available = False


def load_and_split_data(csv_path):
    df = pd.read_csv(csv_path)

    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if isinstance(y_test, pd.Series):
        y_test = y_test.to_frame(name='target')

    # Concatenate X_val and y_val horizontally
    val_data = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    # Write to CSV
    print("Writing validation data to a file")
    val_data.to_csv('../data/validation_data.csv', index=False)

    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)

    return X_train, y_train
    
# ------------------------------
# Load Dataset
# ------------------------------
#df = pd.read_csv("../data/heart.csv")

#X = df.drop("HeartDisease", axis=1)
#y = df["HeartDisease"]
X,y = load_and_split_data("../data/heart.csv")

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# ------------------------------
# Save Feature Columns
# ------------------------------
os.makedirs("saved_models", exist_ok=True)

feature_columns = X.columns.tolist()
with open("saved_models/feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

# ------------------------------
# Scaling
# ------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open("saved_models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ------------------------------
# Define Models
# ------------------------------
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

if xgb_available:
    models["xgboost"] = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
else:
    print("⚠️ XGBoost not available. Skipping XGBoost training.")

# ------------------------------
# Train & Save Models
# ------------------------------
for name, model in models.items():
    model.fit(X_scaled, y)
    with open(f"saved_models/{name}.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Saved {name} model")

print("🎉 Training completed successfully.")
