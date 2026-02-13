import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

def load_and_split_data(csv_path):
    df = pd.read_csv(csv_path)

    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    X = pd.get_dummies(X, drop_first=True)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y
    )

    if isinstance(y_val, pd.Series):
    y_val = y_val.to_frame(name='target')

    # Concatenate X_val and y_val horizontally
    val_data = pd.concat([X_val.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)

    # Write to CSV
    val_data.to_csv('validation_data.csv', index=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }
