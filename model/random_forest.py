from sklearn.ensemble import RandomForestClassifier
from train_utils import load_and_split_data, evaluate_model

X_train, X_test, y_train, y_test = load_and_split_data("data/heart.csv")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

metrics = evaluate_model(model, X_test, y_test)
print("Random Forest:", metrics)
