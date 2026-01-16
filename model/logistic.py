from sklearn.linear_model import LogisticRegression
from train_utils import load_and_split_data, evaluate_model

X_train, X_test, y_train, y_test = load_and_split_data("data/heart.csv")

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

metrics = evaluate_model(model, X_test, y_test)
print("Logistic Regression:", metrics)
