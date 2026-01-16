from sklearn.naive_bayes import GaussianNB
from train_utils import load_and_split_data, evaluate_model

X_train, X_test, y_train, y_test = load_and_split_data("data/heart.csv")

model = GaussianNB()
model.fit(X_train, y_train)

metrics = evaluate_model(model, X_test, y_test)
print("Naive Bayes:", metrics)
