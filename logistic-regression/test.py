from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from logistic_regression import LogisticRegression


from utils.classification import accuracy


dataset = load_breast_cancer()

X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


classification = LogisticRegression()
classification.fit(X_train, y_train)

y_pred = classification.predict(X_test)

print(accuracy(y_test, y_pred))