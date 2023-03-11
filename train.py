from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
import pandas as pd


dataset = load_iris(as_frame=True)
X = dataset.data.to_numpy()
y = dataset.target.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = RandomForestClassifier()
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)
print(f"test acc: {acc}")
