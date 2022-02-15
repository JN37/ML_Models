# %% Import libs

# Example from https://www.datacamp.com/community/tutorials/random-forests-classifier-python

from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# %% Load and prepare data

iris = datasets.load_iris()

data = pd.DataFrame({
    'sepal length': iris.data[:, 0],
    'sepal width': iris.data[:, 1],
    'petal length': iris.data[:, 2],
    'petal width': iris.data[:, 3],
    'species': iris.target
})

X = data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y = data['species']  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# %% Train model

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# %% Predict and check accuracy
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
