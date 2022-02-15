# %%

# Example from https://towardsdatascience.com/a-short-tutorial-on-naive-bayes-classification-with-implementation-2f69183d8ce1

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB

# %% Load data and separate it

data = pd.read_csv('naivebayes.csv')

x1 = data.iloc[:, 0]
x2 = data.iloc[:, 1]
x3 = data.iloc[:, 2]
x4 = data.iloc[:, 3]
y = data.iloc[:, 4]

# Encode the categorical variables

le = LabelEncoder()
x1 = le.fit_transform(x1)
x2 = le.fit_transform(x2)
x3 = le.fit_transform(x3)
x4 = le.fit_transform(x4)
y = le.fit_transform(y)

# Gather everything in a dataframe
X = pd.DataFrame(list(zip(x1, x2, x3, x4)))

# %% Train model and check its output

model = CategoricalNB()
model.fit(X, y)

# Predict Output
predicted = model.predict([[1, 0, 0, 1]])
print("Predicted Value:", model.predict([[1, 0, 0, 1]]))
print(model.predict_proba([[1, 0, 0, 1]]))
