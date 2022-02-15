# && Import

# Example from https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

# %% Data

cancer = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=109)  # 70% training and 30% test

# Create SVM classifier
clf = svm.SVC(kernel='linear')  # Check the above link for an explanation of what a kernel is

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Check accuracy, precision and recall

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
