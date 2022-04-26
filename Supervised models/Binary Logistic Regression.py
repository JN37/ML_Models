# %%

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# %%

# Generate random data

x = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# %%

# Create model and train it

model = LogisticRegression(solver='liblinear', random_state=0, penalty='l2') # penalty parameter decides which regularization is applied
model.fit(x, y)

# %%

# Evaluate model

model.predict_proba(x)
model.predict(x)
model.score(x, y)

# Confusion matrix, shows the numbers of the following:

# True negatives in the upper-left position
# False negatives in the lower-left position
# False positives in the upper-right position
# True positives in the lower-right position

cm = confusion_matrix(y, model.predict(x))

# Visualize it

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()

# If you want to improve the model accuracy, go back and use model = LogisticRegression() with different inputs,such as another solver or different regularization




