# %%

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import AffinityPropagation

np.random.seed(1)

# %%

# Configuration options for the algorithm
num_samples_total = 50
cluster_centers = [(20, 20), (4, 4)]
num_classes = len(cluster_centers)

# Generate data
X, targets = make_blobs(n_samples=num_samples_total, centers=cluster_centers, n_features=num_classes, center_box=(0, 1),
                        cluster_std=1)

# %%

# Set up model
ap = AffinityPropagation(max_iter=250)
ap.fit(X)
cluster_centers_indices = ap.cluster_centers_indices_
n_clusters = len(cluster_centers_indices)

# %%

# Predict the clusters
predictions = ap.predict(X)

# Plot the training data
colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', predictions))
plt.scatter(X[:, 0], X[:, 1], c=colors, marker="o", picker=True)
plt.title(f'Estimated number of clusters = {n_clusters}')
plt.xlabel('Temperature yesterday')
plt.ylabel('Temperature today')
plt.show()
