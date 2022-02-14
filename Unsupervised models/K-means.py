# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# %%

# Generate random data

X = -2 * np.random.rand(100, 2)
X1 = 1 + 2 * np.random.rand(50, 2)
X[50:100, :] = X1
plt.scatter(X[:, 0], X[:, 1], s=50, c="b")
plt.show()

# %%

# Create model

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# Display clusters

clust = kmeans.cluster_centers_

plt.scatter(X[:, 0], X[:, 1], s=50, c="b")
plt.scatter(clust[0, 0], clust[0, 1], s=200, c="g", marker="s")
plt.scatter(clust[1, 0], clust[1, 1], s=200, c="r", marker="s")
plt.show()
