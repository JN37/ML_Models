# %% Import

import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import estimate_bandwidth


# %% Set up clusters and visualize

# Set up coordinates that the generated data will revolve around
coordinates = [[2, 2, 3], [6, 7, 8], [5, 10, 13]]

X, _ = make_blobs(n_samples=120, centers=coordinates, cluster_std=0.60)

data_fig = plt.figure(figsize=(12, 10))
ax = data_fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o', color='green')
plt.show()

# %% Implement Mean-Shift to predict the cluster and define the centroids of the clusters.

# Estimate bandwidth parameter

bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

msc = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X)
cluster_centers = msc.cluster_centers_
labels = msc.labels_
cluster_label = np.unique(labels)
n_clusters = len(cluster_label)
# n_clusters = 3, as it should be

# %% Visualize the clusters

msc_fig = plt.figure(figsize=(12, 10))

ax = msc_fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o', color='yellow')

ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
           cluster_centers[:, 2], marker='o', color='green',
           s=300, linewidth=5, zorder=10)
plt.title('Estimated number of clusters: %d' % n_clusters)
plt.show()
