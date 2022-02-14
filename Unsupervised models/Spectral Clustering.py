# %% Import libs

from numpy import random
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs

# %% Create some data

random.seed(1)
x, _ = make_blobs(n_samples=400, centers=4, cluster_std=1.5)

plt.scatter(x[:, 0], x[:, 1])
plt.show()

# %%

# Apply clustering method

sc = SpectralClustering(n_clusters=4).fit(x)
print(sc)

SpectralClustering(affinity='rbf', assign_labels='kmeans', coef0=1, degree=3,
                   eigen_solver=None, eigen_tol=0.0, gamma=1.0,
                   kernel_params=None, n_clusters=4, n_components=None,
                   n_init=10, n_jobs=None, n_neighbors=10, random_state=None)

# %%

# Visualize clusters

labels = sc.labels_
plt.scatter(x[:, 0], x[:, 1], c=labels)
plt.show()

# %%

# Try changing the number of clusters

f = plt.figure()
f.add_subplot(2, 2, 1)
for i in range(2, 6):
    sc = SpectralClustering(n_clusters=i).fit(x)
    f.add_subplot(2, 2, i - 1)
    plt.scatter(x[:, 0], x[:, 1], s=5, c=sc.labels_, label="n_cluster-" + str(i))
    plt.legend()

plt.show()
