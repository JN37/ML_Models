# %% Import libs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

# %% Get data and normalize it. Not normalizing here would mean that the model could be biased towards variables with higher magnitude

data = pd.read_csv('Wholesale_customers_data.csv')
data_scaled = normalize(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

# %% Draw the dendrogram so we can decide the number of clusters

# Here, inspect the plot and make sure to draw a threshold line at the top to separate the data into two segments, after you've done that, uncomment the below code above plt.show()
plt.figure(figsize=(10, 7))
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
# plt.axhline(y=6, color='r', linestyle='--') In this case, we cut the line at y = 6 and thus end up with two clusters
plt.show()

# Apply the clustering method

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit_predict(data_scaled)

# Visualize the clusters

plt.figure(figsize=(10, 7))
plt.scatter(data_scaled['Milk'], data_scaled['Grocery'], c=cluster.labels_)
plt.show()