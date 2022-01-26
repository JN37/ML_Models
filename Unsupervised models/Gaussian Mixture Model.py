# %%

# Example from https://www.analyticsvidhya.com/blog/2019/10/gaussian-mixture-models-clustering/

# The link to the file "Clustering_gmm" is found in the article

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

data = pd.read_csv('Clustering_gmm.csv')

plt.figure(figsize=(7, 7))
plt.scatter(data["Weight"], data["Height"])
plt.xlabel('Weight')
plt.ylabel('Height')
plt.title('Data Distribution')
plt.show()

# %% Set up GMM, train it and predict clusters

gmm = GaussianMixture(n_components=4)
gmm.fit(data)

labels = gmm.predict(data)
frame = pd.DataFrame(data)
frame['cluster'] = labels
frame.columns = ['Weight', 'Height', 'cluster']

# Plot the clusters
color = ['blue', 'green', 'cyan', 'black']
for k in range(0, 4):
    data = frame[frame["cluster"] == k]
    plt.scatter(data["Weight"], data["Height"], c=color[k])
plt.show()
