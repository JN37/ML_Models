# %%

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# %% Load and fiddle with data

breast = load_breast_cancer()

breast_data = breast.data

breast_labels = breast.target
labels = np.reshape(breast_labels, (569, 1))
final_breast_data = np.concatenate([breast_data, labels], axis=1)
breast_dataset = pd.DataFrame(final_breast_data)

# %% Features

# Check features in the dataset

features = breast.feature_names
print(features)

# Add labels to the dataset

features_labels = np.append(features, 'label')
breast_dataset.columns = features_labels

# Encode labels to show if sample is benign or malignant

breast_dataset['label'].replace(0, 'Benign', inplace=True)
breast_dataset['label'].replace(1, 'Malignant', inplace=True)

# %% Normalizing

# First, normalize the data
x = breast_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x)  # normalizing the features

# Check that the data now has mean 0 and std 1

print(np.mean(x), np.std(x))

# Convert new features into a table

feat_cols = ['feature' + str(i) for i in range(x.shape[1])]

normalised_breast = pd.DataFrame(x, columns=feat_cols)

# %% Apply PCA

pca_breast = PCA(n_components=2)
p_components = pca_breast.fit_transform(x)

principal_breast_Df = pd.DataFrame(data=p_components, columns=['principal component 1', 'principal component 2'])

# Print the explained variance ratio

print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_)) # The printed values mean that pca1 hold 44% of the information, and pca2 19%, and that 37% of the total information were lost when we dropped the third dimension

# Visualize the principal components

plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)
targets = ['Benign', 'Malignant']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = breast_dataset['label'] == target
    plt.scatter(principal_breast_Df.loc[indicesToKeep, 'principal component 1']
                , principal_breast_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})
