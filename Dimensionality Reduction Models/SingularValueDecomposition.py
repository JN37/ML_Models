# %% Different ways of applying SVD

# %% Using Numpy

import numpy as np

# Creating original matrix
A = np.array([[3, 4, 3], [1, 2, 3], [4, 2, 1]])

# Perform SVD on the original matrix, and obtain U,V (orthogonal matrices for rotation) and D (diagonal, for scaling)
U, D, VT = np.linalg.svd(A)

# Checking if we can remake the original matrix using U,D,VT
A_remake = (U @ np.diag(D) @ VT)
print(A_remake)

# %% Using scikit-learn

import numpy as np
from sklearn.decomposition import TruncatedSVD

# Creating original matrix
A = np.array([[3, 4, 3], [1, 2, 3], [4, 2, 1]])

# Fitting the SVD
svd = TruncatedSVD(n_components=2)
A_transformed = svd.fit_transform(A)
