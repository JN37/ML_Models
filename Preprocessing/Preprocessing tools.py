# %%

### Identify and handle missing values ###

import numpy as np

# Delete the row containing missing values, or

# Interpolate between two values, use:

xp = [1, 2, 3]
fp = [3, 2, 0]
ip_value = np.interp(2.5, xp, fp)

# %%

### Normalizing data ###

# Here, you have a few different options:

from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler

# Normalizer = Normalize samples individually to unit norm.

X = [[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]]
normalizer = Normalizer().fit(X)
X_norm = normalizer.transform(X)

# StandardScaler = Standardize features by removing the mean and scaling to unit variance

# The standard score of a sample x is calculated as:
# z = (x - u) / s
# where u is the mean of the training samples or zero if with_mean=False,
# and s is the standard deviation of the training samples or one if with_std=False.

scaler_data = [[0, 0], [0, 0], [1, 1], [1, 1]]
scaler = StandardScaler()
scaled_values = scaler.fit_transform(scaler_data)

# MinMaxScaler = Transform features by scaling each feature to a given range.

# The transformation is given by:
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

min_max_data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
mm_scaler = MinMaxScaler()
mm_scaled_data = mm_scaler.fit_transform(min_max_data)

# %%

### Encoding ###

from numpy import asarray
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# Ordinal Encoding = Encode categorical features as an integer array.

ord_data = asarray([['red'], ['green'], ['blue']])
ord_encoder = OrdinalEncoder()
ord_result = ord_encoder.fit_transform(ord_data)

# One-Hot Encoding = Encode categorical features as a one-hot numeric array.

onehot_data = asarray([['red'], ['green'], ['blue']])
onehot_encoder = OneHotEncoder(sparse=False)
onehot_result = onehot_encoder.fit_transform(onehot_data)

# Dummy Variable Encoding

# Same as OneHot, but drops one number as it is enough to represent "red" with [0, 0]
# if blue is [1, 0, 0] and green is [0, 1, 0]. Always represents C categories with C-1 binary variables.

dummy_data = asarray([['red'], ['green'], ['blue']])
dummy_encoder = OneHotEncoder(drop='first', sparse=False)
dummy_result = dummy_encoder.fit_transform(dummy_data)

# %%

### Binarization ###

from sklearn.preprocessing import Binarizer
import pandas
import numpy
