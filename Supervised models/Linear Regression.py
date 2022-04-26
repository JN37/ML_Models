from datetime import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import Sequential
from keras import layers

np.set_printoptions(precision=3, suppress=True)

# %%

# Load data

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)

# %%

### Data preprocessing ##

dataset = raw_dataset.copy()

# Check for missing values and drop them if there are any

dataset.isna().sum()
dataset = dataset.dropna()

# The "Origin" column is categorical, not numeric, so we have to one-hot encode them. Check the example of one-hot
# encoding in Preprocessing tools.py.

dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')

# After encoding is done, you can now see which country the car model originates from, rather than just see a numeric

# %%

### Create training- and test set ###

train_dataset = dataset.sample(frac=0.8,
                               random_state=0)  # frac = how much of data is used for training, in this case 80%
test_dataset = dataset.drop(train_dataset.index)

# %%

### Inspection of data ###

sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
train_dataset.describe().transpose()

# Split features from labels

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# %%

# Normalize = make sure the features are of the same range

# It is good practice to normalize features that use different scales and ranges.
# One reason this is important is because the features are multiplied by the model weights. So, the scale of the outputs and the scale of the gradients are affected by the scale of the inputs.
# Although a model might converge without feature normalization, normalization makes training much more stable.

checked_features = train_dataset.describe().transpose()[['mean', 'std']]
normalizer = tf.keras.layers.Normalization(axis=-1)

normalizer.adapt(np.array(train_features))  # Fitting
normalizer.mean.numpy()  # Store the mean/variance in the layer

# %%

### Linear regression with one variable ###

# Predict MPG from Horsepower

# 1. Normalize the 'Horsepower' input features using the tf.keras.layers.Normalization preprocessing layer.
# 2. Apply a linear transformation () to produce 1 output using a linear layer (tf.keras.layers.Dense).

horsepower = np.array(train_features['Horsepower'])

horsepower_normalizer = tf.keras.layers.Normalization(input_shape=[1, ], axis=None)
horsepower_normalizer.adapt(horsepower)

# Build model. To check it, use model_name.summary()

horsepower_model = Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])

horsepower_model.predict(horsepower[:10])

horsepower_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error'
)

# %%

# Train model

model_history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split=0.2)

hist = pd.DataFrame(model_history.history)
hist['epoch'] = model_history.epoch


# %%
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)


plot_loss(model_history)

# %%

# Check results


test_results = {'horsepower_model': horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels, verbose=0)}

x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)


def plot_horsepower(x, y):
    plt.scatter(train_features['Horsepower'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.legend()


plot_horsepower(x, y)
