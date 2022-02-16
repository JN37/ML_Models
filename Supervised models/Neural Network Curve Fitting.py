# %%

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import math

# %% Generate data
x_data = np.linspace(-10, 10, num=1000)
y_data = 0.1 * x_data * np.cos(x_data) + 0.1 * np.random.normal(size=1000)

# %% Set up neural net

model = keras.Sequential()
model.add(keras.layers.Dense(units=1, activation='linear', input_shape=[1]))
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dense(units=64, activation='elu'))
model.add(keras.layers.Dense(units=1, activation='linear'))
model.compile(loss='mse', optimizer="adam")

# Look at model
model.summary()

# %% Train model

model.fit(x_data, y_data, epochs=100, verbose=1)

# %% Predit output and plot it
y_predicted = model.predict(x_data)

plt.scatter(x_data[::1], y_data[::1])
plt.plot(x_data, y_predicted, 'r', linewidth=4)
plt.grid()
plt.show()