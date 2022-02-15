# %% Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

# %% Download data

data = yf.download(tickers='GOOG', period='5y', interval='1d')
# Filter so we only get closing price
data = data['Close']

# %% Data fiddling

# Divide up in training/test data and create numpy arrays to hold them
training_data = data.head(int(len(data) * 0.8))
training_data = training_data[:].values

test_data = data.tail(int(len(data) * 0.2))
test_data = test_data[:].values

# %%

scaler = MinMaxScaler()

# Normalize the values
training_data = scaler.fit_transform(training_data.reshape(-1, 1))

# Prepare the data to be the right format for TensorFlow
x_training_data = []
y_training_data = []

for i in range(40, len(training_data)):
    x_training_data.append(training_data[i - 40:i, 0])

    y_training_data.append(training_data[i, 0])

x_training_data = np.array(x_training_data)
y_training_data = np.array(y_training_data)

x_training_data = np.reshape(x_training_data, (x_training_data.shape[0], x_training_data.shape[1], 1))

# %% Build the RNN

rnn = Sequential()

# Add the first layer. Use Dropout regularization to prevent overfitting
rnn.add(LSTM(units=45, return_sequences=True, input_shape=(x_training_data.shape[1], 1)))
rnn.add(Dropout(0.2))

# Add three more layers with subsequent dropouts
rnn.add(LSTM(units=45, return_sequences=True))
rnn.add(Dropout(0.2))

rnn.add(LSTM(units=45, return_sequences=True))
rnn.add(Dropout(0.2))

rnn.add(LSTM(units=45))
rnn.add(Dropout(0.2))

# Add last (output) layer
rnn.add(Dense(units=1))

# Compile the model using the Adam optimizer, and since we're predicting a continuous variable, we can use mean squared error (MSE) to calculate the loss
rnn.compile(optimizer='adam', loss='mean_squared_error')

# %% Fit model to training data
rnn.fit(x_training_data, y_training_data, epochs=100, batch_size=32)

# Test model predictions against test data

# Fix up test data
unscaled_training_data = pd.DataFrame(training_data)
unscaled_test_data = pd.DataFrame(test_data)
all_data = pd.concat((unscaled_training_data, unscaled_test_data), axis=0)
x_test_data = all_data[len(all_data) - len(test_data) - 40:].values
x_test_data = np.reshape(x_test_data, (-1, 1))
x_test_data = scaler.transform(x_test_data)

final_x_test_data = []

for i in range(40, len(x_test_data)):
    final_x_test_data.append(x_test_data[i - 40:i, 0])

final_x_test_data = np.array(final_x_test_data)

final_x_test_data = np.reshape(final_x_test_data, (final_x_test_data.shape[0], final_x_test_data.shape[1], 1))

# %% Make predictions
predictions = rnn.predict(final_x_test_data)
plt.plot(predictions)

# Unscale to make the predictions meaningful

unscaled_predictions = scaler.inverse_transform(predictions)
plt.plot(unscaled_predictions)

plt.plot(unscaled_predictions, color='red', label="Predictions")
plt.plot(test_data, color='black', label="Real Data")

plt.title('Google Stock Price Predictions')
