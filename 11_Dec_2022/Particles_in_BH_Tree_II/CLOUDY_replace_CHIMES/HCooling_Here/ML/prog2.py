
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import tensorflow as tf


df = pd.read_csv('HCmu.csv')

# ['lognH', 'rkpc', 'logNHtot', 'logT', 'logHeating', 'logCooling', 'mu']
print(df.columns)

lognH = df['lognH'].values
rkpc = df['rkpc'].values
logNHtot = df['logNHtot'].values
logT = df['logT'].values
logHeating = df['logHeating'].values
logCooling = df['logCooling'].values
mu = df['mu'].values


X = df.iloc[:, :4].values
y = logCooling

seed = 10
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(4,)),  # First hidden layer with 128 neurons and ReLU activation
    tf.keras.layers.Dense(64, activation='relu'),  # Second hidden layer with 64 neurons and ReLU activation
    tf.keras.layers.Dense(1)  # Output layer with a single neuron since we are predicting a continuous value
])

# Compile the model
model.compile(optimizer='adam',  # Optimizer
              loss='mse',  # Mean Squared Error for regression problems
              metrics=['mae'])  # Mean Absolute Error as an additional metric

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)  # Train for 100 epochs and use 20% of training data for validation

# Evaluate the model on the test data
test_loss, test_mae = model.evaluate(X_test, y_test)

print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Save the model
model.save('my_model.h5')

s()



Tgrid = np.linspace(2, 9, 500)
res = []
for i in range(len(Tgrid)):
  res.append([3.0, 0.5, 20.05, Tgrid[i]])

res = np.array(res)
print(res.shape)
ypred = model.predict(res)

nx = np.where((lognH == 3.0) & (rkpc == 0.5) & (logNHtot == 20.0))[0]

print(len(nx))

if True:
  plt.scatter(logT[nx], logCooling[nx], s = 20, color = 'blue')
  plt.scatter(logT[nx], logHeating[nx], s = 10, color = 'red')
  plt.scatter(Tgrid, ypred, s = 5, color = 'lime')
  plt.show()


