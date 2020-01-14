from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import time as t
import pandas as pd

X = pd.read_csv('../data/x_test.csv', sep=' ')
Y = pd.read_csv('../data/y_test.csv', sep=' ')

model = Sequential()
model.add(Dense(256, input_dim=784, activation="tanh"))
model.add(Dense(10, activation="sigmoid"))
model.compile(optimizer='adam', loss='mse')
model.fit(X, Y, epochs=100, batch_size=256)
y_pred = model.predict(X)
np.set_printoptions(2, suppress=True)
for i in range(10):
    print("Expected:")
    print(np.array(Y.iloc[i]).T)
    print("Got:")
    print(y_pred[i].T)
    print("\n--------------------------\n")
