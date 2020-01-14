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
start = t.time()
model.fit(X, Y, epochs=1000, batch_size=256)
end = t.time()
print("It took {} ms".format((end - start) * 1000))
