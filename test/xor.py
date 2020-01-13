from keras.models import Sequential
from keras.layers import Dense
import numpy as np

X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
Y = np.array([[1], [1], [0], [0]])

model = Sequential()
model.add(Dense(8, input_dim=2, activation="tanh"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X, Y, epochs=2000, batch_size=1)
Y_pred = model.predict(X)
print(Y_pred)
for w in model.weights:
    print(w)
