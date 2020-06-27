import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_anomaly():
    i = random.randint(0, 1)
    return [x_t_max, x_t_min][i]

N = 1
T = 100
x_t_max = 100
x_t_min = -100
x_N = []
x_t = []
variation = (0.01, 0.1)

for t in range(T):
    val = x_t[t-1] + (random.uniform(variation[0], variation[1]) * random.randint(-1, 1)) if t > 0 else 0

    if val > x_t_max:
        val = x_t_max
    elif val < x_t_min:
        val = x_t_min

    x_t.append(val)

for n in range(N):
    x_N.append([x_t.copy()])
    for t in range(T):
        if random.randint(0, 100) == 42:
            x_N[n][0][t] = create_anomaly()

x_train = np.array(x_N)
y_train = np.array([x_t])

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# print(x_train)
# print(x_train.shape)
# print(y_train)
# print(y_train.shape); exit()

model = tf.keras.models.Sequential()

model.add(LSTM(128, input_shape=x_train.shape[1:], activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='softmax'))


opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-5)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30)