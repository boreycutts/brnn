import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_model(x_train, y_train):
    x_train = tf.keras.utils.normalize(x_train, axis=1)

    # mnist = tf.keras.datasets.mnist
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train = tf.keras.utils.normalize(x_train, axis=1)
    # x_test = tf.keras.utils.normalize(x_test, axis=1)

    # print(x_train)
    # print(x_train.shape)
    # print(y_train)
    # print(y_train.shape); exit()

    model = tf.keras.models.Sequential()

    model.add(LSTM(128, input_shape=x_train.shape[1:], activation="relu", return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation="softmax"))


    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-5)
    model.compile(loss="mean_absolute_error", optimizer=opt, metrics=["accuracy"])

    logdir="logs/fit/loooooooog"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    model.fit(
        x_train, 
        y_train, 
        epochs=3,
        callbacks=[tensorboard_callback])