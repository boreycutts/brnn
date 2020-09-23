import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD

import matplotlib.pyplot as plt

def create_model(input_shape, lstm_nodes):
    model = tf.keras.models.Sequential()

    model.add(Bidirectional(LSTM(lstm_nodes, batch_input_shape=input_shape, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(128, activation="sigmoid"))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(1, activation="linear"))

    return model


def train_model(model, x_train, y_train, epochs, d):
    opt = tf.keras.optimizers.Adadelta()
    loss = 'mse'
    model.compile(loss=loss, optimizer=opt, metrics=['mse'])

    history = None

    while True:
        try:
            history = model.fit(
                x_train, 
                y_train, 
                epochs=epochs)
            break
        except KeyboardInterrupt:
            print('\n\nTraining Stopped\n')
            break

    model.summary()

    output = model.predict(x_train)

    plt.plot(range(len(d.get('signal_filtered'))), d.get('signal_filtered'), 'g')
    plt.title('Filtered Signal')

    plt.figure()
    plt.plot(range(len(output)), output)
    plt.title('Network Output')

    plt.figure()
    plt.plot(range(len(d.get('signal_with_noise'))), d.get('signal_with_noise'), 'r')
    plt.title('Network Input')

    plt.figure()
    plt.plot(range(len(history.history['loss'])), history.history['loss'])
    plt.title('Loss vs Epochs')

    plt.show()
    