import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD

import matplotlib.pyplot as plt

def create_model(input_shape):
    model = tf.keras.models.Sequential()

    forward_layer = LSTM(128, activation='relu', return_sequences=True)
    backward_layer = LSTM(128, activation='relu', return_sequences=True,
                        go_backwards=True)
    model.add(Bidirectional(forward_layer, backward_layer=backward_layer,
                            input_shape=(input_shape)))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.1))

    model.add(Dense(1, activation="linear"))

    return model


def train_model(model, x_train, y_train, epochs, d):
    # opt = tf.keras.optimizers.Adam(learning_rate=5, decay=1e-6)
    opt = tf.keras.optimizers.SGD(
        learning_rate=0.01, momentum=0.0, nesterov=False
    )
    # model.compile(loss="mean_squared_error", optimizer=opt, metrics=["accuracy"])
    # model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=opt, metrics=["accuracy"])
    # opt = SGD(lr=0.01)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=opt, metrics=["accuracy"])

    model.fit(
        x_train, 
        y_train, 
        epochs=epochs)

    output = model.predict(x_train)

    x = []
    y = []

    for i in range(len(output)):
        x.append(i)
        y.append(output[i][0][0])
    
    plt.plot(x,d.get('signal'))
    plt.title('signal')
    plt.figure()
    plt.plot(d.get('signal_with_noise'))
    plt.title('signal wit noise')
    plt.figure()
    plt.plot(x, y)
    plt.title('robot signal')
    plt.show()

    # # for layer in model.layers:
    # #     keras_function = tf.keras.backend.function([model.input], [layer.output])
    # #     outputs.append(keras_function([x_train, 1]))

    # # keras_function = tf.keras.backend.function([model.input], [model.layers[len(model.layers) - 1].output])
    # keras_function = tf.keras.backend.function([model.input], [model.layers[0].output])
    # output = keras_function([x_train, 1])

    # print(output[0])
    # print('------------------------------')
    # print(output[0][0])
    # print('------------------------------')
    # print(output[0][0][0])
    # print('------------------------------')
    # print(output[0][0][0][0])
    
    # x = []
    # y = []
    # for i in range(len(output[0])):
    #     x.append(i)
    #     y.append(output[0][i][0][0])
    
    # print('OUTPUT_y')
    # print(y)
    
    # plt.plot(d.get('signal_with_noise'))
    # plt.figure()
    # plt.plot(x, y)
    # plt.show()
    