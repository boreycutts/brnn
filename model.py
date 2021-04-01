import numpy as np
import random
from datetime import datetime
from pytz import timezone
import os.path
from os import path
import math
import time
import matplotlib.pyplot as plt
import sys

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import SGD

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def load_model(model_name):
    if not model_name:
        model_name = input('Enter the model name to load a model for testing (or leave blank to create a new model): ')
    if model_name:
        print('\n\nLoading model...\n\n')
        try:
            model = tf.keras.models.load_model('models/' + model_name)
            print('\n\nModel summary:\n\n')
            model.summary()
            print(model.layers[0].get_weights())
            print(len(model.layers[0].get_weights().flatten()))
            return model
        except:
            if model_name:
                print('No model found')
            return None    

    return None

def save_model(model, name=None):
    if name:
        model.save('models/' + name)
        return name

    model_name = input('\n\nSave model as: ')
    while True:
        if path.exists('models/' + model_name):
            response = input('\n\nA model already exists with the same name would you like to overwrite? (y/n) ')
            if response.lower().replace(' ', '') == 'y':
                print('\n\nSaving model...\n\n')
                if model:
                    model.save('models/' + model_name)
                return model_name
            else:
                model_name = input('\n\nSave model as: ')
        else:
            print('\n\nSaving model...\n\n')
            if model:
                model.save('models/' + model_name)
            return model_name

def create_model(model_type, input_shape, dense_nodes):
    print('\n\nCreating new model...\n\n')

    model = tf.keras.models.Sequential()

    if model_type == "CNN":
        model.add(Conv1D(filters=32, kernel_size=2, activation="relu"))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(dense_nodes, activation="relu"))
    elif model_type == "LSTM":
        model.add(Bidirectional(LSTM(8, return_sequences=False, activation="relu")))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(dense_nodes, activation="relu"))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
    else:
        print("Model type not supported")
        exit()

    model.add(Dense(1, activation="linear"))

    return model

class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        self.testtime = 0
        self.timetaken = time.time()
    def on_epoch_begin(self, epoch, logs={}):
        self.timetaken = time.time()
    def on_epoch_end(self,epoch,logs = {}):
        self.times.append((time.time() - self.timetaken)*1000)
    def on_train_end(self,logs = {}):
        print('Average Training Time:')
        print(np.mean(self.times))
    def on_test_end(self, logs={}):
        print('Evaluation Time (ms): ' + str((time.time() - self.timetaken)*1000))

def train_model(model, x_train, y_train, epochs, opt, batch_size):
    print('\n\nTraining model...\n\n')
    opt = tf.keras.optimizers.Adadelta() if opt == 'adadelta' else tf.keras.optimizers.Adam(learning_rate=1e-6)
    loss = 'mse'
    model.compile(loss=loss, optimizer=opt)

    history = None
    timetaken = timecallback()

    while True:
        try:
            ## Train with one dataset
            history = model.fit(
                x_train, 
                y_train, 
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[timetaken]
            )
            break
        except KeyboardInterrupt:
            print('\n\nTraining Stopped\n\n')
            break

    model.summary()

    return (model, history)

def test_model(model, x_test, y_test, data_obj, history, batch_size, model_name, figure_name, component_name, plot):
    print('\n\nTesting model...\n\n')
    timetaken = timecallback()
    while True:
        try:
            model.evaluate(
                x_test, 
                y_test,
                batch_size=batch_size,
                callbacks=[timetaken]
            )
            break
        except KeyboardInterrupt:
            print('\n\nTesting Stopped\n\n')
            break

    predict_time = time.time()
    output = model.predict(x_test, batch_size=batch_size).flatten()

    diff = np.array([])
    mse = []
    mse_x = []
    for i in range(0, len(output)-128, 128):
        diff = np.concatenate((diff, data_obj["component_output"][i:i+128] - output[i:i+128]))
        mse.append((diff**2).mean())
        mse_x.append(i+128)

    print("Prediction Time (ms): " + str((time.time() - predict_time)*1000))

    print("Signal to Noise Ratio: " + str(signaltonoise(output)))

    if not os.path.exists("figures/" + model_name):
        os.makedirs("figures/" + model_name)

    fig, axs = plt.subplots(3)

    axs[0].plot(data_obj["component_input"], 'r')
    axs[0].set_title("Input")

    axs[1].plot(data_obj["component_output"], 'g')
    axs[1].set_title(component_name + " Output")

    axs[2].plot(output)
    axs[2].tick_params(axis='y', colors="#1f77b4")

    ax2 = axs[2].twinx()
    ax2.plot(mse_x, mse, "orange", label="Error")
    ax2.tick_params(axis='y', colors='orange')
    ax2.legend()
    axs[2].set_title("Network Output")
    
    fig.tight_layout()

    plt.savefig("figures/" + model_name + "/" + figure_name)

    if history:
        plt.figure()
        plt.plot(range(len(history.history['loss'])), history.history['loss'])
        plt.title('Loss vs Epochs')
        plt.savefig("figures/" + model_name + "/" + figure_name + ' Loss vs Epochs.png')
        np.save("figures/" + model_name + "/" + figure_name + " Loss vs Epochs", history.history['loss'])

    if plot:
        plt.show()

def compute_magnitude(model, x_test, data_obj, batch_size, frequency, model_name, plot):
    output = model.predict(x_test, batch_size=batch_size)
    
    x = data_obj['component_input'][2000:]
    y = data_obj['component_output'][2000:]
    y_hat = output.flatten()[2000:]

    magnitude = 20*math.log10((max(y) - min(y))/(max(x) - min(x)))
    magnitude_hat = 20*math.log10((max(y_hat) - min(y_hat))/(max(x) - min(x)))

    print('Low Pass Magnitude = ' + str(magnitude))
    print('Network Magnitude = ' + str(magnitude_hat))

    if not os.path.exists("figures/" + model_name + "/freq"):
        os.makedirs("figures/" + model_name + "/freq")

    frequency = str(frequency)
    plt.figure()
    plt.plot(data_obj["component_output"], 'g')
    plt.title('Lowpass Output')
    plt.savefig("figures/" + model_name + "/freq/" + frequency + " Lowpass Output.png")

    plt.figure()
    plt.plot(output)
    plt.title('Network Output')
    plt.savefig("figures/" + model_name + "/freq/" + frequency + ' Network Output.png')

    plt.figure()
    plt.plot(data_obj["component_input"], 'r')
    plt.title('Input')
    plt.savefig("figures/" + model_name + "/freq/" + frequency + ' Input.png')

    if plot:
        plt.show()

    plt.close('all')

    return (magnitude, magnitude_hat)
