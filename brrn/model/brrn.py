import numpy as np
import random
from datetime import datetime
from pytz import timezone
# from random_word import RandomWords
import os.path
from os import path
import math

from brrn.data.create_data import create_data
from brrn.data.format_data import format_data

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import SGD

import matplotlib.pyplot as plt

import time

BATCH_SIZE = 128

def load_model(model_name):
    if not model_name:
        model_name = input('Enter the model name to load a model for testing (or leave blank to create a new model): ')
    if model_name:
        try:
            print('\n\nLoading model...\n\n')
            model = tf.keras.models.load_model('models/' + model_name)
            print('\n\nModel summary:\n\n')
            model.summary()
            return model
        except:
            if model_name:
                print('No model found')
            return None    

    return None

def create_model(input_shape, lstm_nodes, dense_nodes):
    print('\n\nCreating new model...\n\n')

    model = tf.keras.models.Sequential()

    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))

    # model.add(Bidirectional(LSTM(lstm_nodes, return_sequences=False, activation='relu')))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())

    # model.add(Dense(50, activation="sigmoid"))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())

    model.add(Dense(1, activation="linear"))

    return model

class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        self.testtime = 0
        # use this value as reference to calculate cummulative time taken
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

def train_model(model, x_train, y_train, epochs, opt, save=False):
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
                batch_size=BATCH_SIZE,
                callbacks=[timetaken]
            )
            break
        except KeyboardInterrupt:
            print('\n\nTraining Stopped\n\n')
            break

    model.summary()

    if(save):
        model_name = input('\n\nSave model as (or leave blank to save using timestamp): ')
        while True:
            if path.exists('models/' + model_name):
                response = input('\n\nA model already exists with the same name would you like to overwrite? (y/n) ')
                if response.lower().replace(' ', '') == 'y':
                    break
                else:
                    model_name = input('\n\nSave model as (or leave blank to save using timestamp): ')
            else:
                break

        print('\n\nSaving model...\n\n')
        model.save('models/' + model_name)

    return (model, history)

def test_model(model, x_test, y_test, data_obj, history, name):
    print('\n\nTesting model...\n\n')
    timetaken = timecallback()
    while True:
        try:
            model.evaluate(
                x_test, 
                y_test,
                batch_size=BATCH_SIZE,
                callbacks=[timetaken]
            )
            break
        except KeyboardInterrupt:
            print('\n\nTesting Stopped\n\n')
            break

    output = model.predict(x_test, batch_size=BATCH_SIZE)

    x = data_obj['signal_with_noise'][2000:]
    y = data_obj['signal_filtered'][2000:]
    y_hat = output.flatten()[2000:]

    # magnitude = 20*math.log10((max(y) - min(y))/(max(x) - min(x)))
    # magnitude_hat = 20*math.log10((max(y_hat) - min(y_hat))/(max(x) - min(x)))

    # print('Magnitude = ' + str(magnitude))
    # print('Magnitude_hat = ' + str(magnitude_hat))


    plt.plot(range(len(data_obj.get('signal_filtered'))), data_obj.get('signal_filtered'), 'g')
    plt.title('Lowpass Output')
    plt.savefig(name + ' Lowpass Output.png')

    plt.figure()
    plt.plot(range(len(output)), output)
    plt.title('Network Output')
    plt.savefig(name + ' Network Output.png')

    plt.figure()
    plt.plot(range(len(data_obj.get('signal_with_noise'))), data_obj.get('signal_with_noise'), 'r')
    plt.title('Input')
    plt.savefig(name + ' Input.png')

    if history:
        plt.figure()
        plt.plot(range(len(history.history['loss'])), history.history['loss'])
        plt.title('Loss vs Epochs')
        plt.savefig(name + ' Loss vs Epochs.png')

    plt.show()
    