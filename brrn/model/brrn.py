import numpy as np
import random
from datetime import datetime
from pytz import timezone
# from random_word import RandomWords
import os.path
from os import path

from brrn.data.create_data import create_data
from brrn.data.format_data import format_data

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD

import matplotlib.pyplot as plt

# def generate_model_name():
#     random_word = None
#     try:
#         r = RandomWords()
#         random_word = r.get_random_word()
#     except:
#         random_word = 'null'
     
#     time_format = "%Y-%m-%d_%H-%M-%S"
#     now_time = datetime.now(timezone('US/Eastern'))

#     return now_time.strftime(time_format) + '_' + random_word

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

    model.add(Bidirectional(LSTM(lstm_nodes, batch_input_shape=input_shape, return_sequences=False, activation='relu')))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(dense_nodes, activation="sigmoid"))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(1, activation="linear"))

    return model


def train_model(model, x_train, y_train, epochs, opt, save=False):
    print('\n\nTraining model...\n\n')
    opt = tf.keras.optimizers.Adadelta() if opt == 'adadelta' else tf.keras.optimizers.Adam(learning_rate=1e-5)
    loss = 'mse'
    model.compile(loss=loss, optimizer=opt)

    history = None

    while True:
        try:
            ## Train with one dataset
            history = model.fit(
                x_train, 
                y_train, 
                epochs=epochs
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

def test_model(model, x_test, y_test, data_obj, history):
    print('\n\nTesting model...\n\n')
    while True:
        try:
            model.evaluate(
                x_test, 
                y_test
            )
            break
        except KeyboardInterrupt:
            print('\n\nTesting Stopped\n\n')
            break

    output = model.predict(x_test)

    plt.plot(range(len(data_obj.get('signal_filtered'))), data_obj.get('signal_filtered'), 'g')
    plt.title('Lowpass Output')

    plt.figure()
    plt.plot(range(len(output)), output)
    plt.title('Network Output')

    plt.figure()
    plt.plot(range(len(data_obj.get('signal_with_noise'))), data_obj.get('signal_with_noise'), 'r')
    plt.title('Input')

    if history:
        plt.figure()
        plt.plot(range(len(history.history['loss'])), history.history['loss'])
        plt.title('Loss vs Epochs')

    plt.show()
    