from brrn.data.create_data import create_data
from brrn.data.format_data import format_data
from brrn.model.brrn import load_model, create_model, train_model, test_model
import numpy as np

# Constants
INPUT_LENGTH = 8
INPUT_SAMPLE_RATE = 10e6
TIMESTEPS = 32

# freedosuperlite
# LSTM_NODES = 1
# DENSE_NODES = 16
# EPOCHS_DELTA = 2000                
# EPOCHS_ADAM = 500      

# freedolite
LSTM_NODES = 8
DENSE_NODES = 32
EPOCHS_DELTA = 2000                
EPOCHS_ADAM = 500      

# freedo
# LSTM_NODES = 32
# DENSE_NODES = 128
# EPOCHS_DELTA = 1200                
# EPOCHS_ADAM = 100                

def run():
    model = load_model()
    history = None
    train = False

    if model:
        response = input('Would you like to continue training? (y/n) ')
        if response.lower().replace(' ', '') == 'y':
            train = True

    if not model or train:
        data_obj_train = create_data(input_length=INPUT_LENGTH, fs=INPUT_SAMPLE_RATE, fnoise_min=2e6, fnoise_max=5e6, f1=500e3, cutoff=1e6, T=1e-4)
        (x_train, y_train) = format_data(TIMESTEPS, data_obj_train)

        if not model:
            model = create_model(x_train.shape, LSTM_NODES, DENSE_NODES)

        (model, history) = train_model(model, x_train, y_train, EPOCHS_DELTA, 'adadelta')
        (model, history) = train_model(model, x_train, y_train, EPOCHS_ADAM, 'adam')

    data_obj_test = create_data(input_length=INPUT_LENGTH, fs=INPUT_SAMPLE_RATE, fnoise_min=2e6, fnoise_max=5e6, f1=500e3, cutoff=1e6, T=1e-4)
    (x_test, y_test) = format_data(TIMESTEPS, data_obj_test)

    test_model(model, x_test, y_test, data_obj_test, history)

