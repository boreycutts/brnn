from brrn.data.create_data import create_data
from brrn.data.format_data import format_data
from brrn.model.brrn import load_model, create_model, train_model, test_model
import numpy as np

# Constants
INPUT_LENGTH = 8
INPUT_SAMPLE_RATE = 10e6
LSTM_NODES = 32
EPOCHS = 3000                

def run():
    model = load_model()
    history = None

    if not model:
        data_obj_train = create_data(input_length=INPUT_LENGTH, fs=INPUT_SAMPLE_RATE, fnoise_min=2e6, fnoise_max=5e6, f1=500e3, cutoff=1e6, T=1e-4)
        (x_train, y_train) = format_data(LSTM_NODES, data_obj_train)

        model = create_model(x_train.shape, LSTM_NODES)

        (model, history) = train_model(model, x_train, y_train, EPOCHS)

    data_obj_test = create_data(input_length=INPUT_LENGTH, fs=INPUT_SAMPLE_RATE, fnoise_min=2e6, fnoise_max=5e6, f1=500e3, cutoff=1e6, T=1e-4)
    (x_test, y_test) = format_data(LSTM_NODES, data_obj_test)

    test_model(model, x_test, y_test, data_obj_test, history)

