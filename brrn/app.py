from brrn.data.create_data import create_data
from brrn.data.format_data import format_data
from brrn.model.brrn import load_model, create_model, train_model, test_model
import numpy as np

# Constants
INPUT_LENGTH = 16
INPUT_SAMPLE_RATE = 96e3
LSTM_NODES = 32
EPOCHS = 400                

def run():
    model = load_model()
    history = None

    if not model:
        data_obj_train = create_data(input_length=INPUT_LENGTH, fs=INPUT_SAMPLE_RATE)
        (x_train, y_train) = format_data(LSTM_NODES, data_obj_train)

        model = create_model(x_train.shape, LSTM_NODES)

        (model, history) = train_model(model, x_train, y_train, EPOCHS)

    data_obj_test = create_data(input_length=INPUT_LENGTH, fs=INPUT_SAMPLE_RATE)
    (x_test, y_test) = format_data(LSTM_NODES, data_obj_test)

    test_model(model, x_test, y_test, data_obj_test, history)

