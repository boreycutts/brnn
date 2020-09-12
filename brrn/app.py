from brrn.data.create_data import create_data
from brrn.data.format_data import format_data
from brrn.model.brrn import create_model, train_model
import numpy as np

# Constants
BATCH_SIZE = 10
EPOCHS = 5
INPUT_LENGTH = 32

def run():
    data_obj = create_data(INPUT_LENGTH)
    (x_train, y_train) = format_data(BATCH_SIZE, data_obj)
    model = create_model(x_train.shape[1:])
    train_model(model, x_train, y_train, EPOCHS, data_obj)

