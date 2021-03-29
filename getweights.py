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

from data import create_data, format_data, create_freq_data
from model import test_model

model = tf.keras.models.load_model("models/CNNK4F8")
weights = model.get_weights()

for i in range(len(weights)):
    weights[i] = weights[i].astype("half")

model2 = model
model2.set_weights(weights)

data_obj_test = create_data("OOK", "LPF")
(x_test, y_test) = format_data(data_obj_test, 64)

test_model(model, x_test, y_test, data_obj_test, None, 128, "test", 'Adam', "test", False)
test_model(model2, x_test, y_test, data_obj_test, None, 128, "test1", 'Adam', "test", False)