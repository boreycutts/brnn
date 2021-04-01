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

import progressbar

model = tf.keras.models.load_model("models/CNN64128")
weights = model.layers[0].get_weights()

model2 = tf.keras.models.Sequential()
model2.add(Conv1D(  filters=64, 
                    kernel_size=2, 
                    activation="relu", 
                    weights=weights))

data_obj_test = create_data("OOK", "LPF")
(x_test, y_test) = format_data(data_obj_test, 64)
timetaken = time.time()
output = model2.predict(x_test)
timetaken = time.time() - timetaken
print(str(timetaken/len(x_test)*1e6) + " μs")

# np.save("data", output)


# model = tf.keras.models.load_model("models/CNNK4F8")
# weights = model.get_weights()

# for i in range(len(weights)):
#     weights[i] = weights[i].astype("half")

# model2 = model
# model2.set_weights(weights)

# data_obj_test = create_data("OOK", "LPF")
# (x_test, y_test) = format_data(data_obj_test, 64)

# test_model(model, x_test, y_test, data_obj_test, None, 128, "test", 'Adam', "test", False)
# test_model(model2, x_test, y_test, data_obj_test, None, 128, "test1", 'Adam', "test", False)