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

# while True:
#     num_str = input("Float: ")
#     num = np.float16(num_str)
#     print(bin(num.view('H'))[2:].zfill(16))

## Get binary and float inputs
# data_obj_test = create_data("OOK", "LPF")
# (x_test, y_test) = format_data(data_obj_test, 64)
# x_test = x_test.reshape(16000, 64).astype(np.float16)

# x_test_bin = []
# for input_ in x_test:
#     bin_inputs = []
#     for num in input_:
#         bin_inputs.append(bin(num.view('H'))[2:].zfill(16))
#     x_test_bin.append(bin_inputs)
# x_test_bin = np.array(x_test_bin)

# np.savetxt("x_test.csv", x_test, delimiter=",")
# np.savetxt("x_test_bin.csv", x_test_bin, delimiter=",", fmt="%s")


## Get weights and benchmark output
model = tf.keras.models.load_model("models/CNN321284Filters")
filters, biases = model.layers[0].get_weights()

weights = model.layers[0].get_weights()
# print(weights[0])
# np.savetxt("weights.csv", weights[0], delimiter=",")

model2 = tf.keras.models.Sequential()
model2.add(Conv1D(  filters=4, 
                    kernel_size=2, 
                    activation="relu", 
                    weights=weights))

data_obj_test = create_data("OOK", "LPF")
(x_test, y_test) = format_data(data_obj_test, 64)

# print(x_test)
# np.savetxt("x_test.csv", x_test, delimiter=",")

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