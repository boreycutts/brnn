from model import load_model
from data import create_data, format_data
import matplotlib.pyplot as plt
import numpy as np

data_obj_test = create_data("OOK", "LPF")
(x_test, y_test) = format_data(data_obj_test, 32)

fig, axs = plt.subplots(8, 1)

model = load_model("CNN3232")
output = model.predict(x_test, batch_size=32)

axs[0].plot(output, label="CNN3232")

plt.show()
