from brrn.data.create_data import create_data
from brrn.model.brrn import create_model

def run():
    (x_T, y_t) = create_data(1, 100, 100, -100, (0.01, 1))
    create_model(x_T, y_t)

