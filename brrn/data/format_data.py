import numpy as np

def format_data(batch_size, data_obj):
    signal_with_noise = data_obj.get('signal_with_noise')
    signal_filtered = data_obj.get('signal_filtered')

    x_train = []
    y_train = []

    for i in range(len(signal_with_noise)):
        timestep = []
        for t in range(batch_size):
            timestep.append( signal_with_noise[i-t] if i - t >= 0 else 0 )

        x_train.append(timestep)

    x_train = np.array(x_train)

    for i in range(len(signal_filtered)):
        timestep = signal_filtered[i]
        y_train.append(timestep)

    y_train = np.array(y_train)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    # print(x_train.shape)
    # print(y_train.shape)
    # exit()
    return (x_train, y_train)