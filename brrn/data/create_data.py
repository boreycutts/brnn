import numpy as np
import matplotlib.pyplot as plot
from scipy.signal import butter,filtfilt
import random

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def create_data(input_length=32, fs=48000, amplitude=4, offset=0.25, T=0.017, f0=0, f1=2000, fnoise_min=5000, fnoise_max=10000, cutoff=2500):
    print('\n\nCreating data...\n\n')
    nyq = 0.5 * fs
    n = int(T * fs)

    x = np.array([])
    y = np.array([])
    bitstream = np.random.randint(2, size=input_length)

    for i in range(input_length):
        x_i = np.arange(T * fs) / fs + (i * T)

        if bitstream[i] == 1:
            y_i = np.sin(2 * np.pi * f1 * x_i)/amplitude + offset
        else:
            y_i = np.sin(2 * np.pi * f0 * x_i)/amplitude + offset

        x = np.append(x, x_i)
        y = np.append(y, y_i)

    fnoise = random.randint(fnoise_min, fnoise_max)
    noise = np.sin(2 * np.pi * fnoise * x)/amplitude + offset
    noise_min = np.sin(2 * np.pi * fnoise_min * x)/amplitude + offset
    noise_max = np.sin(2 * np.pi * fnoise_max * x)/amplitude + offset
    signal_with_noise = y + noise

    signal_filtered = butter_lowpass_filter(signal_with_noise, cutoff, fs, 2)
    # signal_filtered = np.zeros(len(x))
    # signal_filtered = noise_max

    return {
        'bitstream': bitstream, 
        'signal': y, 
        'signal_with_noise': signal_with_noise, 
        'signal_filtered': signal_filtered
    }