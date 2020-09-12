import numpy as np
import matplotlib.pyplot as plot
from scipy.signal import butter,filtfilt
import random

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

scale = 4                               # Amplitude scale
offset = 0.25                           # Signal offset
T = 0.017                               # Period
fs = 48000                              # Sample frequency
f0 = 0                                  # FSK frequency 0
f1 = 2000                               # FSK frequency 1
fnoise = random.randint(5000, 10000)    # Noise frequency
bits = 32                               # Message length
cutoff = 2500                           # Filter cutoff frequency
nyq = 0.5 * fs                          # Nyquist frequency
n = int(T * fs)                         # Total number of samples

def create_data(input_length):
    x = np.array([])
    y = np.array([])
    bitstream = np.random.randint(2, size=input_length)

    for i in range(input_length):
        x_i = np.arange(T * fs) / fs + (i * T)

        if bitstream[i] == 1:
            y_i = np.sin(2 * np.pi * f1 * x_i)/scale + offset
        else:
            y_i = np.sin(2 * np.pi * f0 * x_i)/scale + offset

        x = np.append(x, x_i)
        y = np.append(y, y_i)

    noise = np.sin(2 * np.pi * fnoise * x)/scale + offset
    signal_with_noise = y + noise

    signal_filtered = butter_lowpass_filter(signal_with_noise, cutoff, fs, 2)

    return {
        'bitstream': bitstream, 
        'signal': y, 
        'signal_with_noise': signal_with_noise, 
        'signal_filtered': signal_filtered
    }