import numpy as np
import matplotlib.pyplot as plt
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

    x = np.array([])
    y = np.array([])
    mixer = np.array([])
    bitstream = np.random.randint(2, size=input_length)
    # bitstream = np.array([0,1,1,1,1,0,0,1])

    for i in range(input_length):
        x_i = np.arange(T * fs) / fs + (i * T)

        if bitstream[i] == 1:
            y_i = np.sin(2 * np.pi * f1 * x_i)/amplitude + offset
        else:
            y_i = np.sin(2 * np.pi * f0 * x_i)/amplitude + offset

        mixer_i = np.sin(2 * np.pi * f1 * x_i)/amplitude + offset


        x = np.append(x, x_i)
        y = np.append(y, y_i)
        mixer = np.append(mixer, mixer_i)

    noise_gaussian = np.random.normal(0, 0.01, len(x))
    signal_with_noise = y + noise_gaussian

    signal_mixer = signal_with_noise * mixer
    signal_filtered = butter_lowpass_filter(signal_mixer, 100000, fs, 2)
    signal_filtered = np.zeros(len(signal_filtered))

    return {
        'bitstream': bitstream, 
        'signal': y, 
        'signal_with_noise': signal_mixer, 
        'signal_filtered': signal_filtered
    }