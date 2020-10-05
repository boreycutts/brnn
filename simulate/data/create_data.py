from simulate.data.random_word import get_random_words
from scipy.signal import butter,filtfilt
import numpy as np
import matplotlib.pyplot as plt
from random import randint

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def generate_bitstreams():
    words = get_random_words(10)
    bitstreams = []
    for word in words:
        bits = []
        for char in word:
            bits_string = list(''.join(format(ord(char), 'b')))
            bits += ([int(i) for i in bits_string])

        bitstreams.append({
            'word': word,
            'bitstream': bits
        })

    return bitstreams

def generate_signals(bitstreams, fs=10e6, amplitude=4, offset=0.25, T=1e-4, f0=0, f1=500e3, cutoff=1e5):
    signals = []

    enable_malware = int(len(bitstreams)/2)

    for n in range(len(bitstreams)):
        x = np.array([])
        y = np.array([])
        mixer = np.array([])

        word = bitstreams[n].get('word')
        bitstream = bitstreams[n].get('bitstream')

        for i in range(len(bitstream)):
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

        if n > enable_malware:
            signal_filtered = np.zeros(len(signal_mixer))
        else:
            signal_filtered = butter_lowpass_filter(signal_mixer, cutoff, fs, 2)

        # if n == 0:
        #     print(word)
        #     print(bitstream)
        #     plt.plot(signal_mixer)
        #     plt.figure()
        #     plt.plot(signal_filtered)
        #     plt.show()

        signals.append({
            'word': word,
            'bitstream': bitstream,
            'input': signal_mixer, 
            'expected_output': signal_filtered
        })

    return signals