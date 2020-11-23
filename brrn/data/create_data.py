import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io.wavfile import read
import random


def lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b = signal.firwin(80, normal_cutoff, window=('kaiser', 8))
    y = signal.lfilter(b, 1, data)
    return y

    # w, h = signal.freqz(b)
    # x = w * fs * 1.0 / (2 * np.pi)
    # plt.plot(x, 20 * np.log10(abs(h)))
    # plt.grid(True)
    # plt.xlabel('Frequency (MHz)')
    # plt.ylabel('Magnitude (dB)')
    # plt.title('Magnitude Response of DDC Filter')
    # plt.show()
    # exit()
    


def create_data(input_length=32, fs=48000, amplitude=2, offset=0.25, T=0.017, f0=0, f1=2000, fnoise_min=5000, fnoise_max=10000, cutoff=2500):
    print('\n\nCreating data...\n\n')

    ## OOk Data Generation
    # x = np.array([])
    # y = np.array([])
    # mixer = np.array([])
    # # bitstream = np.random.randint(2, size=input_length)
    # bitstream = np.zeros(input_length) + 1

    # for i in range(input_length):
    #     x_i = np.arange(T * fs) / fs + (i * T)

    #     if bitstream[i] == 1:
    #         y_i = np.sin(2 * np.pi * f1 * x_i)/amplitude + offset
    #     else:
    #         y_i = np.sin(2 * np.pi * f0 * x_i)/amplitude + offset

    #     mixer_i = np.sin(2 * np.pi * f1 * x_i)/amplitude + offset


    #     x = np.append(x, x_i)
    #     y = np.append(y, y_i)
    #     mixer = np.append(mixer, mixer_i)

    # noise_gaussian = np.random.normal(0, 0.01, len(x))
    # signal_with_noise = y + noise_gaussian

    # signal_mixer = signal_with_noise * mixer
    # signal_filtered = lowpass_filter(signal_mixer, cutoff, fs, 2)

    # return {
    #     'signal': y, 
    #     'signal_with_noise': signal_mixer, 
    #     'signal_filtered': signal_filtered
    # }

    ## Audio Data Generation
    samplerate, data = read("test.wav")
    signal = np.array(data,dtype=float)
    signal_normalized = (signal - np.min(signal))/np.ptp(signal)
    signal_normalized_left_channel = np.concatenate(signal_normalized, axis=0)

    x = np.arange(len(signal_normalized_left_channel))/fs
    carrier_wave = np.sin(2 * np.pi * f1 * x)/2 + 0.5

    modulated_signal = carrier_wave * signal_normalized_left_channel

    noise_gaussian = np.random.normal(0, 0.01, len(modulated_signal))
    signal_with_noise = modulated_signal + noise_gaussian

    signal_mixer = signal_with_noise * carrier_wave

    signal_filtered = lowpass_filter(signal_mixer, cutoff, fs, 2)

    # plt.plot(signal_mixer)
    # plt.figure()
    # plt.plot(signal_filtered)
    # plt.show()
    # exit()

    return {
        'signal': signal_normalized_left_channel, 
        'signal_with_noise': signal_mixer, 
        'signal_filtered': signal_filtered
    }