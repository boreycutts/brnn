import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io.wavfile import read
import random

# Constants
T = 1e-4
input_length = 16
fs = 10e6
f0 = 0
f1 = 500e3
cutoff = 100e3

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b = signal.firwin(80, normal_cutoff, window=('kaiser', 8))
    y = signal.lfilter(b, 1, data)

    # w, h = signal.freqz(b)
    # x = w * fs * 1.0 / (2 * np.pi)
    # plt.plot(x, 20 * np.log10(abs(h)))
    # plt.grid(True)
    # plt.xlabel('Frequency (MHz)')
    # plt.ylabel('Magnitude (dB)')
    # plt.title('Magnitude Response of DDC Filter')
    # plt.show()
    # exit()

    return y

def create_data(dataset, component):
    print('\n\nCreating data...\n\n')
    
    if dataset == "OOK":
        x = np.array([])
        y = np.array([])
        mixer = np.array([])
        bitstream = np.random.randint(2, size=input_length)

        print("OOK Bitstream = " + str(bitstream))

        for i in range(input_length):
            x_i = np.arange(T * fs) / fs + (i * T)
            if bitstream[i] == 1:
                y_i = np.sin(2 * np.pi * f1 * x_i)
            else:
                y_i = np.sin(2 * np.pi * f0 * x_i)

            mixer_i = np.sin(2 * np.pi * f1 * x_i)

            x = np.append(x, x_i)
            y = np.append(y, y_i)
            mixer = np.append(mixer, mixer_i)

        y = normalize(y)
        mixer = normalize(mixer)

        noise_gaussian = np.random.normal(0, 0.01, len(x))
        signal_with_noise = y + noise_gaussian

        signal_mixer = signal_with_noise * mixer
        signal_filtered = lowpass_filter(signal_mixer, cutoff, fs, 2)

        component_input = None
        component_output = None

        if component == "LPF":
            component_input = signal_mixer
            component_output = signal_filtered
        elif component == "Mixer":
            component_input = signal_with_noise
            component_output = signal_mixer
        elif component == "DDC":
            component_input = signal_with_noise
            component_output = signal_filtered
        else:
            print("Component not supported")
            exit()

        return {
            'signal': y, 
            'component_input': component_input, 
            'component_output': component_output
        }
    elif dataset == "Audio":
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

        if component == "LPF":
            component_input = signal_mixer
            component_output = signal_filtered
        elif component == "Mixer":
            component_input = signal_with_noise
            component_output = signal_mixer
        elif component == "DDC":
            component_input = signal_with_noise
            component_output = signal_filtered
        else:
            print("Component not supported")
            exit()

        return {
            'signal': signal_normalized_left_channel, 
            'component_input': component_input, 
            'component_output': component_output
        }
    else:
        print("Dataset not supported")
        exit()

def format_data(data_obj, timesteps):
    component_input = data_obj.get('component_input')
    component_output = data_obj.get('component_output')

    x_train = []
    y_train = []

    for i in range(len(component_input)):
        timestep = []
        for t in range(timesteps):
            timestep.append( component_input[i-t] if i - t >= 0 else 0 )

        x_train.append(timestep)

    x_train = np.array(x_train)

    for i in range(len(component_output)):
        timestep = component_output[i]
        y_train.append(timestep)

    y_train = np.array(y_train)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    return (x_train, y_train)

def create_freq_data(frequency):
    x = np.array([])
    y = np.array([])
    # mixer = np.array([])

    for i in range(input_length):
        x_i = np.arange(T * fs) / fs + (i * T)
        y_i = np.sin(2 * np.pi * frequency * x_i)
        # mixer_i = np.sin(2 * np.pi * f1 * x_i)

        x = np.append(x, x_i)
        y = np.append(y, y_i)
        # mixer = np.append(mixer, mixer_i)

    y = normalize(y)
    # mixer = normalize(mixer)

    # signal_mixer = y * mixer
    # signal_filtered = lowpass_filter(signal_mixer, cutoff, fs, 2)

    signal_filtered = lowpass_filter(y, cutoff, fs, 2)

    return {
        'signal': y, 
        'component_input': y, 
        'component_output': signal_filtered
    }