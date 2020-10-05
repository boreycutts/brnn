import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

def decode_signal(signal):
    bits = []
    for i in range(500, len(signal), 1000): # HARDCODED FOR TESTING
        if signal[i] > 0.08:
            bits.append(1)
        else:
            bits.append(0)
    
    bitstream = ''.join([str(bit) for bit in bits])
    ascii_list = [bitstream[i:i+7] for i in range(0, len(bitstream), 7)]
    decoded_string = ''
    for byte in ascii_list:
        integer = int('0b' + byte, 2)
        decoded_string += integer.to_bytes((integer.bit_length() + 7) // 8, 'big').decode()

    return decoded_string

def format_data(batch_size, data_obj):
    signal_with_noise = data_obj['input']
    signal_filtered = data_obj['expected_output']

    x_train = []
    y_train = []

    for i in range(len(signal_with_noise)):
        timestep = []
        for t in range(batch_size):
            timestep.append([ signal_with_noise[i-t] if i - t >= 0 else 0 ])

        x_train.append(timestep)

    x_train = np.array(x_train)

    for i in range(len(signal_filtered)):
        timestep = [[ signal_filtered[i] ]]
        y_train.append(timestep)

    y_train = np.array(y_train)
    
    return (x_train, y_train)

def simulate_algorithm(model, signals):
    malware_detected = False
    for obj in signals:
        word = obj['word']
        signal = obj['input']
        expected_output = obj['expected_output']

        print('\nTransmitting: ' + word)   

        if not malware_detected:
            (x_test, y_test) = format_data(32, obj)
            loss = model.evaluate(x_test, y_test)
            if loss > 1e-5:
                print('\nMalware detected! Using network output...\n')
                malware_detected = True
        
        if malware_detected:
            (x_test, y_test) = format_data(32, obj)
            expected_output = model.predict(x_test)

        decoded_string = decode_signal(expected_output)
        print('Received: ' + decoded_string + '\n')


