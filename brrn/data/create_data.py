import random
import numpy as np

def create_anomaly(max_val, min_val):
    i = random.randint(0, 1)
    return [max_val, min_val][i]

def create_data(sensor_count, timesteps, sensor_max_val, sensor_min_val, variation):
    x_T = []
    x_t = []

    for t in range(timesteps):
        val = x_t[t-1] + (random.uniform(variation[0], variation[1]) * random.randint(-1, 1)) if t > 0 else 0

        if val > sensor_max_val:
            val = sensor_max_val
        elif val < sensor_min_val:
            val = sensor_min_val

        x_t.append(val)

    for t in range(timesteps):
        x_T.append([])
        for n in range(sensor_count):
            if random.randint(0, 100) == 42:
                x_T[t].append([create_anomaly(sensor_max_val, sensor_min_val)])
            else:
                x_T[t].append([x_t[t]])
    
    return (np.array(x_T), np.array(x_t)) 