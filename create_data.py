import matplotlib.pyplot as plt
import random

def create_anomaly():
    i = random.randint(0, 1)
    return [y_t_max, y_t_min][i]

N = 17
T = 1000
y_t_max = 10
y_t_min = -10
y_N = []
y_t = []
variation = (0.01, 0.1)

for t in range(T):
    val = y_t[t-1] + (random.uniform(variation[0], variation[1]) * random.randint(-1, 1)) if t > 0 else 0

    if val > y_t_max:
        val = y_t_max
    elif val < y_t_min:
        val = y_t_min

    y_t.append(val)

for n in range(N):
    y_N.append(y_t.copy())
    for t in range(T):
        if random.randint(0, 100) == 42:
            y_N[n][t] = create_anomaly()

for n in range(N):
    plt.figure(n)
    plt.plot(y_N[n])

plt.show()