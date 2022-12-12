import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq
import matplotlib.animation as animation

# def generate_sine_wave(freq, sample_rate, duration):
#     x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
#     frequencies = x * freq
#     # 2pi because np.sin takes radians
#     y = np.sin((2 * np.pi) * frequencies)
#     return x, y

# x, y = generate_sine_wave(1000, 44100, 5)
# plt.plot(x, y)
# plt.show()

# # Number of samples in normalized_tone

# yf = fft(y)
# xf = fftfreq(44100*5, 1 / 44100)
# plt.plot([xf, np.abs(yf)])
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot()

# def animate(j):
#     a = []
#     b = []
#     for i in range(10):
#         a.append(i)
#         b.append(i)
#     ax.clear()
#     ax.plot(a, b)

# ani = animation.FuncAnimation(fig, animate, interval=10000)
# plt.show()

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(j):
    data = [[i for i in range(10)], [i for i in range(10)]]
    xar = []
    yar = []
    for i in range(10):
        # xar.append(data[])
        pass
    ax1.clear()
    ax1.plot(xar,yar)
ani = animation.FuncAnimation(fig, animate, interkal=1000)
plt.show()