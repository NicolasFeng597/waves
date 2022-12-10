import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq

def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians
    y = np.sin((2 * np.pi) * frequencies)
    return x, y

x, y = generate_sine_wave(1000, 44100, 5)
# plt.plot(x, y)
# plt.show()

# Number of samples in normalized_tone

yf = fft(y)
xf = fftfreq(44100*5, 1 / 44100)
plt.plot([xf, np.abs(yf)])
plt.show()