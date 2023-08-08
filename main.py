import matplotlib.pyplot as plt
from formats import Sinusoid, Recording
import numpy as np
from scipy.fft import rfft, rfftfreq
sine = Sinusoid(5, 22050, 1)
from scipy.io.wavfile import read as scipy_read
starwars = Recording('audio samples/Piano.wav', True)

sine.mix(Sinusoid(5, 22050, 5000), Sinusoid(8, 22050, 22050/2 - 1))
starwars.dynamic_ft(window_size=5512)