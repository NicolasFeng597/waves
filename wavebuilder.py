import matplotlib.pyplot as plt
from scipy.io.wavfile import read as scipy_read
import numpy as np
from scipy.fft import rfft, rfftfreq
from math import ceil
#TODO: #have wave and fourier in the same plot, update every 1/2 second or so, have a line that indicates where the fourier is analyzing

#parent class: WaveBuilder - you can choose to generate waves/import from there in subclasses
class wavebuilder:
  def __init__(self, time, sample_rate, normalize, value, frequency=None, path=None):
    self.time = time
    self.sample_rate = sample_rate
    self.frequency = frequency
    self.path = path
    self.normalize = normalize
    self.value = value
    
  def fourier_transform(self, start=0, stop=-1): #sample count is preferably 2^x (1024, 2048, etc.)
    if stop == -1:
      stop = int(self.time * self.sample_rate)
      print(stop)
    yf = rfft(self.value[1][start:stop])
    xf = rfftfreq(stop - start, 1 / self.sample_rate)
    return [xf, np.abs(yf)]

  def mix(self, *waves): #waves objects
    for i in waves:
      if i.sample_rate != self.sample_rate:
        raise AttributeError('Sample rates don\'t match')
    self.value = np.asarray([i.value[1] for i in waves]).sum(axis=0)

  #plots waves
  def plot(wave, start=0, stop=-1, step=1): #waves objects
    for i in wave:
      if stop == -1:
        stop = i.length
      plt.plot(np.linspace(0, i.t, i.length, endpoint=False)[start:stop:step], i.data[start:stop:step])

  @staticmethod
  def plot_ft(*w, start=0, stop=-1): #width of segment to perform ft
    for i in w:
      if stop == -1:
        stop = i.length
      j = i.ft(start=start, stop=stop)
      plt.plot(j[0], j[1])
    
  def dynamic_ft(w, width=-1): #ft that updates depending on the duration and width of the wave
    if width == -1:
      width = w.length

    ft_plot = plt.figure()
    subplot = ft_plot.add_subplot(0)
    
    data = []
    for i in range(ceil(w.length / width)):
      data.append()
    subplot.plot()