import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import numpy as np
from scipy.fft import rfft, rfftfreq
from math import ceil

class waves:
  def __init__(self, t=None, s=None, f=None, path=None, norm = False):
    if ((t is not None or s is not None or f is not None) and path is not None):
      raise ValueError("Did not declare only tsf or path")
    if ((t is None or s is None or f is None) and (t is not None or s is not None or f is not None)):
      raise ValueError("Did not declare all values of tsf")
    if ((t is None and s is None and f is None) and (path is None)):
      raise ValueError("Did not declare either tsf or path")
    
    self.f = f #frequency
    self.path = path #path of file
    
    if path is None:
      self.s = s #sample rate
      self.t = t #time
      self.data = self.gen_wave()[1]
    else:
      x = read(path)
      self.data = x[1]
      self.s = x[0]
      self.t = len(x[1]) / x[0]
    
    self.length = len(self.data)

    self.norm = norm
    if norm:
      self.data = np.int16(self.data / self.data.max() * 32767) #normalizes wave
      
  def __str__(self):
    return "Time: " + str(self.t) + ", Sample Rate: " + str(self.s) + ", Frequency: " + str(self.f) + ", Path: " + str(self.path) + ", Normalized: " + str(self.norm)
    
  def gen_wave(self): #generates a sine wave
    if self.f is None:
      return [None, None]
    x = np.linspace(0, self.t, self.s * self.t, endpoint=False) #nparray of size s * t, each index with a value i/(t*s)
    freq = x * self.f #multiplies all in x by f
    return [x, np.sin(2 * np.pi * freq)]
  
  #fourier transform
  def ft(self, start=0, stop=-1): #stop-start is preferably 2^x (1024, 2048, etc.)
    if stop == -1:
      stop = self.length
    yf = rfft(self.data[start:stop])
    xf = rfftfreq(stop - start, 1 / self.s)
    return [xf, np.abs(yf)]

  #adds waves
  @staticmethod
  def mix(*w): #waves objects
    for i in w:
      if i.s != w[0].s:
        raise AttributeError("Wave has a sample rate of " + str(i.s) + ", while the first wave has a sample rate of " + str(w[0].s))
    return np.asarray([i.data for i in w]).sum(axis=0)

  #plots waves
  @staticmethod
  def plot_wave(*w, start=0, stop=-1, step=1): #waves objects
    #have wave and fourier in the same plot, update every 1/2 second or so, have a line that indicates where the fourier is analyzing
    for i in w:
      if stop == -1:
        stop = i.length
      plt.plot(np.linspace(0, i.t, i.length, endpoint=False)[start:stop:step], i.data[start:stop:step])
    # for i in w:
    #   chunks = w.s * w.t % 1024
    #   if w.s * w.t % 1024 != 0:
    #     chunks += 1

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