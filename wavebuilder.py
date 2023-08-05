import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq
from math import ceil
#TODO: have wave and fourier in the same plot, update every 1/2 second or so, have a line that indicates where the fourier is analyzing
#TODO: combine both wav and sinusoid into new class Composite?

#parent class: WaveBuilder - you can choose to generate waves/import from there in subclasses
class wavebuilder:
  #combines wave objects
  def mix(self, *waves):
    for i in waves:
      if not issubclass(i.__class__, wavebuilder):
        raise ValueError("must inherit wavebuilder")
      if i.__class__ != self.__class__:
        raise NotImplementedError("Sinusoid and Recording addition not yet implemented")
      
    waves = list(waves)
    waves.insert(0, self)
    
    maxtime_index = 0
    for current_index, i in enumerate(waves):
      if i.sample_rate != self.sample_rate:
        raise ValueError('Sample rates don\'t match')
      if not issubclass(i.__class__, wavebuilder):
        raise ValueError("waves must inherit wavebuilder")
      
      if i.time > waves[maxtime_index].time:
        maxtime_index = current_index
    
    self.value[0] = waves[maxtime_index].value[0]
    
    for current_index, i in enumerate(waves):
      zeros_to_pad = int(len(waves[maxtime_index].value[1]) - len(i.value[1]))
      waves[current_index].value[1] = np.append(i.value[1], [0 for _ in range(zeros_to_pad)])
      
    self.value[1] = np.asarray([i.value[1] for i in waves]).sum(axis=0)
    self.time = waves[maxtime_index].time
    if self.frequency is not None: #Sinusoid
      self.frequency = [i.frequency for i in waves]
    elif self.path is not None: #Recording
      self.path = [i.path for i in waves]
  
  #plots waves
  def plot(self, start=0, stop=-1, format="sample"):
    if stop == -1 and format == "sample":
      stop = len(self)
    elif stop == -1 and format == "second":
      stop = self.time
      
    if format == "second":
      plt.plot(self.value[0][start * self.sample_rate:stop * self.sample_rate], 
                self.value[1][start * self.sample_rate:stop * self.sample_rate])
    else:
      plt.plot(self.value[0][start:stop], self.value[1][start:stop])
  
    plt.show()

  #plots fourier transform
  def plot_ft(self, start=0, stop=-1, format="sample"):
    data = self.__fourier_transform(start, stop, format)
    print(len(data[0]))
    plt.plot(data[0], data[1])
    plt.show()
  
  #calculates right fourier transform
  def __fourier_transform(self, start=0, stop=-1, format="sample"): #sample count is preferably 2^x (1024, 2048, etc.)
    if stop == -1 and format == "sample":
      stop = len(self)
    elif stop == -1 and format == "second":
      stop = self.time
    if format == "second":
      yf = rfft(self.value[1][start * self.sample_rate:stop * self.sample_rate])
      xf = rfftfreq(stop * self.sample_rate - start * self.sample_rate, 1 / self.sample_rate)
      return [xf, np.abs(yf)]
    else:
      yf = rfft(self.value[1][start:stop])
      xf = rfftfreq(stop - start, 1 / self.sample_rate)
      return [xf, np.abs(yf)]
  
  def dynamic_ft(self):
    pass