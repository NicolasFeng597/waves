from scipy.io.wavfile import read as scipy_read
import numpy as np
from wavebuilder import wavebuilder

class Sinusoid(wavebuilder): #periodic
    def __init__(self, time, sample_rate, frequency, normalize=True):
      self.time = time
      self.sample_rate = sample_rate
      self.frequency = frequency
      self.normalize = normalize
      
      #building sinusoid, result is put into (time, position) 2darray called value
      times = np.linspace(0, time, sample_rate * time, endpoint=False)
      self.value = [times, np.sin(times * frequency * 2 * np.pi)] #converts frequency values into ascending rads before taking the sine
      
      if normalize: #not normalized to int16 due to FFT
        self.value[1] = self.value[1] / max(self.value[1])
        
      super().__init__(time, sample_rate, normalize, self.value, frequency)
      
    def __str__(self):
      return 'Time: ' + str(self.time) + ', Sample Rate: ' + str(self.sample_rate) + ', Frequency: ' + str(self.frequency) + ', Normalized: ' + str(self.normalize)

class Recording(wavebuilder): #non-periodic
  def __init__(self, path, normalize):
    self.path = path
    data = scipy_read(path)
    self.sample_rate = data[0]
    self.time = len(data[1]) / data[0]
    self.normalize = normalize
    self.value = [np.linspace(0, self.time, len(data[1]), endpoint=False), np.array(data[1])]
    
    if normalize:
        self.value[1] = self.value[1] / max(self.value[1])
        
    super().__init__(time=self.time, sample_rate=self.sample_rate, frequency=None, path=path, normalize=normalize, value=self.value)
    
  def __str__(self):
      return 'Time: ' + str(self.time) + ', Sample Rate: ' + str(self.sample_rate) + ', Path: ' + str(self.path) + ', Normalized: ' + str(self.normalize)