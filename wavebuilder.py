#most recent change: git havoc, finished fft and looking into harmonics and note identification

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq
from math import ceil
import wavio

#storing image files, exporting into video
from io import BytesIO
import cv2 as cv
import os
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from moviepy.editor import VideoFileClip, AudioFileClip

#TODO: have wave and fourier in the same plot, update every 1/2 second or so, have a line that indicates where the fourier is analyzing
#TODO: combine both wav and sinusoid into new class Composite?
#TODO: note identification within fourier

#parent class: WaveBuilder - you can choose to generate waves/import from there in subclasses
class wavebuilder:
  #combines wave objects
  def mix(self, *waves):
    for i in waves:
      if not issubclass(i.__class__, wavebuilder):
        raise ValueError('must inherit wavebuilder')
      if i.__class__ != self.__class__:
        raise NotImplementedError('Sinusoid and Recording addition not yet implemented')
      
    waves = list(waves)
    waves.insert(0, self)
    
    maxtime_index = 0
    for current_index, i in enumerate(waves):
      if i.sample_rate != self.sample_rate:
        raise ValueError('Sample rates don\'t match')
      if not issubclass(i.__class__, wavebuilder):
        raise ValueError('waves must inherit wavebuilder')
      
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
  def plot(self, start=0, stop=-1, format='sample'):
    if stop == -1 and format == 'sample':
      stop = len(self)
    elif stop == -1 and format == 'second':
      stop = self.time
      
    if format == 'second':
      plt.plot(self.value[0][start * self.sample_rate:stop * self.sample_rate], 
                self.value[1][start * self.sample_rate:stop * self.sample_rate])
    else:
      plt.plot(self.value[0][start:stop], self.value[1][start:stop])
  
    plt.show()

  #calculates right fourier transform
  def __fourier_transform(self, start=0, stop=-1, format='sample'): #sample count is preferably 2^x (1024, 2048, etc.)
    if stop == -1 and format == 'sample':
      stop = len(self)
    elif stop == -1 and format == 'second':
      stop = self.time
    if format == 'second':
      yf = rfft(self.value[1][start * self.sample_rate:stop * self.sample_rate])
      xf = rfftfreq(stop * self.sample_rate - start * self.sample_rate, 1 / self.sample_rate)
      return [xf, np.abs(yf)]
    else:
      yf = rfft(self.value[1][start:stop])
      xf = rfftfreq(stop - start, 1 / self.sample_rate)
      return [xf, np.abs(yf)]
    
  #plots fourier transform
  def plot_ft(self, start=0, stop=-1, format='sample'):
    data = self.__fourier_transform(start, stop, format)
    plt.plot(data[0], data[1])
    plt.show()
  
  #plays a video of a sliding-window fourier transform
  def dynamic_ft(self, window_size=2048, format='sample', rewrite=True, preview=False, destroy_temp=True, progress=False):
    if format == 'second':
      window_size *= self.sample_rate
    
    #calculating transforms
    transforms = []
    max_value = -1
    for i in range(ceil(len(self) / window_size)):
      fourier = self.__fourier_transform(start=window_size*i, stop=min((i + 1) * window_size, len(self)))
      print(fourier)
      transforms.append(fourier)
      max_value = max(np.max(transforms[i][1]), max_value)
    
    #normalizing
    for i in range(len(transforms)):
      transforms[i][1] /= max_value

    #adding dir and overriding existing images (if rewrite = True)
    try:
      os.mkdir('temp_images')
    except FileExistsError:
      if rewrite:
        for file in os.listdir('temp_images'):
          os.remove('temp_images/' + file)
    
    #plotting and saving images
    fig, ax = plt.subplots()
    if rewrite:
      for i in range(len(transforms)):
        if progress and i % 50 == 0:
          print(f"WaveBuilder - Writing image {i} out of {len(transforms)}")
        plt.ylim(0, 1)
        plt.plot(transforms[i][0], transforms[i][1], color='blue')
        plt.text(0.5, 1.01, f'time: {round(i * window_size / self.sample_rate, 2)}s', horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
        
        current_image = BytesIO()
        canvas = FigureCanvasAgg(fig)
        canvas.print_png(current_image)
        current_image = Image.open(current_image)
        fig.clear()
        current_image.save(f'temp_images/image{i}.png')
        current_image.close()
      print("WaveBuilder - Finished writing images")
    
    #video preview
    if preview:
      for i in range(len(transforms)):
        abn = cv.imread(f'temp_images/image{i}.png')
        cv.imshow("window", abn)
        cv.waitKey(int((window_size / self.sample_rate) * 1000))
    
    #building video
    video = cv.VideoWriter('temp_images/temp_video.mp4', cv.VideoWriter_fourcc(*'mp4v'), self.sample_rate / window_size,
                            Image.open('temp_images/image0.png').size)
    if progress:
      print("WaveBuilder - Building video")
    for i in range(len(transforms)):
      video.write(cv.imread(f'temp_images/image{i}.png'))
      
    video.release()
    
    video_clip = VideoFileClip('temp_images/temp_video.mp4')
    if self.type == "Sinusoid":
      wavio.write('temp_images/temp_audio.wav', self.value[1], self.sample_rate, sampwidth=2)
      audio_clip = AudioFileClip('temp_images/temp_audio.wav')
    else:
      audio_clip = AudioFileClip(self.path)
      
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile('Final Video.mp4')
    video_clip.close()
    audio_clip.close()
      
    #destroy temp files
    if destroy_temp:
      for i in range(len(transforms)):
        os.remove(f'temp_images/image{i}.png')
      os.remove("temp_images/temp_video.mp4")
