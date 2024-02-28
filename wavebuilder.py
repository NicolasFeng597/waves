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
import ffmpeg
import subprocess
from scipy.signal import find_peaks

#parent class: WaveBuilder - you can choose to generate waves/import from there in subclasses
class wavebuilder:
  #calculating _NOTES
  _a = np.append(
      np.flip(np.array([round(440/(1.0594631**i), 7) for i in range(1, 58)]), 0),
      [round(440*(1.0594631**i), 7) for i in range(68)]) #list of all note frequencies before 22100hz
    
  _b = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
          ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']]
  _c = []
  for i in range(11):
    for j in range(12):
      _c.append(_b[1][j] + str(_b[0][i]))
  _c = _c[:127] #same number of notes as frequencies
    
  _NOTES = dict(zip(_c, _a))
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
  def _fourier_transform(self, start=0, stop=-1, format='sample'): #sample count is preferably 2^x (1024, 2048, etc.)
    if stop == -1 and format == 'sample':
      stop = len(self)
    if stop == -1 and format == 'second':
      stop = self.time
    if format == 'second':
      start = start * self.sample_rate
      stop = stop * self.sample_rate

    yf = rfft(self.value[1][start:stop])
    xf = rfftfreq(stop - start, 1 / self.sample_rate)

    yf /= stop - start #normalizing
    return [xf, np.abs(yf)]

  #plots fourier transform
  def plot_ft(self, start=0, stop=-1, format='sample'):
    data = self._fourier_transform(start, stop, format)
    plt.plot(data[0], data[1])
    plt.show()
  
  #plays a video of a sliding-window fourier transform
  def dynamic_ft(self, window_size=2048, format='sample', rewrite=True, show_preview=False, destroy_temp=True, show_progress=False, save_sample=None, limit=22050):
    """
    Generates a video and audio file from a given audio sample using a dynamic Fourier transform.

    Parameters:
    - window_size (int): The size of the window to use for the Fourier transform. Defaults to 2048.
    - format (str): The format of the audio sample. Can be 'sample' or 'second'. Defaults to 'sample'.
    - rewrite (bool): Determines whether to overwrite existing images and video files. Defaults to True.
    - show_preview (bool): Determines whether to display a preview of the video. Defaults to False.
    - destroy_temp (bool): Determines whether to delete temporary files after generating the video. Defaults to True.
    - show_progress (bool): Determines whether to display progress information during the generation process. Defaults to False.
    - save_sample (str): The name of the file to permanently store in the 'audio samples data' directory. If None, files are saved in the 'audio samples data/temp_images' directory. Defaults to None.
    - limit (int): The maximum Hz in each Fourier transform's domain. Defaults to 22050 (half of 44100; the limit of human hearing).

    Returns:
    - None

    """
    
    if format == 'second':
      window_size *= self.sample_rate
    
    #converts limit to index of frequency bin with Hz 'limit'
    limit = round(limit * window_size / self.sample_rate)
    
    #calculating transforms
    transforms = []
    max_value = -1
    for i in range(ceil(len(self) / window_size)):
      fourier = self._fourier_transform(start=window_size*i, stop=min((i + 1) * window_size, len(self)))
      transforms.append([i[:limit] for i in fourier])
      max_value = max(np.max(transforms[i][1]), max_value)
    
    #normalizing (commented out since we're already normalizing in _fourier_transform)
    # for i in range(len(transforms)):
    #   transforms[i][1] /= max_value

    #adding dir and overriding existing images (if rewrite = True)
    try:
      if save_sample is None:
        os.mkdir('temp_images')
      else:
        os.mkdir('audio samples data/' + save_sample)
        os.mkdir('audio samples data/' + save_sample + '/images')
    except FileExistsError:
      if rewrite:
        if save_sample is None:
          for file in os.listdir('temp_images'):
            os.remove('temp_images/' + file)
        else:
          for file in os.listdir('audio samples data/' + save_sample + '/images'):
            os.remove('audio samples data/' + save_sample + '/images/' + file)
    
    #plotting and saving images
    fig, ax = plt.subplots()
    if rewrite:
      for i in range(len(transforms)):
        if show_progress and i % 50 == 0:
          print(f"(Waves) Class WaveBuilder, Function dynamic_ft(): Writing image {i} out of {len(transforms)}")
        plt.ylim(0, 1)
        plt.plot(transforms[i][0], transforms[i][1], color='blue')
        plt.text(0.5, 1.01, f'time: {round(i * window_size / self.sample_rate, 2)}s', horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
        
        current_image = BytesIO()
        canvas = FigureCanvasAgg(fig)
        canvas.print_png(current_image)
        current_image = Image.open(current_image)
        fig.clear()
        if save_sample is None:
          current_image.save(f'temp_images/image{i}.png')
        else:
          current_image.save(f'audio samples data/{save_sample}/images/image{i}.png')
        current_image.close()
        
      print("Waves) Class WaveBuilder, Function dynamic_ft(): Finished writing images")
    
    #video preview
    if show_preview:
      for i in range(len(transforms)):
        if save_sample is None:
          abn = cv.imread(f'temp_images/image{i}.png')
        else:
          abn = cv.imread(f'audio samples data/{save_sample}/images/image{i}.png')
        cv.imshow("window", abn)
        cv.waitKey(int((window_size / self.sample_rate) * 1000))
    
    #building video
    
    #creating audio
    if self.type == "Sinusoid":
      wavio.write('temp_images/temp_audio.wav', self.value[1], self.sample_rate, sampwidth=2)
        
    if save_sample is None: #files are saved in the 'audio samples data/temp_images' directory
      #temp images -> images and video
      subprocess.run(['ffmpeg',
                      '-y',
                      f'-r {self.sample_rate / window_size}',
                      '-f image2',
                      f's {Image.open("temp_images/image0.png").size[0]}x{Image.open("temp_images/image0.png").size[1]}',
                      '-i audio samples data/temp_images/image%01d.png',
                      f'-i {"temp_images/temp_audio.wav" if self.type == "Sinusoid" else self.path}',
                      '-c:v libx264',
                      '-pix_fmt: yuv420p',
                      'Current Video.mp4'])
    else:
      print("asdf " + str(self.path))
      subprocess.run(['ffmpeg',
                      '-y',
                      '-r', f'{self.sample_rate / window_size}',
                      '-f', 'image2',
                      '-thread_queue_size', '512',
                      '-s', f'{Image.open(f"audio samples data/{save_sample}/images/image0.png").size[0]}x{Image.open(f"audio samples data/{save_sample}/images/image0.png").size[1]}',
                      "-i", f"audio samples data/{save_sample}/images/image%01d.png",
                      '-i', f'{"temp_images/temp_audio.wav" if self.type == "Sinusoid" else self.path}',
                      '-channel_layout', 'mono',
                      '-ac', '1', #only mono audio supported
                      '-c:v', 'libx264',
                      '-pix_fmt', 'yuv420p',
                      '-b:v', '150k',
                      f'audio samples data/{save_sample}/{save_sample}.mp4'])

    #destroy temp files
    if save_sample is None and destroy_temp:
      for i in range(len(transforms)):
        os.remove(f'temp_images/image{i}.png')
      os.remove("temp_images/temp_video.mp4")

  '''java
  \<obsidian pseudo code\>
  #buckets[i] is not real, instead use rfftfreq
  **Threshold/note ID:**
  Map<str (note + octave), float Hz> notes
  Map<str note_name, float magnitude> note_values
  const float threshold #(determine through testing)

  i for all buckets:
    note_values<closest note to ith bucket, magnitude = bucket>
    if (ith magnitude) > threshold: display the note
  (func returns fundamental for use in harmonic series detection, as follows)

  **Harmonic Series detection**:
  str fundamental_note_name, int fundamental_bucket
  Map<str harmonic_name, [int harmonic_bucket, float harmonic_magnitude]> harmonics

  int current_harmonic = 2; #first harmonic is fundamental
  while Hz <= 20000:
    int current_harmonic_Hz = ith_harmonic_Hz_func(current_harmonic)
    harmonics<notes<current_harmonic_Hz>, [bucket[current_harmonic_Hz], wave.value[current_bucket]>
  '''

  def _single_note_id(self, ft):
    """
    Generate a dict of note name and octave + frequency
    """
    
    # only up to 5000 hz (5000 * window size/sample rate) (put window size into params)
    ft[0] = ft[0][:round(5000*1024/self.sample_rate)]
    ft[1] = ft[1][:round(5000*1024/self.sample_rate)]

    #find peaks
    bucket, _ = find_peaks(ft[1]) #the bucket where the peak was found
    bucket_size = self.sample_rate / 1024 #(window size really)
    peaks = dict(zip(bucket * bucket_size, [ft[1][i] for i in bucket])) #dict of peaks <hz, magnitude>
    plt.plot(ft[0], ft[1])
    plt.plot(peaks.keys(), peaks.values(), 'rx')
    plt.show()
    