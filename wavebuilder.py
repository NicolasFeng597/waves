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
from scipy.signal import peak_prominences
from math import log2

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
    # V this bottom fig/ax is for the regular graph
    # fig, ax = plt.subplots()
    if rewrite:
      for i in range(len(transforms)):
        if show_progress and i % 50 == 0:
          print(f"(Waves) Class WaveBuilder, Function dynamic_ft(): Writing image {i} out of {len(transforms)}")
        # this is the regular graph, currently testing the note id one
        # plt.ylim(0, 1)
        # plt.plot(transforms[i][0], transforms[i][1], color='blue')

        fig, ax = self._single_note_id([transforms[i][0], transforms[i][1]])
        plt.text(0.5, 1.01, f'time: {round(i * window_size / self.sample_rate, 2)}s', horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)

        current_image = BytesIO()
        canvas = FigureCanvasAgg(fig)
        canvas.print_png(current_image)
        current_image = Image.open(current_image)
        # fig.clear()
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

  def _single_note_id(self, ft, format='sample', window_size=2048, limit=22050):
    if format == 'second':
      window_size *= self.sample_rate
    
    #converts limit to index of frequency bin with Hz 'limit'
    limit = round(limit * window_size / self.sample_rate)

    # only up to 5000 hz (5000 * window size/sample rate) (put window size into params)
    ft[0] = ft[0][:limit]
    ft[1] = ft[1][:limit]
    
    #find peaks
    bucket, _ = find_peaks(ft[1], prominence=0.001) #the bucket where the peak was found

    peak_hz = bucket * self.sample_rate / window_size
    peak_magnitudes = [ft[1][i] for i in bucket]

    #naively find hz of fundamental
    # fundamental = peak_hz[peak_magnitudes.index(max(peak_magnitudes))] #note: this doesn't work if there's no peaks
    # print('all peak hz: ' + str(peak_hz))
    # print('fundamental hz: ' + str(fundamental))

    #find peak prominence
    # prominence, left_bases, right_bases = peak_prominences(x=ft[1], peaks=bucket)
    # print('prominences: ' + str(prominence))
    # print('bases: ' + str(left_bases) + ' ' + str(right_bases))
    # print('magnitudes: ' + str(peak_magnitudes))

    #plot
    fig, ax = plt.subplots()
    fig.set_figheight(4.8)
    fig.set_figwidth(14)
    plt.ylim(0, 1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Absolute Amplitude')
    plt.plot(ft[0], ft[1])
    plt.plot(peak_hz, peak_magnitudes, 'rx')
    for i in range(len(peak_hz)):
      plt.text(peak_hz[i], peak_magnitudes[i] + 0.01, self._hz_to_note(peak_hz[i])[0] + '\n' + str(round(self._hz_to_note(peak_hz[i])[1], 1)) + '¢', ha='center') #note names and cents
      plt.axvline(peak_hz[i], 0, peak_magnitudes[i] / plt.ylim()[1], color='limegreen', linestyle='dotted') #lines
      plt.text(peak_hz[i], 0, f'{peak_hz[i]:.2f}', color='limegreen', ha='center', va='top', transform=plt.gca().transAxes) #x-axis label
    return fig, ax

  def _hz_to_note(self, hz):
    """
    Returns the closest note and the cents difference for a given frequency in hertz.

    Parameters:
    hz (float): The frequency in hertz.

    Returns:
    tuple: A tuple containing the closest note and the cents difference.
    """

    #self._FREQUENCIES is between 16.3515929 hz and 21096.1711678 hz and has 125 values
    if hz < 16.3515929:
      raise ValueError(f'Hz given ({hz}) is lower than minimum note C0 (Hz = 16.3515929)')
    if hz > 21096.1711678:
      raise ValueError(f'Hz given ({hz}) is higher than maximum note E10 (Hz = 21096.1711678)')
    if round(hz, 7) == 21096.1711678: #edge case
      return ('E10', 0)

    for i in range(125):
      if hz >= self._FREQUENCIES[i] and hz <= self._FREQUENCIES[i + 1]:
        if abs(hz - self._FREQUENCIES[i]) > abs(hz - self._FREQUENCIES[i + 1]):
          return (self._NOTE_NAMES[i + 1], 1200 * log2(hz / self._FREQUENCIES[i + 1]))
        else:
          return (self._NOTE_NAMES[i], 1200 * log2(hz / self._FREQUENCIES[i]))