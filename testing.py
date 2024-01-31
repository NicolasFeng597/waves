#practical application of audio analysis (using gpt)
#get mic audio, perform transforms, then analyze notes

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import spectrogram

def plot_spectrogram(stream, rate, chunk_size=1024):
    fig, ax = plt.subplots()
    f, t, Sxx = spectrogram(np.frombuffer(stream.read(chunk_size), dtype=np.int16), rate)
    im = plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='auto')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Real-time Spectrogram from Microphone')
    # plt.colorbar(mappable=im, label='Intensity [dB]')
    def update_plot(frame):
        f, t, Sxx = spectrogram(np.frombuffer(stream.read(chunk_size), dtype=np.int16), rate)
        plt.clf()
        # plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='auto')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('Real-time Spectrogram from Microphone')
        # plt.colorbar(label='Intensity [dB]')
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='auto')
    
    ani = FuncAnimation(fig, update_plot)
    plt.show()


if __name__ == "__main__":
    # Set up the audio stream
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    input=True,
                    frames_per_buffer=1024)

    # Analyze the audio (for fourier transform)
    # plot_audio(stream, 1024, 44100)
    
    #spectrogram
    plot_spectrogram(stream, 44100)

    # Close the stream and terminate PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()
    
#paste before main 
###################################### FOR FOURIER TRANSFORM PLOTTING WITH LIVE MIC ######################################
# #r rfft on given sample audio
# def audio_analysis(audio_data, chunk_size, rate):
#     mag = np.abs(np.fft.rfft(audio_data))
#     mag = mag / np.max(mag)
#     freq = np.fft.rfftfreq(chunk_size, 1 / rate)
#     return (freq, mag)

# def plot_audio(stream, chunk_size, rate):
#     fig, ax = plt.subplots()
#     x = np.arange(0, chunk_size, 1)
#     line, = ax.plot(np.fft.rfftfreq(chunk_size, 1 / rate), np.abs(np.fft.rfft([1 for i in range(chunk_size)]))) #doing fft once initially so it can get the right size
#     ax.set_ylim(0, 1)
#     ax.set_title('Real-Time Fourier Transform')
#     ax.set_xlabel('Frequency (Hz)')
#     ax.set_ylabel('Magnitude (Normalized)')
#     ax.set_xticks = np.arange(0, rate / 2, rate / 10)
#     print(rate)
    
#     def update_plot(frame):
#         nonlocal stream
#         audio_data = np.frombuffer(stream.read(chunk_size), dtype=np.int16)
#         audio_data = audio_data / 32767 #normalize
        
#         analyzed_data = audio_analysis(audio_data, chunk_size, rate)
#         line.set_ydata(analyzed_data[1])
        
#         return line,

#     ani = FuncAnimation(fig, update_plot, blit=True)
#     plt.show()