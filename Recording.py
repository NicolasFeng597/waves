"""
Audio recording module.

This module contains the Recording class for parameterizing audio file metadata.
"""

from scipy.io.wavfile import read as wav_read
from librosa import load as mp3_load
from librosa.util import normalize as mp3_normalize
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gc
from scipy.signal import find_peaks
from math import log2
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip


class Recording:
    """
    A class representing an audio recording.
    Accepted formats: .wav, .mp3

    This class contains data and metadata for audio recordings, including:
    - path (str): file path
    - sample_rate (int): sample rate
    - data (np.ndarray): audio data of shape (channels, samples)
    - normalize (bool): normalize audio data
    - channels (int): number of channels
    - extension (str): file extension
    """

    path: str
    sample_rate: int
    data: np.ndarray
    normalize: bool
    channels: int
    extension: str

    # list of note names (from "E0" to "E10"; edges of human hearing) and frequency (in hz)
    # tuned with equal temperament relative to A440
    _NOTE_NAMES = []
    _FREQUENCIES = []

    def __init__(self, path: str, normalize: bool = False) -> None:
        """
        Initialize a Recording instance.

        Parameters:
        -
        """
        self.path = path
        self.normalize = normalize

        # Get file extension
        self.extension = os.path.splitext(path)[1].lower()

        # scipy for wav files, librosa for mp3 files
        if self.extension == ".wav":
            # scipy format: (samples, channels)
            self.sample_rate, self.data = wav_read(path)
            self.data = self.data.T
        elif self.extension == ".mp3":
            # librosa format: (channels, samples)
            # always assume multi-channel
            self.data, self.sample_rate = mp3_load(path, sr=None, mono=False)
        else:
            raise ValueError(
                f"Unsupported file format: {self.extension}. Only .wav and .mp3 are supported."
            )

        if len(self.data.shape) == 1:
            self.channels = 1  # Mono
        else:
            self.channels = self.data.shape[0]

        # normalizes per channel
        if self.normalize:
            # must switch to floating point
            self.data = self.data.astype(np.float32)

            if self.extension == ".wav":
                if self.channels > 1:
                    for i in range(self.channels):
                        self.data[i, :] /= np.max(np.abs(self.data[i, :]))
                else:
                    self.data /= np.max(np.abs(self.data))
            elif self.extension == ".mp3":
                self.data = mp3_normalize(self.data)

        # populating _NOTE_NAMES and _FREQUENCIES
        _NOTE_REFERENCES = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
        ]

        for i in range(11):
            for j in range(12):
                self._NOTE_NAMES.append(
                    _NOTE_REFERENCES[1][j] + str(_NOTE_REFERENCES[0][i])
                )

        _TWELTH_ROOT_OF_TWO = 1.0594631
        _A440 = 440

        # for a given note, the frequency is A440 * 2^(n/12),
        # where n is the number of semitones from A440
        self._FREQUENCIES = np.append(
            np.flip(
                # frequencies before A440; 57 notes goes down to C0
                np.array(
                    [round(_A440 / (_TWELTH_ROOT_OF_TWO**i), 7) for i in range(1, 58)]
                ),
                0,
            ),
            # frequencies after A440; 68 notes goes up to C10
            np.array([round(_A440 * (_TWELTH_ROOT_OF_TWO**i), 7) for i in range(75)]),
        )

        # take out everything below E0 and above E10 (~<20Hz and ~>20000Hz)
        self._NOTE_NAMES = self._NOTE_NAMES[4:125]
        self._FREQUENCIES = self._FREQUENCIES[4:125]

    def generate_video(
        self,
        output_file_name: str,
        n_fft: int = 2048,
        hop_length: int = 512,
        max_peaks: int = 10,
        window_max_frequency: int = -1,  # if -1, use the max frequency of the recording
    ) -> None:
        """
        Generates a video of the FFT of the recording, with note identification.
        """

        D = np.abs(librosa.stft(self.data, n_fft=n_fft, hop_length=hop_length))
        freq_bins = librosa.fft_frequencies(sr=self.sample_rate, n_fft=n_fft)

        # normalize D to 0-1 based on the max value across all frames
        max_val = np.max(D[0])
        D[0] /= max_val

        # D.shape[2] is the number of frames taken in the STFT
        # TODO: monochannel?
        frame_times = librosa.frames_to_time(
            np.arange(D.shape[2]),
            sr=self.sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
        )

        # creates matplotlib animation
        fig, ax = plt.subplots()

        # first frame
        # D[0] is the first channel, with shape (len(freq_bins), len(frame_times))
        ax.set_title(f"Time: {frame_times[0]:.2f}s")
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_ylim(0, 1.1)

        if window_max_frequency == -1:
            ax.set_xlim(0, freq_bins[-1])
        else:
            ax.set_xlim(0, window_max_frequency)

        plt.ylabel("Magnitude")
        plt.xlabel("Frequency (Hz)")

        # plot of frequency data
        (freq_plot,) = ax.plot(freq_bins, [0 for i in range(len(freq_bins))])

        # plot of note name peaks and cents for tuning
        # find all peaks for first frame

        # allocate max amount of peaks
        peak_markers = [ax.plot([], [], "rx")[0] for _ in range(max_peaks)]
        peak_texts = [ax.text([], [], "") for _ in range(max_peaks)]

        def update(frame):
            # update frequency plot
            freq_plot.set_ydata(D[0][:, frame])
            ax.set_title(f"Time: {frame_times[frame]:.2f}s")

            # update peak plot
            peak_indexes, _ = find_peaks(D[0][:, frame], prominence=0.001)

            peak_hz = [freq_bins[i] for i in peak_indexes]
            peak_magnitudes = [D[0][i, frame] for i in peak_indexes]

            for i in range(max_peaks):
                if i < len(peak_indexes):
                    peak_markers[i].set_xdata([peak_hz[i]])
                    peak_markers[i].set_ydata([peak_magnitudes[i]])
                    peak_markers[i].set_visible(True)
                    note_name, cents = self._hz_to_note(peak_hz[i])
                    peak_texts[i].set_text(f"{note_name}\n{round(cents, 1)}¢")
                    peak_texts[i].set_position((peak_hz[i], peak_magnitudes[i] + 0.01))
                    peak_texts[i].set_visible(True)
                else:
                    peak_markers[i].set_visible(False)
                    peak_texts[i].set_visible(False)

            return (freq_plot, *peak_markers, *peak_texts)

        ani = animation.FuncAnimation(
            fig=fig,
            func=update,
            frames=D.shape[2],
            interval=1000 * (frame_times[1] - frame_times[0]),
            blit=True,
            repeat=False,
        )

        ani.save(f"{output_file_name}_temp.mp4")

        # saving audio
        video = VideoFileClip(f"{output_file_name}_temp.mp4")
        audio = AudioFileClip(self.path)
        video = video.with_audio(audio)
        video.write_videofile(f"{output_file_name}.mp4")
        video.close()
        audio.close()
        os.remove(f"{output_file_name}_temp.mp4")

        # garbage collection
        plt.close(fig)
        del ani
        gc.collect()

    def _hz_to_note(self, hz: float) -> tuple[str, float]:
        """
        Returns the closest note and the cents difference for a given frequency in hertz.

        Parameters:
        hz (float): The frequency in hertz.

        Returns:
        tuple: A tuple containing the closest note and the cents difference.
        """

        for i in range(len(self._FREQUENCIES) - 1):
            if hz >= self._FREQUENCIES[i] and hz <= self._FREQUENCIES[i + 1]:
                if abs(hz - self._FREQUENCIES[i]) > abs(hz - self._FREQUENCIES[i + 1]):
                    # cents equation is 1200 * log2(reference frequency / closest note frequency)
                    return (
                        self._NOTE_NAMES[i + 1],
                        1200 * log2(hz / self._FREQUENCIES[i + 1]),
                    )
                else:
                    return (self._NOTE_NAMES[i], 1200 * log2(hz / self._FREQUENCIES[i]))
