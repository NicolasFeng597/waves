#most recent change: added 22050 limit to dynamic_ft
#inspiration taken from: https://github.com/jeffheaton/present/blob/master/youtube/video/fft-frequency.ipynb

from formats import Sinusoid, Recording

sine = Sinusoid(3, 44100, 440.00)
sine.single_note_id(1)
rec = Recording('audio samples/Piano Scale.wav', True)
print(rec)

# rec.dynamic_ft(2048, 'sample', True, False, True, True, 'Piano Scale', 22050)
rec.plot_ft()

# self, start=0, stop=-1, format='sample'
# if stop == -1:
#     stop = len(self)
# if format == 'second':
#     start = start * self.sample_rate
#     stop = stop * self.sample_rate