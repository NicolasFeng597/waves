#most recent change: added 22050 limit to dynamic_ft
#inspiration taken from: https://github.com/jeffheaton/present/blob/master/youtube/video/fft-frequency.ipynb

from formats import Sinusoid, Recording

# debugging /audio samples data recordings, kinda corrupt?
rec = Recording('audio samples/15000Hz.wav', True)
print(rec)
rec.dynamic_ft(rewrite=True,save_sample='15000Hz')