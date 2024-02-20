#most recent change: updated sample recordings
#inspiration taken from: https://github.com/jeffheaton/present/blob/master/youtube/video/fft-frequency.ipynb

#currently: implementing note identification



from formats import Sinusoid, Recording
rec = Recording('audio samples/Piano Scale.wav', True)
rec.single_note_id(rec._fourier_transform(100000, 101024))