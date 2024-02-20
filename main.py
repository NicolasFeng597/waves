#most recent change: updated sample recordings
#inspiration taken from: https://github.com/jeffheaton/present/blob/master/youtube/video/fft-frequency.ipynb

#currently: implementing note identification
#added 1/N for normalizing fourier, testing against existing samples


from formats import Sinusoid, Recording
rec = Recording('audio samples/Piano Scale.wav', True)
rec.dynamic_ft()
rec.single_note_id(rec._fourier_transform(100000, 101024))