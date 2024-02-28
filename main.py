#inspiration taken from: https://github.com/jeffheaton/present/blob/master/youtube/video/fft-frequency.ipynb

#most recent changes: 
#switched normalizing from dynamic_ft() to _fourier_transform()
#made the notes dict a static final

#currently: implementing note identification


from formats import Sinusoid, Recording
rec = Recording('audio samples/Piano Scale.wav', True)
print(rec)
rec._single_note_id(rec._fourier_transform(44100*2, 44100*2 + 1024)) #frame at 2 sec