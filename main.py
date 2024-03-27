#inspiration taken from: https://github.com/jeffheaton/present/blob/master/youtube/video/fft-frequency.ipynb

#most recent changes: 
#switched normalizing from dynamic_ft() to _fourier_transform()
#made the notes dict a static final
#switched notes dict to _NOTE_NAMES and _FREQUENCIES
#finished fluff for _single_note_id()
#implemented note identification; using _single_note_id() for video frames

#currently:


from formats import Sinusoid, Recording
rec = Recording('audio samples/A440 Piano Note.wav', True)
print(rec)
rec.dynamic_ft(show_progress=True, save_sample='A440 Piano Note -- Notes Identified', rewrite=True, limit=5000)