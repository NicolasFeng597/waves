from formats import Sinusoid, Recording


sine = Sinusoid(3, 44100, 440.00)
sine.mix([Sinusoid(3, 44100, )])
rec = Recording('audio samples/Piano Song.wav', True)

rec.dynamic_ft(progress=True)
