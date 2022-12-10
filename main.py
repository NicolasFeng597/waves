import matplotlib.pyplot as plt
from waves import waves

a440 = waves(5, 44100, 440)
star_wars = waves(path="StarWars3.wav")
# print(star_wars)
# waves.plot_wave(star_wars, start = 50000, stop = 50500)
waves.plot_ft(a440)
waves.plot_ft(star_wars)
plt.show()
