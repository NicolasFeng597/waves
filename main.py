import matplotlib.pyplot as plt
from formats import Sinusoid
from formats import Recording

sine = Sinusoid(5, 44100, 1)
starwars = Recording('StarWars3.wav', True)
print(starwars)
sinef = sine.fourier_transform()
starf = starwars.fourier_transform()
plt.plot(starf[0], starf[1])

plt.show()