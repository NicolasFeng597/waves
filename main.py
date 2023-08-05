import matplotlib.pyplot as plt
from formats import Sinusoid
from formats import Recording
import numpy as np
sine = Sinusoid(5, 22050, 1)

starwars = Recording('StarWars3.wav', True)

sine.mix(Sinusoid(5, 22050, 5000), Sinusoid(8, 22050, 22050/2 - 1))
sine.plot_ft()