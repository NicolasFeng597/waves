import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
fig, ax = plt.subplots()
ims=[]

for i in range(4):
    ttl = plt.text(0.5, 1.01, i, horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
    ims.append([plt.plot(np.random.randint(0,10,5), np.random.randint(0,20,5)), ttl])
    #plt.cla()


ani = animation.ArtistAnimation(fig, ims, interval=500, blit=False)
plt.show()