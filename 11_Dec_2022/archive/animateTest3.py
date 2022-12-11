import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import time

x = np.arange(128)
fig, ax = plt.subplots()

for it in range(50):
    ax.plot(x, x+it)
    # fig.canvas.draw_idle()  # required to work on mpl < 1.5
    fig.canvas.flush_events()
    time.sleep(0.1)

plt.ioff()
#plt.show()
