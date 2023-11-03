
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(42)

beg = 0
end = 1

N = 100 # number of particles

x = np.array([random.uniform(beg, end) for _ in range(N)])
y = np.array([random.uniform(beg, end) for _ in range(N)])

GRIDSIZE = 10

for i in range(N):

  cell_x = int(x[i] * GRIDSIZE)
  cell_y = int(y[i] * GRIDSIZE)
  
  print(x[i], y[i], cell_x, cell_y)
  print()

xi, yi, cell_x, cell_y = [0.6399997598540929, 0.6311029572700989, 6, 6]

xc = (cell_x+0.5)/10
yc = (cell_y+0.5)/10


if True:
  plt.figure(figsize = (6, 6))
  plt.scatter(x, y, s = 20, color = 'k')
  plt.scatter(xc, yc, s = 60, color = 'r', alpha = 0.6)
  plt.scatter(xi, yi, s = 20, color = 'lime', alpha = 0.6)
  
  
  for xt in np.arange(0, 1.1, 0.1):
    plt.axvline(x = xt, linestyle = '-', color = 'blue')
    plt.axhline(y = xt, linestyle = '-', color = 'blue')
  
  plt.show()



