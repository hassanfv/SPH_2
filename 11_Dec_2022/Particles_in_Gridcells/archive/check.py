import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(10)


#----- get_grid_cell
def get_grid_cell(x, y, GridSize):

  '''Determine the grid cell for a given particle'''

  cell_x = int((x - min(x)) / GridSize)
  cell_y = int((y - min(y)) / GridSize)

  return cell_x, cell_y


#----- draw_cells
def draw_cells():

  stp = (max(x) - min(x)) / GridSize
  for i in range(10):
    plt.axvline(x = min(x)+stp/2 + i*stp, linestyle = '-', color = 'blue', alpha = 0.4)
  
  stp = (max(y) - min(y)) / GridSize
  for i in range(10):
    plt.axhline(y = min(y)+stp/2 + i*stp, linestyle = '-', color = 'blue', alpha = 0.4)




N = 100

GridSize = 10

beg = -1.0
end = 1.0

x = np.array([random.uniform(beg, end) for _ in range(N)])
y = np.array([random.uniform(beg, end) for _ in range(N)])




plt.figure(figsize = (7, 7))

plt.scatter(x, y, s = 10, color = 'k')
draw_cells()

plt.show()


