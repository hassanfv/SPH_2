import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

random.seed(10)


#===== get_grid_cell
def get_grid_cell(x, y, x_p, y_p, GridSize):

  '''Determine the grid cell for a given particle'''

  maxi = max(max(x), max(y))

  cell_x = int(x_p * GridSize/maxi)
  cell_y = int(y_p * GridSize/maxi)

  return cell_x, cell_y


#===== draw_grid_lines
def draw_grid_lines():

  stp_x = (max(x) - min(x)) / GridSize
  stp_y = (max(y) - min(y)) / GridSize
  
  stp = max(stp_x, stp_y)
  
  for i in range(GridSize):
    plt.axvline(x = min(x) + i*stp, linestyle = '-', color = 'blue', alpha = 0.4)
  
  for i in range(GridSize):
    plt.axhline(y = min(y) + i*stp, linestyle = '-', color = 'blue', alpha = 0.4)


#===== show_one_cell
def show_one_cell(x, y, cell_x, cell_y, GridSize):

  '''This marks the lower left point (i.e. the starting point) of the cell'''

  maxi = max(max(x), max(y))

  xstp = (max(x) - min(x)) / GridSize
  ystp = (max(y) - min(y)) / GridSize

  xc = cell_x / GridSize * maxi
  yc = cell_y / GridSize * maxi
  
  plt.scatter(xc, yc, s = 200, color = 'cyan', alpha = 1)


#===== plot_single_particle
def plot_single_particle(x_p, y_p):
  plt.scatter(x_p, y_p, s = 20, color = 'r')


#===== fill_cell_with_color
def fill_cell_with_color(x, y, cell_x, cell_y, GridSize, color='red'):
  """Fill the grid cell with the specified color on the existing plot."""
  
  maxi = max(max(x), max(y))
  
  cell_size_x = (max(x) - min(x)) / GridSize
  cell_size_y = (max(y) - min(y)) / GridSize
  
  # Calculate the minimum coordinates of the cell
  min_x = cell_x / GridSize * maxi
  min_y = cell_y / GridSize * maxi
  
  # Create a rectangle (cell) and fill it with the color
  rect = patches.Rectangle((min_x, min_y), cell_size_x, cell_size_y, linewidth=1, edgecolor='black', facecolor=color, alpha = 0.3)
  
  # Get current axes and add the rectangle to it
  ax = plt.gca()  # Get current axes
  ax.add_patch(rect)




N = 100

GridSize = 10

beg = -1.0
end = 1.0

x = np.array([random.uniform(beg, end) for _ in range(N)])
y = np.array([random.uniform(beg, end) for _ in range(N)])

x = x - min(x)
y = y - min(y)

i_p = 46
x_p = x[i_p]
y_p = y[i_p]

cell_x, cell_y = get_grid_cell(x, y, x_p, y_p, GridSize)


xstp = (max(x) - min(x)) / GridSize
ystp = (max(y) - min(y)) / GridSize

print(f'xstp = {xstp},  ystp = {ystp}')
  
print(x_p, y_p, cell_x, cell_y)



plt.figure(figsize = (7, 7))

plt.scatter(x, y, s = 10, color = 'k')

draw_grid_lines()

show_one_cell(x, y, cell_x, cell_y, GridSize)

plot_single_particle(x_p, y_p)

fill_cell_with_color(x, y, cell_x, cell_y, GridSize, color='red')

plt.show()






