import numpy as np
import matplotlib.pyplot as plt

with open("dataX.bin", "rb") as file:
    # Read the sizes and W_cell
    N = np.fromfile(file, dtype=np.int32, count=1)[0]
    Ncell = np.fromfile(file, dtype=np.int32, count=1)[0]
    nSplit = np.fromfile(file, dtype=np.int32, count=1)[0]
    W_cell = np.fromfile(file, dtype=np.float32, count=1)[0]

    # Read arrays
    x = np.fromfile(file, dtype=np.float32, count=N)
    y = np.fromfile(file, dtype=np.float32, count=N)
    groupedIndex = np.fromfile(file, dtype=np.int32, count=N)
    scanOffset = np.fromfile(file, dtype=np.int32, count=Ncell+1)


print(scanOffset)


i = 37

ndx_particles_in_cell_i = groupedIndex[scanOffset[i]:scanOffset[i+1]]

x_tmp = x[ndx_particles_in_cell_i]
y_tmp = y[ndx_particles_in_cell_i]

# Plotting
plt.figure(figsize = (8, 8))

plt.scatter(x, y, s = 10, color = 'k')

plt.scatter(x_tmp, y_tmp, s = 10, color = 'red')

for i in range(-int(nSplit/2), int(nSplit/2)+1):

    plt.axvline(x = i * W_cell, linestyle = '--', color = 'b')
    plt.axhline(y = i * W_cell, linestyle = '--', color = 'b')


plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot of x and y')
plt.show()


