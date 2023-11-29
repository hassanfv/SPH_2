import numpy as np
import matplotlib.pyplot as plt

# Read the binary file
with open("data.bin", "rb") as file:
    N = np.fromfile(file, dtype=np.int32, count=1)[0]  # Read N
    x = np.fromfile(file, dtype=np.float32, count=N)   # Read x array
    y = np.fromfile(file, dtype=np.float32, count=N)   # Read y array

# Plotting
plt.figure(figsize = (8, 8))

plt.scatter(x, y, s = 10, color = 'k')

nSplit = 10
W_cell = 2.0 / nSplit

for i in range(-int(nSplit/2), int(nSplit/2)+1):

    plt.axvline(x = i * W_cell, linestyle = '--', color = 'b')
    plt.axhline(y = i * W_cell, linestyle = '--', color = 'b')


plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot of x and y')
plt.show()

