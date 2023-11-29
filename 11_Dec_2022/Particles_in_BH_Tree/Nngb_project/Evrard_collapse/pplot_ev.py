import numpy as np
import matplotlib.pyplot as plt

# Open the binary file
with open("IC_Evrard_65752.bin", "rb") as file:
    # Read number_particles
    number_particles = np.fromfile(file, dtype=np.int32, count=1)[0]

    # Read x, y, z arrays
    data = np.fromfile(file, dtype=np.float32, count=3*number_particles).reshape(-1, 3)
    x, y, z = data[:,0], data[:,1], data[:,2]

# Scatter plot x and y
plt.scatter(x, y, s = 0.1, color = 'k')

#plt.scatter(x, y, s = 0.1, color = 'k')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot of x and y')
plt.show()

