%matplotlib notebook

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Open the binary file
with open("IC_Evrard_65752.bin", "rb") as file:
    # Read number_particles
    number_particles = np.fromfile(file, dtype=np.int32, count=1)[0]

    # Read array of types (Typ)
    typ = np.fromfile(file, dtype=np.int32, count=number_particles)

    # Read x, y, z coordinates
    xx = np.fromfile(file, dtype=np.float32, count=number_particles)
    yy = np.fromfile(file, dtype=np.float32, count=number_particles)
    zz = np.fromfile(file, dtype=np.float32, count=number_particles)

    # Read velocities
    vx = np.fromfile(file, dtype=np.float32, count=number_particles)
    vy = np.fromfile(file, dtype=np.float32, count=number_particles)
    vz = np.fromfile(file, dtype=np.float32, count=number_particles)

    # Read other parameters
    Uthermal = np.fromfile(file, dtype=np.float32, count=number_particles)
    h = np.fromfile(file, dtype=np.float32, count=number_particles)
    eps = np.fromfile(file, dtype=np.float32, count=number_particles)
    mass = np.fromfile(file, dtype=np.float32, count=number_particles)




# Open the binary file and read the data (as you did in your code)
# ...

# Creating a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot x, y, z
ax.scatter(xx, yy, zz, s=0.1, color='k')

# Highlight a specific particle (e.g., the one at index 48999)
i = 48999
ax.scatter(xx[i], yy[i], zz[i], s=20, color='red')

# Setting labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.title('3D Scatter Plot')

# Show plot
plt.show()


