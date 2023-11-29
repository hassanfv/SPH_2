import numpy as np
import matplotlib.pyplot as plt

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


print("np.sort(h) = ", np.sort(h))


# Scatter plot x and y
plt.scatter(xx, yy, s=0.1, color='k')

i = 48999

plt.scatter(xx[i], yy[i], s=20, color='red')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot of x and y')
plt.show()

