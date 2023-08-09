
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

filename = './OutXX/G-0.002499.bin'
filename = './OutXX/G-0.000200.bin'

Unit_u_in_cgs = 1.0324E+12
UnitDensity_in_cgs = 1.624E-24
unitVelocity_in_cm_per_s = 1016080.02

def loadArraysFromBinary(filename):
    with open(filename, "rb") as file:
        # Read N
        N = np.fromfile(file, dtype=np.int32, count=1)[0]

        # Create arrays for each of the data types
        Typ = np.fromfile(file, dtype=np.int32, count=N)
        x = np.fromfile(file, dtype=np.float32, count=N)
        y = np.fromfile(file, dtype=np.float32, count=N)
        z = np.fromfile(file, dtype=np.float32, count=N)
        vx = np.fromfile(file, dtype=np.float32, count=N)
        vy = np.fromfile(file, dtype=np.float32, count=N)
        vz = np.fromfile(file, dtype=np.float32, count=N)
        rho = np.fromfile(file, dtype=np.float32, count=N)
        h = np.fromfile(file, dtype=np.float32, count=N)
        u = np.fromfile(file, dtype=np.float32, count=N)
        mass = np.fromfile(file, dtype=np.float32, count=N)

    return N, Typ, x, y, z, vx, vy, vz, rho, h, u, mass

# Usage
N, Typ, x, y, z, vx, vy, vz, rho, h, u, mass = loadArraysFromBinary(filename)

nn = np.where(u != 0.0)[0]

x = x[nn]
y = y[nn]
z = z[nn]

vx = vx[nn]
vy = vy[nn]
vz = vz[nn]

# 1. Calculate the distance of each particle from the center
distances = np.sqrt(x**2 + y**2 + z**2)

# 2. Calculate the radial velocity for each particle
# First, compute the normalized position vectors
norm_positions = np.vstack([x, y, z]).T
norm_positions /= distances[:, np.newaxis]

# Radial velocity is the dot product of velocity and normalized position
radial_velocities = np.einsum('ij,ij->i', np.vstack([vx, vy, vz]).T, norm_positions)

radial_velocities = radial_velocities * unitVelocity_in_cm_per_s / 100. / 1000. # in km/s

print(radial_velocities)

# 3. Plot the radial velocity against the distance
plt.figure(figsize=(10, 7))
plt.scatter(distances, radial_velocities, c='blue', alpha=0.5, s = 2)
plt.title("Radial Velocity vs. Distance from Center")
plt.xlabel("Distance from Center")
plt.ylabel("Radial Velocity")
plt.grid(True)
plt.show()









