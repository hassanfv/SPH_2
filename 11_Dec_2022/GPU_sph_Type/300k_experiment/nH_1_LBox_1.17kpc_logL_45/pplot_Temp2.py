
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
import time

TA = time.time()

filename = './OutXX/G-0.002499.bin'

Unit_u_in_cgs = 1.0324E+12
UnitDensity_in_cgs = 1.624E-24

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

rho = rho * UnitDensity_in_cgs
u = u * Unit_u_in_cgs


kB = 1.3807e-16
mu = 0.61
mH = 1.673534e-24

gamma = 5./3.

print('nmber of Typ = -1 particles = ', np.sum(Typ == -1))

#nT = np.where(Typ != -1)[0]

nT = np.where(u != 0.0)[0]

nT = nT#[1:] # To remove black hole as it has u = rho = 0!!!!

x = x[nT]
y = y[nT]
z = z[nT]
rho = rho[nT]
u = u[nT]

Temp = (gamma - 1) * mH / kB * mu * u

nz = np.where(np.abs(z) < 0.10)[0]

x = x[nz]
y = y[nz]
z = z[nz]
rho = rho[nz]
Temp = Temp[nz]

print(len(Temp))

# Create a grid of points where the image will be interpolated
#grid_x, grid_y = np.mgrid[-0.6:0.6:1000j, -0.6:0.6:1000j]
grid_x, grid_y = np.mgrid[0.0:0.6:1000j, 0.0:0.6:1000j]

# Interpolate data to get values at each point on the grid
grid_temp = griddata((x, y), np.log10(Temp), (grid_x, grid_y), method='cubic')

# Plotting
plt.figure(figsize=(10,7))
plt.imshow(grid_temp, extent=(0.0, 0.6, 0.0, 0.6), origin='lower', aspect='auto', cmap='inferno', vmin=1, vmax=10)
cbar = plt.colorbar()
cbar.set_label('Temperature', rotation=270, labelpad=15)
plt.title("Interpolated Particle Temperatures")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")

print('Elapsed time = ', time.time() - TA)

# Show the plot
plt.show()









