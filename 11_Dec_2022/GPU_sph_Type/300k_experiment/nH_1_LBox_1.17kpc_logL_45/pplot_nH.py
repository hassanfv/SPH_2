
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

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

nz = np.where(np.abs(z) < 0.040)[0]

x = x[nz]
y = y[nz]
z = z[nz]
rho = rho[nz]
Temp = Temp[nz]

nH = rho / mu / mH


# Create the scatter plot
plt.figure(figsize=(10,7))
sc = plt.scatter(x, y, c=np.log10(nH), cmap='jet', s = 0.1)

# Add a colorbar to show temperature values
cbar = plt.colorbar(sc)
cbar.set_label('Density', rotation=270, labelpad=15)

# Set labels and title
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Particle Density")

# Show the plot
plt.show()









