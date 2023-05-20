import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as colors

def read_arrays_from_binary(filename):
    # Read the binary file
    with open(filename, 'rb') as file:
        # Read N and NG from the file
        N = np.frombuffer(file.read(4), dtype=np.int32)[0]
        NG = np.frombuffer(file.read(4), dtype=np.int32)[0]

        # Read the arrays from the file
        x = np.frombuffer(file.read(N * 4), dtype=np.float32)
        y = np.frombuffer(file.read(N * 4), dtype=np.float32)
        z = np.frombuffer(file.read(N * 4), dtype=np.float32)
        vx = np.frombuffer(file.read(N * 4), dtype=np.float32)
        vy = np.frombuffer(file.read(N * 4), dtype=np.float32)
        vz = np.frombuffer(file.read(N * 4), dtype=np.float32)
        rho = np.frombuffer(file.read(NG * 4), dtype=np.float32)
        h = np.frombuffer(file.read(NG * 4), dtype=np.float32)
        u = np.frombuffer(file.read(NG * 4), dtype=np.float32)

    return x, y, z, vx, vy, vz, rho, h, u, N, NG

# Specify the input file name
filename = 'Outputs/G-0.001208.bin'

# Read the arrays from the binary file
x, y, z, vx, vy, vz, rho, h, u, N, NG = read_arrays_from_binary(filename)

x = np.array(x)[:NG]
y = np.array(y)[:NG]
z = np.array(z)[:NG]
masses = np.full(NG, 1.0 / NG)

xx = x[:NG]
yy = y[:NG]
zz = z[:NG]
nz = np.where(np.abs(zz) <= 0.01)[0]
x = xx[nz]
y = yy[nz]
z = zz[nz]
masses = masses[nz]

# Generate random particle positions and masses (replace this with your own data)
num_particles = len(x)

# Create a 2D grid
#x_grid = np.linspace(min(x), max(x), 100)
#y_grid = np.linspace(min(y), max(y), 100)

xy = 0.15
x_grid = np.linspace(-xy, xy, 500)
y_grid = np.linspace(-xy, xy, 500)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

# Initialize the mass accumulation array
mass_accumulation = np.zeros_like(x_mesh)

# Iterate over particles and accumulate mass on the grid
for i in range(num_particles):
    # Calculate the distances between particle i and all points on the grid
    distances = np.sqrt((x_mesh - x[i]) ** 2 + (y_mesh - y[i]) ** 2)

    # Accumulate mass based on the inverse of distance
    mass_accumulation += masses[i] / distances

# Create a 2D heatmap plot with logarithmic color scale
plt.imshow(mass_accumulation, cmap='viridis', extent=[min(x_grid), max(x_grid), min(y_grid), max(y_grid)],
           norm=colors.LogNorm(vmin=mass_accumulation.min(), vmax=mass_accumulation.max()))

plt.colorbar(label='Mass Accumulation')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Mass Accumulation Projection')
plt.show()

