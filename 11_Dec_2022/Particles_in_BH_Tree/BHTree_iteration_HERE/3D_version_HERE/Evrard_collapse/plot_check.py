import struct
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

filename = "IC_Evrard_523984.bin"  # Replace with your actual file name

jj = 505851

df = pd.read_csv('ndx.csv')
ndx = df['n'].values
ndx = [np.int32(tmp) for tmp in ndx]

# Open the binary file
with open(filename, 'rb') as f:
    # Read the total number of particles (int)
    num_particles = struct.unpack('i', f.read(4))[0]

    # Skip the 'Typ' array (assuming it's of int type)
    f.read(num_particles * 4)

    # Read the x, y, and z coordinates (floats)
    xx = np.array(struct.unpack(f'{num_particles}f', f.read(num_particles * 4)))
    yy = np.array(struct.unpack(f'{num_particles}f', f.read(num_particles * 4)))
    zz = np.array(struct.unpack(f'{num_particles}f', f.read(num_particles * 4)))

    # Skip vx, vy, vz, Uthermal and eps arrays (all floats)
    f.read(num_particles * 4 * 5)

    # Read the h values
    h = np.array(struct.unpack(f'{num_particles}f', f.read(num_particles * 4)))


print(np.sort(h))
print()
print('2*h[jj] = ', 2*h[jj])


dx = xx - xx[jj]
dy = yy - yy[jj]
dz = zz - zz[jj]

r = np.sqrt(dx*dx + dy*dy + dz*dz)

nx = np.where(r <= 2*h[jj])[0]

print(nx)

# Create a scatter plot
plt.figure(figsize = (8, 8))
plt.scatter(xx, yy, s=0.1, color = 'k')
plt.scatter(xx[ndx], yy[ndx], s=5, color = 'lime')
plt.scatter(xx[jj], yy[jj], s=50, color = 'red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of x and y coordinates')
plt.show()

