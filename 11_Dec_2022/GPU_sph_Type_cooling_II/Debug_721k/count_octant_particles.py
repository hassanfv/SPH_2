
import numpy as np
import matplotlib.pyplot as plt

#filename = 'G-0.001313.bin' # No cooling
#filename = './OutAGN_1.0kpc_New/G-0.000300.bin' # With cooling

#filename = './OutAGN_1.0kpc_res_by_2/G-0.004450.bin'

filename = './Out712k_mgpu/G-0.015000.bin'


import struct

def readBinaryFile(filename):
    with open(filename, 'rb') as f:
        # Read N and N_ionFrac
        N, N_ionFrac = struct.unpack('ii', f.read(2 * 4))  # 4 bytes each for two integers

        # Read arrays
        Typ = np.array(struct.unpack(f'{N}i', f.read(N * 4)))
        x = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        y = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        z = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        vx = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        vy = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        vz = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        rho = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        h = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        u = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        mass = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        ionFrac = np.array(struct.unpack(f'{N_ionFrac}f', f.read(N_ionFrac * 4)))

    # Return the data
    return N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac 



# Usage
N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac = readBinaryFile(filename)

ionFrac = ionFrac.reshape((N, 14))

print('Typ == 0 ===> ', np.sum(Typ == 0))

print('ionFrac.shape = ', ionFrac.shape)

n = np.where(u != 0.0)[0]
rho = rho[n]
u = u[n]
h = h[n]

ionFrac = ionFrac[n, :]

x = x[n]
y = y[n]
z = z[n]

vx = vx[n]
vy = vy[n]
vz = vz[n]


ndx_BH = 701166
xx = x[ndx_BH:]
yy = y[ndx_BH:]
zz = z[ndx_BH:]
uu = u[ndx_BH:]

#===== Counting particles located in each Octant ===
# Initialize a dictionary to hold counts for each octant
octant_counts = {i: 0 for i in range(1, 9)}

# Count particles in each octant
for x, y, z in zip(xx, yy, zz):
    if x > 0 and y > 0 and z > 0:
        octant_counts[1] += 1
    elif x < 0 and y > 0 and z > 0:
        octant_counts[2] += 1
    elif x < 0 and y < 0 and z > 0:
        octant_counts[3] += 1
    elif x > 0 and y < 0 and z > 0:
        octant_counts[4] += 1
    elif x > 0 and y > 0 and z < 0:
        octant_counts[5] += 1
    elif x < 0 and y > 0 and z < 0:
        octant_counts[6] += 1
    elif x < 0 and y < 0 and z < 0:
        octant_counts[7] += 1
    elif x > 0 and y < 0 and z < 0:
        octant_counts[8] += 1

print(octant_counts)

kB = 1.3807e-16
mu = 0.61
mH = 1.673534e-24

unit_u = 7.02136e+12
gamma = 5./3.
Temp = (gamma - 1) * mH / kB * mu * uu * unit_u


plt.figure(figsize=(10, 8))

# Create a scatter plot. The color of each point will depend on the corresponding T value.
scatter = plt.scatter(xx, yy, c=np.log10(Temp), cmap='rainbow', s=5)
#scatter = plt.scatter(xx, yy, c=np.log10(nH_cgs), cmap='rainbow', s=2)


# Add a colorbar to the plot to show the relationship between color and T value.
#plt.colorbar(scatter, label='T Value')
plt.colorbar(scatter, label='nH Value')

#----- drawing a circle ----
R = 0.1635
center = (0, 0)
theta = np.linspace(0, 2*np.pi, 100)
xc = center[0] + R * np.cos(theta)
yc = center[1] + R * np.sin(theta)

plt.plot(xc, yc, color = 'blue', linewidth = 5, alpha = 0.4)

plt.axvline(x = 0, linestyle = '--', color = 'b')
plt.axhline(y = 0, linestyle = '--', color = 'b')
#---------------------------

xy = 0.20

plt.xlim(-xy, xy)
plt.ylim(-xy, xy)


plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot of X and Y, colored by T value')

plt.savefig('figOut.png')

plt.show()



