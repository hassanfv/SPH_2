
import numpy as np
import matplotlib.pyplot as plt
import struct

#filename = './WithCooling/G-0.001320.bin'

filename = 'G-0.015044.bin'

unit_velocity_cgs = 1.34181e+06 # cm/s #!!!!!!!!!!!!!!!!!!!!!!!!
unit_u = 1.80046e+12 #!!!!!!!!!!!!!!!!!!!!!!!!
unit_rho = 2.83261e-24 # !!!!!!!!!!!!!!!!!!!

def readBinaryFile(filename):
    with open(filename, 'rb') as f:
        file_content = f.read()  # Read the entire file at once

    # Use a memoryview to avoid copies
    buffer = memoryview(file_content)

    # Unpack N and N_ionFrac
    N, N_ionFrac = struct.unpack_from('ii', buffer)
    offset = 8  # Start after the first two integers

    # Function to read and advance the offset
    def read_array(dtype, size, itemsize):
        nonlocal offset
        array = np.frombuffer(buffer, dtype=dtype, count=size, offset=offset)
        offset += size * itemsize
        return array

    # Read arrays
    Typ = read_array(np.int32, N, 4)
    x = read_array(np.float32, N, 4)
    y = read_array(np.float32, N, 4)
    z = read_array(np.float32, N, 4)
    vx = read_array(np.float32, N, 4)
    vy = read_array(np.float32, N, 4)
    vz = read_array(np.float32, N, 4)
    rho = read_array(np.float32, N, 4)
    h = read_array(np.float32, N, 4)
    u = read_array(np.float32, N, 4)
    mass = read_array(np.float32, N, 4)
    ionFrac = read_array(np.float32, N_ionFrac, 4)
    ngbDebug = read_array(np.int32, N, 4)

    return N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac, ngbDebug

# Usage
N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac, ngbDebug = readBinaryFile(filename)


print('Typ == 0 ===> ', np.sum(Typ == 0))

print('ionFrac.shape = ', ionFrac.shape)

ntmp = np.where(h > 0.03)[0]
print('ntmp = ', ntmp)

n = np.where(u != 0.0)[0]
rho = rho[n]
u = u[n]

h = h[n]

print(np.sort(h))

x = x[n]
y = y[n]
z = z[n]


vx = vx[n]
vy = vy[n]
vz = vz[n]

nz = np.where(np.abs(z) < 0.03)[0]

x = x[nz]
y = y[nz]
z = z[nz]

h = h[nz]

vx = vx[nz]
vy = vy[nz]
vz = vz[nz]

vv = np.sqrt(vx*vx + vy*vy + vz*vz)

print()
print('sorted(vv) = ', np.sort(vv))
print()

u = u[nz]
rho = rho[nz]

nx = np.where(rho == max(rho))[0]
print(f'median(u) = {np.median(u)}, max(u) = {u[nx]},  =====> nx = {nx}')

print('sort(u) = ', np.sort(u))

print(np.sum(Typ == -1))

nx = np.where(rho == max(rho))[0]
print(f'median(rho) = {np.median(rho)}, max(rho) = {rho[nx]}  NOTE: rho is different from nH ! rho here is in code unit !!!')
print(np.sort(rho))
#print(np.sum(rho >= 4.0))

kB = 1.3807e-16
mu = 0.61
mH = 1.673534e-24

gamma = 5./3.
Temp = (gamma - 1) * mH / kB * mu * u * unit_u
print('sort T = ', np.sort(Temp))#[-5:])
print('median(Temp) = ', np.median(Temp))


rho_cgs = rho * unit_rho
XH = 0.7
nH_cgs = rho_cgs * XH / mH


nT = np.where(Temp < 12000)[0]
print(nT)


#nn = np.where((x > 0.2) & ( Temp > 1e6))[0]  # nn =  [300013 300018 300022 300049 300104 300116 300163 300167 300178]
#print('nn = ', nn)

unit_density_in_cgs = unit_rho

nH = rho * unit_density_in_cgs * XH /mH

print(f'max(nH) = {max(rho * unit_density_in_cgs * XH /mH)}')
print()

XH = 0.7

print()
print('sort(Temp) = ', np.sort(Temp))
print()


ntmp = np.where((Temp < 30000) & (nH > 0.01) & (nH < 5))[0]
#ntmp = np.where((Temp < 500))[0]
print('ntmp = ', ntmp)

print()
print('Number of particles with T > 1e6 K (Note that this is for the thing layer) = ', len(np.where(Temp > 1e6)[0]))
print()
print('Number of particles with T > 1e9 K (Note that this is for the thing layer) = ', len(np.where(Temp > 1e9)[0]))
print()

plt.figure(figsize=(10, 8))

# Create a scatter plot. The color of each point will depend on the corresponding T value.
#scatter = plt.scatter(x, y, c=np.log10(Temp), cmap='rainbow', s=2)
scatter = plt.scatter(x, y, c=np.log10(nH_cgs), cmap='rainbow', s=2)



# Add a colorbar to the plot to show the relationship between color and T value.
plt.colorbar(scatter, label='nH Value')

scatter = plt.scatter(x[ntmp], y[ntmp], c='b', s=0.1)

xy = 0.32

#plt.xlim(-xy, 0)
#plt.ylim(-xy, 0)

plt.xlim(-xy, xy)
plt.ylim(-xy, xy)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot of X and Y, colored by T value')

plt.savefig('fig.png')

plt.show()




