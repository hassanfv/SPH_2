
import numpy as np
import matplotlib.pyplot as plt
import struct

#filename = '/mnt/Linux_Shared_Folder_2022/Outputs/G-0.016005.bin'

filename = './Outputs/G-0.003200.bin'

unit_velocity_cgs = 2.31909e+06 # cm/s #!!!!!!!!!!!!!!!!!!!!!!!!
unit_u = 5.37816e+12 #!!!!!!!!!!!!!!!!!!!!!!!!
unit_rho = 8.46128e-24 # !!!!!!!!!!!!!!!!!!!

def read_binary_file(filename):
    with open(filename, 'rb') as file:
        file_content = file.read()
    
    buffer = memoryview(file_content)
    offset = 0

    # Function to read and advance the offset
    def read_array(dtype, size, itemsize):
        nonlocal offset
        array = np.frombuffer(buffer, dtype=dtype, count=size, offset=offset)
        offset += size * itemsize
        return array

    # Read N
    N = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
    offset += 4  # Size of int32

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

    return x, y, z, vx, vy, vz, rho, h, u, mass, Typ, N

# Usage
x, y, z, vx, vy, vz, rho, h, u, mass, Typ, N = read_binary_file(filename)

print('Typ == 0 ===> ', np.sum(Typ == 0))
print()

n = np.where(u != 0.0)[0]
rho = rho[n]
u = u[n]
mass = mass[n]

print('sort(mass) = ', np.sort(mass))

h = h[n]
print('sort(h) = ', np.sort(h))
print()
print('sort(rho) = ', np.sort(rho))

x = x[n]
y = y[n]
z = z[n]

G = 1.0
gamma = 5./3.

N_J_m = 4
N_ngb = 64

mSPH = mass

Pfloor = (36./np.pi**5)**(1./3.) * (G / gamma) * (N_J_m * N_ngb * mSPH)**(2./3.) * rho**(4./3.)

Pgas = (gamma - 1.) * rho * u

print('Pfloor = ', np.sort(Pfloor))
print('Pgas = ', np.sort(Pgas))



for i in range(len(Pgas)):
  if Pgas[i] < 5e-2:
    print(Pgas[i], Pfloor[i])







