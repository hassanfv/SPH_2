
import numpy as np
import matplotlib.pyplot as plt
import struct

filename = './Outputs_1181X/G-0.006379.bin'

#filename = './Outputs/G-0.016406.bin'

unit_velocity_cgs = 1.34181e+06 # cm/s #!!!!!!!!!!!!!!!!!!!!!!!!
unit_u = 1.80046e+12 #!!!!!!!!!!!!!!!!!!!!!!!!
unit_rho = 2.83261e-24 # !!!!!!!!!!!!!!!!!!!

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
    
    dudt = read_array(np.float32, N, 4)
    dudt_pre = read_array(np.float32, N, 4)
    uBad = read_array(np.float32, N, 4)
    uAad = read_array(np.float32, N, 4)
    u = read_array(np.float32, N, 4)
    
    mass = read_array(np.float32, N, 4)

    return x, y, z, vx, vy, vz, rho, h, dudt, dudt_pre, uBad, uAad, u, mass, Typ, N

# Usage
x, y, z, vx, vy, vz, rho, h, dudt, dudt_pre, uBad, uAad, u, mass, Typ, N = read_binary_file(filename)

print('Typ == 0 ===> ', np.sum(Typ == 0))
print()

n = np.where(u != 0.0)[0]
rho = rho[n]

uBad = uBad[n]
uAad = uAad[n]
u = u[n]


mass = mass[n]

print('sort(mass) = ', np.sort(mass))
print()

h = h[n]
print('sort(h) = ', np.sort(h))
print()
print('sort(rho) = ', np.sort(rho))
print()

x = x[n]
y = y[n]
z = z[n]

vx = vx[n]
vy = vy[n]
vz = vz[n]

vr = (vx*vx + vy*vy + vz*vz)**0.5

dudt_cool = u - uAad

print('sort(dudt_cool) = ', np.sort(dudt_cool))
print()

t_cool = u / dudt_cool # cooling time scale
print('t_cool = ', t_cool)
print()

# Reference: Hutchings and Thomas - 2000 - In-shock cooling in numerical simulations

gamma = 5./3.
Nngb = 64
l_sph = 2.0 * h / Nngb**(1./3.) # l_sph is the diameter of one SPH particle
cs = (gamma * (gamma - 1.0) * u)**0.5

t_shock = l_sph / vr # shock crossing time

print('sort(t_shock) = ', np.sort(t_shock))






