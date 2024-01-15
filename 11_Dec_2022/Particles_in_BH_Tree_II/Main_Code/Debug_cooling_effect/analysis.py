
import numpy as np
import matplotlib.pyplot as plt
import struct

filename = './Outputs_1181/G-0.000400.bin'

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

dudt = dudt[n]
dudt_pre = dudt_pre[n]
uBad = uBad[n]
uAad = uAad[n]
u = u[n]

mass = mass[n]

h = h[n]

x = x[n]
y = y[n]
z = z[n]

vx = vx[n]
vy = vy[n]
vz = vz[n]

i = 10000

ut = dudt[i]
ut_pre = dudt_pre[i]
uB = uBad[i]
uA = uAad[i]
uC = u[i] # this is actually uAC

print(f'ut = {ut:.4f}, ut_pre = {ut_pre:.4f}, uBad = {uB:.4f}, uAad = {uA:.4f}, uAC = {uC:.4f}')






