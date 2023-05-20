
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

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
filename = 'Outputs/G-1.380117.bin'

# Read the arrays from the binary file
x, y, z, vx, vy, vz, rho, h, u, N, NG = read_arrays_from_binary(filename)

print(f'min_u = {min(u)},    max_u = {max(u)},    median_u = {np.median(u)}')

print(x[-10:])

print('x shape = ', x.shape)

plt.hist(u, bins = np.arange(0, 3, 0.1))
#plt.ylim(0, 200)
plt.show()


xy = 0.50

plt.scatter(x[:NG], y[:NG], s = 0.01, color = 'k')
#plt.scatter(x[NG:], y[NG:], s = 0.01, color = 'orange')

plt.xlim(-xy, xy)
plt.ylim(-xy, xy)

plt.show()


r = np.hstack((x[:NG].reshape(-1, 1), y[:NG].reshape(-1, 1), z[:NG].reshape(-1, 1)))

print(r.shape)

print(rho)

d = np.square(r)
d = np.sum(d, axis = 1)
d = d**0.5

plt.scatter(d, rho, s = 0.05, color = 'k')
plt.xscale('log')
plt.yscale('log')
plt.xlim(5e-4, 20)
plt.ylim(1e-8, 5e3)
plt.show()





