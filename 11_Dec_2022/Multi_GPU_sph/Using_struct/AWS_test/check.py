import struct
import numpy as np

def read_binary_file(filename):
    with open(filename, 'rb') as file:
        # Read N from the file
        N = struct.unpack('i', file.read(4))[0]

        # Initialize the arrays
        Typ = [0]*N
        x = [0.0]*N
        y = [0.0]*N
        z = [0.0]*N
        vx = [0.0]*N
        vy = [0.0]*N
        vz = [0.0]*N
        rho = [0.0]*N
        h = [0.0]*N
        u = [0.0]*N
        mass = [0.0]*N

        # Read the data from the file
        for i in range(N):
            Typ[i] = struct.unpack('i', file.read(4))[0]
        for i in range(N):
            x[i] = struct.unpack('f', file.read(4))[0]
        for i in range(N):
            y[i] = struct.unpack('f', file.read(4))[0]
        for i in range(N):
            z[i] = struct.unpack('f', file.read(4))[0]
        for i in range(N):
            vx[i] = struct.unpack('f', file.read(4))[0]
        for i in range(N):
            vy[i] = struct.unpack('f', file.read(4))[0]
        for i in range(N):
            vz[i] = struct.unpack('f', file.read(4))[0]
        for i in range(N):
            rho[i] = struct.unpack('f', file.read(4))[0]
        for i in range(N):
            h[i] = struct.unpack('f', file.read(4))[0]
        for i in range(N):
            u[i] = struct.unpack('f', file.read(4))[0]
        for i in range(N):
            mass[i] = struct.unpack('f', file.read(4))[0]

        return x, y, z, vx, vy, vz, rho, h, u, mass, Typ, N


# Example usage:
filename = "Test.bin"
x, y, z, vx, vy, vz, rho, h, u, mass, Typ, N = read_binary_file(filename)

# Printing the read data for verification
print("N:", N)
#print("rho:", rho)
#print("h:", h)
#print("u:", u)
#print("mass:", mass)

h = np.array(h)

nx = np.where((h >= 0.01110) & (h <= 0.01112))[0]

print(len(nx))










