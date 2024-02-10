import struct
import numpy as np

unitLength = 3.086e+21
UnitDensity = 6.21563e-24

XH = 0.7
mH = 1.673534e-24

def readArraysFromBinary(filename):
    with open(filename, 'rb') as file:
        # Read N and N_ionFrac
        N, N_ionFrac = struct.unpack('ii', file.read(8))
        
        # Function to read an array of floats
        def read_float_array(size):
            return np.frombuffer(file.read(size * 4), dtype=np.float32)
        
        # Function to read an array of integers
        def read_int_array(size):
            return np.frombuffer(file.read(size * 4), dtype=np.int32)  # Assuming int is 4 bytes
        
        # Read the arrays
        Typ = read_int_array(N)
        x = read_float_array(N)
        y = read_float_array(N)
        z = read_float_array(N)
        vx = read_float_array(N)
        vy = read_float_array(N)
        vz = read_float_array(N)
        rho = read_float_array(N)
        h = read_float_array(N)
        u = read_float_array(N)
        mass = read_float_array(N)
        gradRhoNorm = read_float_array(N)
        ionFrac = read_float_array(N_ionFrac)
        
    return x, y, z, vx, vy, vz, rho, h, u, mass, gradRhoNorm, ionFrac, Typ, N, N_ionFrac


filename = './Outputs/G-0.010540.bin'

x, y, z, vx, vy, vz, rho, h, u, mass, gradRhoNorm, ionFrac, Typ, N, N_ionFrac = readArraysFromBinary(filename)


print('Typ == 0 ===> ', np.sum(Typ == 0))
print('ionFrac.shape = ', ionFrac.shape)

n = np.where(u != 0.0)[0]
rho = rho[n]
gradRhoNorm = gradRhoNorm[n]
gradRhoNorm = np.sqrt(gradRhoNorm)

# Calculate the 90th percentile value of rho
percentile_90_value = np.percentile(rho, 99.9)
nt = (np.abs(rho - percentile_90_value)).argmin()
nH = rho[nt] * UnitDensity * XH / mH

#nt = np.where(rho == max(rho))[0][0]
#nt = rho[nt] * UnitDensity * XH / mH


print()
print(f'h (in pc) = {(h[nt]*unitLength/3.086e18):.3f}   its log(NHtot) = {np.log10(nH * h[nt]*unitLength):.3f}')
print(f'rho/gradRho (in pc) = {(rho[nt]/gradRhoNorm[nt]*unitLength/3.086e18):.3f}    its log(NHtot) = {np.log10(nH*(rho[nt]/gradRhoNorm[nt]*unitLength)):.3f}')
print(f'rho = {rho[nt]:.3f}    its nH = {nH:.3f}')
print(f'gRo = {gradRhoNorm[nt]:.3f}')








