import struct
import numpy as np

# Open the binary file
with open('coolHeatGridNewdebug.bin', 'rb') as f:
    # Read and unpack the dimensions
    N_kpc, N_nH, N_Z, N_T, N_M, N_time = struct.unpack('iiiiii', f.read(6 * 4))

    # Skip the data you don't need (kpcs, densities, metallicities, temperatures, timeArr)
    f.read(4 * (N_kpc + N_nH + N_Z + N_T + N_time))  # float is 4 bytes

    # Read uEvolution
    uEvolution_data = f.read(4 * N_kpc * N_T * N_nH * N_Z * N_time)
    uEvolution = np.frombuffer(uEvolution_data, dtype=np.float32).reshape(N_kpc, N_T, N_nH, N_Z, N_time)

    # Read muArr
    muArr_data = f.read(4 * N_kpc * N_T * N_nH * N_Z * N_time)
    muArr = np.frombuffer(muArr_data, dtype=np.float32).reshape(N_kpc, N_T, N_nH, N_Z, N_time)

    # Skip metalz data
    f.read(4 * N_kpc * N_T * N_nH * N_Z * N_M * N_time)  # float is 4 bytes

    # Read TEvolution
    TEvolution_data = f.read(4 * N_kpc * N_T * N_nH * N_Z * N_time)
    TEvolution = np.frombuffer(TEvolution_data, dtype=np.float32).reshape(N_kpc, N_T, N_nH, N_Z, N_time)

# Now you have uEvolution, muArr, and TEvolution as multidimensional numpy arrays



t, i, j, k, l = 1, 50, 41, 1, 0  # specify the indices you want to access

gamma = 5./3.
kB = 1.3807e-16
mH = 1.6726e-24
mu = 0.6144

print(TEvolution[1, 40, 41, 1, :])




