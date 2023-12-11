import struct

def read_element(filename, t, i, j, k, l):
    with open(filename, 'rb') as f:
        # Read dimensions
        N_kpc, N_nH, N_Z, N_T, N_M, N_time = struct.unpack('iiiiii', f.read(24))

        # Skip over kpcsF, densities, metallicities, temperatures, timeArr_in_sec
        f.seek((N_kpc + N_nH + N_Z + N_T + N_time) * 4, 1)

        # Skip over uEvolutionX data until the desired indices
        offset = (t * N_T * N_nH * N_Z * N_time +
                  i * N_nH * N_Z * N_time +
                  j * N_Z * N_time +
                  k * N_time +
                  l) * 4
        f.seek(offset, 1)  # '1' indicates relative positioning

        # Read and return the desired element
        return struct.unpack('f', f.read(4))[0]

# Example usage
filename = 'coolHeatGridNew.bin'
t, i, j, k, l = 1, 50, 41, 1, 0  # specify the indices you want to access

gamma = 5./3.
kB = 1.3807e-16
mH = 1.6726e-24
mu = 0.6144

for l in range(6):

  element = read_element(filename, t, i, j, k, l)
  
  T = mu * (gamma - 1.) * mH * element / kB
  
  print(T)
  #print(f'{element:.3E}')




