
import numpy as np
import matplotlib.pyplot as plt
import glob
import struct

filz = np.sort(glob.glob('/mnt/Linux_Shared_Folder_2022/Outputs_recombined/*.bin'))

filz = np.sort(glob.glob('./OutputsX2/*.bin'))

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
        
        ngb = np.array(struct.unpack(f'{N}i', f.read(N * 4)))

    # Return the data
    return N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac, ngb




for filename in filz:
  N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac, ngb = readBinaryFile(filename)

  #print('Typ == 0 ===> ', np.sum(Typ == 0))

  #print('ionFrac.shape = ', ionFrac.shape)

  n = np.where(u != 0.0)[0]
  rho = rho[n]
  u = u[n]
  h = h[n]

  ngb = ngb[n]

  print('sort(ngb) = ', np.sort(ngb))
  print()





