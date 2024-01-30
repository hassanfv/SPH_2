import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import pickle
import os
import struct


filename = './Outputs/G-0.049590.bin'

unit_velocity_cgs = 3.21164e+06 # cm/s #!!!!!!!!!!!!!!!!!!!!!!!!
unit_u = 1.03146e+13 #!!!!!!!!!!!!!!!!!!!!!!!!
unit_rho = 1.62276e-23 # !!!!!!!!!!!!!!!!!!!
unit_length = 3.086e+21 #!!!!!!!!!!!!!!!!!!!

XH = 0.7

kB = 1.3807e-16
mu = 0.61
mH = 1.673534e-24
gamma = 5./3.



#===== find_clumps
def find_clumps(particles, cold_particles_indices, min_particles):
    """
    Identify clumps in particle data based on temperature and density thresholds.

    Parameters:
    - particles: numpy array of shape (n_particles, 5) where each row represents a particle and
                 columns represent x, y, z coordinates, temperature, and density, respectively.
    - min_particles: Minimum number of particles to consider a group a clump.

    Returns:
    - clumps: A list of lists, where each inner list contains the indices of particles in a clump.
    """

    in_clump = np.zeros(len(particles), dtype=bool)
    clumps = []

    for i in cold_particles_indices:
        if in_clump[i]:
            continue  # Skip if particle is already in a clump

        # Start a new clump with the current particle
        current_clump = [i]
        in_clump[i] = True

        # Iteratively add nearby particles to the clump
        for j in current_clump:
            for k in cold_particles_indices:
                if in_clump[k]:
                    continue  # Skip if particle is already in a clump

                distance = np.sqrt(np.sum((particles[j, :3] - particles[k, :3]) ** 2))
                if distance <= min(particles[j, 3], particles[k, 3]):  # Using the sum of the smoothing lengths of particles j, and k.
                    current_clump.append(k)
                    in_clump[k] = True

        # Check if the current clump meets the minimum size requirement
        if len(current_clump) >= min_particles:
            clumps.append(current_clump)

    return clumps

#===== readBinaryFile
def readBinaryFile(filename):
    with open(filename, "rb") as file:
        # Read N and N_ionFrac
        N, N_ionFrac = np.fromfile(file, dtype=np.int32, count=2)

        # Create arrays for each of the data types
        Typ = np.fromfile(file, dtype=np.int32, count=N)
        x = np.fromfile(file, dtype=np.float32, count=N)
        y = np.fromfile(file, dtype=np.float32, count=N)
        z = np.fromfile(file, dtype=np.float32, count=N)
        vx = np.fromfile(file, dtype=np.float32, count=N)
        vy = np.fromfile(file, dtype=np.float32, count=N)
        vz = np.fromfile(file, dtype=np.float32, count=N)
        rho = np.fromfile(file, dtype=np.float32, count=N)
        h = np.fromfile(file, dtype=np.float32, count=N)
        u = np.fromfile(file, dtype=np.float32, count=N)
        mass = np.fromfile(file, dtype=np.float32, count=N)
        ionFrac = np.fromfile(file, dtype=np.float32, count=N_ionFrac)

    return N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac


# 0    1   2    3    4    5    6      7      8   9    10   11    12   13
# HI  HII  CI  CII CIII  CIV  SiII  SiIII  SiIV  NV  OVI  FeII  MgI  MgII

N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac = readBinaryFile(filename)

print('Typ == 0 ===> ', np.sum(Typ == 0))

n = np.where(u != 0.0)[0]
rho = rho[n]
u = u[n]

ionFrac = ionFrac.reshape((N, 14))
ionFrac = ionFrac[n, :]

print(ionFrac.shape)

h = h[n]
mass = mass[n]

print('sort(mass) = ', np.sort(mass))

N = len(h)

#-----
# We will multiply inClump this by nH_cgs so that particle already in a clump are skipped!
if not os.path.exists('Clumps.pkl'):
  with open('Clumps.pkl', 'wb') as f:
    inClump = np.ones(N)
    dictInitial = {'inClump': inClump, 'nxclumps': []} # nxclumps is initially empty!
    pickle.dump(dictInitial, f)

with open('Clumps.pkl', 'rb') as f:
  dict_tmp = pickle.load(f)
  inClump = dict_tmp['inClump']
  nxclumps_previous = dict_tmp['nxclumps']
#-----

print('len(nxclumps_previous) = ', len(nxclumps_previous))


x = x[n]
y = y[n]
z = z[n]

r = np.vstack((x, y, z))
r = np.transpose(r)
print(r.shape)

vx = vx[n]
vy = vy[n]
vz = vz[n]

v = np.vstack((vx, vy, vz))
v = np.transpose(v)
print(v.shape)

rho_cgs = rho * unit_rho
nH_cgs = rho_cgs * XH / mH

nH_cgs = nH_cgs * inClump # To exclude particles already belonging to a clump from the previous run!

nt = np.where(nH_cgs == max(nH_cgs))[0][0] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

rtmp = np.sqrt((x - x[nt])**2 + (y - y[nt])**2 + (z - z[nt])**2)
ntmpx = np.argsort(rtmp)
ntmpx = ntmpx[:4000]

ntmp = ntmpx[np.where(nH_cgs[ntmpx] > 500)[0]]

particles = np.zeros((N, 4))

particles[:, 0] = x
particles[:, 1] = y
particles[:, 2] = z
particles[:, 3] = h

clumps = find_clumps(particles, ntmp, min_particles=200)
maxi = 0
for ic, clump in enumerate(clumps):

  if len(clump) > maxi:
    maxi = len(clump)
    ndxMaxClump = ic
  print('len(clump) = ', len(clump))

nxclumps = clumps[ndxMaxClump] # getting the clump with the max number of particles.---> nxclump contain the index of the particles in the clump.
nxclumpsX= nxclumps.copy() # used for plotting only!

inClump[nxclumps] = 0.0

#--- Output to a file
nxclumps_previous.append(nxclumps)
dictx = {'inClump': inClump, 'nxclumps': nxclumps_previous}
with open('Clumps.pkl', 'wb') as f:
  pickle.dump(dictx, f)

#-----> To be read by the ElemAbund.cu code in C++ CUDA <------
with open('Clumps.bin', 'wb') as file:
  for sublist in nxclumps_previous:
    # Store the length of the sublist first
    file.write(struct.pack('I', len(sublist)))
    # Then write the sublist elements
    for item in sublist:
      file.write(struct.pack('I', item))
#-------


#--------- drawing the line of sight -------------
p0 = np.array([0, 0, 0])
p1 = np.array([x[nt], y[nt], z[nt]])

tt = np.linspace(0, 2, 2000)

xt = p0[0] + (p1[0] - p0[0]) * tt
yt = p0[1] + (p1[1] - p0[1]) * tt
zt = p0[2] + (p1[2] - p0[2]) * tt

pt = np.vstack((xt, yt, zt))
pt = np.transpose(pt)
print(pt.shape)

rrt = np.sqrt(pt[:, 0] * pt[:, 0] + pt[:, 1] * pt[:, 1] + pt[:, 2] * pt[:, 2])
#-------------------------------------------------

if False:
  plt.figure(figsize = (6, 6))

  #----
  plt.scatter(x[nxclumpsX], y[nxclumpsX], s = 1, color = 'k')
  plt.plot(xt, yt, color = 'red') #----- Line of Sight ---------
  delta = 0.05
  plt.xlim(x[nt] - delta, x[nt]+delta)
  plt.ylim(y[nt] - delta, y[nt]+delta)
  plt.show()
  #----




