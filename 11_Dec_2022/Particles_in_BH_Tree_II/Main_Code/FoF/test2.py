
import numpy as np
import matplotlib.pyplot as plt
import struct
import pickle

filename = 'G-0.015044.bin'

kB = 1.3807e-16
mu = 0.61 # !!!!!!!!!!!!!!!!!!!!!!!!!! You may want to calculate better mu using ionFrac or ...!
mH = 1.673534e-24
gamma = 5./3.

unit_velocity_cgs = 1.34181e+06 # cm/s #!!!!!!!!!!!!!!!!!!!!!!!!
unit_u = 1.80046e+12 #!!!!!!!!!!!!!!!!!!!!!!!!
unit_rho = 2.83261e-24 # !!!!!!!!!!!!!!!!!!!

def readBinaryFile(filename):
    with open(filename, 'rb') as f:
        file_content = f.read()  # Read the entire file at once

    # Use a memoryview to avoid copies
    buffer = memoryview(file_content)

    # Unpack N and N_ionFrac
    N, N_ionFrac = struct.unpack_from('ii', buffer)
    offset = 8  # Start after the first two integers

    # Function to read and advance the offset
    def read_array(dtype, size, itemsize):
        nonlocal offset
        array = np.frombuffer(buffer, dtype=dtype, count=size, offset=offset)
        offset += size * itemsize
        return array

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
    ionFrac = read_array(np.float32, N_ionFrac, 4)
    ngbDebug = read_array(np.int32, N, 4)

    return N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac, ngbDebug

N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac, ngbDebug = readBinaryFile(filename)


def isFriend(xj, yj, zj, xi, yi, zi, hj, hi):

  dist = ((xj - xi)*(xj - xi) + (yj - yi)*(yj - yi) + (zj - zi)*(zj - zi))**0.5
  
  return dist < min(hj, hi)



print('Typ == 0 ===> ', np.sum(Typ == 0))

n = np.where(u != 0.0)[0]

N = len(n)

rho = rho[n]
u = u[n]
h = h[n]
print(np.sort(h))

x = x[n]
y = y[n]
z = z[n]

rho_cgs = rho * unit_rho
XH = 0.7
nH = rho_cgs * XH / mH

T = (gamma - 1) * mH / kB * mu * u * unit_u

T_crit = 10000 # K
nH_crit = 100

nx = np.where((T < T_crit) & (nH > nH_crit))[0]
print()
print(f'We found {len(nx)} particles satisfying the criteria!')
print()

nH = nH[nx]

x = x[nx]
y = y[nx]
z = z[nx]

h = h[nx]
T = T[nx]

nmax_nH = np.where(nH >= 100)[0] # !!!!!!!!!!!!!!!!!!!!!! NOTE we have nH above!!!!!!!!!!!!!!!!!!!!
print(f'max(nH) = ', max(nH))
print()
#print(f'maximun nH occurs for particle {nmax_nH}')
#print()



# Assuming x, y, z, h, and T are 1D numpy arrays of the same length
# x, y, z are position coordinates, h is the smoothing length, and T is the temperature

def distance(i, j):
    """Calculate the Euclidean distance between two particles."""
    return ((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2) ** 0.5

def find_friends_of_friends(particle_index):
    """Find friends and friends-of-friends of a specific particle."""
    num_particles = len(x)
    friends = set()
    friends_of_friends = set()

    # Find immediate friends of the particle
    for j in range(num_particles):
        if j != particle_index and distance(particle_index, j) <= min(1*h[particle_index], 1*h[j]):
            friends.add(j)

    # Find friends of these friends
    for friend in friends:
        for k in range(num_particles):
            if k not in friends and k != particle_index and distance(friend, k) <= min(1*h[friend], 1*h[k]):
                friends_of_friends.add(k)

    return friends, friends_of_friends



i = 305 # Change this index to the particle you are interested in
friends, friends_of_friends = find_friends_of_friends(i)
print(f"particle {i} (particle {nx[i]} in the main array) has   {len(friends)} friends.")
nclump = np.array(list(friends))
nclump = [nx[k] for k in nclump]
with open('clumps.pkl', 'wb') as f:
  pickle.dump(nclump, f)

#s()


for i in nmax_nH:

  #i = 826 # Change this index to the particle you are interested in
  friends, friends_of_friends = find_friends_of_friends(i)

  if len(friends) > 50:
    print(f"particle {i} (particle {nx[i]} in the main array) has   {len(friends)} friends.")
    #print(f"Friends of particle {i}: {friends},  particle {i} has {len(friends)} friends.")
    #print(f"Friends of friends of particle {i}: {friends_of_friends}")



