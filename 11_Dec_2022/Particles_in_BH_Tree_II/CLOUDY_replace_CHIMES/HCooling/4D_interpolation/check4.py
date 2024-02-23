
import numpy as np
import matplotlib.pyplot as plt


#===== find_closest_index
def find_closest_index(X, a):

  differences = np.abs(X - a)
  closest_index = np.argmin(differences)
  
  return closest_index


#===== interpolate_4d_hypercube
def interpolate_4d_hypercube(p, dx, dy, dz, dw):
    """
    Perform 4D interpolation on point p within a hypercube defined by the coordinates x, y, z, w.
    
    :param p: An array of shape (16,) representing the values at the 16 points of the hypercube
              following the binary encoding convention for indexing.
    :param x: The x-coordinate for interpolation, normalized between 0 and 1.
    :param y: The y-coordinate for interpolation, normalized between 0 and 1.
    :param z: The z-coordinate for interpolation, normalized between 0 and 1.
    :param w: The w-coordinate for interpolation, normalized between 0 and 1.
    :return: The interpolated value.
    """
    
    # Interpolate along the x-axis
    c00 = p[0] * (1 - dx) + p[8] * dx
    c01 = p[1] * (1 - dx) + p[9] * dx
    c10 = p[2] * (1 - dx) + p[10] * dx
    c11 = p[3] * (1 - dx) + p[11] * dx
    c20 = p[4] * (1 - dx) + p[12] * dx
    c21 = p[5] * (1 - dx) + p[13] * dx
    c30 = p[6] * (1 - dx) + p[14] * dx
    c31 = p[7] * (1 - dx) + p[15] * dx

    # Correct interpolation along the y-axis
    c0 = c00 * (1 - dy) + c10 * dy
    c1 = c01 * (1 - dy) + c11 * dy
    c2 = c20 * (1 - dy) + c30 * dy
    c3 = c21 * (1 - dy) + c31 * dy

    # Interpolate along the z-axis
    c = c0 * (1 - dz) + c2 * dz
    d = c1 * (1 - dz) + c3 * dz

    # Finally, interpolate along the w-axis
    c_final = c * (1 - dw) + d * dw

    return c_final



nH = np.arange(0, 6.1, 0.1)
T = np.arange(2, 11.1, 0.1)
r = np.arange(0.0, 1.1, 0.1)
NH= np.arange(16, 23.1, 0.2)
N_nH = len(nH)
N_T = len(T)
N_r = len(r)
N_NH = len(NH)

Ntot = N_nH * N_T * N_r * N_NH

print()
print(f'Number of CLOUDY models to be constructed = {Ntot}')
print()

#---- Constructing u so that we have something to test!!
u = np.zeros((N_nH, N_T, N_r, N_NH))
for i in range(len(nH)):
  for j in range(len(T)):
    for k in range(len(r)):
      for l in range(len(NH)):
        u[i, j, k, l] = nH[i]**0.5 + T[j]**0.5 + r[k]**0.5 + NH[l]**0.2

#print('sort(u) = ', np.sort(u))
#plt.hist(u, bins = 30)
#plt.show()
#-----------------------------


nH_p = 0.52
T_p = 3.47
r_p = 0.043
NH_p = 18.35

nxnH0 = -1
for j in range(N_nH):
  if (nxnH0 == -1) & (nH_p <= nH[j]):
    nxnH0 = j - 1

if nxnH0 == -1:
  nxnH0 = N_nH - 1

nxT0 = -1
for j in range(N_T):
  if (nxT0 == -1) & (T_p <= T[j]):
    nxT0 = j - 1

if nxT0 == -1:
  nxT0 = N_T - 1

nxr0 = -1
for j in range(N_r):
  if (nxr0 == -1) & (r_p <= r[j]):
    nxr0 = j - 1

if nxr0 == -1:
  nxr0 = N_r - 1

nxNH0 = -1
for j in range(N_NH):
  if (nxNH0 == -1) & (NH_p <= NH[j]):
    nxNH0 = j - 1

if nxNH0 == -1:
  nxNH0 = N_NH - 1


nxnH1 = nxnH0 + 1
nxT1  = nxT0 + 1
nxr1  = nxr0 + 1
nxNH1 = nxNH0 + 1


print(u.shape)

print()
print(nxnH0, nxT0, nxr0, nxNH0)

P = np.zeros(16)
s = 0
for i in [nxnH0, nxnH1]:
  for j in [nxT0, nxT1]:
    for k in [nxr0, nxr1]:
      for l in [nxNH0, nxNH1]:
        #print(i, j, k, l)
        P[s] = u[i, j, k, l]
        s += 1


print()
print(P)
print()

dx = (nH_p - nH[nxnH0]) / (nH[nxnH1] - nH[nxnH0])
dy = (T_p - T[nxT0]) / (T[nxT1] - T[nxT0])
dz = (r_p - r[nxr0]) / (r[nxr1] - r[nxr0])
dw = (NH_p - NH[nxNH0]) / (NH[nxNH1] - NH[nxNH0])

print(dx, dy, dz, dw)
print()

u_val = interpolate_4d_hypercube(P, dx, dy, dz, dw)

u_real = nH_p**0.5 + T_p**0.5 + r_p**0.5 + NH_p**0.2

print(f'u_val (interpolated) = {u_val:.5f},   u_val (real) = {u_real:.5f},  uncertainty = {((np.abs(u_val-u_real))/(u_val+u_real)):.5f}')







