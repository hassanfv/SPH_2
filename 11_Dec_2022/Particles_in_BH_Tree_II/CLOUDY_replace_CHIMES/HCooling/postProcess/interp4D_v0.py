
import numpy as np
import matplotlib.pyplot as plt
import pickle


#===== interpolate_4d_hypercube
def interpolate_4d_hypercube(p, dx, dy, dz, dw):
    """
    Perform 4D interpolation on point p within a hypercube defined by the coordinates x, y, z, w.
    
    :param p: An array of shape (16,) representing the values at the 16 points of the hypercube
              following the binary encoding convention for indexing.
    :param dx: The x-coordinate for interpolation, normalized between 0 and 1.
    :param dy: The y-coordinate for interpolation, normalized between 0 and 1.
    :param dz: The z-coordinate for interpolation, normalized between 0 and 1.
    :param dw: The w-coordinate for interpolation, normalized between 0 and 1.
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


with open('Ready.pkl', 'rb') as f:
  dictx = pickle.load(f)

nH = dictx['nH']
T = dictx['T']
r = dictx['rkpc']
NH = dictx['NH']

Gam = dictx['Gam'] # heating
Lam = dictx['Lam'] # cooling

N_nH = len(nH)
N_T = len(T)
N_r = len(r)
N_NH = len(NH)

Ntot = N_nH * N_T * N_r * N_NH
print()
print(f'Number of CLOUDY models to be constructed = {Ntot}')
print()
#-----------------------------


nH_p = 3.0
T_p = 3.7
r_p = 0.3
NH_p = 21.8


#--- nH
nxnH0 = -1
for j in range(N_nH):
  if (nxnH0 == -1) & (nH_p <= nH[j]):
    nxnH0 = j

if nxnH0 > 0: # To handle cases in which nH_p <= min(nH)
  nxnH0 = nxnH0 - 1

if nxnH0 == -1: # To handle cases in which nH_p >= max(nH)
  nxnH0 = N_nH - 2 # 1 will be added in nxnH1!

#--- T
nxT0 = -1
for j in range(N_T):
  if (nxT0 == -1) & (T_p <= T[j]):
    nxT0 = j

if nxT0 > 0: # To handle cases in which T_p <= min(T)
  nxT0 = nxT0 - 1

if nxT0 == -1: # To handle cases in which T_p >= max(T)
  nxT0 = N_T - 2

#--- rkpc
nxr0 = -1
for j in range(N_r):
  if (nxr0 == -1) & (r_p <= r[j]):
    nxr0 = j

if nxr0 > 0: # To handle cases in which r_p <= min(r)
  nxr0 = nxr0 - 1

if nxr0 == -1: # To handle cases in which r_p >= max(r)
  nxr0 = N_r - 2

#--- NH
nxNH0 = -1
for j in range(N_NH):
  if (nxNH0 == -1) & (NH_p <= NH[j]):
    nxNH0 = j

if nxNH0 > 0: # To handle cases in which NH_p < min(NH)
  nxNH0 = nxNH0 - 1

if nxNH0 == -1: # To handle cases in which NH_p > max(NH)
  nxNH0 = N_NH - 2


nxnH1 = nxnH0 + 1
nxT1  = nxT0 + 1
nxr1  = nxr0 + 1
nxNH1 = nxNH0 + 1


print()
print(f'nH_p = {nH_p},   nH_low = {nH[nxnH0]},   nH_up = {nH[nxnH1]}')
print(f'T_p = {T_p},   T_low = {T[nxT0]},   T_up = {T[nxT1]}')
print(f'r_p = {r_p},   r_low = {r[nxr0]},   r_up = {r[nxr1]}')
print(f'NH_p = {NH_p},   NH_low = {NH[nxNH0]},   NH_up = {NH[nxNH1]}')
print()

S1 = N_T * N_r * N_NH
S2 = N_r * N_NH
S3 = N_NH

PG = np.zeros(16)
PL = np.zeros(16)
s = 0
for i in [nxnH0, nxnH1]:
  for j in [nxT0, nxT1]:
    for k in [nxr0, nxr1]:
      for l in [nxNH0, nxNH1]:
        Lnx = i * S1 + j * S2 + k * S3 + l
        PG[s] = Gam[Lnx] # heating!
        PL[s] = Lam[Lnx] # cooling!
        s += 1

print()
print('PG = ', PG)
print()
print('PL = ', PL)
print()

dx = (nH_p - nH[nxnH0]) / (nH[nxnH1] - nH[nxnH0])
dy = (T_p - T[nxT0]) / (T[nxT1] - T[nxT0])
dz = (r_p - r[nxr0]) / (r[nxr1] - r[nxr0])
dw = (NH_p - NH[nxNH0]) / (NH[nxNH1] - NH[nxNH0])

print('dx, dy, dz, dw = ', dx, dy, dz, dw)
print('nH = ', nH_p, nH[nxnH0], nH[nxnH1], nH[nxnH0])
print('T = ', T_p, T[nxT0], T[nxT1], T[nxT0])
print('r = ', r_p, r[nxr0], r[nxr1], r[nxr0])
print('NH = ', NH_p, NH[nxNH0], NH[nxNH1], NH[nxNH0])
print()
print('nxnH0 = ', nxnH0)
print('nxT0 = ', nxT0)
print('nxr0 = ', nxr0)
print('nxNH0 = ', nxNH0)

print()
print(f'nH_p = {nH_p},  T_p = {T_p},  r = {r_p},  NH = {NH_p}')
print()

Gam_val = interpolate_4d_hypercube(PG, dx, dy, dz, dw)
Lam_val = interpolate_4d_hypercube(PL, dx, dy, dz, dw)

print(f'Gam_val (interpolated) = {Gam_val:.5f}')
print(f'Lam_val (interpolated) = {Lam_val:.5f}')







