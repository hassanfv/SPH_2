
import numpy as np
import pickle
import matplotlib.pyplot as plt


with open('coolHeatGridX.pkl', 'rb') as f:
    data = pickle.load(f)

nH = data['densities']
Z = data['metallicities']
Temp = data['temperatures'] # NOTE: This is log T
uEvol = data['uEvolution']
tArr_sec = data['timeArr_in_sec']
muA = data['muArr']
kpc = data['kpc']

print('kpc = ', kpc)
print()
print('nH = ', nH)
print()
print('Z = ', Z)
print()
print('Temp = ', Temp)
print()
print('uEvol.shape = ', uEvol.shape)
print('muA.shape = ', muA.shape)
print()

#--- test values ---
kB = 1.3807e-16
mu = 0.61
mH = 1.673534e-24
#unit_u = 1032418615683.733
gamma = 5./3.

# uEvol ===> (ndx_kpc, ndx_T, ndx_nH, ndx_Z, ndx_time)

N_kpc = len(kpc)
N_nH = len(nH)
N_Z = len(Z)
N_T = len(Temp)
N_Time = len(tArr_sec)

#print(f'N_kpc = {N_kpc},  N_nH = {N_nH}, N_Z = {N_Z}, N_T = {N_T}, N_Time = {N_Time}')


#----- Used for debugging !!!
def find_mu(T, u):

  return kB * T / (gamma - 1.0) / mH / u



def hcooler(r_p, u_p, nH_p, Z_p, dt_sec, nH, Z, Temp, uEvol, tArr_sec, N_nH, N_Z, N_T, N_Time, muA):

  ndx_kpc = -1
  ndx_nH = -1
  ndx_u = -1
  ndx_t = -1
  
  #====== kpc ========
  for i in range(N_kpc):
    if (ndx_kpc == -1) & (r_p <= kpc[i]):
      ndx_kpc = i
  
  if ndx_kpc != 0:
    delta1 = np.abs(kpc[ndx_kpc] - r_p)
    delta2 = np.abs(kpc[ndx_kpc - 1] - r_p)
    if delta2 < delta1:
      ndx_kpc -= 1

  print(f'r_p = {r_p},  ndx_kpc (XXXX) = ', ndx_kpc)

  #====== nH =========
  for i in range(N_nH):
      if (ndx_nH == -1) & (nH_p <= nH[i]): # This work because in the beginning nH_p is always less than or maybe equal to nH[i]! TRIVIAL !!
          ndx_nH = i

  if ndx_nH != 0:
    delta1 = np.abs(nH[ndx_nH] - nH_p)
    delta2 = np.abs(nH[ndx_nH - 1] - nH_p)
    if delta2 < delta1:
        ndx_nH -= 1
  
  print(f'nH_p = {nH_p},  ndx_nH (XXXX) = ', ndx_nH)

  #======= Z =========
  ndx_Z = 1 # Assuming [Z/H] -1
  
  print(f'Z_p = {Z_p},  ndx_Z (XXXX) = ', ndx_Z)

  #======== u =========
  tmp = uEvol[0, :, ndx_nH, ndx_Z, :]

  U = tmp[:, 0] # The list of all initial u, i.e. corresponding to T
  
  print(f'U = ', U)

  for i in range(N_T): # Note that N_T = len(U)!
      
      if (ndx_u == -1) & (u_p <= U[i]):
          ndx_u = i

  #if ndx_u != 0:
  #  delta1 = np.abs(U[ndx_u] - u_p)
  #  delta2 = np.abs(U[ndx_u - 1] - u_p)

  #  if delta2 < delta1:
  #      ndx_u -= 1
  
  '''
  for i, utt in enumerate(U):
  
    print(f'{i},  U = {utt:.4E}')
    
  print()
  print(f'u_p = {u_p:.4E},  ndx_u (XXXX) = ', ndx_u)
  '''

  #======== time ============
  for i in range(N_Time):
      
      if (ndx_t == -1) & (dt_sec <= tArr_sec[i]):
          ndx_t = i

  print(ndx_kpc, ndx_u, ndx_nH, ndx_Z, ndx_t)
  print(uEvol.shape, muA.shape)
  print()
  
  mu_tmp = muA[ndx_kpc, ndx_u, ndx_nH, ndx_Z, ndx_t]
  
  if ndx_u > 0:
    uEv_1 = uEvol[ndx_kpc, ndx_u-1, ndx_nH, ndx_Z, ndx_t]
    uEv_2 = uEvol[ndx_kpc, ndx_u, ndx_nH, ndx_Z, ndx_t]
    
    diff = U[ndx_u] - U[ndx_u - 1]
    
    fhi = (u_p - U[ndx_u - 1]) / diff
    flow = 1.0 - fhi
    
    print(f'uEv_1 = {uEv_1:.4E},  uEv_2 = {uEv_2:.4E},  u_p = {u_p:.4E}')
    print()
    print(f'fhi = {fhi},  flow = {flow}')
    print()
    print()
    
    uEv = flow * uEv_1 + fhi * uEv_2
  else:
    uEv = uEvol[ndx_kpc, ndx_u, ndx_nH, ndx_Z, ndx_t]

  return uEv, mu_tmp

unit_u = 1.5272E+13


ndx_kpc = 0
ndx_nH = 26
ndx_u = 18
ndx_Z = 1

u_test = uEvol[ndx_kpc, ndx_u, ndx_nH, ndx_Z, :] / unit_u
tArr_yrs = tArr_sec / 3600 / 24 / 365.25

#plt.scatter(tArr_yrs, u_test, s = 10, color = 'k')
#plt.show()

#--- test values ---
r_p = 0.8 # kpc
T_p = 10000.0
u_p = 1.0 / mu / (gamma - 1) * kB / mH * T_p
nH_p = 25.
Z_p = -1.0
dt_sec = 6.3176E+09 # seconds
#-------------------

print(f'HHHH u_p = {u_p:.3E}')


uu, mu = hcooler(r_p, u_p, nH_p, Z_p, dt_sec, nH, Z, Temp, uEvol, tArr_sec, N_nH, N_Z, N_T, N_Time, muA)

TBefore = (gamma - 1) * mH / kB * mu * u_p
TAfter = (gamma - 1) * mH / kB * mu * uu 

print()


print(f'u(Before) = {u_p:.4E}, u(After) = {uu:.4E},  mu = {mu:.3f}, T(After) = {TAfter:.2f}')
print()
print(f'u(Before code unit) = {u_p/unit_u:.2f}, u(After code unit) = {uu/unit_u:.2f}')
print()
print(f'T(Before) = {TBefore:.2f}, T(After) = {TAfter:.2f}')



