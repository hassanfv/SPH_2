
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



def hcooler(r_p, u_p, nH_p, Z_p, dt_sec):

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
    
    uEv = flow * uEv_1 + fhi * uEv_2
  else:
    uEv = uEvol[ndx_kpc, ndx_u, ndx_nH, ndx_Z, ndx_t]

  return uEv, mu_tmp





