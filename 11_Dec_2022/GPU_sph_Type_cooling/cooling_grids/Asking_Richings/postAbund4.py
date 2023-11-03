
import numpy as np
import pickle
import matplotlib.pyplot as plt


with open('coolHeatGrid.pkl', 'rb') as f:
    data = pickle.load(f)

nH = data['densities']
Z = data['metallicities']
Temp = data['temperatures']
uEvol = data['uEvolution']
tArr_sec = data['timeArr_in_sec']
muA = data['muArr']

gamma = 5./3.
kB = 1.3807e-16
mH = 1.6726e-24

print('nH = ', nH)
print()
print('Z = ', Z)
print()
print(Temp)
print()
print('tArr_sec = ', tArr_sec)
print()

ndx_nH = 2
ndx_Z = 1
ndx_T = 2

print(f'Temp[ndx_T] = {Temp[ndx_T]},  nH[ndx_nH] = {nH[ndx_nH]},  Z[ndx_Z] = {Z[ndx_Z]}')

# uEvol ===> (ndx_T, ndx_nH, ndx_Z, ndx_time)


print(muA.shape, uEvol.shape)

ux = uEvol[ndx_T, ndx_nH, ndx_Z, :]
mux = muA[ndx_T, ndx_nH, ndx_Z, :]

Tx = mux * (gamma - 1) * mH / kB * ux

print(f'Tx = {Tx}')
print()

t_Myr = tArr_sec / 3600 / 24 / 365.25 / 1e6

plt.plot(t_Myr, Tx)

plt.xlabel('Time [Myrs]')
plt.ylabel('Temperature [Kelvin]')

plt.savefig('r_0.2kpc.jpg')

plt.show()

s()


N_nH = len(nH)
N_Z = len(Z)
N_T = len(Temp)
N_Time = len(tArr_sec)

#print(f'N_nH = {N_nH}, N_Z = {N_Z}, N_T = {N_T}, N_Time = {N_Time}')


def hcooler(u_p, nH_p, Z_p, dt_sec, nH, Z, Temp, uEvol, tArr_sec, N_nH, N_Z, N_T, N_Time, muA):

  ndx_nH = -1
  ndx_u = -1
  ndx_t = -1

  #====== nH =========
  for i in range(N_nH):
      if (ndx_nH == -1) & (nH_p <= nH[i]): # This work because in the beginning nH_p is always less than or maybe equal to nH[i]! TRIVIAL !!
          ndx_nH = i

  if ndx_nH != 0:
    delta1 = np.abs(nH[ndx_nH] - nH_p)
    delta2 = np.abs(nH[ndx_nH - 1] - nH_p)
    if delta2 < delta1:
        ndx_nH -= 1

  #======= Z =========
  ndx_Z = 1 # Assuming [Z/H] -1

  #======== u =========
  tmp = uEvol[:, ndx_nH, ndx_Z, :]

  U = tmp[:, 0] # The list of all initial u, i.e. corresponding to T

  for i in range(N_T): # Note that N_T = len(U)!
      
      if (ndx_u == -1) & (u_p <= U[i]):
          ndx_u = i

  if ndx_u != 0:
    delta1 = np.abs(U[ndx_u] - u_p)
    delta2 = np.abs(U[ndx_u - 1] - u_p)

    if delta2 < delta1:
        ndx_u -= 1

  #======== time ============
  for i in range(N_Time):
      
      if (ndx_t == -1) & (dt_sec <= tArr_sec[i]):
          ndx_t = i

  print(ndx_u, ndx_nH, ndx_Z, ndx_t)
  print(uEvol.shape, muA.shape)
  print()

  return uEvol[ndx_u, ndx_nH, ndx_Z, ndx_t], muA[ndx_u, ndx_nH, ndx_Z, ndx_t]


#--- test values ---
u_p = 1.0e14
nH_p = 190.
Z_p = -1.0
dt_sec = 1.0e11 # seconds
#-------------------

uu, mu = hcooler(u_p, nH_p, Z_p, dt_sec, nH, Z, Temp, uEvol, tArr_sec, N_nH, N_Z, N_T, N_Time, muA)

print(f'u (Before) = {u_p:.4E}, u (After) = {uu:.4E},  mu = {mu}')



