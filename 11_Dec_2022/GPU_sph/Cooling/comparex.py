

import numpy as np
from photolibs2 import *
import matplotlib.pyplot as plt
import pandas as pd
import time

XH = 0.76
mH = 1.6726e-24 # gram

UnitTime_in_yrs = 500 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
dt_t  = UnitTime_in_yrs * 3600. * 24. * 365.24

nHcgs = 0.1 # cm^-3   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
rhot = nHcgs * mH / XH
print('nHcgs = ', nHcgs)

T = 50000. # K

uad = convert_Temp_to_u(T, nHcgs, XH)

print(f'uad = {uad:.3E}')

print('UnitTime_in_yrs = ', UnitTime_in_yrs)

ux = DoCooling_h(rhot, uad, dt_t, XH)

delta_u = uad - ux
print()
print('delta_u = ', delta_u)

print()
print(f'rho = {rhot:.4E}')

print()
print('u (before) = ', uad)
print('u (After) = ', ux)

print()
print()
print("*************************************")
print("********** GRID COOLING *************")
print("*************************************")
print()

ut = uad
rhot = rhot

dfG = pd.read_csv('sortedCoolingGrid_1k_1k.csv')
# ['u', 'rho', 'dt', 'delta_u']

uad = dfG['u_ad'].values
rho = dfG['rho'].values
dt = dfG['dt'].values
delta_u = dfG['delta_u'].values

N = len(uad)

i = 0

TT = time.time()

#while((i < N) & ((ut < uad[i]) | (ut > uad[i+1]) | (rhot < rho[i]) | (rhot > rho[i+1]))):

n_uad = []
for i in range(N-1):

	if (ut >= uad[i]) & (ut <= uad[i+1]):
		n_uad.append(i)


uad = uad[n_uad]
rho = rho[n_uad]
delta_u = delta_u[n_uad]


NN = len(uad)

i = 0
while ((i < NN-1) & ((rhot < rho[i]) | (rhot > rho[i+1]))):

	i += 1


print('Elapsed time = ', time.time() - TT)

nx = i - 1

print('nx = ', nx)
print()

print('RHO = ', rho[nx])

print('DELTA_U = ', delta_u[nx])
print()
print(f'u BEFORE = {uad[nx]}')
print(f'u AFTER = {uad[nx]-delta_u[nx]}')
	



