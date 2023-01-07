

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





uad = 2.4789E+13
rhot = 2.4968E-22
delta = 2.2955E+13
uxx = uad - delta
print('------------------')
print(f'uxx = {uxx:.4E}')
print('uxx/uad = ', uxx/uad)
print('------------------')




ux = DoCooling_h(rhot, uad, dt_t, XH)

delta_u = uad - ux
print()
print(f'delta_u = {delta_u:.4E}')

print()
print(f'rho = {rhot:.4E}')

print()
print(f'u (before) = {uad:.4E}')
print(f'u (After) = {ux:.4E}')
print()
print(f'u (before) = {uad/uad}')
print(f'u (After) = {ux/uad}')

s()

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


for i in range(N-1):

	print(f'MM = , {ut:.4E}, {uad[i]:.4E}, {uad[i+1]:.4E}')
	if (ut >= uad[i]) & (ut <= uad[i+1]):
		print('HH = ', rhot, rho[i], rho[i+1])
		if (rhot >= rho[i]) & (rhot <= rho[i+1]):
			nx = i
			break

print('Elapsed time = ', time.time() - TT)

print('nx = ', nx)
print()

print('RHO = ', rho[nx])

print('DELTA_U = ', delta_u[nx])
print()
print(f'u BEFORE = {uad[nx]}')
print(f'u AFTER = {uad[nx]-delta_u[nx]}')
	



