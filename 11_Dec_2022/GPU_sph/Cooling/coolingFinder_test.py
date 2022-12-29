
import pandas as pd
import numpy as np
import time

df = pd.read_csv('sortedCoolingGrid_1k_1k.csv')

# ['u_ad', 'rho', 'dt', 'delta_u']

uad = df['u_ad'].values
rho = df['rho'].values
dt = df['dt'].values
delta_u = df['delta_u'].values

ux = uad - delta_u

ut = 1.1888E+13
rhot = 9.0333E-23

# SET STH TO HANDLE ut and rhot BIGGER THAN THE MIN and MAX values.

uGrid = np.unique(uad)
rhoGrid = np.unique(rho)


TA = time.time()

#------ Getting u index --------
for i in range(len(uGrid)-1):

	if (ut >= uGrid[i]) & (ut <= uGrid[i+1]):
		
		nx_u = i
		break

#------ Getting rho index --------
for i in range(len(rhoGrid)-1):

	if (rhot >= rhoGrid[i]) & (rhot <= rhoGrid[i+1]):
		
		nx_rho = i
		break

print('Elapsed time = ', time.time() - TA)

nxx = nx_u * 1000 + nx_rho

print(f'uad = {uad[nxx]:.4E}')
print(f'rho = {rho[nxx]:.4E}')
print(f'ux = {ux[nxx]:.4E}')
print(f'delta_u = {delta_u[nxx]:.4E}')






