
import pandas as pd
import numpy as np
import time

# The difference with v2 is that here we handle the cases in which u decreases by more than 50% in a single time step.

#------- applyCooling ------- (# Decrease in u by more than 50% is not allowed !)
def applyCooling2(uad, rho, delta_u, ref_dt_cgs, uGrid, rhoGrid, MIN_uad, MAX_uad, MIN_rho, MAX_rho, ut_cgs, rhot_cgs, current_dt_cgs):
	
	coeff = current_dt_cgs / ref_dt_cgs
	
	if ut_cgs < MIN_uad:
		ut_cgs = MIN_uad
	
	elif ut_cgs > MAX_uad:
		ut_cgs = MAX_uad

	elif rhot_cgs < MIN_rho:
		rhot_cgs = MIN_rho
	
	elif rhot_cgs > MAX_rho:
		rhot_cgs = MAX_rho

	#--- Getting u index
	for i in range(len(uGrid)-1):

		if (ut_cgs >= uGrid[i]) & (ut_cgs <= uGrid[i+1]):
			
			nx_u = i
			break

	#--- Getting rho index
	for j in range(len(rhoGrid)-1):

		if (rhot_cgs >= rhoGrid[j]) & (rhot_cgs <= rhoGrid[j+1]):
			
			nx_rho = j
			break

	nxx = nx_u * 1000 + nx_rho
	
	uAf = uad[nxx] - coeff * delta_u[nxx]
	
	

	return uAf




df = pd.read_csv('sortedCoolingGrid_J_0.00001.csv', header = None)

print(df.head())

df.columns = ['u_ad', 'rho', 'dt', 'delta_u']


uad = df['u_ad'].values
rho = df['rho'].values
dt = df['dt'].values
ref_dt_cgs = dt[0]
delta_u = df['delta_u'].values

uGrid = np.unique(uad)
rhoGrid = np.unique(rho)

MIN_uad = uGrid[0]
MAX_uad = uGrid[-1]
MIN_rho = rhoGrid[0]
MAX_rho = rhoGrid[-1]


if True:
	#Unit_u_in_cgs = 4.30125e+08
	#Unit_P_in_cgs = 2.91088e-15
	#unitVelocity = 20739.5
	#unitTime_in_s = 1.48799e+15

	u_Before = 5964.69
	rho = 0.227516
	UnitDensity_in_cgs = 6.76752e-24
	Unit_u_in_cgs = 4.30125e+08
	ref_dt_cgs = 6.31135e+09
	current_dt_cgs = 2.97597e+11
	#u_After = -29778.5


ut_cgs = 5964.69 * Unit_u_in_cgs
rhot_cgs = 0.227516 * UnitDensity_in_cgs

current_dt_cgs = current_dt_cgs

TA = time.time()

u_after_cooling = applyCooling(uad, rho, delta_u, ref_dt_cgs, uGrid, rhoGrid, MIN_uad, MAX_uad, MIN_rho, MAX_rho, ut_cgs, rhot_cgs, current_dt_cgs)

print('Elapsed time = ', time.time() - TA)

print()
print(f'rhot_cgs = {rhot_cgs:.4E}')
print()
print(f'uad = {ut_cgs:.4E}')
print(f'uad = {u_after_cooling:.4E}')
print()

print(f'uad = {ut_cgs/Unit_u_in_cgs:.4E}')
print(f'uad = {u_after_cooling/Unit_u_in_cgs:.4E}')




