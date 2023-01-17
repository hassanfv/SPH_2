import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle


UnitMass_in_g = 7.3150687572e+35
UnitRadius_in_cm = 9.258e+19
grav_const_in_cgs = 6.6738e-08


unitTime_in_s = np.sqrt(UnitRadius_in_cm**3 / grav_const_in_cgs / UnitMass_in_g);
ref_dt_cgs = 200.0 * 365.24 * 24.0 * 3600.0 # i.e 200 years.
MAX_dt_code_unit = ref_dt_cgs / unitTime_in_s;
dt = MAX_dt_code_unit


UnitDensity_in_cgsT = UnitMass_in_g / UnitRadius_in_cm**3
Unit_u_in_cgsT = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm;

filz = np.sort(glob.glob('/mnt/Linux_Shared_Folder_2022/check_Outputs_delete/*.csv'))

res = []

jj = 77375

for j in range(0, len(filz), 10):

	df = pd.read_csv(filz[j])

	t = float(filz[j].split('/')[-1][2:-4])
	
	rho = df['rho'].values
	u = df['u'].values
	upre = df['uprevious'].values
	dudt = df['dudt'].values
	uBeforeCool = df['uBeforeCooling'].values
	
	res.append([t, uBeforeCool[jj], u[jj], upre[jj], rho[jj], dudt[jj]])
	#res.append([t, max(u), max(upre), rho[jj], dudt[jj]])

res = np.array(res)

#print(res)

t = res[:, 0]
uBeforeCool = res[:, 1]
u = res[:, 2]
upre = res[:, 3]
rho = res[:, 4]
ut = res[:, 5]

for k in range(len(u)):

	upreX = upre[k] * Unit_u_in_cgsT
	uX = u[k] * Unit_u_in_cgsT
	rhoX = rho[k] * UnitDensity_in_cgsT
	dU = ut[k] * dt * Unit_u_in_cgsT
	uBeforeCoolX = uBeforeCool[k] * Unit_u_in_cgsT
	
	print(f'{t[k]:.4f}, {uBeforeCoolX:.4E}, {uX:.4E}, {upreX:.4E}, {rhoX:.4E}, {dU:.4E}, {uX - uBeforeCoolX:.4E}, {(uX - uBeforeCoolX)/uBeforeCoolX:.4E}')
	#print(f'{upreX}, {uX}') #, {rhoX:.4E}, {dU:.4E}, {uX - upreX:.4E}')


plt.scatter(t, u, s = 5, color = 'black')
#plt.scatter(t, upre, s = 5, color = 'blue')
plt.scatter(t, uBeforeCool, s = 10, color = 'red')
plt.xlabel('t')
plt.ylabel('u_j')

#plt.ylim(0, 4e5)

#plt.yscale('log')

plt.savefig('u_j_vs_t.png')
plt.show()



