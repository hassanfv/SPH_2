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

filz = np.sort(glob.glob('./Outputs_orig/*.csv'))

res = []

jj = 52405

for j in range(0, len(filz), 1):

	df = pd.read_csv(filz[j])

	t = float(filz[j].split('/')[-1][2:-4])
	
	rho = df['rho'].values
	u = df['u'].values
	
	#res.append([t, u[jj], rho[jj]])
	res.append([t, max(u), max(rho)])

res = np.array(res)

t = res[:, 0]
rho = res[:, 2]


with open('rho_j_with_cooling.pkl', 'wb') as f:
	pickle.dump({'t': t, 'rho': rho}, f)


plt.scatter(t, rho, s = 5, color = 'black')
plt.xlabel('t')
plt.ylabel('rho_j')

#plt.ylim(0, 4e5)

#plt.yscale('log')

plt.savefig('rho_j_vs_t.png')
plt.show()



