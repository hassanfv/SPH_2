
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
import time


#unitTime_in_Myr =  0.07673840095663824 # Myr

M_sun = 1.98992e+33 # gram
UnitMass_in_g = 1.0 * M_sun       # !!!!!!!!!!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!
UnitRadius_in_cm = 4.95e+16  #!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3

print(f'UnitDensity_in_cgs = {UnitDensity_in_cgs} g/cm^3')


rho_0 = 3.82e-18 # g/cm^-3


#filz = np.sort(glob.glob('./Outputs/*.pkl'))

#filz = np.sort(glob.glob('/mnt/Linux_Shared_Folder_2022/Outputs_60k_UA/*.pkl'))
filz = np.sort(glob.glob('./Outputs_60k_UB_h/Outputs/*.pkl'))

res = []

for j in range(0, len(filz), 10):  # 35.54 + 1 = 36.54

	print('j = ', j)

	with open(filz[j], 'rb') as f:
		data = pickle.load(f)


	r = data['pos']
	h = data['h']

	x = r[:, 0]
	y = r[:, 1]
	z = r[:, 2]
	rho = data['rho']
	unitTime = data['unitTime']
	t = data['current_t'] * unitTime
	t_ff = data['t_ff']
	
	rho = np.sort(rho)*UnitDensity_in_cgs
	
	max_rho = np.max(rho)
	
	res.append([(t/t_ff), np.log10(max_rho/rho_0)])

res = np.array(res)

t_tff = res[:, 0]
rho_rho0 = res[:, 1]

plt.scatter(t_tff, rho_rho0, s = 10)

plt.xlim(0.4, 1.5)
plt.ylim(0, 9)

plt.savefig('result.png')

plt.show()





