
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
import time


#unitTime_in_Myr =  0.07673840095663824 # Myr

M_sun = 1.98992e+33 # gram
UnitMass_in_g = 2000.0 * M_sun       # !!!!!!!!!!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!
UnitRadius_in_cm = 4.54 * 3.086e18  #!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3

print(f'UnitDensity_in_cgs = {UnitDensity_in_cgs} g/cm^3')


filz = np.sort(glob.glob('./Outputs_Me_19k_Mach_3/*.pkl'))

res = []

for j in range(0, len(filz), 1):  # 35.54 + 1 = 36.54

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
	t = data['current_t'] * unitTime / 3600. / 24. / 365.25 / 1e6 # in Myrs
	
	rho = np.sort(rho)*UnitDensity_in_cgs
	
	max_rho = np.max(rho)
	
	res.append([t, np.log10(max_rho)])

res = np.array(res)

t = res[:, 0]
rho = res[:, 1]

print(10**max(rho))

plt.scatter(t, rho, s = 10)

plt.xlim(0.0, 1.4)
plt.ylim(-21.2, -17.5)

plt.xlabel('Myrs')
plt.ylabel('max density')

plt.savefig('result.png')

plt.show()





