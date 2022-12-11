
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
import time
import h5py


#unitTime_in_Myr =  0.07673840095663824 # Myr

#M_sun = 1.98992e+33 # gram
#UnitMass_in_g = 10.0 * M_sun       # !!!!!!!!!!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!
#UnitRadius_in_cm = 0.22 * 3.086e18  #!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!
#UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3

#print(f'UnitDensity_in_cgs = {UnitDensity_in_cgs} g/cm^3')


unitTime_in_Myrs = 0.48886 # Myrs
UnitDensity_in_cgs = 6.296e-20


filz = np.sort(glob.glob('./output_Gad_65k_eps_0.001/*.hdf5'))

res = []

for j in range(0, len(filz), 1):  # 35.54 + 1 = 36.54

	print('j = ', j)

	fil = h5py.File(filz[j], 'r')

	rho = np.array(list(fil['PartType0']['Density']))

	tStep = 0.005 # From Gadget parameter file.
	t = j * tStep * unitTime_in_Myrs
	
	rho = np.sort(rho)*UnitDensity_in_cgs
	
	max_rho = np.max(rho)
	
	res.append([t, np.log10(max_rho)])

res = np.array(res)

t = res[:, 0]
rho = res[:, 1]

print(10**max(rho))

plt.scatter(t, rho, s = 10)

plt.xlim(0.0, 0.37)
plt.ylim(-19, -14)

plt.xlabel('Myrs')
plt.ylabel('max density')

plt.savefig('result.png')

plt.show()





