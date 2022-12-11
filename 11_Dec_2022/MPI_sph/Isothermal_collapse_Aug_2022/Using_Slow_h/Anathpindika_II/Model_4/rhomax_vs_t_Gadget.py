
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
import time
import h5py

#!!!! FOR NEW IC CHECK AND MODIFY IF REQUIRED !!!!!!!!

unitTime_in_Myrs = 3.22 # Myrs  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
UnitDensity_in_cgs = 1.44e-21 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

tStep = 0.005 # From Gadget parameter file. #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


filz = np.sort(glob.glob('./output_Gad_19k/*.hdf5'))

res = []

for jj in range(0, len(filz), 2):  # 35.54 + 1 = 36.54

	print('j = ', jj)

	file = h5py.File(filz[jj], 'r')

	rho = list(file['PartType0']['Density'])
	rho = np.sort(rho) * UnitDensity_in_cgs
	
	max_rho = np.max(rho)
	
	t = jj * tStep * unitTime_in_Myrs
	print(f'currentTime = {round(t, 3)} Myrs.')
	print()
	
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





