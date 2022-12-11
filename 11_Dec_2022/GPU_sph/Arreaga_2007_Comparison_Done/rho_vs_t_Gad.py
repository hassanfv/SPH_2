
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import time

unitTime_in_Myrs = 0.0306585 # Myrs
UnitDensity_in_cgs = 1.6e-17
UnitRadius_in_cm =  4.99e+16
unitVelocity =  51575.68

df_info = pd.read_csv('/mnt/Linux_Shared_Folder_2022/GPU_sph/180k_Gad_Arreaga/infox.csv')
x7 = df_info['x7'].values
n7 = np.where(x7 == 180000)[0]

t_arr = df_info['x3'].values

res = []

TA = time.time()

for jj in range(0, 1650, 20):

	print(jj)

	nam = 'snap_' + str(jj).zfill(3) + '.hdf5'

	file = h5py.File('/mnt/Linux_Shared_Folder_2022/GPU_sph/180k_Gad_Arreaga/' + nam, 'r')

	t = t_arr[n7][jj]

	rho = np.array(list(file['PartType0']['Density'])) * UnitDensity_in_cgs
	
	res.append([t, np.max(rho)])

print('TA = ', time.time() - TA)
	
res = np.array(res)

t = res[:, 0]
rho = res[:, 1]

dictx = {'t': t, 'rho': rho}
with open('rho_vs_t_Gad.pkl', 'wb') as f:
	pickle.dump(dictx, f)

plt.scatter(t, np.log10(rho), s = 5, color = 'black')
plt.savefig('1111_Gad.png')
plt.show()



