
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import time
import pandas as pd
import numpy as np


unitTime_in_Myr = 0.03066 # Myr
UnitDensity_in_cgs = 1.6e-17

filz = np.sort(glob.glob('/mnt/Linux_Shared_Folder_2022/GPU_sph/Arreaga/180k_eps_0.001/*.csv'))

res = []

for j in range(0, len(filz), 10):

	print('j = ', j)

	df = pd.read_csv(filz[j])

	x = df['x'].values
	y = df['y'].values
	z = df['z'].values
	
	h = df['h'].values
	rho = df['rho'].values
	
	t = float(filz[j].split('/')[-1][2:-4])
	
	rho = rho * UnitDensity_in_cgs
		
	#print('rho = ', np.sort(rho))
	
	res.append([t, np.max(rho)])

res = np.array(res)

t = res[:, 0]
rho = res[:, 1]

dictx = {'t': t, 'rho': rho}
with open('rho_vs_t_hfv.pkl', 'wb') as f:
	pickle.dump(dictx, f)

plt.scatter(t, np.log10(rho), s = 5, color = 'black')
plt.savefig('1111.png')
plt.show()







