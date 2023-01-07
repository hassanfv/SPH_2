
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle


filz = np.sort(glob.glob('./Outputs/*.csv'))

res = []

jj = 70327

for nam in filz:

	df = pd.read_csv(nam)

	t = float(nam.split('/')[-1][2:-4])
	
	rho = df['rho'].values
	u = df['u'].values
	
	res.append([t, u[jj]])

res = np.array(res)

with open('u_j_vs_t_with_cooling.pkl', 'wb') as f:
	pickle.dump(res, f)

print(res)

t = res[:, 0]
u = res[:, 1]

plt.scatter(t, u, s = 20, color = 'black')
#plt.xlim(0.0, 0.0075)
#plt.ylim(2350.0, 2400)
plt.xlabel('t')
plt.ylabel('u')
plt.savefig('u_j_vs_t.png')
plt.show()







