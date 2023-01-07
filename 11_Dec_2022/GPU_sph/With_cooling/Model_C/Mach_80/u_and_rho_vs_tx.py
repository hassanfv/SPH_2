
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob


filz = np.sort(glob.glob('./Outputs/*.csv'))

res = []

for nam in filz:

	df = pd.read_csv(nam)

	t = float(nam.split('/')[-1][2:-4])
	
	rho = df['rho'].values
	u = df['u'].values
	
	res.append([t, max(u), max(rho)])

res = np.array(res)

print(res)

t = res[:, 0]
u = res[:, 1]
rho = res[:, 2]

plt.scatter(t, u/max(u), s = 20, color = 'black', label = 'u')
plt.scatter(t, rho/max(rho), s = 20, color = 'blue', label = 'rho')
#plt.xlim(0.0, 0.0075)
#plt.ylim(2350.0, 2400)
plt.xlabel('t')
plt.ylabel('u and rho')
plt.savefig('u_and_rho_vs_t.png')
plt.legend()
plt.show()







