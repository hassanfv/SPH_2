import numpy as np
import pickle
import matplotlib.pyplot as plt


with open('rho_vs_t_with_cooling.pkl', 'rb') as f:
	res = pickle.load(f)

t1 = res[:, 0]
rho1 = res[:, 1]

with open('./No_cooling/rho_vs_t_without_cooling.pkl', 'rb') as f:
	res = pickle.load(f)

t2 = res[:, 0]
rho2 = res[:, 1]

plt.scatter(t1, rho1, s = 10, color = 'black')
plt.scatter(t2, rho2, s = 10, color = 'blue')

plt.yscale('log')

#plt.ylim(0, 4)

plt.show()




