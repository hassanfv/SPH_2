import numpy as np
import pickle
import matplotlib.pyplot as plt


with open('rho_j_with_cooling.pkl', 'rb') as f:
	res = pickle.load(f)

t1 = res['t']
rho1 = res['rho']

with open('./No_cooling_dt_0.1_T/rho_j_Adiabatic.pkl', 'rb') as f:
	res = pickle.load(f)

t2 = res['t']
rho2 = res['rho']

plt.scatter(t1, rho1, s = 10, color = 'black', label = 'with cooling')
plt.scatter(t2, rho2, s = 10, color = 'blue', label = 'Adiabatic')

plt.yscale('log')

#plt.ylim(0, 4)

plt.legend()

plt.savefig('rho_comparison.png')

plt.show()




