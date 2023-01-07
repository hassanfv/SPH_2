import numpy as np
import pickle
import matplotlib.pyplot as plt


with open('u_j_with_cooling.pkl', 'rb') as f:
	res = pickle.load(f)

t1 = res['t']
u1 = res['u']

with open('./No_cooling_dt_0.1_T/u_j_Adiabatic.pkl', 'rb') as f:
	res = pickle.load(f)

t2 = res['t']
u2 = res['u']

plt.scatter(t1, u1, s = 10, color = 'black', label = 'With cooling')
plt.scatter(t2, u2, s = 10, color = 'blue', label = 'Adiabatic')

#plt.ylim(0, 4)

#plt.yscale('log')

plt.legend()

plt.savefig('png')

plt.show()




