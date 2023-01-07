import numpy as np
import pickle
import matplotlib.pyplot as plt


with open('u_vs_t_with_cooling.pkl', 'rb') as f:
	res = pickle.load(f)

t1 = res[:, 0]
u1 = res[:, 1]/1e6

with open('./No_cooling/u_vs_t_without_cooling.pkl', 'rb') as f:
	res = pickle.load(f)

t2 = res[:, 0]
u2 = res[:, 1]/1e6

plt.scatter(t1, u1, s = 10, color = 'black')
plt.scatter(t2, u2, s = 10, color = 'blue')

#plt.ylim(0, 4)

#plt.yscale('log')

plt.show()




