
import numpy as np
import matplotlib.pyplot as plt
import pickle


with open('Main_IC_Grid_17k.pkl', 'rb') as f:
	data = pickle.load(f)

r = data['r']
rho = data['rho']

d = (r[:, 0]*r[:, 0] + r[:, 1]*r[:, 1] + r[:, 2]*r[:, 2])**0.5

plt.scatter(d, rho, s = 0.1, color = 'k')

plt.show()



