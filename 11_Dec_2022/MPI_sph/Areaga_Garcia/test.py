
import numpy as np
import matplotlib.pyplot as plt
import pickle


with open('Boss_IC_10000.pkl', 'rb') as f:
	data = pickle.load(f)


r = data['r']
v = data['v']


plt.scatter(r[:, 0], r[:, 1], s = 1)
plt.show()


