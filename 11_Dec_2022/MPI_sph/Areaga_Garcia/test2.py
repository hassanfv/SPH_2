
import numpy as np
import matplotlib.pyplot as plt
import pickle


with open('Data_Turner.pkl', 'rb') as f:
	data = pickle.load(f)


r = data['r']
h = data['h']


plt.scatter(r[:, 0], r[:, 1], s = 1)
plt.show()


