
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar


with open('shocktube_ce_outputX.pkl', 'rb') as f:
	df = pickle.load(f)
	
x = df['x']
P = df['p']
u = df['e']
v = df['v']
rho = df['rho']



#plt.scatter(x, u, s = 1, color = 'black')
plt.scatter(x, v, s = 1, color = 'black')
#plt.xlim(-0.4, 0.6)
#plt.ylim(0.0, 1.1)

plt.show()

