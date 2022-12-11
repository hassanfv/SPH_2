
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import time
import readchar
from libsx import *


TA = time.time()

with open('Data_Turner.pkl', 'rb') as f:
	data = pickle.load(f)


r = data['r']

x = r[:, 0]
y = r[:, 1]
z = r[:, 2]

rr = (x*x + y*y + z*z)**0.5


N = r.shape[0]
m = 1.0 / N + np.zeros(N)

h = do_smoothingX((r, r))

rho = getDensity(r, m, h)


print('Elapsed time = ', time.time() - TA)


plt.scatter(rr, rho, s = 0.4, color = 'k')

plt.savefig('fig.png')

plt.show()


	
	




