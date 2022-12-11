
import numpy as np
import matplotlib.pyplot as plt
import pickle


with open('Data_Turner.pkl', 'rb') as f:
	data = pickle.load(f)

r = pos = data['r']
h = data['h']

xx = pos[:, 0]
yy = pos[:, 1]
zz = pos[:, 2]

plt.scatter(xx, yy, s = 0.1)
plt.show()

res = []

thet = []

for i in range(len(xx)):

	rr = (xx[i]**2 + yy[i]**2 + zz[i]**2)**0.5

	theta =  (0.0 + 180. * np.random.random()) * np.pi/180. # the angle from z-axis
	phi = 0.0 + (360.-0.)*np.pi/180. * np.random.random()
	
	thet.append(phi*180./np.pi)

	x_dist = rr * np.cos(phi) * np.sin(theta)
	y_dist = rr * np.sin(phi) * np.sin(theta)
	z_dist = rr * np.cos(theta)
	
	res.append([x_dist, y_dist, z_dist])


thet = np.array(thet)
plt.hist(thet)
plt.show()

res = np.array(res)

x = res[:, 0]
y = res[:, 1]
z = res[:, 2]

print

plt.scatter(x, z, s = 0.1)
plt.show()








