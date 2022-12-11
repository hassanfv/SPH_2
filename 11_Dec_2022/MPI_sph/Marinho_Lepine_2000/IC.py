
import numpy as np
import matplotlib.pyplot as plt
import pickle
np.random.seed(42)


nPart = 5000

Rc = 10. # in code unit
# Note that the following calculation for r in "r = Rc * ksi1**0.5" line is for the asumption that M = 10 in code unit. See Marinho et al (2000; Appendix D.).

res = []

for i in range(nPart):

	ksi1, ksi2, ksi3 = np.random.random(3)

	r = Rc * ksi1**0.5
	theta = np.arccos(1. - 2. * ksi2)
	phi = 2. * np.pi * ksi3
	
	x = r * np.sin(theta) * np.cos(phi)
	y = r * np.sin(theta) * np.sin(phi)
	z = r * np.cos(theta)
	
	res.append([x, y, z])


res1 = np.array(res)
res2 = res1.copy()
res1[:, 0] = res1[:, 0] + 10. # shifting the cloud 10. unit to the right side (i.e. positive x).
res2[:, 0] = res2[:, 0] - 10. # shifting the cloud 10. unit to the left side (i.e. negative x).

res = np.vstack((res1, res2))

vel1 = np.zeros_like(res1)
vel2 = np.zeros_like(res2)
vel1[:, 0] = -5.0 # km/s !!!!!!!!!!!!!!!!!! WHY THIS IS in km/s !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
vel2[:, 0] = 5.0

vel = np.vstack((vel1, vel2))


dictx = {'r': res, 'v': vel}

with open('Marinho_IC_' + str(int(nPart)) + '.pkl', 'wb') as f:
	pickle.dump(dictx, f)


plt.figure(figsize = (8, 8))

plt.scatter(res[:, 0], res[:, 1], s = 0.05, color = 'black')

plt.xlim(-24, 24)
plt.ylim(-24, 24)

plt.show()	







