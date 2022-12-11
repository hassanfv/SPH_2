
# Solution of the isothermal Lane-Emden equation.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time
from libsx import *
np.random.random(42)
import pickle

# See: https://scicomp.stackexchange.com/questions/2424/solving-non-linear-singular-ode-with-scipy-odeint-odepack

def dSdx(x, S):
	
	y1, y2 = S
	
	return [y2, -2./x * y2 + np.exp(-y1)]


y1_0 = 0.
y2_0 = 0.
S_0 = (y1_0, y2_0)


x = np.linspace(.00001, 10., 10000)

sol = odeint(dSdx, y0 = S_0, t = x, tfirst = True)

y1_sol = sol.T[0]
y2_sol = sol.T[1]


#----- mu_from_ksi
def mu_from_ksi(x, y2_sol, ksi): #y2_sol is d_psi/d_ksi

	# finding the closest value
	x1 = x - ksi
	nx = np.where(x1 > 0.)[0]
	
	return ksi * ksi * y2_sol[nx[0] - 1]


#----- ksi_from_mu
def ksi_from_mu(x, y2_sol, mu):

	mu1 = x * x * y2_sol - mu
	nx = np.where(mu1 > 0.)[0]
	
	return x[nx[0]-1]


TA = time.time()

ksi_B = 3.

nPart = 1500 # number of gas particles.

res = []

for j in range(nPart):

	mu_ksi_B = mu_from_ksi(x, y2_sol, ksi_B) # Note that x here is the ksi array from above !!! 
	
	Rand_r = np.random.random()
	mu_ksi = mu_ksi_B * Rand_r
	ksi = ksi_from_mu(x, y2_sol, mu_ksi)
	
	r =  ksi
	
	Rand_theta = np.random.random()
	cos_theta = 1. - 2.*Rand_theta
	theta = np.arccos(cos_theta)
	
	Rand_phi = np.random.random()
	phi = 2. * np.pi * Rand_phi
	
	# Now that we have (r, theta, phi), we convert them to (x, y, z)
	xx = r * np.sin(theta) * np.cos(phi)
	yy = r * np.sin(theta) * np.sin(phi)
	zz = r * np.cos(theta)
	
	res.append([xx, yy, zz])


res = np.array(res)
x = res[:, 0]
y = res[:, 1]
z = res[:, 2]


Th1 = time.time()
#-------- h (initial) -------
h = do_smoothingX((res, res))  # This plays the role of the initial h so that the code can start !
#----------------------------
print('Th1 = ', time.time() - Th1)

print('h = ', np.sort(h))

hB = np.median(h)
print('hB = ', hB)


rcloud_2 = res.copy()
rcloud_2[:, 0] += (2.*3.0 + 2.*hB)

r_1_2 = np.vstack((res, rcloud_2))

print(r_1_2.shape)

x12 = r_1_2[:, 0]
y12 = r_1_2[:, 1]
z12 = r_1_2[:, 2]


#---- Speed of Sound ------
mH = 1.6726e-24 # gram
kB = 1.3807e-16  # cm2 g s-2 K-1
T_0 = 54. # K, see Table_1 in Anathpindika - 2009

# Note that for pure molecular hydrogen mu=2. For molecular gas with ~10% He by mass and trace metals, mu ~ 2.7 is often used.
muu = 2.7
mH2 = muu * mH

c_0 = (kB * T_0 / mH2)**0.5

print('Sound speed = ', c_0)
#--------------------------

Mach = 25.
vel_ref = Mach * c_0 # The speed of each cloud. Note that the clouds are in a collision course so v1 = -v2.

v_cld_1 = np.zeros_like(res)
v_cld_2 = v_cld_1.copy()

v_cld_1[:, 0] = vel_ref
v_cld_2[:, 0] = -vel_ref

vel = np.vstack((v_cld_1, v_cld_2))


dictx = {'r': r_1_2, 'v': vel}

with open('Data.pkl', 'wb') as f:
	pickle.dump(dictx, f)


print('Elapsed time = ', time.time() - TA)

plt.figure(figsize = (14, 6))


m = 1.0/nPart + np.zeros(nPart)

h = do_smoothingX((res, res))
rho = getDensity(res, m, h)

r = (res[:, 0]**2 + res[:, 1]**2 + res[:, 2]**2)

plt.scatter(r, rho, s = 0.1, color = 'k')
#plt.scatter(rcloud_2[:, 0], rcloud_2[:, 1], s = 0.1)
#plt.scatter(res[:, 0], res[:, 1], s = 0.1)

#plt.yscale('log')

plt.show()


