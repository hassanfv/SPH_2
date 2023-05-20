
# Shear viscosity correction (i.e. Balsara switsch) is incorporated.
# New h algorithm is employed !

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os
from libsx import *
from shear import *

np.random.seed(42)

#---- Constants -----------
eta = 0.1
gama = 5.0/3.0
alpha = 1.0
beta = 2.0
G = 1.0
eps_AGN = 0.05
#---------------------------
t = 0.0
dt = 1e-4
tEnd = 1.0
Nt = int(np.ceil(tEnd/dt)+1)


filz = np.sort(os.listdir('./Outputs'))
try:
	for k in range(len(filz)):
		os.remove('./Outputs/' + filz[k])
except:
	pass


with open('IC.pkl', 'rb') as f:
    data = pickle.load(f)

rG = data['r_gas']
rD = data['r_dm']
vG = data['v_gas']
vD = data['v_dm']
m = data['m']
mBH = data['mBH']
u  = data['u']
epsD = data['eps_dm'] # epsilon (i.e. gravitational softening length) for DM. For gas it will be set equal to the smoothing length h!
h = data['h']
L_Edd = data['L_Edd'] # in energy per unit mass per unit time similar to u !
epsG = h.copy()

epsilon = np.hstack((epsG, epsD))

print('The file is read .....')
print()

NG = rG.shape[0]
ND = rD.shape[0]

r = np.vstack((rG, rD))
N = r.shape[0] # Note that this N does not include one particle, i.e. the BH.

v = np.vstack((vG, vD))

# Inserting a Black Hole at the center
rBH = np.array([[0, 0, 0]])
r = np.vstack((r, rBH))
vBH = np.array([[0, 0, 0]])
v = np.vstack((v, vBH))
print(m.shape)
m = np.hstack((m, np.array([mBH])))
print(m.shape)
print(m)
print(mBH)
epsilon = np.hstack((epsilon, np.array(epsilon[-1])))

hBH = smoothing_BH(rBH, r[:NG])
weights = InjectEnergy_weights(rBH, r[:NG], hBH, h)
weights /= np.sum(weights)
print('hBH = ', hBH)
#-------------------------------------

#-------- rho (GAS) ---------
Trho = time.time()
rho = getDensity(r[:NG], m[:NG], h)
print('Trho = ', time.time() - Trho)
#----------------------

#------- acc_g --------
TG = time.time()
acc_g = getAcc_g_smth(r, m, G, epsilon)
print('TG = ', time.time() - TG)
#----------------------	

#--------- P ----------
P = getPressure(rho, u, gama)
#----------------------

#--------- c ----------
c = np.sqrt(gama * (gama - 1.0) * u)
#----------------------

#--- divV & curlV -----
divV, curlV = div_curlVel(r[:NG], v[:NG], rho, m[:NG], h)
#----------------------

#------ acc_sph -------
acc_sph = getAcc_sph_shear(r[:NG], v[:NG], rho, P, c, h, m[:NG], divV, curlV, alpha)
#----------------------

#-------- acc ---------
acc = acc_g
acc[:NG] = acc[:NG] + acc_sph
#----------------------

#--------- ut ---------
ut = get_dU_shear(r[:NG], v[:NG], rho, P, c, h, m[:NG], divV, curlV, alpha)
#----------------------

#--------- u ----------
u += ut * dt
u += eps_AGN * L_Edd * dt * weights * 1000 # Injecting AGN energy! BH
u_previous = u.copy() # since u_previous and ut_previous is only used in rank = 0, we do not need to broadcast them.
ut_previous = ut.copy()
#----------------------

t = 0.0
ii = 0

TA = time.time()

while t < tEnd:

	TB = time.time()

	#--------- v ----------
	v += acc * dt/2.0
	#----------------------

	#--------- r ----------
	r += v * dt
	r[-1, :] = [0, 0, 0] # BH position must not be updated! We do not move the BH!
	#----------------------

	#--------- h ----------
	h = h_smooth_fast(r[:NG], h)
	hBH = smoothing_BH(rBH, r[:NG])
	weights = InjectEnergy_weights(rBH, r[:NG], hBH, h)
	weights /= np.sum(weights)
	
	nw = np.where(weights > 1e-5)[0]
	print(nw)
	print(len(nw))
	print(weights[nw])
	
	
	x = r[:NG, 0]
	y = r[:NG, 1]
	z = r[:NG, 2]
	dd = (x*x + y*y + z*z)**0.5
	nd = np.argsort(dd)[:64]
	print('d = ', dd[nd])
	print('weights = ', weights[nd])
	print('sum(weights) = ', np.sum(weights[nd]))
	print('u = ', u[nd])
	print('hBH = ', hBH)
	
	#nx = np.argsort(u)[-64:]
	nx = np.argsort(u)[(NG-64):]
	xt = x[nx]
	yt = y[nx]
	zt = z[nx]
	
	if False:
		plt.scatter(x, y, s = 0.1, color = 'k')
		plt.scatter(xt, yt, s = 15, color = 'red')
		plt.scatter(x[nd], y[nd], s = 10, color = 'lime')
		plt.scatter(x[nw], y[nw], s = 4, color = 'blue')
		xy = 0.2
		plt.xlim(-xy, xy)
		plt.ylim(-xy, xy)
		plt.show()
	
	
	
	#----------------------
	
	#-------- rho ---------
	rho = getDensity(r[:NG], m[:NG], h)
	#----------------------
	
	#------- acc_g --------
	epsilon[:NG] = h
	acc_g = getAcc_g_smth(r, m, G, epsilon)
	#----------------------

	#--------- P ----------
	P = getPressure(rho, u, gama)
	#----------------------

	#--------- c ----------
	c = np.sqrt(gama * (gama - 1.0) * u)
	#----------------------

	#--- divV & curlV -----
	divV, curlV = div_curlVel(r[:NG], v[:NG], rho, m[:NG], h)
	#----------------------
	
	#------ acc_sph -------
	acc_sph = getAcc_sph_shear(r[:NG], v[:NG], rho, P, c, h, m[:NG], divV, curlV, alpha)
	#----------------------

	#-------- acc ---------
	acc = acc_g
	acc[:NG, :] = acc[:NG, :] + acc_sph
	#----------------------
	
	#--------- v ----------
	v += acc * dt/2.0
	#----------------------
	
	#--------- ut ---------
	ut = get_dU_shear(r[:NG], v[:NG], rho, P, c, h, m[:NG], divV, curlV, alpha)
	#----------------------

	#--------- u ----------
	u = u_previous + 0.5 * dt * (ut + ut_previous)
	u += eps_AGN * L_Edd * dt * weights * 1000 # Injecting AGN energy! BH
	#u[nw] = u[nw] + eps_AGN * L_Edd * dt * weights[nw]  * 1000 # Injecting AGN energy! BH
	print(np.sort(u))
	u_previous = u.copy() # since u_previous and ut_previous is only used in rank = 0, we do not need to broadcast them.
	ut_previous = ut.copy()
	#----------------------
	
	t += dt
	
	if not (ii%20):
		print('h/c = ', np.sort(h/c))

	print('Loop time (1) = ', time.time() - TB)
	
	print()
	print()
	print()

	ii += 1
	dictx = {'pos': r, 'v': v, 'm': m, 'u': u, 'dt': dt, 'current_t': t, 'rho': rho, 'NG': NG, 'ND': ND}
	with open('./Outputs/' + str(ii).zfill(5) + '.pkl', 'wb') as f:
		pickle.dump(dictx, f)
	
	#print('Loop time (2) = ', time.time() - TB)

print('elapsed time = ', time.time() - TA)




