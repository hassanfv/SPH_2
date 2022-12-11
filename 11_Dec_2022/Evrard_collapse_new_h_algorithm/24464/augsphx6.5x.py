
# New h algorithm is employed !
# The difference with augsphx6.4.py is that here we also use epsilonij instead of epsilon
# The difference with augsphx6.3.py is that here we use hij instead of h

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os
#from cool_libs import *
from numba import jit, njit
import concurrent.futures



#===== W_I
@njit
def W_I(posz): # posz is a tuple containg two arrays: r and h.

	pos = posz[0]
	h = posz[1]
	N = pos.shape[0]
	
	WI = np.zeros((N, N))

	for i in range(N):
		for j in range(N):

			dx = pos[i, 0] - pos[j, 0]
			dy = pos[i, 1] - pos[j, 1]
			dz = pos[i, 2] - pos[j, 2]
			rr = np.sqrt(dx**2 + dy**2 + dz**2)
			
			hij = 0.5 * (h[i] + h[j])

			sig = 1.0/np.pi
			q = rr / hij
			
			if q <= 1.0:
				WI[i][j] = sig / hij**3 * (1.0 - (3.0/2.0)*q**2 + (3.0/4.0)*q**3)

			if (q > 1.0) and (q <= 2.0):
				WI[i][j] = sig / hij**3 * (1.0/4.0) * (2.0 - q)**3

	return  WI


#===== gradW_I
@njit
def gradW_I(posz): # posz is a tuple containg two arrays: r and h.

	pos = posz[0]
	h = posz[1]
	N = pos.shape[0]
	
	gWx = np.zeros((N, N))
	gWy = np.zeros((N, N))
	gWz = np.zeros((N, N))

	for i in range(N):
		for j in range(N):

			dx = pos[i, 0] - pos[j, 0]
			dy = pos[i, 1] - pos[j, 1]
			dz = pos[i, 2] - pos[j, 2]
			rr = np.sqrt(dx**2 + dy**2 + dz**2)

			sig = 1.0/np.pi
			hij = 0.5 * (h[i] + h[j])
			q = rr / hij
			
			if q <= 1.0:
				nW = sig/hij**5 * (-3.0 + 9.0/4.0 * q)
				gWx[i][j] = nW * dx
				gWy[i][j] = nW * dy
				gWz[i][j] = nW * dz

			if (q > 1.0) & (q <= 2.0):
				nW = -3.0*sig/4.0/hij**5 * (2.0 - q)**2 / (q+1e-20)
				gWx[i][j] = nW * dx
				gWy[i][j] = nW * dy
				gWz[i][j] = nW * dz

	return gWx, gWy, gWz



#===== getDensity
def getDensity(r, pos, m, h):  # You may want to change your Kernel !!

    M = r.shape[0]
    
    rho = np.sum(m * W_I((r, h)), 1)
    
    return rho




#===== Standard PI_ij (Monaghan & Gingold 1983)
@njit
def PI_ij(pos, v, rho, c, m, h, eta, alpha, beta):

	N = pos.shape[0]
	
	PIij = np.zeros((N, N))

	for i in range(N):

		for j in range(N):
		
			rijx = pos[i, 0] - pos[j, 0]
			rijy = pos[i, 1] - pos[j, 1]
			rijz = pos[i, 2] - pos[j, 2]
			
			vijx = v[i, 0] - v[j, 0]
			vijy = v[i, 1] - v[j, 1]
			vijz = v[i, 2] - v[j, 2]
			
			vij_rij = vijx*rijx + vijy*rijy + vijz*rijz
			
			rr = np.sqrt(rijx**2 + rijy**2 + rijz**2)
			
			hij = 0.5 * (h[i] + h[j])
			rhoij = 0.5 * (rho[i] + rho[j])
			cij = 0.5 * (c[i] + c[j])
			
			muij = hij * vij_rij / (rr*rr + hij*hij * eta*eta)
			
			if vij_rij <=0:
			
				PIij[i][j] = (-alpha * cij * muij + beta * muij*muij) / rhoij

	return PIij




#===== PI_ij as in Gadget 2.0 or 4.0
@njit
def PI_ijXXXXXX(pos, v, rho, c, m, h, eta, alpha, beta):

	N = pos.shape[0]
	
	PIij = np.zeros((N, N))

	for i in range(N):
		for j in range(N):
		
			rijx = pos[i, 0] - pos[j, 0]
			rijy = pos[i, 1] - pos[j, 1]
			rijz = pos[i, 2] - pos[j, 2]
			
			vijx = v[i, 0] - v[j, 0]
			vijy = v[i, 1] - v[j, 1]
			vijz = v[i, 2] - v[j, 2]
			
			vij_rij = vijx*rijx + vijy*rijy + vijz*rijz
			
			rr = np.sqrt(rijx**2 + rijy**2 + rijz**2)
			
			wij = vij_rij / (rr+1e-15)
			vij_sig = c[i] + c[j] - 3.0 * wij
			rhoij = 0.5 * (rho[i] + rho[j])
			
			if vij_rij < 0:
			
				PIij[i][j] = -0.5 * alpha * vij_sig * wij / rhoij

	return PIij



#===== getPressure
def getPressure(rho, u, gama):

    P = (gama - 1.0) * rho * u

    return P
    


#===== getAcc_sph
@njit
def getAcc_sph(pos, v, rho, P, PIij, h, m, gama, eta, alpha, beta):

	N = pos.shape[0]
	
	gWx, gWy, gWz = gradW_I((pos, h))
	
	ax = np.zeros(N)
	ay = np.zeros(N)
	az = np.zeros(N)

	for i in range(N):
	
		axt = 0.0
		ayt = 0.0
		azt = 0.0
		for j in range(N):
			
			axt -= m[j] * (P[i]/rho[i]**2 + P[j]/rho[j]**2 + PIij[i][j]) * gWx[i][j]
			ayt -= m[j] * (P[i]/rho[i]**2 + P[j]/rho[j]**2 + PIij[i][j]) * gWy[i][j]
			azt -= m[j] * (P[i]/rho[i]**2 + P[j]/rho[j]**2 + PIij[i][j]) * gWz[i][j]
	
		ax[i] = axt
		ay[i] = ayt
		az[i] = azt
	
	ax = ax.reshape((N, 1))
	ay = ay.reshape((N, 1))
	az = az.reshape((N, 1))
	
	a = np.hstack((ax, ay, az))

	return a
	


#===== get_dU
@njit
def get_dU(pos, v, rho, P, PIij, h, m, gama, eta, alpha, beta):

	N = pos.shape[0]
	
	gWx, gWy, gWz = gradW_I((pos, h))
	
	dudt = np.zeros(N)

	for i in range(N):
		du_t = 0.0
		for j in range(N):
		
			vxij = v[i, 0] - v[j, 0]
			vyij = v[i, 1] - v[j, 1]
			vzij = v[i, 2] - v[j, 2]
			
			vij_gWij = vxij*gWx[i][j] + vyij*gWy[i][j] + vzij*gWz[i][j]
			
			du_t += m[j] * (P[i]/rho[i]**2 + PIij[i][j]/2.) * vij_gWij

		dudt[i] = du_t
		
	return dudt



#===== getKE
@njit
def getKE(v, m):

	N = v.shape[0]

	KE = 0.0
	for i in range(N):
		KE += 0.5 * m[i] * (v[i, 0]**2 + v[i, 1]**2 + v[i, 2]**2)
	
	return KE



#===== getPE
@njit
def getPE(pos, m, G, epsilon):

	N = pos.shape[0]

	dx = np.empty(3)
	PE = 0.0

	for i in range(N):
		for j in range(i+1, N):
			
			dx = pos[i, 0] - pos[j, 0]
			dy = pos[i, 1] - pos[j, 1]
			dz = pos[i, 2] - pos[j, 2]
			
			rr = dx**2 + dy**2 + dz**2				
			rr = np.sqrt(rr)
			
			fk = 0.0
			
			if rr != 0.0 :
				inv_r = 1.0 / rr

			epsilonij = 0.5 * (epsilon[i] + epsilon[j])
			q = rr / epsilonij
			
			if (q <= 1.0) & (q != 0.0):
				fk = m[j] * ((-2.0/epsilonij) * ( (1.0/3.0)*q**2 - (3.0/20.0)*q**4 + (1.0/20.0)*q**5 ) + 7.0/5.0/epsilonij)

			if (q > 1.) and (q <= 2.):
				fk = m[j]*((-1.0/15.0)*inv_r - (1.0/epsilonij) * ((4.0/3.0)*q**2 - q**3 + (3.0/10.0)*q**4 - (1.0/30.0)*q**5) + 8.0/5.0/epsilonij)

			if q > 2.:
				fk = m[j] * inv_r

			PE -= G * m[i] * fk

	return PE



#===== getAcc_g_smth
@njit
def getAcc_g_smth(pos, mass, G, epsilon):

	N = pos.shape[0]
	field = np.zeros_like(pos)
	dx = np.empty(3)
	fk = 0.0

	for i in range(N):
		for j in range(i+1, N):
			rr = 0.0
			for k in range(3):
			
				dx[k] = pos[j, k] - pos[i, k]
				rr += dx[k]**2
				
			rr = np.sqrt(rr)
			for k in range(3):

				inv_r3 = 1.0 / rr**3

				epsilonij = 0.5 * (epsilon[i] + epsilon[j])
				q = rr / epsilonij
				
				if q <= 1.0:
					fk = (1.0/epsilonij**3) * ( (4.0/3.0) - (6.0/5.0)*q**2 + (1.0/2.0)*q**3 )

				if (q > 1.) and (q <= 2.):
					fk = inv_r3 * ( (-1.0/15.0) + (8.0/3.0)*q**3 - 3.0*q**4 + (6.0/5.0)*q**5 - (1.0/6.0)*q**6 )

				if q > 2.:
					fk = inv_r3

				field[i, k] += G * fk * dx[k] * mass[j]
				field[j, k] -= G * fk * dx[k] * mass[i]
	return field



#===== smooth_hX (non-parallel)
@njit
def do_smoothingX(poz):

    pos = poz[0]
    subpos = poz[1]

    N = pos.shape[0]
    M = subpos.shape[0]
    hres = []

    for i in range(M):
        dist = np.zeros(N)
        for j in range(N):
        
            dx = pos[j, 0] - subpos[i, 0]
            dy = pos[j, 1] - subpos[i, 1]
            dz = pos[j, 2] - subpos[i, 2]
            dist[j] = np.sqrt(dx**2 + dy**2 + dz**2)

        hres.append(np.sort(dist)[64])

    return np.array(hres) * 0.5




#===== smooth_hX
@njit
def do_smoothing(poz):

    pos = poz[0]
    subpos = poz[1]

    N = pos.shape[0]
    M = subpos.shape[0]
    hres = []

    for i in range(M):
        dist = np.zeros(N)
        for j in range(N):
        
            dx = pos[j, 0] - subpos[i, 0]
            dy = pos[j, 1] - subpos[i, 1]
            dz = pos[j, 2] - subpos[i, 2]
            dist[j] = np.sqrt(dx**2 + dy**2 + dz**2)

        hres.append(np.sort(dist)[50])

    return hres


#===== smoothing_in_parallel
def smooth_h(pos):

	nCPUs = 6
	N = pos.shape[0]
	lenx = int(N / nCPUs)
	posez = []

	for k in range(nCPUs - 1):
		posez.append((pos, pos[(k*lenx):((k+1)*lenx), :]))
	posez.append((pos, pos[((nCPUs-1)*lenx):, :]))
		
	with concurrent.futures.ProcessPoolExecutor() as executor:
		
		res = executor.map(do_smoothing, posez)
		
		out = []
		for ff in res:

			out += ff
		
	out = np.array(out) * 0.5
	
	return out



#===== h_smooth_fast 
@njit
def h_smooth_fast(pos, h):

	N = pos.shape[0]

	Nth_up = 50 + 10.
	Nth_low = 50 - 10.

	hres = np.zeros_like(h)
	
	n_Max_iteration = 100

	for i in range(N):

		hi = h[i]
		dist = np.zeros(N)

		for j in range(N):

			dx = pos[j, 0] - pos[i, 0]
			dy = pos[j, 1] - pos[i, 1]
			dz = pos[j, 2] - pos[i, 2]
			dist[j] = (dx*dx + dy*dy + dz*dz)**0.5

		Nngb = np.sum(dist < 2.0*hi)

		while (Nngb > Nth_up) or (Nngb < Nth_low):
		
			if Nngb > Nth_up:

				hi -= 0.003 * hi

			if Nngb < Nth_low:
				
				hi += 0.003 * hi

			Nngb = np.sum(dist < 2.0*hi)

		hres[i] = hi

	return hres






np.random.seed(42)

#---- Constants -----------
eta = 0.1
gama = 5.0/3.0
alpha = 1.0
beta = 2.0
G = 1.0
#---------------------------
t = 0.0
dt = 0.001
tEnd = 3.0
Nt = int(np.ceil(tEnd/dt)+1)


filz = np.sort(os.listdir('./Outputs'))
try:
	for k in range(len(filz)):
		os.remove('./Outputs/' + filz[k])
except:
	pass


with open('Evrard_24464.pkl', 'rb') as f:   # !!!!!! Change epsilon
    res = pickle.load(f)
resx = res['x'].reshape((len(res['x']),1))
resy = res['y'].reshape((len(res['x']),1))
resz = res['z'].reshape((len(res['x']),1))

print('The file is read .....')
print()

rSPH = np.hstack((resx, resy, resz))
rDM = rSPH.copy()
N = len(rSPH)

epsilonSPH = np.zeros(N) + 0.005
#epsilonDM = np.zeros((1, N)) + 0.20
epsilon = epsilonSPH #np.hstack((epsilonSPH, epsilonDM))


MSPH = 1.0 # total gas mass
#MDM = 0.9 # total DM mass

rr = np.sqrt(resx**2 + resy**2 + resz**2).reshape((1, N))
omega = 0.5 # angular velocity.
vel = rr * omega

sin_T = resy.T / rr
cos_T = resx.T / rr

vx = np.abs(vel * sin_T)
vy = np.abs(vel * cos_T)
vz = 0.0 * vx


nregA = (resx.T >= 0.0) & (resy.T >= 0.0)
vx[nregA] = -vx[nregA]

nregB = (resx.T < 0.0) & (resy.T >= 0.0)
vx[nregB] = -vx[nregB]
vy[nregB] = -vy[nregB]

nregC = (resx.T < 0.0) & (resy.T < 0.0)
vy[nregC] = -vy[nregC]

vSPH = np.hstack((vx.T, vy.T, vz.T))
vDM = vSPH.copy()

uFloor = 0.05 #0.00245 # This is also the initial u.   NOTE to change this in 'do_sth' function too !!!!!!!!!!!!!!!!!!!!!
u = np.zeros(N) + uFloor # 0.0002405 is equivalent to T = 1e3 K

#h = smooth_h(rSPH)                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
h = do_smoothingX((rSPH, rSPH))  # This plays the role of the initial h so that the code can start !
h = h_smooth_fast(rSPH, h)
mSPH = np.zeros(N) + MSPH/N
#mDM = np.zeros((1, N)) + MDM/N

r = rSPH #np.vstack((rSPH, rDM))
v = 0.0 * vSPH #np.vstack((vSPH, vDM))
m = mSPH #np.hstack((mSPH, mDM))

rho = getDensity(r[:N, :], r[:N, :], mSPH, h)

TG = time.time()
acc_g = getAcc_g_smth(r, m, G, epsilon)
print('TG = ', time.time() - TG)

P = getPressure(rho, u, gama)
c = np.sqrt(gama * (gama - 1.0) * u)
PIij = PI_ij(r, v, rho, c, m, h, eta, alpha, beta)
acc_sph = getAcc_sph(rSPH, vSPH, rho, P, PIij, h, m, gama, eta, alpha, beta)
acc = acc_g.copy()
acc[:N, :] = acc[:N, :] + acc_sph

KE = getKE(v, m)
PE = getPE(r, m, G, epsilon)
KE_save = np.zeros((Nt+1))
PE_save = np.zeros((Nt+1))
KE_save[0] = KE
PE_save[0] = PE

U_save = np.zeros(Nt+1)
U_tot = np.sum(m * u)
U_save[0] = U_tot
t_all = np.arange(Nt+1)*dt

t = 0.0

u_previous = u.copy()
uold = u.copy()
ut = get_dU(r[:N, :], v[:N, :], rho, P, PIij, h, m, gama, eta, alpha, beta)
ut_previous = ut.copy()

TA = time.time()

i = 0

while t < tEnd:

	TB = time.time()
	
	v += acc * dt/2.0

	r += v * dt

	T1 = time.time()
	#h = smooth_h(r[:N, :])
	#h = do_smoothingX((r[:N, :], r[:N, :]))
	h = h_smooth_fast(r[:N, :], h)
	print('T1 = ', time.time() - T1)
	
	T2 = time.time()
	rho = getDensity(r[:N, :], r[:N, :], mSPH, h)
	print('T2 = ', time.time() - T2)

	T3 = time.time()
	acc_g = getAcc_g_smth(r, m, G, epsilon)
	print('T3 = ', time.time() - T3)
	
	
	P = getPressure(rho, u, gama)
	c = np.sqrt(gama * (gama - 1.0) * u)
	TP = time.time()
	PIij = PI_ij(r[:N, :], v[:N, :], rho, c, m, h, eta, alpha, beta)
	print('TP = ', time.time() - TP)
	T4 = time.time()
	ut = get_dU(r[:N, :], v[:N, :], rho, P, PIij, h, m, gama, eta, alpha, beta)
	print('T4 = ', time.time() - T4)
	uold += dt * ut
	u = u_previous + 0.5 * dt * (ut + ut_previous)
	#print(u)
	#print(np.sort(u.flatten()))

	u_previous = u.copy()
	ut_previous = ut.copy()

	T5 = time.time()
	acc_sph = getAcc_sph(r[:N, :], v[:N, :], rho, P, PIij, h, m, gama, eta, alpha, beta)
	print('T5 = ', time.time() - T5)
	acc = acc_g.copy()
	acc[:N, :] = acc[:N, :] + acc_sph

	T6 = time.time()
	KE = getKE(v, m)
	PE = getPE(r, m, G, epsilon)
	print('T6 = ', time.time() - T6)
	KE_save[i+1] = KE
	PE_save[i+1] = PE
	
	U_tot = np.sum(m * u)
	U_save[i+1] = U_tot

	v += acc * dt/2.0
	
	print('h/c = ', np.sort(h/c))
	
	print('Current time = ', t)

	t += dt
	
	i += 1

	dictx = {'pos': r, 'v': v, 'm': m, 'uDirectcool': u, 'dt': dt, 'current_t': t, 'rho': rho}
	with open('./Outputs/' + str(i).zfill(5) + '.pkl', 'wb') as f:
		pickle.dump(dictx, f)
	
	print('Loop time = ', time.time() - TB)
	print()

E_Dictx = {'U': U_save, 'PE': PE_save, 'KE': KE_save, 't': t_all}
with open('Energy.pkl', 'wb') as f:
	pickle.dump(E_Dictx, f)

print('elapsed time = ', time.time() - TA)




