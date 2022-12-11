

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import pickle
import time

#===== W_I_1D
@njit
def W_I_1D(posz): # posz is a tuple containg two arrays: r and h.

	pos = posz[0]
	h = posz[1]
	N = pos.shape[0]
	
	WI = np.zeros((N, N))

	for i in range(N):
		for j in range(N):

			dx = pos[i] - pos[j]
			rr = np.sqrt(dx**2)
			
			hij = 0.5 * (h[i] + h[j])

			sig = 2. / 3. # for 1D
			q = rr / hij
			
			if q <= 1.0:
				WI[i][j] = sig / hij**1 * (1.0 - (3.0/2.0)*q**2 + (3.0/4.0)*q**3)

			if (q > 1.0) and (q <= 2.0):
				WI[i][j] = sig / hij**1 * (1.0/4.0) * (2.0 - q)**3
	return  WI



#===== gradW_I_1D
@njit(parallel=True)
def gradW_I_1D(posz): # posz is a tuple containg two arrays: r and h.

	pos = posz[0]
	h = posz[1]
	N = pos.shape[0]
	
	gWx = np.zeros((N, N))

	for i in range(N):
		for j in range(N):

			dx = pos[i] - pos[j]
			rr = np.sqrt(dx**2)

			sig = 2. / 3. # for 1D
			hij = 0.5 * (h[i] + h[j])
			q = rr / hij
			
			if q <= 1.0:
				nW = sig/hij**2 * (-3.0 + 9.0/4.0 * q)
				gWx[i][j] = nW * dx

			if (q > 1.0) & (q <= 2.0):
				nW = -3.0*sig/4.0/hij**2 * (2.0 - q)**2 / (q+1e-20)
				gWx[i][j] = nW * dx
	return gWx



#===== PI_ij_1D as in Gadget 2.0 or 4.0
@njit(parallel=True)
def PI_ij_1D(pos, v, rho, c, m, h, eta, alpha, beta): # Note that abs_vij and vij_sig are de-activated here.

	N = pos.shape[0]
	
	PIij = np.zeros((N, N))

	for i in range(N):
		for j in range(N):
		
			rijx = pos[i] - pos[j]
			
			vijx = v[i] - v[j]
			
			vij_rij = vijx*rijx
			
			rr = np.sqrt(rijx**2)
			
			wij = vij_rij / (rr+1e-15)
			vij_sig = c[i] + c[j] - 3.0 * wij
			rhoij = 0.5 * (rho[i] + rho[j])
			
			if vij_rij < 0:
			
				PIij[i][j] = -0.5 * alpha * vij_sig * wij / rhoij

	return PIij #, vij_sig, abs_vij   # vij_sig is needed to calculate the time step with Courant condition.



#===== getAcc_sph_1D
@njit(parallel=True)
def getAcc_sph_1D(pos, v, rho, P, PIij, h, m, gama, eta, alpha, beta):

	N = pos.shape[0]
	
	gWx = gradW_I_1D((pos, h))
	
	ax = np.zeros(N)

	for i in range(N):
	
		axt = 0.0
		for j in range(N):
			
			axt -= m[j] * (P[i]/rho[i]**2 + P[j]/rho[j]**2 + PIij[i][j]) * gWx[i][j]
	
		ax[i] = axt
	
	#ax = ax.reshape((N, 1))
	
	a = ax

	return a



#===== get_dU_1D
@njit(parallel=True)
def get_dU_1D(pos, v, rho, P, PIij, h, m, gama, eta, alpha, beta):

	N = pos.shape[0]
	
	gWx = gradW_I_1D((pos, h))
	
	dudt = np.zeros(N)

	for i in range(N):
		du_t = 0.0
		for j in range(N):
		
			vxij = v[i] - v[j]
			
			vij_gWij = vxij*gWx[i][j]
			
			du_t += m[j] * (P[i]/rho[i]**2 + PIij[i][j]/2.) * vij_gWij

		dudt[i] = du_t
		
	return dudt



#===== smooth_hX_1D (non-parallel)
@njit(parallel=True)
def do_smoothingX_1D(poz):

    pos = poz[0]
    subpos = poz[1]

    N = pos.shape[0]
    M = subpos.shape[0]
    hres = []

    for i in range(M):
        dist = np.zeros(N)
        for j in range(N):
        
            dx = pos[j] - subpos[i]
            dist[j] = np.sqrt(dx**2)

        hres.append(np.sort(dist)[40])

    return np.array(hres) * 0.5


#===== getDensity_1D
def getDensity_1D(r, pos, m, h):  # You may want to change your Kernel !!

    M = r.shape[0]
    
    rho = np.sum(m * W_I_1D((r, h)), 1)
    
    return rho



#===== getPressure
def getPressure(rho, u, gama):

    P = (gama - 1.0) * rho * u

    return P



#---- Constants -----------
eta = 0.001
gama = 5.0/3.0
alpha = 1.0
beta = 1.0
#G = 1.0
#---------------------------


r_left = np.linspace(-1.0, 0, 1000)
dxl = r_left[1] - r_left[0]
r_right = np.linspace(0, 1.0, 1000)[1:]
dxr = r_right[1] - r_right[0]

#------ Left ------
rho_left = np.ones_like(r_left)
P_left   = np.ones_like(r_left)

u_left = P_left / (gama - 1.) / rho_left

h_left = do_smoothingX_1D((r_left, r_left))

m_left = 0.00100 + np.zeros_like(P_left)

rho = getDensity_1D(r_left, r_left, m_left, h_left)

#print(np.min(rho_left), np.max(rho_left), np.median(rho_left))
#print(np.min(rho), np.max(rho), np.median(rho))

#------ Right ------
rho_right = np.zeros_like(r_right) + 0.25
P_right   = np.zeros_like(r_right) + 0.1795

u_right = P_right / (gama - 1.) / rho_right

h_right = do_smoothingX_1D((r_right, r_right))

m_right = 0.000250 + np.zeros_like(P_right)

rho = getDensity_1D(r_right, r_right, m_right, h_right)

print(np.min(rho_right), np.max(rho_right), np.median(rho_right))
print(np.min(rho), np.max(rho), np.median(rho))


r = np.append(r_left, r_right)
m = np.append(m_left, m_right)

rho = np.append(rho_left, rho_right)
P = np.append(P_left, P_right)

u = P / (gama - 1.) / rho

v = np.zeros_like(r)

h = do_smoothingX_1D((r, r))
rho = getDensity_1D(r, r, m, h)

#plt.scatter(rho, rhoX, s = 1)
#plt.show()
#s()


#acc_g = getAcc_g_smth(r, m, G, epsilon)

P = getPressure(rho, u, gama)
c = np.sqrt(gama * (gama - 1.0) * u)

PIij = PI_ij_1D(r, v, rho, c, m, h, eta, alpha, beta)

acc = getAcc_sph_1D(r, v, rho, P, PIij, h, m, gama, eta, alpha, beta)

u_previous = u.copy()
uold = u.copy()
ut = get_dU_1D(r, v, rho, P, PIij, h, m, gama, eta, alpha, beta)
ut_previous = ut.copy()

jj = 0

tEnd = 3.0
t = 0.0

dt = 0.001

DirOut = './Output/'

TA = time.time()
while t < tEnd:

	TB = time.time()

	v += acc * dt/2.0

	r += v * dt
	
	h = do_smoothingX_1D((r, r))
	
	#acc_g = getAcc_g_smth(r, mDYN, G, epsilon)

	rho = getDensity_1D(r, r, m, h)
	P = getPressure(rho, u, gama)
	c = np.sqrt(gama * (gama - 1.0) * u)

	PIij = PI_ij_1D(r, v, rho, c, m, h, eta, alpha, beta)

	ut = get_dU_1D(r, v, rho, P, PIij, h, m, gama, eta, alpha, beta)

	u = u_previous + 0.5 * dt * (ut + ut_previous)

	u_previous = u.copy()
	ut_previous = ut.copy()

	acc = getAcc_sph_1D(r, v, rho, P, PIij, h, m, gama, eta, alpha, beta)

	dictx = {'r': r, 'v': v, 'acc': acc, 'm': m, 'u': u, 'dt': dt, 't': t, 'rho': rho, 'P': P}
	
	with open(DirOut + str(jj).zfill(5) + '.pkl', 'wb') as f:
		pickle.dump(dictx, f)

	jj += 1
	t += dt
	
	print()
	print('Current time = ', t)
	print()











