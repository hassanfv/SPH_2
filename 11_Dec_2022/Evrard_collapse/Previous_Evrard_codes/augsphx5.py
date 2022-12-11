
# The difference with augsphx4.py is that here we use viscosity similar to Gadget 4.

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os
#from cool_libs import *
from numba import jit, prange
import concurrent.futures


#===== getPairwiseSeparations_h
def getPairwiseSeparations_h(r, pos):
    
    M = r.shape[0]
    N = pos.shape[0]
    
    rx = r[:, 0].reshape((M, 1))
    ry = r[:, 1].reshape((M, 1))
    rz = r[:, 2].reshape((M, 1))
    
    posx = pos[:, 0].reshape((N, 1))
    posy = pos[:, 1].reshape((N, 1))
    posz = pos[:, 2].reshape((N, 1))
    
    dx = rx - posx.T
    dy = ry - posy.T
    dz = rz - posz.T
    
    return dx, dy, dz
    


#===== W_I
def W_I(dx, dy, dz, h):

    rr = np.sqrt(dx**2 + dy**2 + dz**2)

    sig = 1.0/np.pi
    q = rr / h

    Wq1 = 1.0 - (3.0/2.0)*q**2 + (3.0/4.0)*q**3
    ntt = np.where(q > 1.0)
    rows = ntt[0]
    cols = ntt[1]
    Wq1[rows, cols] = 0.0


    Wq2 = (1.0/4.0) * (2.0 - q)**3
    ntt = np.where(np.logical_or(q <= 1.0, q > 2.0))
    rows = ntt[0]
    cols = ntt[1]
    Wq2[rows, cols] = 0.0

    return  sig / h**3 * (Wq1 + Wq2)



#===== gradW_I
def gradW_I(dx, dy, dz, h):

    rr = np.sqrt(dx**2 + dy**2 + dz**2)
    
    sig = 1.0/np.pi
    q = rr / h

    ngW1 = sig/h**5 * (-3.0 + 9.0/4.0 * q)
    ntt = np.where(q > 1.0)
    rows = ntt[0]
    cols = ntt[1]
    ngW1[rows, cols] = 0.0

    ngW2 = -3.0*sig/4.0/h**5 * (2.0 - q)**2 / (q+1e-20)
    ntt = np.where(np.logical_or(q <= 1.0, q > 2.0))
    rows = ntt[0]
    cols = ntt[1]
    ngW2[rows, cols] = 0.0

    ngW = (ngW1 + ngW2)

    gWx = ngW * dx
    gWy = ngW * dy
    gWz = ngW * dz

    return gWx, gWy, gWz



#===== getDensity
def getDensity(r, pos, m, h):  # You may want to change your Kernel !!

    hij = 0.5 * (h + h.T)
    
    dx, dy, dz = getPairwiseSeparations_h(r, pos)
    M = r.shape[0]
    
    rho = np.sum(m * W_I(dx, dy, dz, hij), 1).reshape((M, 1))
    
    return rho



#===== PI_ij
def PI_ijTTTT(pos, v, rho, c, m, h, eta, alpha, beta):

    N = pos.shape[0]

    hij = 0.5 * (h + h.T)

    x = pos[:, 0].reshape((N, 1))
    y = pos[:, 1].reshape((N, 1))
    z = pos[:, 2].reshape((N, 1))

    rijx = x - x.T
    rijy = y - y.T
    rijz = z - z.T

    vx = v[:, 0].reshape((N, 1))
    vy = v[:, 1].reshape((N, 1))
    vz = v[:, 2].reshape((N, 1))

    vijx = vx - vx.T
    vijy = vy - vy.T
    vijz = vz - vz.T

    vij_rij = vijx*rijx + vijy*rijy + vijz*rijz
    r2ij = rijx**2 + rijy**2 + rijz**2

    h2ij = hij**2

    mu_ij = hij * vij_rij / (r2ij + h2ij * eta**2)
    mu_ij[mu_ij > 0] = 0.0

    rhoij = (rho + rho.T) / 2.0
    cij = (c.T + c) / 2.0

    PI_ij = (1.0/rhoij) * (-alpha * mu_ij * cij + beta * mu_ij**2)

    return PI_ij




#===== PI_ij Gadget 2
def PI_ij(pos, v, rho, c, m, h, eta, alpha, beta):

    N = pos.shape[0]
    
    hij = 0.5 * (h + h.T)

    x = pos[:, 0].reshape((N, 1))
    y = pos[:, 1].reshape((N, 1))
    z = pos[:, 2].reshape((N, 1))

    rijx = x - x.T
    rijy = y - y.T
    rijz = z - z.T

    vx = v[:, 0].reshape((N, 1))
    vy = v[:, 1].reshape((N, 1))
    vz = v[:, 2].reshape((N, 1))

    vijx = vx - vx.T
    vijy = vy - vy.T
    vijz = vz - vz.T

    vdotr = vijx*rijx + vijy*rijy + vijz*rijz
    r2ij = rijx**2 + rijy**2 + rijz**2
    r = np.sqrt(r2ij)
    
    fac_mu = 1.0
    mu_ij = fac_mu * vdotr / (r+1e-10)
    mu_ij[mu_ij > 0] = 0.0
    
    vsig = c + c.T - 3.0 * mu_ij
    
    rhoij = (rho + rho.T) / 2.0
    vis = 0.50 * alpha * vsig * (-mu_ij) / rhoij
    
    return vis



#===== getPressure
def getPressure(rho, u, gama):

    P = (gama - 1.0) * rho * u.T

    return P


#===== getAcc_sph
def getAcc_sph(pos, v, rho, u, h, m, gama, eta, alpha, beta):    # You may want to change your Kernel !!

    N = pos.shape[0]
    hij = (h + h.T)/2.0
    dx, dy, dz = getPairwiseSeparations_h(pos, pos)

    P = getPressure(rho, u, gama)
    c = np.sqrt(gama * (gama - 1.0) * u)

    PIij = PI_ij(pos, v, rho, c, m, h, eta, alpha, beta)

    gWx, gWy, gWz = gradW_I(dx, dy, dz, hij)

    ax = -np.sum(m * (P/rho**2 + P.T/rho.T**2 + PIij) * gWx, 1).reshape((N, 1))
    ay = -np.sum(m * (P/rho**2 + P.T/rho.T**2 + PIij) * gWy, 1).reshape((N, 1))
    az = -np.sum(m * (P/rho**2 + P.T/rho.T**2 + PIij) * gWz, 1).reshape((N, 1))

    a = np.hstack((ax, ay, az))
    
    return a



#===== get_dU
def get_dU(pos, v, rho, u, m, h, eta, alpha, beta):     # You may want to change your Kernel !!

    N = pos.shape[0]
    hij = (h + h.T)/2.0
    c = np.sqrt(gama * (gama - 1.0) * u)

    PIij = PI_ij(pos, v, rho, c, m, hij, eta, alpha, beta)

    P = getPressure(rho, u, gama)

    vx = v[:, 0].reshape((N, 1))
    vy = v[:, 1].reshape((N, 1))
    vz = v[:, 2].reshape((N, 1))

    dvx = vx - vx.T
    dvy = vy - vy.T
    dvz = vz - vz.T

    dx, dy, dz = getPairwiseSeparations_h(pos, pos)
    gWx, gWy, gWz = gradW_I(dx, dy, dz, hij)

    vij_gWij = dvx*gWx + dvy*gWy + dvz*gWz

    dudt = np.sum(m * (P/rho**2 + PIij/2.0) * vij_gWij, 1)

    return dudt



#===== getEnergy_smth
def getEnergy_smth(pos, v, mass, G, epsilon):

    N = pos.shape[0]

    KE = 0.5 * np.sum(np.sum(mass.reshape((N, 1)) * v**2, 1))

    dx, dy, dz = getPairwiseSeparations_h(pos, pos)

    rr = np.sqrt(dx**2 + dy**2 + dz**2)   # Dont forget -G * m
    inv_r = rr.copy()
    inv_r[inv_r > 0] = 1.0 / (inv_r[inv_r > 0])  # This is the same as (1.0/rr)

    q = rr / epsilon

    nphi1 = (-2.0/epsilon) * ( (1.0/3.0)*q**2 - (3.0/20.0)*q**4 + (1.0/20.0)*q**5 ) + 7.0/5.0/epsilon
    ntt = np.where(np.logical_or(q > 1.0, rr == 0))
    rows = ntt[0]
    cols = ntt[1]
    nphi1[rows, cols] = 0.0


    nphi2 = (-1.0/15.0)*inv_r - (1.0/epsilon) * ( (4.0/3.0)*q**2 - q**3 + (3.0/10.0)*q**4 - (1.0/30.0)*q**5 ) + 8.0/5.0/epsilon
    ntt = np.where(np.logical_or(q <= 1.0, q > 2.0))
    rows = ntt[0]
    cols = ntt[1]
    nphi2[rows, cols] = 0.0

    nphi3 = inv_r
    ntt = np.where(q <= 2.0)
    rows = ntt[0]
    cols = ntt[1]
    nphi3[rows, cols] = 0.0

    nphi = nphi1 + nphi2 + nphi3

    PE = -G * np.sum(np.sum(np.triu(mass.reshape((N, 1)) * mass * nphi), 1))

    return KE, PE



#===== getAcc_g_smth
def getAcc_g_smth(pos, mass, G, epsilon):

    N = pos.shape[0]

    dx, dy, dz = getPairwiseSeparations_h(pos, pos)

    rr = np.sqrt(dx**2 + dy**2 + dz**2)   # Dont forget -G * m
    inv_r3 = rr.copy()
    inv_r3[inv_r3 > 0] = 1.0 / (inv_r3[inv_r3 > 0])**3  # This is the same as (1.0/(rr)**3)

    q = rr / epsilon

    nacc1 = (1.0/epsilon**3) * ( (4.0/3.0) - (6.0/5.0)*q**2 + (1.0/2.0)*q**3 )
    ntt = np.where(np.logical_or(q > 1.0, rr == 0))
    rows = ntt[0]
    cols = ntt[1]
    nacc1[rows, cols] = 0.0


    nacc2 = inv_r3 * ( (-1.0/15.0) + (8.0/3.0)*q**3 - 3.0*q**4 + (6.0/5.0)*q**5 - (1.0/6.0)*q**6 )
    ntt = np.where(np.logical_or(q <= 1.0, q > 2.0))
    rows = ntt[0]
    cols = ntt[1]
    nacc2[rows, cols] = 0.0

    nacc3 = inv_r3
    ntt = np.where(q <= 2.0)
    rows = ntt[0]
    cols = ntt[1]
    nacc3[rows, cols] = 0.0

    nacc = nacc1 + nacc2 + nacc3

    accx = -np.sum(G * mass * nacc * dx, 1).reshape((N, 1))
    accy = -np.sum(G * mass * nacc * dy, 1).reshape((N, 1))
    accz = -np.sum(G * mass * nacc * dz, 1).reshape((N, 1))

    acc = np.hstack((accx, accy, accz))

    return acc



#===== smooth_h
def smooth_h(pos):

    N = pos.shape[0]
    dx, dy, dz = getPairwiseSeparations_h(pos, pos)
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    hres = []

    for j in range(N):

        rrt = dist[j, :]
        rrx = np.sort(rrt)
        hres.append(rrx[64])

    return np.array(hres).reshape((1, N)) * 0.5





np.random.seed(42)

#---- Constants -----------
eta = 0.1
gama = 5.0/3.0
alpha = 1.0
beta = 2.0
G = 1.0
#---------------------------
t = 0.0
dt = 0.01
tEnd = 3.0
Nt = int(np.ceil(tEnd/dt)+1)


filz = np.sort(os.listdir('./Outputs'))
try:
	for k in range(len(filz)):
		os.remove('./Outputs/' + filz[k])
except:
	pass

i = 0


with open('Evrard_1472.pkl', 'rb') as f:
    res = pickle.load(f)
resx = res['x'].reshape((len(res['x']),1))
resy = res['y'].reshape((len(res['x']),1))
resz = res['z'].reshape((len(res['x']),1))


rSPH = np.hstack((resx, resy, resz))
rDM = rSPH.copy()
N = len(rSPH)

epsilonSPH = np.zeros((1, N)) + 0.10
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
u = np.zeros((1, N)) + uFloor # 0.0002405 is equivalent to T = 1e3 K

h = smooth_h(rSPH)                 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
hij = (h.T + h)/2.0
mSPH = np.zeros((1, N)) + MSPH/N
#mDM = np.zeros((1, N)) + MDM/N

r = rSPH #np.vstack((rSPH, rDM))
v = 0.0 * vSPH #np.vstack((vSPH, vDM))
#v = np.random.randn(N, 3) * 0.10
m = mSPH #np.hstack((mSPH, mDM))

rho = getDensity(r[:N, :], r[:N, :], mSPH, h)

acc_g = getAcc_g_smth(r, m, G, epsilon)
acc_sph = getAcc_sph(rSPH, vSPH, rho, u, h, mSPH, gama, eta, alpha, beta)
acc = acc_g.copy()
acc[:N, :] = acc[:N, :] + acc_sph


KE, PE = getEnergy_smth(r, v, m, G, epsilon)
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
ut = get_dU(r[:N, :], v[:N, :], rho, u, mSPH, h, eta, alpha, beta)
ut_previous = ut.copy()

TA = time.time()

i = 0

while t < tEnd:

	TB = time.time()
	
	v += acc * dt/2.0

	r += v * dt

	h = smooth_h(r[:N, :])
	
	rho = getDensity(r[:N, :], r[:N, :], mSPH, h)

	acc_g = getAcc_g_smth(r, m, G, epsilon)
	
	ut = get_dU(r[:N, :], v[:N, :], rho, u, mSPH, h, eta, alpha, beta)
	uold += dt * ut
	u = u_previous + 0.5 * dt * (ut + ut_previous)
	
	print(u)
	print('Current time = ', t)

	u_previous = u.copy()
	ut_previous = ut.copy()

	acc_sph = getAcc_sph(r[:N, :], v[:N, :], rho, u, h, mSPH, gama, eta, alpha, beta)
	acc = acc_g.copy()
	acc[:N, :] = acc[:N, :] + acc_sph

	KE, PE = getEnergy_smth(r, v, m, G, epsilon)
	KE_save[i+1] = KE
	PE_save[i+1] = PE
	
	U_tot = np.sum(m * u)
	U_save[i+1] = U_tot

	v += acc * dt/2.0

	t += dt
	
	i += 1

	# save to files
	dictx = {'pos': r, 'v': v, 'm': m, 'uDirectcool': u, 'dt': dt, 'current_t': t, 'rho': rho}
	with open('./Outputs/' + str(i).zfill(5) + '.pkl', 'wb') as f:
		pickle.dump(dictx, f)
	
	print('TB = ', time.time() - TB)
	print()

E_Dictx = {'U': U_save, 'PE': PE_save, 'KE': KE_save, 't': t_all}
with open('Energy.pkl', 'wb') as f:
	pickle.dump(E_Dictx, f)

print('elapsed time = ', time.time() - TA)






