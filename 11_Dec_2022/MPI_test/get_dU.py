
import numpy as np
from numba import njit

#===== get_dU
@njit
def get_dU(pos, v, rho, P, c, h, m, gama, eta, alpha, beta):

	N = pos.shape[0]
	
	#gWxn, gWyn, gWzn = gradW_I((pos, h))
	
	dudt = np.zeros(N)

	for i in range(N):
		du_t = 0.0
		for j in range(N):
		
			#----- gradW section -----
			dx = pos[i, 0] - pos[j, 0]
			dy = pos[i, 1] - pos[j, 1]
			dz = pos[i, 2] - pos[j, 2]
			rr = np.sqrt(dx**2 + dy**2 + dz**2)
			
			sig = 1.0/np.pi
			hij = 0.5 * (h[i] + h[j])
			q = rr / hij
			
			gWx = gWy = gWz = 0.0 # in case none of the following two if conditions satisfies !
			
			if q <= 1.0:
			
				nW = sig/hij**5 * (-3.0 + 9.0/4.0 * q)
				gWx = nW * dx
				gWy = nW * dy
				gWz = nW * dz
				
			if (q > 1.0) & (q <= 2.0):
				
				nW = -3.0*sig/4.0/hij**5 * (2.0 - q)**2 / (q+1e-20)
				gWx = nW * dx
				gWy = nW * dy
				gWz = nW * dz
			#-------------------------
		
			vxij = v[i, 0] - v[j, 0]
			vyij = v[i, 1] - v[j, 1]
			vzij = v[i, 2] - v[j, 2]
			
			vij_gWij = vxij*gWx + vyij*gWy + vzij*gWz
			
			#--------- PIij ----------
			vij_rij = vxij*dx + vyij*dy + vzij*dz
			
			cij = 0.5 * (c[i] + c[j])
			
			muij = hij * vij_rij / (rr*rr + hij*hij * eta*eta)
			
			rhoij = 0.5 * (rho[i] + rho[j])
			
			PIij = 0.0
			if vij_rij <=0:
			
				PIij = (-alpha * cij * muij + beta * muij*muij) / rhoij
			#-------------------------
			
			du_t += m[j] * (P[i]/rho[i]**2 + PIij/2.) * vij_gWij

		dudt[i] = du_t
		
	return dudt






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


