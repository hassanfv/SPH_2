
import numpy as np
import time
import pickle
import os
from numba import jit, njit

#===== getDensityx
@njit
def getDensity(pos, m, h):  # You may want to change your Kernel !!

	N = pos.shape[0]
	rho = np.zeros(N)

	for i in range(N):

		s = 0.0

		for j in range(N):

			dx = pos[i, 0] - pos[j, 0]
			dy = pos[i, 1] - pos[j, 1]
			dz = pos[i, 2] - pos[j, 2]
			rr = (dx**2 + dy**2 + dz**2)**0.5
			
			hij = 0.5 * (h[i] + h[j])

			sig = 1.0/np.pi
			q = rr / hij

			WIij = 0.0

			if q <= 1.0:
				WIij = sig / hij**3 * (1.0 - (3.0/2.0)*q**2 + (3.0/4.0)*q**3)

			if (q > 1.0) and (q <= 2.0):
				WIij = sig / hij**3 * (1.0/4.0) * (2.0 - q)**3
				
			s += m[j] * WIij

		rho[i] = s

	return rho





#===== getDensity_mpi
@njit
def getDensity_mpi(nbeg, nend, pos, m, h):  # You may want to change your Kernel !!

	N = pos.shape[0]
	M = nend - nbeg
	rho = np.zeros(M)

	for i in range(nbeg, nend):

		s = 0.0

		for j in range(N):

			dx = pos[i, 0] - pos[j, 0]
			dy = pos[i, 1] - pos[j, 1]
			dz = pos[i, 2] - pos[j, 2]
			rr = (dx**2 + dy**2 + dz**2)**0.5
			
			hij = 0.5 * (h[i] + h[j])

			sig = 1.0/np.pi
			q = rr / hij

			WIij = 0.0

			if q <= 1.0:
				WIij = sig / hij**3 * (1.0 - (3.0/2.0)*q**2 + (3.0/4.0)*q**3)

			if (q > 1.0) and (q <= 2.0):
				WIij = sig / hij**3 * (1.0/4.0) * (2.0 - q)**3
				
			s += m[j] * WIij

		rho[i-nbeg] = s

	return rho




#===== getPressure
def getPressure(rho, u, gama):

    P = (gama - 1.0) * rho * u

    return P
    


#===== getAcc_sph (Non-parallel) (with Gadget 2.0 and 4.0 PIij)
@njit
def getAcc_sph(pos, v, rho, P, c, h, m, gama, eta, alpha, beta):

	N = pos.shape[0]
	
	ax = np.zeros(N)
	ay = np.zeros(N)
	az = np.zeros(N)

	for i in range(N):
	
		axt = 0.0
		ayt = 0.0
		azt = 0.0
		for j in range(N):
		
			#----- gradW section -----
			dx = pos[i, 0] - pos[j, 0]
			dy = pos[i, 1] - pos[j, 1]
			dz = pos[i, 2] - pos[j, 2]
			rr = (dx**2 + dy**2 + dz**2)**0.5
			
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
			
			#--------- PIij ----------
			vxij = v[i, 0] - v[j, 0]
			vyij = v[i, 1] - v[j, 1]
			vzij = v[i, 2] - v[j, 2]
			
			vij_rij = vxij*dx + vyij*dy + vzij*dz
			
			cij = 0.5 * (c[i] + c[j])
			
			wij = vij_rij / (rr+1e-5)
			vij_sig = c[i] + c[j] - 3.0 * wij
			rhoij = 0.5 * (rho[i] + rho[j])
			
			PIij = 0.0
			if vij_rij <= 0:
			
				PIij = -0.5 * alpha * vij_sig * wij / rhoij
			#-------------------------
			
			axt -= m[j] * (P[i]/rho[i]**2 + P[j]/rho[j]**2 + PIij) * gWx
			ayt -= m[j] * (P[i]/rho[i]**2 + P[j]/rho[j]**2 + PIij) * gWy
			azt -= m[j] * (P[i]/rho[i]**2 + P[j]/rho[j]**2 + PIij) * gWz
	
		ax[i] = axt
		ay[i] = ayt
		az[i] = azt
	
	ax = ax.reshape((N, 1))
	ay = ay.reshape((N, 1))
	az = az.reshape((N, 1))
	
	a = np.hstack((ax, ay, az))

	return a





#===== getAcc_sph_mpi (with Standard PI_ij (Monaghan & Gingold 1983))
@njit
def getAcc_sph_mpiXXX(nbeg, nend, pos, v, rho, P, c, h, m, gama, eta, alpha, beta):

	N = pos.shape[0]
	
	M = nend - nbeg
	
	ax = np.zeros(M)
	ay = np.zeros(M)
	az = np.zeros(M)

	for i in range(nbeg, nend):
	
		axt = 0.0
		ayt = 0.0
		azt = 0.0
		for j in range(N):
		
			#----- gradW section -----
			dx = pos[i, 0] - pos[j, 0]
			dy = pos[i, 1] - pos[j, 1]
			dz = pos[i, 2] - pos[j, 2]
			rr = (dx**2 + dy**2 + dz**2)**0.5
			
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
			
			#--------- PIij ----------
			vxij = v[i, 0] - v[j, 0]
			vyij = v[i, 1] - v[j, 1]
			vzij = v[i, 2] - v[j, 2]
			
			vij_rij = vxij*dx + vyij*dy + vzij*dz
			
			cij = 0.5 * (c[i] + c[j])
			
			muij = hij * vij_rij / (rr*rr + hij*hij * eta*eta)
			
			rhoij = 0.5 * (rho[i] + rho[j])
			
			PIij = 0.0
			if vij_rij <=0:
			
				PIij = (-alpha * cij * muij + beta * muij*muij) / rhoij
			#-------------------------
			
			axt -= m[j] * (P[i]/rho[i]**2 + P[j]/rho[j]**2 + PIij) * gWx
			ayt -= m[j] * (P[i]/rho[i]**2 + P[j]/rho[j]**2 + PIij) * gWy
			azt -= m[j] * (P[i]/rho[i]**2 + P[j]/rho[j]**2 + PIij) * gWz
	
		ax[i-nbeg] = axt
		ay[i-nbeg] = ayt
		az[i-nbeg] = azt
	
	ax = ax.reshape((M, 1))
	ay = ay.reshape((M, 1))
	az = az.reshape((M, 1))
	
	a = np.hstack((ax, ay, az))

	return a





#===== getAcc_sph_mpi (with PI_ij as in Gadget 2.0 or 4.0)
@njit
def getAcc_sph_mpi(nbeg, nend, pos, v, rho, P, c, h, m, gama, eta, alpha, beta):

	N = pos.shape[0]
	
	M = nend - nbeg
	
	ax = np.zeros(M)
	ay = np.zeros(M)
	az = np.zeros(M)

	for i in range(nbeg, nend):
	
		axt = 0.0
		ayt = 0.0
		azt = 0.0
		for j in range(N):
		
			#----- gradW section -----
			dx = pos[i, 0] - pos[j, 0]
			dy = pos[i, 1] - pos[j, 1]
			dz = pos[i, 2] - pos[j, 2]
			rr = (dx**2 + dy**2 + dz**2)**0.5
			
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
			
			#--------- PIij ----------
			vxij = v[i, 0] - v[j, 0]
			vyij = v[i, 1] - v[j, 1]
			vzij = v[i, 2] - v[j, 2]
			
			vij_rij = vxij*dx + vyij*dy + vzij*dz
			
			cij = 0.5 * (c[i] + c[j])
			
			wij = vij_rij / (rr+1e-5)
			vij_sig = c[i] + c[j] - 3.0 * wij
			rhoij = 0.5 * (rho[i] + rho[j])
			
			PIij = 0.0
			if vij_rij <= 0:
			
				PIij = -0.5 * alpha * vij_sig * wij / rhoij
			#-------------------------
			
			axt -= m[j] * (P[i]/rho[i]**2 + P[j]/rho[j]**2 + PIij) * gWx
			ayt -= m[j] * (P[i]/rho[i]**2 + P[j]/rho[j]**2 + PIij) * gWy
			azt -= m[j] * (P[i]/rho[i]**2 + P[j]/rho[j]**2 + PIij) * gWz
	
		ax[i-nbeg] = axt
		ay[i-nbeg] = ayt
		az[i-nbeg] = azt
	
	ax = ax.reshape((M, 1))
	ay = ay.reshape((M, 1))
	az = az.reshape((M, 1))
	
	a = np.hstack((ax, ay, az))

	return a





#===== get_dU (Non-parallel) (with Gadget 2.0 and 4.0 PIij)
@njit
def get_dU(pos, v, rho, P, c, h, m, gama, eta, alpha, beta):

	N = pos.shape[0]
	
	dudt = np.zeros(N)

	for i in range(N):
		du_t = 0.0
		for j in range(N):
		
			#----- gradW section -----
			dx = pos[i, 0] - pos[j, 0]
			dy = pos[i, 1] - pos[j, 1]
			dz = pos[i, 2] - pos[j, 2]
			rr = (dx**2 + dy**2 + dz**2)**0.5
			
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
			vxij = v[i, 0] - v[j, 0]
			vyij = v[i, 1] - v[j, 1]
			vzij = v[i, 2] - v[j, 2]
			
			vij_rij = vxij*dx + vyij*dy + vzij*dz
			
			cij = 0.5 * (c[i] + c[j])
			
			wij = vij_rij / (rr+1e-5)
			vij_sig = c[i] + c[j] - 3.0 * wij
			rhoij = 0.5 * (rho[i] + rho[j])
			
			PIij = 0.0
			if vij_rij <= 0:
			
				PIij = -0.5 * alpha * vij_sig * wij / rhoij
			#-------------------------
			
			du_t += m[j] * (P[i]/rho[i]**2 + PIij/2.) * vij_gWij

		dudt[i] = du_t
		
	return dudt





#===== get_dU_mpi (with Standard PI_ij (Monaghan & Gingold 1983))
@njit
def get_dU_mpiXXXX(nbeg, nend, pos, v, rho, P, c, h, m, gama, eta, alpha, beta):

	N = pos.shape[0]
	M = nend - nbeg
	dudt = np.zeros(M)

	for i in range(nbeg, nend):
		du_t = 0.0
		for j in range(N):
		
			#----- gradW section -----
			dx = pos[i, 0] - pos[j, 0]
			dy = pos[i, 1] - pos[j, 1]
			dz = pos[i, 2] - pos[j, 2]
			rr = (dx**2 + dy**2 + dz**2)**0.5
			
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

		dudt[i-nbeg] = du_t
		
	return dudt




#===== get_dU_mpi (with PI_ij as in Gadget 2.0 or 4.0)
@njit
def get_dU_mpi(nbeg, nend, pos, v, rho, P, c, h, m, gama, eta, alpha, beta):

	N = pos.shape[0]
	M = nend - nbeg
	dudt = np.zeros(M)

	for i in range(nbeg, nend):
		du_t = 0.0
		for j in range(N):
		
			#----- gradW section -----
			dx = pos[i, 0] - pos[j, 0]
			dy = pos[i, 1] - pos[j, 1]
			dz = pos[i, 2] - pos[j, 2]
			rr = (dx**2 + dy**2 + dz**2)**0.5
			
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
			vxij = v[i, 0] - v[j, 0]
			vyij = v[i, 1] - v[j, 1]
			vzij = v[i, 2] - v[j, 2]
			
			vij_rij = vxij*dx + vyij*dy + vzij*dz
			
			cij = 0.5 * (c[i] + c[j])
			
			wij = vij_rij / (rr+1e-5)
			vij_sig = c[i] + c[j] - 3.0 * wij
			rhoij = 0.5 * (rho[i] + rho[j])
			
			PIij = 0.0
			if vij_rij <= 0:
			
				PIij = -0.5 * alpha * vij_sig * wij / rhoij
			#-------------------------
			
			du_t += m[j] * (P[i]/rho[i]**2 + PIij/2.) * vij_gWij

		dudt[i-nbeg] = du_t
		
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
			
			rr = (dx**2 + dy**2 + dz**2)**0.5				
			
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
	field = np.zeros((N, 3))

	for i in range(N):

		for j in range(i+1, N):

			dx = pos[j, 0] - pos[i, 0]
			dy = pos[j, 1] - pos[i, 1]
			dz = pos[j, 2] - pos[i, 2]
			
			rr = (dx*dx + dy*dy + dz*dz)**0.5

			inv_r3 = 1.0 / rr**3

			epsilonij = 0.5 * (epsilon[i] + epsilon[j])
			q = rr / epsilonij
			
			if q <= 1.0:
				fk = (1.0/epsilonij**3) * ( (4.0/3.0) - (6.0/5.0)*q**2 + (1.0/2.0)*q**3 )

			if (q > 1.) and (q <= 2.):
				fk = inv_r3 * ( (-1.0/15.0) + (8.0/3.0)*q**3 - 3.0*q**4 + (6.0/5.0)*q**5 - (1.0/6.0)*q**6 )

			if q > 2.:
				fk = inv_r3

			field[i, 0] += G * fk * dx * mass[j]
			field[j, 0] -= G * fk * dx * mass[i]
			
			field[i, 1] += G * fk * dy * mass[j]
			field[j, 1] -= G * fk * dy * mass[i]
			
			field[i, 2] += G * fk * dz * mass[j]
			field[j, 2] -= G * fk * dz * mass[i]
	return field




#===== getAcc_g_smth_mpi
@njit
def getAcc_g_smth_mpi(nbeg, nend, pos, mass, G, epsilon): ## NOTE that all particles should have the same mass !!!!!!!!!!!!!!!!!!!!!!!!!

	N = pos.shape[0]
	M = nend - nbeg
	field = np.zeros((M, 3))

	for i in range(nbeg, nend):

		for j in range(N):
		
			if i != j:
		
				dx = pos[i, 0] - pos[j, 0]
				dy = pos[i, 1] - pos[j, 1]
				dz = pos[i, 2] - pos[j, 2]
				
				rr = (dx*dx + dy*dy + dz*dz)**0.5

				inv_r3 = 1.0 / rr**3

				epsilonij = 0.5 * (epsilon[i] + epsilon[j])
				q = rr / epsilonij
				
				if q <= 1.0:
					fk = (1.0/epsilonij**3) * ( (4.0/3.0) - (6.0/5.0)*q**2 + (1.0/2.0)*q**3 )

				if (q > 1.) and (q <= 2.):
					fk = inv_r3 * ( (-1.0/15.0) + (8.0/3.0)*q**3 - 3.0*q**4 + (6.0/5.0)*q**5 - (1.0/6.0)*q**6 )

				if q > 2.:
					fk = inv_r3

				field[i-nbeg, 0] -= G * fk * dx * mass[j]
				field[i-nbeg, 1] -= G * fk * dy * mass[j]				
				field[i-nbeg, 2] -= G * fk * dz * mass[j]

	return field




#===== getAcc_g_smth
@njit
def getAcc_g_smthx(pos, mass, G, epsilon):

	N = pos.shape[0]
	field = np.zeros((N, 3))

	for i in range(N):

		for j in range(N):
		
			if i != j:
		
				dx = pos[i, 0] - pos[j, 0]
				dy = pos[i, 1] - pos[j, 1]
				dz = pos[i, 2] - pos[j, 2]
				
				rr = (dx*dx + dy*dy + dz*dz)**0.5

				inv_r3 = 1.0 / rr**3

				epsilonij = 0.5 * (epsilon[i] + epsilon[j])
				q = rr / epsilonij
				
				if q <= 1.0:
					fk = (1.0/epsilonij**3) * ( (4.0/3.0) - (6.0/5.0)*q**2 + (1.0/2.0)*q**3 )

				if (q > 1.) and (q <= 2.):
					fk = inv_r3 * ( (-1.0/15.0) + (8.0/3.0)*q**3 - 3.0*q**4 + (6.0/5.0)*q**5 - (1.0/6.0)*q**6 )

				if q > 2.:
					fk = inv_r3

				field[i, 0] -= G * fk * dx * mass[j]
				field[i, 1] -= G * fk * dy * mass[j]				
				field[i, 2] -= G * fk * dz * mass[j]

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
            dist[j] = (dx**2 + dy**2 + dz**2)**0.5

        hres.append(np.sort(dist)[50])

    return np.array(hres) * 0.5




#===== smoothing_length_mpi (same as do_smoothingX but modified fpr MPI)
@njit
def smoothing_length_mpi(nbeg, nend, pos):

	N = pos.shape[0]
	M = nend - nbeg
	hres = np.zeros(M)

	for i in range(nbeg, nend):

		dist = np.zeros(N)

		for j in range(N):

		    dx = pos[j, 0] - pos[i, 0]
		    dy = pos[j, 1] - pos[i, 1]
		    dz = pos[j, 2] - pos[i, 2]
		    dist[j] = (dx**2 + dy**2 + dz**2)**0.5

		hres[i-nbeg] = np.sort(dist)[50]

	return hres * 0.5



#===== h_smooth_fast 
#@njit
def h_smooth_fast(pos, h):

	N = pos.shape[0]

	Nth_up = 50 + 5
	Nth_low = 50 - 5

	hres = np.zeros(N)
	
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
		
		niter = 0

		while (Nngb > Nth_up) | (Nngb < Nth_low):
		
			print('i, hi, Nngb (Before) = ', i, hi, Nngb)
		
			if Nngb > Nth_up:

				hi -= 0.003 * hi

			if Nngb < Nth_low:
				
				hi += 0.003 * hi

			Nngb = np.sum(dist < 2.0*hi)
			
			niter += 1
			
			print('i, hi, Nngb (After) = ', i, hi, Nngb)
			print()
			time.sleep(3)
			
			#plt.scatter(pos[:, 0], pos[:, 1], s = 0.1)
			#plt.plot(pos[i, 0], pos[i, 1], marker = 'o')
			#plt.show()
			
			#if i == 0:
			#	print('i, hi, Nngb = ', i, hi, Nngb)
			
			if niter > n_Max_iteration:
				print('i, h[i], hi, Nngb = ', i, h[i], hi, Nngb)
				print('!!!!!! Maximum iteration in h computation reached !!!!!!')
				time.sleep(0.5)

		hres[i] = hi

	return hres





#===== h_smooth_fast_h_minimum_set 
@njit
def h_smooth_fast_h_minimum_set(pos, h, minimum_h):

	N = pos.shape[0]

	Nth_up = 50 + 5
	Nth_low = 50 - 5

	hres = np.zeros(N)
	
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
		
		niter = 0

		Checker = 0

		while ((Nngb > Nth_up) or (Nngb < Nth_low)) & (Checker == 0):
		
			if Nngb > Nth_up:

				hi -= 0.003 * hi

			if Nngb < Nth_low:
				
				hi += 0.003 * hi

			Nngb = np.sum(dist < 2.0*hi)
			
			niter += 1
			
			if (niter > 10) & (hi < minimum_h): # We let it to update itself for 10 iteration. If it is still < minimum_h then we skip it.
				hi = minimum_h
				Checker = 1
			
			if niter > n_Max_iteration:
				print('!!!!!! Maximum iteration in h computation reached !!!!!!')
				print('!!!! This is the h and Nngb responsible for this = ', hi, Nngb)
			

		hres[i] = hi

	return hres








#===== h_smooth_fast_mpi 
@njit
def h_smooth_fast_mpi(nbeg, nend, pos, h):

	N = pos.shape[0]

	Nth_up = 50 + 5
	Nth_low = 50 - 5
	
	n_Max_iteration = 100
	
	M = nend - nbeg
	
	hres = np.zeros(int(M))

	for i in range(nbeg, nend):

		hi = h[i]
		dist = np.zeros(N)

		for j in range(N):

			dx = pos[j, 0] - pos[i, 0]
			dy = pos[j, 1] - pos[i, 1]
			dz = pos[j, 2] - pos[i, 2]
			dist[j] = (dx*dx + dy*dy + dz*dz)**0.5

		Nngb = np.sum(dist < 2.0*hi)

		niter = 0

		while (Nngb > Nth_up) or (Nngb < Nth_low):
		
			if Nngb > Nth_up:

				hi -= 0.003 * hi

			if Nngb < Nth_low:
				
				hi += 0.003 * hi

			Nngb = np.sum(dist < 2.0*hi)
			
			niter += 1
			
			if niter > n_Max_iteration:
				print('!!!!!! Maximum iteration in h computation reached !!!!!!')

		hres[i-nbeg] = hi

	return hres






#===== h_smooth_fast_mpi_min_h_set
@njit
def h_smooth_fast_mpi_min_h_set(nbeg, nend, pos, h, minimum_h):

	N = pos.shape[0]

	Nth_up = 50 + 5
	Nth_low = 50 - 5
	
	n_Max_iteration = 100
	
	M = nend - nbeg
	
	hres = np.zeros(int(M))

	for i in range(nbeg, nend):

		hi = h[i]
		dist = np.zeros(N)

		for j in range(N):

			dx = pos[j, 0] - pos[i, 0]
			dy = pos[j, 1] - pos[i, 1]
			dz = pos[j, 2] - pos[i, 2]
			dist[j] = (dx*dx + dy*dy + dz*dz)**0.5

		Nngb = np.sum(dist < 2.0*hi)

		niter = 0
		
		Checker = 0

		while ((Nngb > Nth_up) or (Nngb < Nth_low)) & (Checker == 0):
		
			if Nngb > Nth_up:

				hi -= 0.003 * hi

			if Nngb < Nth_low:
				
				hi += 0.003 * hi

			Nngb = np.sum(dist < 2.0*hi)
			
			niter += 1
			
			if (niter > 10) & (hi < minimum_h): # We let it to update itself for 10 iteration. If it is still < minimum_h then we skip it.
				hi = minimum_h
				Checker = 1
			
			if niter > n_Max_iteration:
				print('!!!!!! Maximum iteration in h computation reached !!!!!!')
				print('!!!! This is the h and Nngb responsible for this = ', hi, Nngb)

		hres[i-nbeg] = hi

	return hres





#===== getAcc_g_smth_mimj_mpi # Here, the mass of the particles do not necesserily need to be the same i.e. m[i] can be different from m[j] !
@njit
def getAcc_g_smth_mimj_mpi(nbeg, nend, pos, mass, G, epsilon):

	N = pos.shape[0]
	field = np.zeros((N, 3))

	for i in range(nbeg, nend):

		for j in range(i+1, N):

			dx = pos[j, 0] - pos[i, 0]
			dy = pos[j, 1] - pos[i, 1]
			dz = pos[j, 2] - pos[i, 2]
			
			rr = (dx*dx + dy*dy + dz*dz)**0.5

			inv_r3 = 1.0 / rr**3

			epsilonij = 0.5 * (epsilon[i] + epsilon[j])
			q = rr / epsilonij
			
			if q <= 1.0:
				fk = (1.0/epsilonij**3) * ( (4.0/3.0) - (6.0/5.0)*q**2 + (1.0/2.0)*q**3 )

			if (q > 1.) and (q <= 2.):
				fk = inv_r3 * ( (-1.0/15.0) + (8.0/3.0)*q**3 - 3.0*q**4 + (6.0/5.0)*q**5 - (1.0/6.0)*q**6 )

			if q > 2.:
				fk = inv_r3

			field[i, 0] += G * fk * dx * mass[j]
			field[j, 0] -= G * fk * dx * mass[i]
			
			field[i, 1] += G * fk * dy * mass[j]
			field[j, 1] -= G * fk * dy * mass[i]
			
			field[i, 2] += G * fk * dz * mass[j]
			field[j, 2] -= G * fk * dz * mass[i]
	return field


