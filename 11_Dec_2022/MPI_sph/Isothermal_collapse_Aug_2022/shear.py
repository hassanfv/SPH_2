
from numba import njit
import numpy as np


#===== divVel & curlVel
@njit
def div_curlVel(pos, v, rho, m, h):

	N = len(m)
	divV = np.zeros(N)
	curlV = np.zeros(N)
	
	for i in range(N):
	
		s = 0.0
		
		curlVx = 0.0
		curlVy = 0.0
		curlVz = 0.0
		
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
			
			vxji = v[j, 0] - v[i, 0]
			vyji = v[j, 1] - v[i, 1]
			vzji = v[j, 2] - v[i, 2]
			
			s += m[j]/rho[i] * (vxji * gWx + vyji * gWy + vzji * gWz)
			
			vxij = v[i, 0] - v[j, 0]
			vyij = v[i, 1] - v[j, 1]
			vzij = v[i, 2] - v[j, 2]
			
			curlVx += m[j]/rho[i] * (vyij * gWz - vzij * gWy)
			curlVy += m[j]/rho[i] * (vzij * gWx - vxij * gWz)
			curlVz += m[j]/rho[i] * (vxij * gWy - vyij * gWx)
			
		
		divV[i] = np.abs(s)  # Note that we have one divVel for each particle.
		curlV[i] = (curlVx * curlVx + curlVy * curlVy + curlVz * curlVz)**0.5

	return divV, curlV






#===== divVel & curlVel for MPI
@njit
def div_curlVel_mpi(nbeg, nend, pos, v, rho, m, h):

	N = pos.shape[0]
	
	M = nend - nbeg

	divV = np.zeros(M)
	curlV = np.zeros(M)
	
	for i in range(nbeg, nend):
	
		s = 0.0
		
		curlVx = 0.0
		curlVy = 0.0
		curlVz = 0.0
		
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
			
			vxji = v[j, 0] - v[i, 0]
			vyji = v[j, 1] - v[i, 1]
			vzji = v[j, 2] - v[i, 2]
			
			s += m[j]/rho[i] * (vxji * gWx + vyji * gWy + vzji * gWz)
			
			vxij = v[i, 0] - v[j, 0]
			vyij = v[i, 1] - v[j, 1]
			vzij = v[i, 2] - v[j, 2]
			
			curlVx += m[j]/rho[i] * (vyij * gWz - vzij * gWy)
			curlVy += m[j]/rho[i] * (vzij * gWx - vxij * gWz)
			curlVz += m[j]/rho[i] * (vxij * gWy - vyij * gWx)
			
		
		divV[i-nbeg] = np.abs(s)  # Note that we have one divVel for each particle.
		curlV[i-nbeg] = (curlVx * curlVx + curlVy * curlVy + curlVz * curlVz)**0.5

	return divV, curlV






#===== getAcc_sph_shear (Non-parallel) (with Gadget 2.0 and 4.0 PIij)
@njit
def getAcc_sph_shear(pos, v, rho, P, c, h, m, divV, curlV, alpha): # shear viscosity-corrected

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
				
			#---- shear-visc. correction -----
			fi = divV[i]/(divV[i] + curlV[i] + 0.0001*c[i]/h[i])
			fj = divV[j]/(divV[j] + curlV[j] + 0.0001*c[j]/h[j])
			fij = 0.5 * (fi + fj)
			PIij = fij * PIij
			
			#if vij_rij <= 0:
			#	print(fi, fj, PIij)
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





#===== getAcc_sph_mpi (with PI_ij as in Gadget 2.0 or 4.0)
@njit
def getAcc_sph_shear_mpi_Gad_art_vis(nbeg, nend, pos, v, rho, P, c, h, m, divV, curlV, alpha): # shear viscosity-corrected.

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
			
			#---- shear-visc. correction -----
			fi = divV[i]/(divV[i] + curlV[i] + 0.0001*c[i]/h[i])
			fj = divV[j]/(divV[j] + curlV[j] + 0.0001*c[j]/h[j])
			fij = 0.5 * (fi + fj)
			PIij = fij * PIij
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







#===== getAcc_sph_mpi (with Standard PI_ij (Monaghan & Gingold 1983))
@njit
def getAcc_sph_shear_mpi(nbeg, nend, pos, v, rho, P, c, h, m, divV, curlV, alpha, beta, eta): # shear viscosity-corrected.

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
			
			#---- shear-visc. correction -----
			fi = divV[i]/(divV[i] + curlV[i] + 0.0001*c[i]/h[i])
			fj = divV[j]/(divV[j] + curlV[j] + 0.0001*c[j]/h[j])
			fij = 0.5 * (fi + fj)
			PIij = fij * PIij
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





#===== get_dU_shear (Non-parallel) (with Gadget 2.0 and 4.0 PIij)
@njit
def get_dU_shear(pos, v, rho, P, c, h, m, divV, curlV, alpha): # shear-corrected

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

			#---- shear-visc. correction -----
			fi = divV[i]/(divV[i] + curlV[i] + 0.0001*c[i]/h[i])
			fj = divV[j]/(divV[j] + curlV[j] + 0.0001*c[j]/h[j])
			fij = 0.5 * (fi + fj)
			PIij = fij * PIij
			#-------------------------
			
			du_t += m[j] * (P[i]/rho[i]**2 + PIij/2.) * vij_gWij

		dudt[i] = du_t
		
	return dudt





#===== get_dU_shear_mpi (with PI_ij as in Gadget 2.0 or 4.0)
@njit
def get_dU_shear_mpi_Gad_art_vis(nbeg, nend, pos, v, rho, P, c, h, m, divV, curlV, alpha): # shear viscosity-corrected.

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
			
			#---- shear-visc. correction -----
			fi = divV[i]/(divV[i] + curlV[i] + 0.0001*c[i]/h[i])
			fj = divV[j]/(divV[j] + curlV[j] + 0.0001*c[j]/h[j])
			fij = 0.5 * (fi + fj)
			PIij = fij * PIij
			#-------------------------
			
			du_t += m[j] * (P[i]/rho[i]**2 + PIij/2.) * vij_gWij

		dudt[i-nbeg] = du_t
		
	return dudt




#===== get_dU_shear_mpi (with Standard PI_ij (Monaghan & Gingold 1983))
@njit
def get_dU_shear_mpi(nbeg, nend, pos, v, rho, P, c, h, m, divV, curlV, alpha, beta, eta):

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
			
			#---- shear-visc. correction -----
			fi = divV[i]/(divV[i] + curlV[i] + 0.0001*c[i]/h[i])
			fj = divV[j]/(divV[j] + curlV[j] + 0.0001*c[j]/h[j])
			fij = 0.5 * (fi + fj)
			PIij = fij * PIij
			#-------------------------
			
			du_t += m[j] * (P[i]/rho[i]**2 + PIij/2.) * vij_gWij

		dudt[i-nbeg] = du_t
		
	return dudt




