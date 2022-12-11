

#===== divVel
@njit(parallel=True)
def divVel(gWx, gWy, gWz, v, rho, m):

	N = len(m)
	divV = np.zeros(N)
	
	for i in range(N):
		s = 0.0
		for j in range(N):
			
			vxji = v[j, 0] - v[i, 0]
			vyji = v[j, 1] - v[i, 1]
			vzji = v[j, 2] - v[i, 2]
			
			s += m[j]/rho[i] * (vxji * gWx[i][j] + vyji * gWy[i][j] + vzji * gWz[i][j])
		
		divV[i] = s  #np.abs(s)  # Note that we have one divVel for each particle.

	return divV



#===== curlVel
@njit(parallel=True)
def curlVel(gWx, gWy, gWz, v, rho, m):

	N = len(m)
	curlV = np.zeros(N)
	
	for i in range(N):
		
		curlVx = 0.0
		curlVy = 0.0
		curlVz = 0.0

		for j in range(N):
		
			vxij = v[i, 0] - v[j, 0]
			vyij = v[i, 1] - v[j, 1]
			vzij = v[i, 2] - v[j, 2]
		
			curlVx += m[j]/rho[i] * (vyij * gWz[i][j] - vzij * gWy[i][j])
			curlVy += m[j]/rho[i] * (vzij * gWx[i][j] - vxij * gWz[i][j])
			curlVz += m[j]/rho[i] * (vxij * gWy[i][j] - vyij * gWx[i][j])
			
		curlV[i] = np.sqrt(curlVx * curlVx + curlVy * curlVy + curlVz * curlVz)

	return curlV



#===== f_tilda
@njit(parallel=True)
def f_tilda(divV, curlV, c, h):

	N = len(h)
	
	fij = np.zeros((N, N))

	for i in range(N):
		for j in range(N):
		
			fi = divV[i]/(divV[i] + curlV[i] + 0.0001*c[i]/h[i])
			fj = divV[j]/(divV[j] + curlV[j] + 0.0001*c[j]/h[j])
			
			fij[i][j] = 0.5 * (fi + fj)

	return fij




#===== PIij_tilda
@njit(parallel=True)
def PIij_tilda(fij, PIij):

	N = fij.shape[0]
	PIij_tilda = np.zeros((N, N))

	for i in range(N):
		for j in range(N):
		
			PIij_tilda[i][j] = fij[i][j] * PIij[i][j]

	return PIij_tilda







