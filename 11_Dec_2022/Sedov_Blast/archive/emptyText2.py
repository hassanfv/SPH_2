
from numba import njit
import concurrent.futures
import numpy as np

#===== do_acc_sph
#@njit
def do_acc_sph(poz):

	pos = poz[0]; nLow = poz[1]; nUp = poz[2]

	N = nUp - nLow
	M = pos.shape[0]
	
	print(N, M)
	
	ax = np.zeros(N)
	ay = np.zeros(N)
	az = np.zeros(N)

	for i in range(nLow, nUp):

		axt = 0.0
		ayt = 0.0
		azt = 0.0
		for j in range(M):

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




#===== Acc_sph_parallel
def Acc_sph_parallel(pos):

	nCPUs = 2
	N = pos.shape[0]
	lenx = int(N / nCPUs)
	posez = []

	for k in range(nCPUs - 1):
		posez.append((pos, k*lenx, (k+1)*lenx))
	posez.append((pos, (nCPUs-1)*lenx, N))
		
	with concurrent.futures.ProcessPoolExecutor() as executor:
		
		res = executor.map(do_acc_sph, posez)
		
		out = np.empty((0, 3))
		for ff in res:

			out = np.append(out, ff, axis = 0)

	return out






