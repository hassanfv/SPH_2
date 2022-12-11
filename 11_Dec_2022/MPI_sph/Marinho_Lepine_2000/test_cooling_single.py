
import numpy as np
import pickle


#============= COOLING FROM TABLE ====================
with open('./CoolingGrids_project/u_after_cool_MPI_dt_0.005_N_200.pkl', 'rb') as f:
	data = pickle.load(f)

NN = int(np.sqrt(data.shape[0]))

rhoG = data[:, 0].reshape(NN, NN)
uBefore = data[:, 1].reshape(NN, NN)
uAfter = data[:, 2].reshape(NN, NN)
dt = data[:, 3].reshape(NN, NN)

u_uniq = np.unique(uBefore)
rho_uniq = np.unique(rhoG)


def do_grid_cooling_single(ut, rhot):


	#-------- n11 i.e. lower left -----------
	n11 = np.where(u_uniq <= ut)[0]
	u11 = np.max(u_uniq[n11])

	n11 = np.where(rho_uniq <= rhot)[0]
	rho11 = np.max(rho_uniq[n11])
	#----------------------------------------

	ndx = np.where((uBefore == u11) & (rhoG == rho11))

	nrow = ndx[0][0]
	ncol = ndx[1][0]

	return uAfter[nrow, ncol]

#=====================================================


ut = 1.40
rhot = 0.015

print(do_grid_cooling_single(ut, rhot))






