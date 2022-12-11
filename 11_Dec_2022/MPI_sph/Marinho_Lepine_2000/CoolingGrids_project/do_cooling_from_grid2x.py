
import numpy as np
import pickle


with open('u_after_cool_MPI_dt_0.005.pkl', 'rb') as f:
	data = pickle.load(f)


rho = data[:, 0].reshape(1000, 1000)
uBefore = data[:, 1].reshape(1000, 1000)
uAfter = data[:, 2].reshape(1000, 1000)
dt = data[:, 3].reshape(1000, 1000)

u_uniq = np.unique(uBefore)
rho_uniq = np.unique(rho)

ut = np.linspace(0.1, 10, 20)
rhot = np.linspace(10, 100, 20)


def do_grid_cooling(ut, rhot):

	N = len(ut)
	ures = np.zeros(N)

	for i in range(N):

		#-------- n11 i.e. lower left -----------
		n11 = np.where(u_uniq <= ut[i])[0]
		u11 = np.max(u_uniq[n11])

		n11 = np.where(rho_uniq <= rhot[i])[0]
		rho11 = np.max(rho_uniq[n11])
		#----------------------------------------

		ndx = np.where((uBefore == u11) & (rho == rho11))

		nrow = ndx[0][0]
		ncol = ndx[1][0]

		ures[i] = uAfter[nrow, ncol]

	return ures




uA = do_grid_cooling(ut, rhot)

for i in range(len(ut)):

	print(ut[i], uA[i])





