
import numpy as np
import matplotlib.pyplot as plt
import readchar


def Lambda(T):
	
	# note that 2.0e-26 MUST include both expressions either side of the + sign as I did here ! Good !
	return 2.0e-26 *  (1e7 * np.exp(-1.184e5/(T+1000.)) + 1.4e-2 * np.sqrt(T) * np.exp(-92./T))  # erg/s.cm3, cooling rate


# find T from n (see Vazquez-Semadeni 2007)
def find_T(nden): # nden = number density

	rho = nden # * mH
	T = 1000. # initial guess (to give the code something to start with).
	T_upper = T
	
	if rho*Lambda(T) - Gamma > 0.:

		while rho*Lambda(T_upper) - Gamma > 0.:
			
			T_upper /= 1.1
		
		T_lower = T_upper
		T_upper = T_lower * 1.1


	if rho*Lambda(T) - Gamma < 0.:

		while rho*Lambda(T_upper) - Gamma < 0.:
		
			T_upper *= 1.1
		
		T_lower = T_upper / 1.1
		T_upper = T_upper

	dT = T
	niter = 0
	MAXITER = 100
	
	while (dT/T > 1.e-3) & (niter < MAXITER):
	
		T = 0.5 * (T_upper + T_lower)

		if Gamma - rho * Lambda(T) > 0.:
			T_upper = T
		else:
			T_lower = T
		
		dT = np.abs(T_upper - T_lower)
		
		niter += 1
	
	if niter >= MAXITER:
		print('MAXITER reached ! Failed to converge !!')
	
	return T
	




mH = 1.6726e-24 # gram
kB = 1.3807e-16  # cm2 g s-2 K-1

Gamma = 2.0e-26 # erg/s, Heating rate

n = np.logspace(-2, 3., 1000)

mu = 1. # for pure H gas.

P = []


for nt in n:
	
	T = find_T(nt)
	rho = nt * mH
	
	P.append(kB/mu/mH*T*rho)

P = np.array(P)

P_kB = P / kB

print(find_T(0.5))

plt.plot(n, P_kB)
plt.xscale('log')
plt.yscale('log')
plt.show()









