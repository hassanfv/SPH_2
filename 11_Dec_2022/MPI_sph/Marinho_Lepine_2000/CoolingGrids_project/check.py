
import numpy as np
import pickle


#-------------
with open('u_rho_vects.pkl', 'rb') as f:
	df = pickle.load(f)

u_vect = df['u_vect']
rho_vect = df['rho_vect']
uZ = df['uZ']
rhoZ = df['rhoZ']
LambdaZ = df['LambdaZ']

def Lambda_from_Grid(ut, rhot):

	n11 = np.where((uZ <= ut) & (rhoZ <= rhot))[0]
	
	return LambdaZ[n11[-1]]
#-------------

ut = 1000.
rhot = 50.

n11 = np.where((uZ <= ut) & (rhoZ <= rhot))[0]

#print(n11)
print(uZ[n11[-1]])
print(rhoZ[n11[-1]])
print()
print(LambdaZ[n11[-1]])
print('***************************')
print(Lambda_from_Grid(ut, rhot))












