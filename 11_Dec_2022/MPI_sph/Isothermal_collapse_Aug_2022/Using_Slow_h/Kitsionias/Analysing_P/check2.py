
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
import time
from numba import njit



#----- P_polytrop_mpi (Anathpindika - 2009 - II)
@njit
def P_polytrop(rho, rho_0, rho_1, c_0, c_s):

	M = len(rho)
	P_res = np.zeros(M)
	c_sound = np.zeros(M)
	
	for i in range(M):
		
		rhot = rho[i]*UnitDensity_in_cgs
		
		if rhot <= rho_0:
			P_res[i] = rhot * c_0*c_0
			c_sound[i] = c_0

		if rhot > rho_0:
			P_res[i] = rhot * ((c_0*c_0 - c_s*c_s)*(rhot/rho_0)**(-2./3.) + c_s*c_s) * (1. + (rhot/rho_1)**(4./3.))**0.5
			c_sound[i] = (((c_0*c_0 - c_s*c_s)*(rhot/rho_0)**(-2./3.) + c_s*c_s) * (1. + (rhot/rho_1)**(4./3.))**0.5)**0.5
	
	P_res = P_res / Unit_P_in_cgs
	c_sound = c_sound / unitVelocity

	return P_res, c_sound


#unitTime_in_Myr =  0.6251515693750652 # Myr
M_sun = 1.989e33 # gram
grav_const_in_cgs = 6.67259e-8 #  cm3 g-1 s-2
Mcld = UnitMass_in_g = 75.0 * M_sun       # !!!!!!!!!!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!
R_0 = 0.58 # see the printed output of step_2_IC_Turner_1995.py
UnitRadius_in_cm = R_0 * 3.086e18  #!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3
unitVelocity = (grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm)**0.5
Unit_u_in_cgs = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm
Unit_P_in_cgs = UnitDensity_in_cgs * Unit_u_in_cgs

c_0 = 0.2 * 1e5 * (Mcld/M_sun)**0.25 # cm/s
c_s = 0.2 * 1e5 # cm/s
rho_1 = 1e-14 # g/cm^3
rho_0 = 1.2e-20


print('Unit_P_in_cgs = ', Unit_P_in_cgs)
print('UnitDensity_in_cgs = ', UnitDensity_in_cgs)
print('Unit_u_in_cgs = ', Unit_u_in_cgs)



filz = np.sort(glob.glob('../Demo/Outputs_demo/*.pkl'))


plt.ion()
fig, ax = plt.subplots(figsize = (7, 6))

kb = ''

for j in range(0, len(filz), 10):

	print('j = ', j)

	with open(filz[j], 'rb') as f:
		data = pickle.load(f)


	r = data['pos']
	h = data['h']
	
	print('h = ', np.sort(h))

	x = r[:, 0]
	y = r[:, 1]
	z = r[:, 2]
	t = data['current_t']
	rho = data['rho']
	unitTime = data['unitTime']
	unitTime_in_Myr = unitTime / 3600. / 24. / 365.25 / 1.e6
	print('rho = ', np.sort(rho))
	#rho = rho*UnitDensity_in_cgs
	#print('rho = ', np.sort(rho))
	
	nnx = np.where(rho*UnitDensity_in_cgs > 4e-19)[0]
	
	ax.cla()

	
	PP, cc = P_polytrop(rho, rho_0, rho_1, c_0, c_s)

	xx = np.arange(len(PP))

	ax.scatter(xx, PP, s = 0.1, color = 'k')
	#ax.scatter(xx, cc, s = 0.1, color = 'b')
	
	if len(nnx) > 0:
		ax.scatter(xx[nnx], PP[nnx], s = 0.1, color = 'red')

	#ax.set_xscale('log')
	#ax.set_yscale('log')
	
	ax.axis(ymin = 0, ymax = 4.0)

	ax.set_title('t = ' + str(np.round(t*unitTime_in_Myr,4)))
	fig.canvas.flush_events()
	time.sleep(0.01)
	
	#if np.round(t*unitTime_in_Myr,2) > 20000:
	kb =readchar.readkey()
	
	if kb == 'q':
		break

plt.savefig('1111.png')





