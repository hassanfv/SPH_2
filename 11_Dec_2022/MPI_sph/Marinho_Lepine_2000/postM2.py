
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
from coolHeat2libs import u_to_temp
from numba import njit
import time


mH = 1.6726e-24 # gram
kB = 1.3807e-16  # cm2 g s-2 K-1

#--- Scaling From Marinho et al. (2000)-----
M_sun = 1.989e33 # gram
UnitRadius_in_cm = 3.086e18 # == 1 pc
UnitTime_in_s = 3.16e13 # == 1Myr
UnitTime_in_kyears = UnitTime_in_s / 3600. / 24. / 365.25 / 1000. # in thounsand years unit.
grav_const_in_cgs = 6.67259e-8 #  cm3 g-1 s-2

UnitMass_in_g = UnitRadius_in_cm**3 / grav_const_in_cgs / UnitTime_in_s**2
Unit_u_in_cgs = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3
#-------------------------------------------

print('UnitDensity_in_cgs = ', UnitDensity_in_cgs)



mH = 1.6726e-24 # gram

print('unit number density = ', UnitDensity_in_cgs/mH)

dirx = './Outputs/'

filz = np.sort(glob.glob(dirx+'*.pkl'))

t = 0.0

j = 0

with open(filz[j], 'rb') as f:
	res = pickle.load(f)

r = res['pos']
u = res['u']
rho = res['rho']
nH = rho * UnitDensity_in_cgs / mH
t = res['current_t']

plt.hist(rho, bins = 20)
plt.show()

s()

x_ = r[:, 0]
y_ = r[:, 1]
z_ = r[:, 2]

nz = np.where((z_ >= -0.25) & (z_ <= 0.25))[0]

x = x_#[nz]
y = y_#[nz]
z = z_#[nz]


#print('u = ', np.min(u), np.max(u))
#print('rho = ', np.sort(rho))
#print()


X = 0.75
Y = 0.25
gamma = 5./3.


#***** T_from_u
def T_from_u(rho, u, unit_rho, unit_U):

	i = 0
	
	Tres = np.zeros_like(rho)

	for rhot, ut in zip(rho, u):

		rho_cgs = rhot * unit_rho
		u_cgs = ut * unit_U

		#---- calculating some rough T_init ----
		y = 0.5 # Just approximation to have some estimate of T_init (doesn't need to be exact).
		mu_approx = X/2. + X*y/2. + Y/4.
		T_init = (gamma - 1.0) * mH / kB * mu_approx * u_cgs
		#---------------------------------------
		
		Tres[i] = u_to_temp(u_cgs, T_init, rho_cgs)
		i += 1
		
	return Tres


TA = time.time()
T = T_from_u(rho, u, UnitDensity_in_cgs, Unit_u_in_cgs)
print('TA = ', time.time() - TA)

print('Temp = ', np.sort(T))

plt.hist(T, bins = 20)
plt.show()


nn = np.where(nH > 3000.)[0]

nn = np.where(T > 80.)[0]

xH = x[nn]
yH = y[nn]
zH = z[nn]


plt.figure(figsize = (8, 8))

plt.scatter(x, y, s = 0.1, color = 'black')
plt.scatter(xH, yH, s = 0.50, color = 'red')

plt.title('t = ' + str(np.round(t*UnitTime_in_kyears,2)))

plt.xlim(-24, 24)
plt.ylim(-24, 24)

#plt.xlim(-4, 4)
#plt.ylim(-4, 4)


plt.show()


	


