
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
from coolHeat2libs import u_to_temp
from numba import njit
import time
from libsx import do_smoothingX


mH = 1.6726e-24 # gram
kB = 1.3807e-16  # cm2 g s-2 K-1

#--- Scaling From Marinho et al. (2000)-----
M_sun = 1.989e33 # gram
UnitRadius_in_cm = 3.086e18 # == 1 pc
UnitTime_in_s = 3.16e13 # == 1Myr
grav_const_in_cgs = 6.67259e-8 #  cm3 g-1 s-2

UnitMass_in_g = UnitRadius_in_cm**3 / grav_const_in_cgs / UnitTime_in_s**2
Unit_u_in_cgs = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3
#-------------------------------------------

mH = 1.6726e-24 # gram

print('unit number density = ', UnitDensity_in_cgs/mH)

dirx = './Outputs/'
#dirx = './Outputs_10000/'
dirx = './Outputs_4000/'

filz = np.sort(glob.glob(dirx+'*.pkl'))

plt.figure(figsize = (8, 8))

t = 0.0

j = -1 

with open(filz[j], 'rb') as f:
	res = pickle.load(f)

r = res['pos']
u = res['u']
rho = res['rho']
nH = rho * UnitDensity_in_cgs / mH
dt = res['dt']

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

t += dt

X = 0.75
Y = 0.25
gamma = 5./3.


if False:
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

	TT = time.time()
	T = T_from_u(rho, u, UnitDensity_in_cgs, Unit_u_in_cgs)
	print('TT = ', time.time() - TT)

	print(np.sort(T))

	#plt.hist(T, bins = 20)
	#plt.show()

#---------------------------------------------
#------------- JEANS CREITERION --------------
#---------------------------------------------
G = 1.
gama = 5./3.
h = do_smoothingX((r, r))
c = np.sqrt(gama * (gama - 1.0) * u)

lamb_J = (np.pi * c*c / G / rho)**0.5

#for i in range(len(h)):
#	print(np.round(lamb_J[i], 4), np.round(4.*h[i], 2))

print()

dif = lamb_J / (4.*h)

print('median(dif) = ', np.median(dif))

plt.hist(dif, bins = np.arange(12), density = True)
plt.show()
s()


print(np.sort(dif))

#s()

#---------------------------------------------
#---------------------------------------------
#---------------------------------------------


nn = np.where(nH > 3500.)[0]

nn = np.where(dif <= 1.1)

#nn = np.where(T > 12.)[0]

xH = x[nn]
yH = y[nn]
zH = z[nn]


plt.scatter(y, z, s = 0.1, color = 'black')
plt.scatter(yH, zH, s = 0.50, color = 'lime')

plt.xlim(-24, 24)
plt.ylim(-24, 24)

plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.title('t = ' + str(t))

plt.show()


	


