
import numpy as np
from scipy.integrate import odeint


def dSdx(x, S):
	
	y1, y2 = S
	
	return [y2, -2./x * y2 + np.exp(-y1)]


y1_0 = 0.
y2_0 = 0.
S_0 = (y1_0, y2_0)


x = np.linspace(.00001, 10., 10000)

sol = odeint(dSdx, y0 = S_0, t = x, tfirst = True)

y1_sol = sol.T[0]
y2_sol = sol.T[1]


#----- mu_from_ksi
def mu_from_ksi(x, y2_sol, ksi): #y2_sol is d_psi/d_ksi

	# finding the closest value
	x1 = x - ksi
	nx = np.where(x1 > 0.)[0]
	
	return ksi * ksi * y2_sol[nx[0] - 1]


#----- ksi_from_mu
def ksi_from_mu(x, y2_sol, mu):

	mu1 = x * x * y2_sol - mu
	nx = np.where(mu1 > 0.)[0]
	
	return x[nx[0]-1]


mH = 1.6726e-24 # gram
kB = 1.3807e-16  # cm2 g s-2 K-1

M_sun = 1.989e33 # gram
G = 6.67259e-8 #  cm3 g-1 s-2

ksi_B = 3.
R_cld = 0.8 * 3.086e18 # cm
mH2 = 2.7 * mH

M_sphere = 50.0 * M_sun # in gram

rB = R_cld # Note that rB is the physical radius of the cloud. R_0 is not the physical radius of the cloud See Anathpindika 2009 paper !!!!!!

T_cloud = G * M_sphere * ksi_B * mH2 / (rB * mu_from_ksi(x, y2_sol, ksi_B) * kB)

print('Cloud Temperature = ', T_cloud)



#------ Cloud central density ---------
grav_const_in_cgs = 6.67259e-8 #  cm3 g-1 s-2
M_cld = 50. * M_sun

T_cld = 54.

c_s = (kB*T_cld/mH2)**0.5

muu = mu_from_ksi(x, y2_sol, ksi_B)

rho_c = c_s**6/4./np.pi/grav_const_in_cgs**3/M_cld**2 * muu**2

print('rho_c = ', rho_c)


#------- Cloud Bonnor-Ebert Mass ----------

# Delete this. it is wrong. there should be rho_c in the denominator !!!!M_BE = c_s**3 / (4.*np.pi*grav_const_in_cgs**3)**0.5 * muu**2 

#print('M_BE (in M_sun) = ', M_BE/M_sun)



#---- Numerical solution from https://www.worldscientific.com/doi/10.1142/S0219876207000960
#y1 = 1./6.*x**2 - 1./120.*x**4 + 1./1890.*x**6 - 61./1632960.*x**8 + 629./224532000.*x**10
#y2 = 1./3.*x - 1./30.*x**3 + 1./315.*x**5 - 61./204120.*x**7 + 629./22453200.*x**9
#--------------------------------------
#T_cloud = G * M_sphere * ksi_B * mH2 / (rB * mu_from_ksi(x, y2, ksi_B) * kB)
#print('Cloud Temperature (Guzel et al.) = ', T_cloud)


#----- THE LANE-EMDEN FUNCTION theta_3.25 (Chandrasekhar 1939) ------
#mu_chandra = 1.56442
#T_cloud = G * M_sphere * ksi_B * mH2 / (rB * mu_chandra * kB)
#print('Cloud Temperature (Chandrasekhar) = ', T_cloud)





