
import numpy as np



#***** z_para
def z_para(T):

	NoP = 10
	s = 0.
	for j in range(0, NoP, 2):

		s += (2.*j+1.) * np.exp(-j*(j+1.)*theta_rot/T)

	return s


#***** ln_z_p
def ln_z_p(T):

	NoP = 10
	s = 0.
	for j in range(0, NoP, 2):

		s += (2.*j+1.) * np.exp(-j*(j+1.)*theta_rot/T)

	return np.log(s)


#***** z_ortho
def z_ortho(T):

	NoP = 10
	s = 0.
	for j in range(1, NoP, 2):

		s += (2.*j+1.) * np.exp(-j*(j+1.)*theta_rot/T)

	return s


#***** ln_z_o
def ln_z_o(T):

	NoP = 10
	s = 0.
	for j in range(1, NoP, 2):

		s += (2.*j+1.) * np.exp(-j*(j+1.)*theta_rot/T)

	return np.log(s)


#***** deriv_h2
def deriv_h2(func, x, h):

	df = (func(x+h) - func(x-h)) / 2. / h

	return df


#***** E_H2_func
def E_H2_func(ln_z_p, ln_z_o, T, h):

	f_p = T*T * deriv_h2(ln_z_p, T, h)
	f_o = T*T * deriv_h2(ln_z_o, T, h)
	f_tot = (z_para(T) * f_p + 3. * z_ortho(T) * f_o) / (z_para(T) + 3. * z_ortho(T))

	E_H2 = kB * (3./2. * T + f_tot + theta_vib / (np.exp(theta_vib/T) - 1.))

	return E_H2



theta_rot = 85.4 # K
theta_vib = 6100. # K
kB = 1.3807e-16  # cm2 g s-2 K-1
mH = 1.6726e-24 # gram

h = 0.1

T = 50.


f_p = T*T * deriv_h2(ln_z_p, T, h)
f_o = T*T * deriv_h2(ln_z_o, T, h)

f_tot = (z_para(T) * f_p + 3. * z_ortho(T) * f_o) / (z_para(T) + 3. * z_ortho(T))


E_H2 = kB * (3./2. * T + f_tot + theta_vib / (np.exp(theta_vib/T) - 1.))

print(f_p, f_o, f_tot)
print(E_H2)
print(E_H2_func(ln_z_p, ln_z_o, T, h))















