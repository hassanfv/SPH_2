
import numpy as np
import time
from libsx_v2 import *
from shear_itime import *



def core_sph_func(r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph, u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt, nG):

	#--------- v ----------
	v += acc * dt/2.0
	#----------------------

	#--------- r ----------
	r += v * dt
	#----------------------

	#--------- h ----------
	h[nG] = h_smooth_fast_itime(r, h, nG) # !!!!!!! DONE !!!!
	#----------------------

	#-------- rho ---------
	rho[nG] = getDensity_itime(r, m, h, nG) # !!!!!!! DONE !!!!
	#----------------------

	#------- acc_g --------
	acc_g[nG, :] = getAcc_g_smth_itime(r, m, G, epsilon, nG) # !!!!!!! DONE !!!!
	#----------------------
	
	#--------- P ----------
	P[nG] = getPressure(rho[nG], u[nG], gama)
	#----------------------

	#--------- c ----------
	c[nG] = np.sqrt(gama * (gama - 1.0) * u[nG])
	#----------------------
	
	#--- divV & curlV -----
	divV[nG], curlV[nG] = div_curlVel_itime(r, v, rho, m, h, nG) # !!!!!!! DONE !!!!
	#----------------------

	#------ acc_sph -------
	acc_sph[nG, :] = getAcc_sph_shear_itime(r, v, rho, P, c, h, m, divV, curlV, alpha, nG) # !!!!!!! DONE !!!!
	#----------------------

	#-------- acc ---------
	acc = acc_g + acc_sph
	#----------------------

	#--------- v ----------
	v += acc * dt/2.0
	#----------------------

	#--------- ut ---------
	ut[nG] = get_dU_shear_itime(r, v, rho, P, c, h, m, divV, curlV, alpha, nG) # !!!!!!! DONE !!!!
	#----------------------

	#--------- u ----------
	u = u_previous + 0.5 * dt * (ut + ut_previous)
	u_previous = u.copy() # since u_previous and ut_previous is only used in rank = 0, we do not need to broadcast them.
	ut_previous = ut.copy()
	#----------------------

	return r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph, u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt






