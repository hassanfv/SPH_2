
# Here, the main LARGE grid is created. We can use this LARGE grid to create smaller grids by randomly picking some particles. This is
# done in next step, i.e. step_5.py

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import time
import os

TA = time.time()

with open('main_IC_tmp2.pkl', 'rb') as f:
	data = pickle.load(f)

res = data['r']
h = data['h']
rho = data['rho']
rho_0 = data['rho_cen']
c_0 = data['c_0']
gamma = data['gamma']
Rcld_in_cm = data['Rcld_in_cm']
Rcld_in_pc = data['Rcld_in_pc']
Mcld_in_g = data['Mcld_in_g']
muu = data['mu']
grav_const_in_cgs = data['grav_const_in_cgs']

N_uniform = res.shape[0]

#x, y, z = res[:, 0], res[:, 1], res[:, 2]

#------- Prepare the IC to output -------

m = np.ones(N_uniform) / N_uniform

hB = 0.0 #np.median(h) # the two clouds will be separated by 2*hB; Now they are just touching to save time!

res2 = res.copy()
res2[:, 0] += (2.*1.0 + 2.*hB) # 1.0 is the radius of the cloud !


#--- Applying the impact parameter on one of the clouds ---
b_param = 0.4 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
res[:, 1] = -b_param/2. + res[:, 1] # see Turner et al - 1995
res2[:, 1] = b_param/2. + res2[:, 1] # see Turner et al - 1995
#----------------------------------------------------------

res12 = np.vstack((res, res2))

c_s = 0.2 * 1e5 # cm/s See Kitsionas et al. #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Mach = 10.0 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
vel_ref = Mach * c_s/2.0 # Only for Kitsionas et al. it is c_s otherwise maybe use c_0 but check !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

v_cld_1 = np.zeros_like(res)
v_cld_2 = v_cld_1.copy()

v_cld_1[:, 0] = vel_ref
v_cld_2[:, 0] = -vel_ref

vel = np.vstack((v_cld_1, v_cld_2))

h = np.concatenate((h, h))
rho = np.concatenate((rho, rho))

xx, yy = res12[:, 0], res12[:, 1]

print()
print(res.shape)
print(res12.shape)

print('Elapsed time = ', time.time() - TA)

plt.figure(figsize = (14, 6))
plt.scatter(xx, yy, s = 0.01, color = 'k')
plt.savefig('fig.png')
plt.show()

#---- Output to a file ------
num = str(int(np.floor(len(vel)/1000))) # Used for out put file name.
m = np.hstack((m, m))

print(res12.shape)

dictx = {'r': res12, 'v': vel, 'h': h, 'm': m, 'rho': rho, 'rho_cen': rho_0, 'c_0': c_0, 'gamma': gamma,
	 'Rcld_in_pc': Rcld_in_pc, 'Rcld_in_cm': Rcld_in_cm, 'Mcld_in_g': Mcld_in_g, 'mu': muu, 'Mach': Mach,
	 'grav_const_in_cgs': grav_const_in_cgs, }


with open('Main_IC_Grid_' + str(num) + 'k_'+ 'b_' + str(b_param) + '_Mach_' + str(int(Mach)) + '.pkl', 'wb') as f:
	pickle.dump(dictx, f)
#----------------------------
#os.remove('main_IC_tmp2.pkl')




